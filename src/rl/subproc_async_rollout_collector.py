from __future__ import annotations

import multiprocessing as mp
import traceback
from multiprocessing.connection import Connection, wait
from typing import Any

import numpy as np
import torch

from src.rl.observation_spec import PolicyObservationSpec, resolve_policy_observation_spec
from src.runtime.builders import build_actor_critic, build_env
from src.structures.action import NormalizedAction
from src.utils.tensor_utils import observation_to_tensor


def _build_attempt_summary(info, *, policy_forward_s: float, worker_id: int) -> dict:
    trial_metadata = info.extra.get("trial_metadata", {})
    return {
        "valid_for_learning": bool(trial_metadata.get("valid_for_learning", True)),
        "trial_status": str(trial_metadata.get("trial_status", "unknown")),
        "failure_reason": trial_metadata.get("failure_reason"),
        "drop_success_after_live": int(info.drop_success),
        "legacy_drop_success_before": info.extra.get("legacy_drop_success_before"),
        "policy_forward_s": float(policy_forward_s),
        "worker_id": int(worker_id),
    }


def _worker_collect_transition(env, actor_critic, device: torch.device, observation_spec: PolicyObservationSpec) -> dict:
    obs = env.reset()
    obs_tensor = observation_to_tensor(obs, spec=observation_spec).to(device)
    policy_start = mp_context_time()
    with torch.no_grad():
        action_tensor, log_prob, value, _ = actor_critic.act(obs_tensor)
    policy_forward_s = mp_context_time() - policy_start
    record_timing = getattr(env, "record_timing", None)
    if callable(record_timing):
        record_timing("policy_forward_s", policy_forward_s)
    action_np = action_tensor.squeeze(0).detach().cpu().numpy().astype(np.float32)
    next_obs, reward, done, info = env.step(NormalizedAction(value=action_np))
    trial_metadata = info.extra.get("trial_metadata", {})
    return {
        "obs": obs,
        "action": action_np,
        "reward": float(reward),
        "next_obs": next_obs,
        "done": bool(done),
        "log_prob": float(log_prob.squeeze(0).detach().cpu().item()),
        "value": float(value.squeeze(0).detach().cpu().item()),
        "info": info,
        "raw_logit_before": float(info.extra.get("raw_logit_before", obs.raw_stability_logit)),
        "raw_logit_after": float(info.extra.get("raw_logit_after", next_obs.raw_stability_logit)),
        "valid_for_learning": bool(trial_metadata.get("valid_for_learning", True)),
        "policy_forward_s": float(policy_forward_s),
        "attempt_summary": _build_attempt_summary(info, policy_forward_s=policy_forward_s, worker_id=-1),
    }


def _async_rollout_worker(
    conn: Connection,
    worker_id: int,
    num_workers: int,
    env_cfg: dict,
    perception_cfg: dict,
    calibration_cfg: dict,
    actor_critic_cfg: dict,
    observation_spec: PolicyObservationSpec,
    worker_policy_device: str,
    env_factory,
    actor_critic_factory,
    base_seed: int,
):
    env = None
    actor_critic = None
    device = torch.device(worker_policy_device)
    rollout_version = -1
    try:
        worker_seed = int(base_seed) + int(worker_id)
        env_result = env_factory(
            env_cfg=env_cfg,
            perception_cfg=perception_cfg,
            calibration_cfg=calibration_cfg,
            worker_id=worker_id,
            num_workers=num_workers,
            worker_seed=worker_seed,
        )
        env = env_result[0] if isinstance(env_result, tuple) else env_result

        actor_critic = actor_critic_factory(
            perception_cfg=perception_cfg,
            actor_critic_cfg=actor_critic_cfg,
            observation_spec=observation_spec,
        )
        actor_critic.to(device)
        actor_critic.eval()
        conn.send(
            {
                "type": "ready",
                "worker_id": worker_id,
                "device": str(device),
            }
        )

        while True:
            command = conn.recv()
            cmd = command.get("cmd")
            if cmd == "sync_snapshot":
                actor_critic.load_state_dict(command["actor_state"])
                actor_critic.to(device)
                actor_critic.eval()
                sync_fn = getattr(env, "sync_calibrator", None)
                if callable(sync_fn):
                    sync_fn(command["calibrator_state"])
                rollout_version = int(command["rollout_version"])
                conn.send(
                    {
                        "type": "synced",
                        "worker_id": worker_id,
                        "rollout_version": rollout_version,
                    }
                )
                continue

            if cmd == "collect_one":
                transition = _worker_collect_transition(
                    env=env,
                    actor_critic=actor_critic,
                    device=device,
                    observation_spec=observation_spec,
                )
                conn.send(
                    {
                        "type": "transition",
                        "worker_id": worker_id,
                        "rollout_version": rollout_version,
                        "device": str(device),
                        "transition": transition,
                    }
                )
                continue

            if cmd == "debug_state":
                calibrator_state = None
                get_state = getattr(getattr(env, "calibrator", None), "get_state", None)
                if callable(get_state):
                    calibrator_state = get_state()
                debug_snapshot = None
                get_debug_snapshot = getattr(env, "get_debug_snapshot", None)
                if callable(get_debug_snapshot):
                    debug_snapshot = get_debug_snapshot()
                conn.send(
                    {
                        "type": "debug_state",
                        "worker_id": worker_id,
                        "rollout_version": rollout_version,
                        "device": str(device),
                        "calibrator_state": calibrator_state,
                        "debug_snapshot": debug_snapshot,
                    }
                )
                continue

            if cmd == "close":
                conn.send({"type": "closed", "worker_id": worker_id})
                break

            raise ValueError(f"Unsupported worker command: {cmd}")
    except EOFError:
        pass
    except Exception as exc:  # pragma: no cover - exercised by integration behavior
        conn.send(
            {
                "type": "error",
                "worker_id": worker_id,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        if env is not None:
            close_fn = getattr(env, "close", None)
            if callable(close_fn):
                close_fn()
        conn.close()


class SubprocAsyncRolloutCollector:
    """Spawn-based async collector with one env per worker process."""

    def __init__(
        self,
        env_cfg: dict,
        perception_cfg: dict,
        calibration_cfg: dict,
        actor_critic_cfg: dict,
        rl_cfg: dict,
        num_workers: int,
        observation_spec: PolicyObservationSpec | None = None,
        env_factory=build_env,
        actor_critic_factory=build_actor_critic,
    ):
        self.env_cfg = env_cfg
        self.perception_cfg = perception_cfg
        self.calibration_cfg = calibration_cfg
        self.actor_critic_cfg = actor_critic_cfg
        self.rl_cfg = rl_cfg
        self.num_workers = int(num_workers)
        if self.num_workers <= 0:
            raise ValueError("SubprocAsyncRolloutCollector requires at least one worker.")
        self.observation_spec = observation_spec or resolve_policy_observation_spec(perception_cfg, actor_critic_cfg)
        self.env_factory = env_factory
        self.actor_critic_factory = actor_critic_factory
        self.worker_policy_device = str(rl_cfg.get("worker_policy_device", rl_cfg.get("device", "cpu")))
        self.max_collect_attempt_factor = int(rl_cfg.get("max_collect_attempt_factor", 10))
        self.base_seed = int(env_cfg.get("seed", 0))
        self._closed = False

        self._ctx = mp.get_context("spawn")
        self._processes = []
        self._connections = []
        self._connection_to_worker = {}

        for worker_id in range(self.num_workers):
            parent_conn, child_conn = self._ctx.Pipe()
            process = self._ctx.Process(
                target=_async_rollout_worker,
                args=(
                    child_conn,
                    worker_id,
                    self.num_workers,
                    env_cfg,
                    perception_cfg,
                    calibration_cfg,
                    actor_critic_cfg,
                    self.observation_spec,
                    self.worker_policy_device,
                    env_factory,
                    actor_critic_factory,
                    self.base_seed,
                ),
                daemon=True,
            )
            process.start()
            child_conn.close()
            self._processes.append(process)
            self._connections.append(parent_conn)
            self._connection_to_worker[parent_conn] = worker_id

        try:
            for conn in self._connections:
                message = self._recv_checked(conn)
                if message.get("type") != "ready":
                    raise RuntimeError(f"Worker failed during startup: {message}")
        except Exception:
            self.close()
            raise

    def collect_batch(
        self,
        target_valid_episodes: int,
        actor_state: dict,
        calibrator_state: dict,
        obs_spec: PolicyObservationSpec | None,
        rollout_version: int,
    ) -> dict:
        if self._closed:
            raise RuntimeError("Collector is already closed.")
        if target_valid_episodes <= 0:
            return {
                "transitions": [],
                "attempts_total": 0,
                "valid_episodes": 0,
                "attempt_summaries": [],
                "rollout_version": int(rollout_version),
            }
        if obs_spec is not None and obs_spec != self.observation_spec:
            raise ValueError("Collector observation spec differs from the worker observation spec.")

        actor_state_cpu = {
            key: value.detach().cpu().clone() if isinstance(value, torch.Tensor) else value
            for key, value in actor_state.items()
        }
        self._broadcast_snapshot(
            actor_state=actor_state_cpu,
            calibrator_state=calibrator_state,
            rollout_version=rollout_version,
        )

        transitions: list[dict] = []
        attempt_summaries: list[dict] = []
        attempts = 0
        max_attempts = max(target_valid_episodes * self.max_collect_attempt_factor, target_valid_episodes)
        in_flight: set[int] = set()

        def maybe_dispatch(worker_id: int) -> bool:
            nonlocal attempts
            if len(transitions) + len(in_flight) >= target_valid_episodes:
                return False
            if attempts >= max_attempts:
                return False
            self._connections[worker_id].send({"cmd": "collect_one"})
            in_flight.add(worker_id)
            attempts += 1
            return True

        for worker_id in range(self.num_workers):
            if not maybe_dispatch(worker_id):
                break

        while len(transitions) < target_valid_episodes:
            if not in_flight:
                raise RuntimeError(
                    f"Exceeded max rollout collection attempts ({max_attempts}) while collecting valid episodes."
                )
            ready_conns = wait([self._connections[worker_id] for worker_id in in_flight])
            for conn in ready_conns:
                message = self._recv_checked(conn)
                if message.get("type") != "transition":
                    raise RuntimeError(f"Unexpected worker payload during rollout: {message}")
                worker_id = int(message["worker_id"])
                in_flight.remove(worker_id)
                transition = message["transition"]
                transition["worker_id"] = worker_id
                transition["rollout_version"] = int(message["rollout_version"])
                transition["worker_device"] = str(message["device"])
                attempt_summary = dict(transition.get("attempt_summary", {}))
                attempt_summary["worker_id"] = worker_id
                attempt_summaries.append(attempt_summary)
                if bool(transition["valid_for_learning"]):
                    transitions.append(transition)
                if len(transitions) < target_valid_episodes:
                    maybe_dispatch(worker_id)

        while in_flight:
            ready_conns = wait([self._connections[worker_id] for worker_id in in_flight])
            for conn in ready_conns:
                message = self._recv_checked(conn)
                if message.get("type") != "transition":
                    raise RuntimeError(f"Unexpected worker payload while draining in-flight tasks: {message}")
                worker_id = int(message["worker_id"])
                in_flight.remove(worker_id)
                transition = message["transition"]
                attempt_summary = dict(transition.get("attempt_summary", {}))
                attempt_summary["worker_id"] = worker_id
                attempt_summaries.append(attempt_summary)

        return {
            "transitions": transitions,
            "attempts_total": attempts,
            "valid_episodes": len(transitions),
            "attempt_summaries": attempt_summaries,
            "rollout_version": int(rollout_version),
        }

    def get_worker_debug_states(self) -> list[dict]:
        if self._closed:
            return []
        for conn in self._connections:
            conn.send({"cmd": "debug_state"})
        states = []
        for conn in self._connections:
            message = self._recv_checked(conn)
            if message.get("type") != "debug_state":
                raise RuntimeError(f"Unexpected worker debug payload: {message}")
            states.append(message)
        return sorted(states, key=lambda item: int(item["worker_id"]))

    def get_worker_process_states(self) -> list[dict[str, Any]]:
        states = []
        for worker_id, process in enumerate(self._processes):
            states.append(
                {
                    "worker_id": int(worker_id),
                    "pid": None if process.pid is None else int(process.pid),
                    "is_alive": bool(process.is_alive()),
                    "exitcode": None if process.exitcode is None else int(process.exitcode),
                }
            )
        return states

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for conn in self._connections:
            try:
                conn.send({"cmd": "close"})
            except (BrokenPipeError, EOFError, OSError):
                continue
        for conn in self._connections:
            try:
                self._recv_checked(conn, allow_closed=True)
            except Exception:
                pass
            finally:
                conn.close()
        for process in self._processes:
            process.join(timeout=5.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)

    def _broadcast_snapshot(self, actor_state: dict, calibrator_state: dict, rollout_version: int) -> None:
        for conn in self._connections:
            conn.send(
                {
                    "cmd": "sync_snapshot",
                    "actor_state": actor_state,
                    "calibrator_state": calibrator_state,
                    "rollout_version": int(rollout_version),
                }
            )
        for conn in self._connections:
            message = self._recv_checked(conn)
            if message.get("type") != "synced":
                raise RuntimeError(f"Unexpected worker sync payload: {message}")

    def _recv_checked(self, conn: Connection, allow_closed: bool = False) -> dict:
        try:
            message = conn.recv()
        except EOFError as exc:
            raise RuntimeError(self._build_unexpected_close_message(conn)) from exc
        if not isinstance(message, dict):
            raise RuntimeError(f"Worker returned non-dict payload: {message!r}")
        if message.get("type") == "error":
            raise RuntimeError(
                f"Worker {message.get('worker_id')} failed: {message.get('error')}\n{message.get('traceback', '')}"
            )
        if allow_closed and message.get("type") == "closed":
            return message
        return message

    def _build_unexpected_close_message(self, conn: Connection) -> str:
        worker_id = self._connection_to_worker.get(conn)
        if worker_id is None:
            return "Worker pipe closed unexpectedly."

        process = self._processes[int(worker_id)]
        process.join(timeout=0.05)
        worker_state = {
            "worker_id": int(worker_id),
            "pid": None if process.pid is None else int(process.pid),
            "is_alive": bool(process.is_alive()),
            "exitcode": None if process.exitcode is None else int(process.exitcode),
        }
        all_states = self.get_worker_process_states()
        return (
            "Worker pipe closed unexpectedly. "
            f"worker_state={worker_state}; all_worker_states={all_states}"
        )


def mp_context_time() -> float:
    import time

    return time.perf_counter()
