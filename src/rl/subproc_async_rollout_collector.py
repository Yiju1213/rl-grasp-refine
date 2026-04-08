from __future__ import annotations

import multiprocessing as mp
import traceback
from dataclasses import dataclass
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


def _worker_collect_transition(
    env,
    actor_critic,
    device: torch.device,
    observation_spec: PolicyObservationSpec,
    *,
    deterministic_policy: bool,
) -> dict:
    obs = env.reset()
    obs_tensor = observation_to_tensor(obs, spec=observation_spec).to(device)
    policy_start = mp_context_time()
    with torch.no_grad():
        action_tensor, log_prob, value, _ = actor_critic.act(obs_tensor, deterministic=deterministic_policy)
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


def _safe_conn_send(conn: Connection, payload: dict) -> bool:
    try:
        conn.send(payload)
    except (BrokenPipeError, EOFError, OSError):
        return False
    return True


@dataclass
class _WorkerRecord:
    slot_id: int
    generation: int
    process: Any
    conn: Connection
    role: str
    ready: bool = False
    created_rollout_version: int = -1

    def build_process_state(self) -> dict[str, Any]:
        self.process.join(timeout=0.0)
        return {
            "worker_id": int(self.slot_id),
            "slot_id": int(self.slot_id),
            "generation": int(self.generation),
            "role": str(self.role),
            "created_rollout_version": int(self.created_rollout_version),
            "pid": None if self.process.pid is None else int(self.process.pid),
            "is_alive": bool(self.process.is_alive()),
            "exitcode": None if self.process.exitcode is None else int(self.process.exitcode),
        }


def _async_rollout_worker(
    conn: Connection,
    worker_id: int,
    worker_generation: int,
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
            worker_generation=worker_generation,
        )
        env = env_result[0] if isinstance(env_result, tuple) else env_result

        actor_critic = actor_critic_factory(
            perception_cfg=perception_cfg,
            actor_critic_cfg=actor_critic_cfg,
            observation_spec=observation_spec,
        )
        actor_critic.to(device)
        actor_critic.eval()
        if not _safe_conn_send(
            conn,
            {
                "type": "ready",
                "worker_id": worker_id,
                "worker_generation": worker_generation,
                "device": str(device),
            },
        ):
            return

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
                if not _safe_conn_send(
                    conn,
                    {
                        "type": "synced",
                        "worker_id": worker_id,
                        "worker_generation": worker_generation,
                        "rollout_version": rollout_version,
                    },
                ):
                    break
                continue

            if cmd == "collect_one":
                transition = _worker_collect_transition(
                    env=env,
                    actor_critic=actor_critic,
                    device=device,
                    observation_spec=observation_spec,
                    deterministic_policy=bool(command.get("deterministic_policy", False)),
                )
                if not _safe_conn_send(
                    conn,
                    {
                        "type": "transition",
                        "worker_id": worker_id,
                        "rollout_version": rollout_version,
                        "device": str(device),
                        "transition": transition,
                    },
                ):
                    break
                continue

            if cmd == "rebuild_scene":
                rebuild_fn = getattr(env, "rebuild_scene", None)
                if not callable(rebuild_fn):
                    raise RuntimeError("Worker environment does not support scene rebuild.")
                rebuild_start = mp_context_time()
                rebuild_fn()
                if not _safe_conn_send(
                    conn,
                    {
                        "type": "scene_rebuilt",
                        "worker_id": worker_id,
                        "worker_generation": worker_generation,
                        "rebuild_wall_s": mp_context_time() - rebuild_start,
                    },
                ):
                    break
                continue

            if cmd == "reset_sampling_sequence":
                reset_fn = getattr(env, "reset_sampling_sequence", None)
                if callable(reset_fn):
                    reset_fn()
                if not _safe_conn_send(
                    conn,
                    {
                        "type": "sampling_sequence_reset",
                        "worker_id": worker_id,
                        "worker_generation": worker_generation,
                    },
                ):
                    break
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
                if not _safe_conn_send(
                    conn,
                    {
                        "type": "debug_state",
                        "worker_id": worker_id,
                        "worker_generation": worker_generation,
                        "rollout_version": rollout_version,
                        "device": str(device),
                        "calibrator_state": calibrator_state,
                        "debug_snapshot": debug_snapshot,
                    },
                ):
                    break
                continue

            if cmd == "close":
                _safe_conn_send(
                    conn,
                    {
                        "type": "closed",
                        "worker_id": worker_id,
                        "worker_generation": worker_generation,
                    },
                )
                break

            raise ValueError(f"Unsupported worker command: {cmd}")
    except EOFError:
        pass
    except Exception as exc:  # pragma: no cover - exercised by integration behavior
        _safe_conn_send(
            conn,
            {
                "type": "error",
                "worker_id": worker_id,
                "worker_generation": worker_generation,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            },
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
        self.scene_rebuild_every_n_iterations = max(int(rl_cfg.get("scene_rebuild_every_n_iterations", 0)), 0)
        self.worker_recycle_every_n_iterations = max(int(rl_cfg.get("worker_recycle_every_n_iterations", 0)), 0)
        self.worker_recycle_slots_per_event = max(int(rl_cfg.get("worker_recycle_slots_per_event", 1)), 1)
        self.worker_recycle_enable_standby_prefetch = bool(
            rl_cfg.get("worker_recycle_enable_standby_prefetch", True)
        )
        self.worker_recycle_prefetch_count = max(int(rl_cfg.get("worker_recycle_prefetch_count", 1)), 0)
        self._closed = False

        self._ctx = mp.get_context("spawn")
        self._active_workers: list[_WorkerRecord | None] = [None] * self.num_workers
        self._standby_workers: dict[int, _WorkerRecord] = {}
        self._slot_age_order = list(range(self.num_workers))
        self._processes: list[Any] = []
        self._connections: list[Connection] = []
        self._connection_to_worker = {}
        self._connection_to_record: dict[Connection, _WorkerRecord] = {}

        for worker_id in range(self.num_workers):
            self._active_workers[worker_id] = self._spawn_worker(
                slot_id=worker_id,
                generation=0,
                role="active",
                created_rollout_version=-1,
            )
        self._refresh_active_views()

        try:
            for record in self._active_worker_records():
                self._await_worker_ready(record)
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
        reset_worker_sequences: bool = False,
        deterministic_policy: bool = False,
        return_overflow_transitions: bool = False,
        per_worker_dispatch_limits: dict[int, int] | None = None,
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
                "scene_rebuild_performed": 0,
                "scene_rebuild_workers": 0,
                "scene_rebuild_wall_s": 0.0,
                "worker_recycle_performed": 0,
                "worker_recycle_slots": 0,
                "worker_recycle_prefetched": 0,
                "worker_recycle_prefetch_ready": 0,
                "worker_recycle_wall_s": 0.0,
                "worker_recycle_wait_ready_wall_s": 0.0,
                "overflow_transitions": [],
            }
        if obs_spec is not None and obs_spec != self.observation_spec:
            raise ValueError("Collector observation spec differs from the worker observation spec.")

        recycle_metrics = self._maybe_recycle_workers(rollout_version=int(rollout_version))
        recycled_slot_ids = set(recycle_metrics.get("worker_recycle_slot_ids", []))
        scene_rebuild_metrics = self._maybe_rebuild_scenes(
            rollout_version=int(rollout_version),
            excluded_slot_ids=recycled_slot_ids,
        )
        if bool(reset_worker_sequences):
            self._reset_worker_sequences()
        actor_state_cpu = {
            key: value.detach().cpu().clone() if isinstance(value, torch.Tensor) else value
            for key, value in actor_state.items()
        }
        self._broadcast_snapshot(
            actor_state=actor_state_cpu,
            calibrator_state=calibrator_state,
            rollout_version=rollout_version,
        )
        worker_recycle_prefetched = self._maybe_prefetch_standby_workers(rollout_version=int(rollout_version))
        self._poll_standby_ready()

        transitions: list[dict] = []
        overflow_transitions: list[dict] = []
        attempt_summaries: list[dict] = []
        attempts = 0
        max_attempts = max(target_valid_episodes * self.max_collect_attempt_factor, target_valid_episodes)
        in_flight: set[int] = set()
        dispatch_limits = (
            {int(worker_id): max(int(limit), 0) for worker_id, limit in per_worker_dispatch_limits.items()}
            if per_worker_dispatch_limits is not None
            else None
        )
        dispatched_by_worker = {worker_id: 0 for worker_id in range(self.num_workers)}

        def maybe_dispatch(worker_id: int) -> bool:
            nonlocal attempts
            if len(transitions) + len(in_flight) >= target_valid_episodes:
                return False
            if attempts >= max_attempts:
                return False
            if dispatch_limits is not None:
                worker_limit = int(dispatch_limits.get(worker_id, 0))
                if dispatched_by_worker[worker_id] >= worker_limit:
                    return False
            self._connections[worker_id].send(
                {
                    "cmd": "collect_one",
                    "deterministic_policy": bool(deterministic_policy),
                }
            )
            in_flight.add(worker_id)
            attempts += 1
            dispatched_by_worker[worker_id] += 1
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
            self._poll_standby_ready()

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
                if bool(transition["valid_for_learning"]) and bool(return_overflow_transitions):
                    overflow_transition = dict(transition)
                    overflow_transition["worker_id"] = worker_id
                    overflow_transition["rollout_version"] = int(message["rollout_version"])
                    overflow_transition["worker_device"] = str(message["device"])
                    overflow_transitions.append(overflow_transition)
            self._poll_standby_ready()

        return {
            "transitions": transitions,
            "attempts_total": attempts,
            "valid_episodes": len(transitions),
            "attempt_summaries": attempt_summaries,
            "rollout_version": int(rollout_version),
            **recycle_metrics,
            **scene_rebuild_metrics,
            "worker_recycle_prefetched": int(worker_recycle_prefetched),
            "worker_recycle_prefetch_ready": int(self._count_ready_standbys()),
            "overflow_transitions": overflow_transitions,
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
        states = [record.build_process_state() for record in self._active_worker_records()]
        states.extend(record.build_process_state() for record in self._standby_workers.values())
        return sorted(
            states,
            key=lambda item: (0 if item["role"] == "active" else 1, int(item["slot_id"]), int(item["generation"])),
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        records_by_conn: dict[int, _WorkerRecord] = {}
        for record in self._active_worker_records():
            records_by_conn[id(record.conn)] = record
        for record in self._standby_workers.values():
            records_by_conn[id(record.conn)] = record
        for record in records_by_conn.values():
            try:
                self._shutdown_worker(record)
            except Exception:
                pass
        self._active_workers = [None] * self.num_workers
        self._standby_workers.clear()
        self._refresh_active_views()

    def _broadcast_snapshot(self, actor_state: dict, calibrator_state: dict, rollout_version: int) -> None:
        for record in self._active_worker_records():
            record.conn.send(
                {
                    "cmd": "sync_snapshot",
                    "actor_state": actor_state,
                    "calibrator_state": calibrator_state,
                    "rollout_version": int(rollout_version),
                }
            )
        for record in self._active_worker_records():
            message = self._recv_checked(record.conn)
            if message.get("type") != "synced":
                raise RuntimeError(f"Unexpected worker sync payload: {message}")

    def _maybe_rebuild_scenes(self, *, rollout_version: int, excluded_slot_ids: set[int] | None = None) -> dict[str, float]:
        interval = int(self.scene_rebuild_every_n_iterations)
        if interval <= 0 or rollout_version <= 0 or (rollout_version % interval) != 0:
            return {
                "scene_rebuild_performed": 0,
                "scene_rebuild_workers": 0,
                "scene_rebuild_wall_s": 0.0,
            }
        rebuild_records = [
            record
            for record in self._active_worker_records()
            if excluded_slot_ids is None or record.slot_id not in excluded_slot_ids
        ]
        if not rebuild_records:
            return {
                "scene_rebuild_performed": 0,
                "scene_rebuild_workers": 0,
                "scene_rebuild_wall_s": 0.0,
            }
        rebuild_start = mp_context_time()
        for record in rebuild_records:
            record.conn.send({"cmd": "rebuild_scene"})
        rebuilt_workers = 0
        for record in rebuild_records:
            message = self._recv_checked(record.conn)
            if message.get("type") != "scene_rebuilt":
                raise RuntimeError(f"Unexpected worker rebuild payload: {message}")
            rebuilt_workers += 1
        return {
            "scene_rebuild_performed": 1,
            "scene_rebuild_workers": rebuilt_workers,
            "scene_rebuild_wall_s": float(mp_context_time() - rebuild_start),
        }

    def _maybe_recycle_workers(self, *, rollout_version: int) -> dict[str, Any]:
        interval = int(self.worker_recycle_every_n_iterations)
        if interval <= 0 or rollout_version <= 0 or (rollout_version % interval) != 0:
            return {
                "worker_recycle_performed": 0,
                "worker_recycle_slots": 0,
                "worker_recycle_wall_s": 0.0,
                "worker_recycle_wait_ready_wall_s": 0.0,
                "worker_recycle_slot_ids": [],
            }

        target_slot_ids = self._oldest_slot_ids(limit=self.worker_recycle_slots_per_event)
        if not target_slot_ids:
            return {
                "worker_recycle_performed": 0,
                "worker_recycle_slots": 0,
                "worker_recycle_wall_s": 0.0,
                "worker_recycle_wait_ready_wall_s": 0.0,
                "worker_recycle_slot_ids": [],
            }

        recycle_start = mp_context_time()
        wait_ready_wall_s = 0.0
        for slot_id in target_slot_ids:
            active_record = self._require_active_record(slot_id)
            standby_record = self._ensure_standby_record(
                slot_id=slot_id,
                generation=int(active_record.generation) + 1,
                created_rollout_version=int(rollout_version),
            )
            wait_ready_wall_s += self._await_worker_ready(standby_record)
            self._shutdown_worker(active_record)
            standby_record.role = "active"
            standby_record.created_rollout_version = int(rollout_version)
            self._active_workers[slot_id] = standby_record
            self._standby_workers.pop(slot_id, None)
            self._move_slot_to_newest(slot_id)

        self._refresh_active_views()
        return {
            "worker_recycle_performed": 1,
            "worker_recycle_slots": len(target_slot_ids),
            "worker_recycle_wall_s": float(mp_context_time() - recycle_start),
            "worker_recycle_wait_ready_wall_s": float(wait_ready_wall_s),
            "worker_recycle_slot_ids": list(target_slot_ids),
        }

    def _maybe_prefetch_standby_workers(self, *, rollout_version: int) -> int:
        interval = int(self.worker_recycle_every_n_iterations)
        if interval <= 0 or not self.worker_recycle_enable_standby_prefetch:
            return 0
        if self.worker_recycle_prefetch_count <= 0:
            return 0

        next_rollout_version = int(rollout_version) + 1
        if next_rollout_version <= 0 or (next_rollout_version % interval) != 0:
            return 0

        target_slot_ids = self._oldest_slot_ids(limit=self.worker_recycle_slots_per_event)
        if not target_slot_ids:
            return 0

        prefetch_limit = min(len(target_slot_ids), self.worker_recycle_prefetch_count)
        existing_prefetch = sum(1 for slot_id in target_slot_ids if slot_id in self._standby_workers)
        spawn_budget = max(prefetch_limit - existing_prefetch, 0)
        spawned = 0
        for slot_id in target_slot_ids:
            if spawned >= spawn_budget:
                break
            if slot_id in self._standby_workers:
                continue
            active_record = self._require_active_record(slot_id)
            self._standby_workers[slot_id] = self._spawn_worker(
                slot_id=slot_id,
                generation=int(active_record.generation) + 1,
                role="standby",
                created_rollout_version=int(rollout_version),
            )
            spawned += 1
        return int(spawned)

    def _poll_standby_ready(self) -> int:
        pending_conns = [record.conn for record in self._standby_workers.values() if not record.ready]
        if not pending_conns:
            return 0
        ready_conns = wait(pending_conns, timeout=0.0)
        newly_ready = 0
        for conn in ready_conns:
            record = self._connection_to_record.get(conn)
            if record is None or record.ready:
                continue
            self._await_worker_ready(record)
            newly_ready += 1
        return newly_ready

    def _count_ready_standbys(self) -> int:
        return sum(1 for record in self._standby_workers.values() if record.ready)

    def _reset_worker_sequences(self) -> None:
        for record in self._active_worker_records():
            record.conn.send({"cmd": "reset_sampling_sequence"})
        for record in self._active_worker_records():
            message = self._recv_checked(record.conn)
            if message.get("type") != "sampling_sequence_reset":
                raise RuntimeError(f"Unexpected worker sampling reset payload: {message}")

    def _spawn_worker(
        self,
        *,
        slot_id: int,
        generation: int,
        role: str,
        created_rollout_version: int,
    ) -> _WorkerRecord:
        parent_conn, child_conn = self._ctx.Pipe()
        process = self._ctx.Process(
            target=_async_rollout_worker,
            args=(
                child_conn,
                slot_id,
                generation,
                self.num_workers,
                self.env_cfg,
                self.perception_cfg,
                self.calibration_cfg,
                self.actor_critic_cfg,
                self.observation_spec,
                self.worker_policy_device,
                self.env_factory,
                self.actor_critic_factory,
                self.base_seed,
            ),
            daemon=True,
        )
        process.start()
        child_conn.close()
        record = _WorkerRecord(
            slot_id=int(slot_id),
            generation=int(generation),
            process=process,
            conn=parent_conn,
            role=str(role),
            ready=False,
            created_rollout_version=int(created_rollout_version),
        )
        self._connection_to_worker[parent_conn] = int(slot_id)
        self._connection_to_record[parent_conn] = record
        return record

    def _ensure_standby_record(self, *, slot_id: int, generation: int, created_rollout_version: int) -> _WorkerRecord:
        existing = self._standby_workers.get(slot_id)
        if existing is not None and int(existing.generation) == int(generation):
            return existing
        if existing is not None:
            self._shutdown_worker(existing)
        standby_record = self._spawn_worker(
            slot_id=slot_id,
            generation=generation,
            role="standby",
            created_rollout_version=created_rollout_version,
        )
        self._standby_workers[slot_id] = standby_record
        return standby_record

    def _await_worker_ready(self, record: _WorkerRecord) -> float:
        if record.ready:
            return 0.0
        ready_start = mp_context_time()
        message = self._recv_checked(record.conn)
        if message.get("type") != "ready":
            raise RuntimeError(f"Worker failed during startup: {message}")
        if int(message.get("worker_id", -1)) != int(record.slot_id):
            raise RuntimeError(f"Worker ready payload slot mismatch: {message}")
        if int(message.get("worker_generation", -1)) != int(record.generation):
            raise RuntimeError(f"Worker ready payload generation mismatch: {message}")
        record.ready = True
        return float(mp_context_time() - ready_start)

    def _shutdown_worker(self, record: _WorkerRecord) -> None:
        conn = record.conn
        process = record.process
        if record.ready:
            try:
                conn.send({"cmd": "close"})
                self._recv_checked(conn, allow_closed=True)
            except Exception:
                pass
        self._connection_to_worker.pop(conn, None)
        self._connection_to_record.pop(conn, None)
        try:
            conn.close()
        except OSError:
            pass
        process.join(timeout=5.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=1.0)

    def _active_worker_records(self) -> list[_WorkerRecord]:
        return [record for record in self._active_workers if record is not None]

    def _require_active_record(self, slot_id: int) -> _WorkerRecord:
        record = self._active_workers[int(slot_id)]
        if record is None:
            raise RuntimeError(f"Active worker slot {slot_id} is not initialized.")
        return record

    def _refresh_active_views(self) -> None:
        self._connections = [record.conn for record in self._active_worker_records()]
        self._processes = [record.process for record in self._active_worker_records()]

    def _oldest_slot_ids(self, *, limit: int) -> list[int]:
        return [int(slot_id) for slot_id in self._slot_age_order[: max(min(int(limit), self.num_workers), 0)]]

    def _move_slot_to_newest(self, slot_id: int) -> None:
        self._slot_age_order = [item for item in self._slot_age_order if int(item) != int(slot_id)]
        self._slot_age_order.append(int(slot_id))

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
        record = self._connection_to_record.get(conn)
        if record is None:
            return "Worker pipe closed unexpectedly."

        record.process.join(timeout=0.05)
        worker_state = record.build_process_state()
        all_states = self.get_worker_process_states()
        return (
            "Worker pipe closed unexpectedly. "
            f"worker_state={worker_state}; all_worker_states={all_states}"
        )


def mp_context_time() -> float:
    import time

    return time.perf_counter()
