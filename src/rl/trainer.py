from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch

from src.rl.advantage import compute_returns_and_advantages
from src.rl.rollout_buffer import RolloutBuffer
from src.structures.action import NormalizedAction
from src.utils.system_diagnostics import collect_system_metrics
from src.utils.tensor_utils import action_tensor_to_numpy, observation_to_tensor


class Trainer:
    """Coordinate rollout collection, PPO updates, and calibrator updates."""

    def __init__(
        self,
        env,
        actor_critic,
        agent,
        buffer,
        calibrator,
        logger,
        cfg: dict,
        collector=None,
        validation_env=None,
        validation_collector=None,
        validation_cfg: dict | None = None,
    ):
        self.env = env
        self.actor_critic = actor_critic
        self.agent = agent
        self.buffer = buffer
        self.calibrator = calibrator
        self.logger = logger
        self.cfg = cfg
        self.collector = collector
        self.validation_env = validation_env
        self.validation_collector = validation_collector
        self.validation_cfg = dict(validation_cfg or {})
        self.observation_spec = getattr(actor_critic, "observation_spec", None)
        self.gamma = float(cfg.get("gamma", 0.99))
        self.lam = float(cfg.get("lam", 0.95))
        self.batch_episodes = int(cfg.get("batch_episodes", 32))
        self.max_collect_attempt_factor = int(cfg.get("max_collect_attempt_factor", 10))
        self.device = torch.device(cfg.get("device", "cpu"))
        self.diagnostics_enabled = bool(getattr(logger, "diagnostics_enabled", True))
        self.validation_enabled = bool(self.validation_cfg.get("enabled", False)) and (
            self.validation_env is not None or self.validation_collector is not None
        )
        self.validation_every_n_iterations = max(int(self.validation_cfg.get("every_n_iterations", 1)), 1)
        self.validation_num_episodes = max(int(self.validation_cfg.get("num_episodes", 0)), 0)
        self.iteration = 0
        self._last_collection_report: dict[str, Any] = {
            "attempts_total": 0,
            "valid_episodes": 0,
            "attempt_summaries": [],
            "rollout_version": -1,
            "scene_rebuild_performed": 0,
            "scene_rebuild_workers": 0,
            "scene_rebuild_wall_s": 0.0,
            "worker_recycle_performed": 0,
            "worker_recycle_slots": 0,
            "worker_recycle_prefetched": 0,
            "worker_recycle_prefetch_ready": 0,
            "worker_recycle_wall_s": 0.0,
            "worker_recycle_wait_ready_wall_s": 0.0,
        }

    def train(self, num_iterations: int, *, start_iteration: int = 0, iteration_callback=None):
        history: list[dict[str, Any]] = []
        for iteration in range(start_iteration, start_iteration + num_iterations):
            self.iteration = iteration
            iteration_start = time.perf_counter()
            collect_start = iteration_start
            collection_report = self.collect_rollout(self.batch_episodes)
            collect_wall_s = time.perf_counter() - collect_start
            batch = self.buffer.get_all()
            returns, advantages = compute_returns_and_advantages(
                rewards=batch["rewards"],
                values=batch["values"],
                dones=batch["dones"],
                gamma=self.gamma,
                lam=self.lam,
            )
            batch["returns"] = returns
            batch["advantages"] = advantages
            update_start = time.perf_counter()
            training_stats = self.agent.update(batch)
            calibrator_post_state = self.update_calibrator(batch)
            update_wall_s = time.perf_counter() - update_start
            validation_stats, validation_wall_s = self.run_validation(calibrator_state=calibrator_post_state)
            iteration_wall_s = time.perf_counter() - iteration_start
            system_stats = {}
            if self.diagnostics_enabled:
                worker_process_states = (
                    self.collector.get_worker_process_states()
                    if self.collector is not None and hasattr(self.collector, "get_worker_process_states")
                    else None
                )
                system_stats = collect_system_metrics(
                    main_device=self.device,
                    worker_process_states=worker_process_states,
                )
            rollout_stats = self._summarize_rollout(
                batch=batch,
                collection_report=collection_report,
                calibrator_post_state=calibrator_post_state,
                timing_stats={
                    "timing/collect_wall_s": collect_wall_s,
                    "timing/update_wall_s": update_wall_s,
                    "timing/validation_wall_s": validation_wall_s,
                    "timing/iteration_wall_s": iteration_wall_s,
                    **system_stats,
                },
            )
            stats = {**rollout_stats, **training_stats, **validation_stats}
            self.log_iteration(stats)
            self._log_episode_samples(batch)
            history.append(stats)
            if callable(iteration_callback):
                iteration_callback(iteration=iteration, stats=stats, history_snapshot=list(history))
            self.buffer.clear()
        return history

    def collect_rollout(self, num_episodes: int) -> dict[str, Any]:
        self.buffer.clear()
        if self.collector is not None:
            report = self._collect_rollout_async(num_episodes)
            self._last_collection_report = report
            return report
        self._sync_calibrator_to_env()
        report = self._collect_rollout_single(num_episodes)
        self._last_collection_report = report
        return report

    def _collect_rollout_async(self, num_episodes: int) -> dict[str, Any]:
        actor_state = {key: value.detach().cpu() for key, value in self.actor_critic.state_dict().items()}
        payload = self.collector.collect_batch(
            target_valid_episodes=num_episodes,
            actor_state=actor_state,
            calibrator_state=self.calibrator.get_state(),
            obs_spec=self.observation_spec,
            rollout_version=int(self.iteration),
        )
        for transition in payload["transitions"]:
            self.buffer.add(
                obs=transition["obs"],
                action=NormalizedAction(value=transition["action"]),
                reward=transition["reward"],
                next_obs=transition["next_obs"],
                done=transition["done"],
                log_prob=transition["log_prob"],
                value=transition["value"],
                info=transition["info"],
                raw_logit_before=transition["raw_logit_before"],
                raw_logit_after=transition["raw_logit_after"],
            )
        return self._collection_report_from_payload(payload)

    def _collect_rollout_single(self, num_episodes: int) -> dict[str, Any]:
        return self._collect_into_buffer(env=self.env, target_buffer=self.buffer, num_episodes=num_episodes)

    def update_calibrator(self, batch: dict | None = None) -> dict[str, Any]:
        batch = self.buffer.get_all() if batch is None else batch
        if batch["raw_logit_after"].size == 0:
            return self.calibrator.get_state()
        logits = batch["raw_logit_after"]
        labels = np.asarray([info.drop_success for info in batch["infos"]], dtype=np.float32)
        self.calibrator.update(logits, labels)
        return self.calibrator.get_state()

    def _sync_calibrator_to_env(self) -> None:
        self._sync_calibrator_state_to_env(self.env, self.calibrator.get_state())

    @staticmethod
    def _sync_calibrator_state_to_env(env, state: dict[str, Any]) -> None:
        sync_fn = getattr(env, "sync_calibrator", None)
        if env is None or not callable(sync_fn):
            return
        sync_fn(state)

    def _should_run_validation(self) -> bool:
        return (
            self.validation_enabled
            and self.validation_num_episodes > 0
            and (int(self.iteration) % self.validation_every_n_iterations) == 0
        )

    def run_validation(self, *, calibrator_state: dict[str, Any]) -> tuple[dict[str, float], float]:
        if not self._should_run_validation():
            return {}, 0.0

        validation_start = time.perf_counter()
        collect_start = validation_start
        if self.validation_collector is not None:
            report, batch = self._collect_validation_async(
                num_episodes=self.validation_num_episodes,
                calibrator_state=calibrator_state,
            )
        else:
            self._sync_calibrator_state_to_env(self.validation_env, calibrator_state)
            report, batch = self._collect_validation_single(num_episodes=self.validation_num_episodes)
        collect_wall_s = time.perf_counter() - collect_start
        stats = self._summarize_rollout(
            batch=batch,
            collection_report=report,
            calibrator_post_state=calibrator_state,
            timing_stats={"timing/collect_wall_s": collect_wall_s},
            prefix="validation/",
        )
        return stats, time.perf_counter() - validation_start

    def _collect_validation_async(
        self,
        *,
        num_episodes: int,
        calibrator_state: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        actor_state = {key: value.detach().cpu() for key, value in self.actor_critic.state_dict().items()}
        payload = self.validation_collector.collect_batch(
            target_valid_episodes=num_episodes,
            actor_state=actor_state,
            calibrator_state=calibrator_state,
            obs_spec=self.observation_spec,
            rollout_version=int(self.iteration),
            reset_worker_sequences=True,
        )
        temp_buffer = RolloutBuffer()
        for transition in payload["transitions"]:
            temp_buffer.add(
                obs=transition["obs"],
                action=NormalizedAction(value=transition["action"]),
                reward=transition["reward"],
                next_obs=transition["next_obs"],
                done=transition["done"],
                log_prob=transition["log_prob"],
                value=transition["value"],
                info=transition["info"],
                raw_logit_before=transition["raw_logit_before"],
                raw_logit_after=transition["raw_logit_after"],
            )
        return self._collection_report_from_payload(payload), temp_buffer.get_all()

    def _collect_validation_single(self, *, num_episodes: int) -> tuple[dict[str, Any], dict[str, Any]]:
        reset_fn = getattr(self.validation_env, "reset_sampling_sequence", None)
        if callable(reset_fn):
            reset_fn()
        temp_buffer = RolloutBuffer()
        report = self._collect_into_buffer(
            env=self.validation_env,
            target_buffer=temp_buffer,
            num_episodes=num_episodes,
        )
        return report, temp_buffer.get_all()

    def _collect_into_buffer(self, *, env, target_buffer, num_episodes: int) -> dict[str, Any]:
        episodes_collected = 0
        attempts = 0
        attempt_summaries: list[dict[str, Any]] = []
        max_attempts = max(num_episodes * self.max_collect_attempt_factor, num_episodes)
        while episodes_collected < num_episodes:
            attempts += 1
            if attempts > max_attempts:
                raise RuntimeError(
                    f"Exceeded max rollout collection attempts ({max_attempts}) while collecting valid episodes."
                )
            obs = env.reset()
            obs_tensor = observation_to_tensor(obs, spec=self.observation_spec).to(self.device)
            policy_start = time.perf_counter()
            with torch.no_grad():
                action_tensor, log_prob, value, _ = self.actor_critic.act(obs_tensor)
            policy_forward_s = time.perf_counter() - policy_start
            action_np = action_tensor_to_numpy(action_tensor).reshape(-1)
            next_obs, reward, done, info = env.step(NormalizedAction(value=action_np))
            trial_metadata = info.extra.get("trial_metadata", {})
            attempt_summaries.append(self._build_attempt_summary(info=info, policy_forward_s=policy_forward_s, worker_id=0))
            if not bool(trial_metadata.get("valid_for_learning", True)):
                continue
            target_buffer.add(
                obs=obs,
                action=NormalizedAction(value=action_np),
                reward=reward,
                next_obs=next_obs,
                done=done,
                log_prob=float(log_prob.squeeze(0).cpu().item()),
                value=float(value.squeeze(0).cpu().item()),
                info=info,
                raw_logit_before=info.extra["raw_logit_before"],
                raw_logit_after=info.extra["raw_logit_after"],
            )
            episodes_collected += 1
        return {
            "attempts_total": attempts,
            "valid_episodes": episodes_collected,
            "attempt_summaries": attempt_summaries,
            "rollout_version": int(self.iteration),
            "scene_rebuild_performed": 0,
            "scene_rebuild_workers": 0,
            "scene_rebuild_wall_s": 0.0,
            "worker_recycle_performed": 0,
            "worker_recycle_slots": 0,
            "worker_recycle_prefetched": 0,
            "worker_recycle_prefetch_ready": 0,
            "worker_recycle_wall_s": 0.0,
            "worker_recycle_wait_ready_wall_s": 0.0,
        }

    @staticmethod
    def _collection_report_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "attempts_total": int(payload["attempts_total"]),
            "valid_episodes": int(payload["valid_episodes"]),
            "attempt_summaries": list(payload.get("attempt_summaries", [])),
            "rollout_version": int(payload["rollout_version"]),
            "scene_rebuild_performed": int(payload.get("scene_rebuild_performed", 0)),
            "scene_rebuild_workers": int(payload.get("scene_rebuild_workers", 0)),
            "scene_rebuild_wall_s": float(payload.get("scene_rebuild_wall_s", 0.0)),
            "worker_recycle_performed": int(payload.get("worker_recycle_performed", 0)),
            "worker_recycle_slots": int(payload.get("worker_recycle_slots", 0)),
            "worker_recycle_prefetched": int(payload.get("worker_recycle_prefetched", 0)),
            "worker_recycle_prefetch_ready": int(payload.get("worker_recycle_prefetch_ready", 0)),
            "worker_recycle_wall_s": float(payload.get("worker_recycle_wall_s", 0.0)),
            "worker_recycle_wait_ready_wall_s": float(payload.get("worker_recycle_wait_ready_wall_s", 0.0)),
        }

    def log_iteration(self, stats: dict):
        self.logger.log_dict(stats, step=self.iteration)
        format_payload = getattr(self.logger, "format_payload", None)
        if callable(format_payload):
            rendered_stats = format_payload(stats)
        else:
            rendered_stats = str(stats)
        self.logger.info(f"Iteration {self.iteration}: {rendered_stats}")

    def _log_episode_samples(self, batch: dict) -> None:
        if not bool(getattr(self.logger, "sample_metrics_enabled", False)):
            return
        log_fn = getattr(self.logger, "log_episode_samples", None)
        if not callable(log_fn):
            return
        samples = []
        infos = batch["infos"]
        for obs, next_obs, action, reward, info in zip(
            batch["obs"],
            batch["next_obs"],
            batch["actions"],
            batch["rewards"],
            infos,
        ):
            reward_breakdown = info.extra.get("reward_breakdown")
            reward_payload = (
                reward_breakdown.as_dict()
                if reward_breakdown is not None
                else {
                    "total": float(reward),
                    "drop": float(info.reward_drop),
                    "stability": float(info.reward_stability),
                    "contact": float(info.reward_contact),
                }
            )
            samples.append(
                {
                    "contact": {
                        "t_cover_before": float(obs.contact_semantic[0]),
                        "t_cover_after": float(next_obs.contact_semantic[0]),
                        "t_edge_before": float(obs.contact_semantic[1]),
                        "t_edge_after": float(next_obs.contact_semantic[1]),
                    },
                    "calibrator": {
                        "raw_logit_before": float(info.extra["raw_logit_before"]),
                        "raw_logit_after": float(info.extra["raw_logit_after"]),
                        "prob_before": float(info.calibrated_stability_before),
                        "prob_after": float(info.calibrated_stability_after),
                        "posterior_trace_snapshot": float(info.posterior_trace),
                    },
                    "reward": reward_payload,
                    "outcome": {
                        "drop_success_after_live": int(info.drop_success),
                        "legacy_drop_success_before": info.extra.get("legacy_drop_success_before"),
                        "trial_status": info.extra.get("trial_metadata", {}).get("trial_status"),
                    },
                    "action": {
                        "value": np.asarray(action, dtype=np.float32).tolist(),
                        "l2": float(np.linalg.norm(action)),
                        "abs_mean": float(np.mean(np.abs(action))),
                        "reward_total": float(reward),
                    },
                }
            )
        log_fn(samples, step=self.iteration)

    @staticmethod
    def _build_attempt_summary(info, *, policy_forward_s: float, worker_id: int) -> dict[str, Any]:
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

    @staticmethod
    def _summarize_rollout(
        *,
        batch: dict,
        collection_report: dict[str, Any],
        calibrator_post_state: dict[str, Any],
        timing_stats: dict[str, float],
        prefix: str = "",
    ) -> dict[str, float]:
        rewards = np.asarray(batch["rewards"], dtype=np.float32)
        infos = list(batch["infos"])
        actions = np.asarray(batch["actions"], dtype=np.float32)
        attempt_summaries = list(collection_report.get("attempt_summaries", []))

        def _mean(values) -> float:
            arr = np.asarray(values, dtype=np.float32)
            return float(np.mean(arr)) if arr.size else 0.0

        def _std(values) -> float:
            arr = np.asarray(values, dtype=np.float32)
            return float(np.std(arr)) if arr.size else 0.0

        def _rate(values) -> float:
            arr = np.asarray(values, dtype=np.float32)
            return float(np.mean(arr)) if arr.size else 0.0

        def _finite_mean(values) -> float:
            arr = np.asarray(values, dtype=np.float32)
            if arr.size == 0:
                return 0.0
            arr = arr[np.isfinite(arr)]
            return float(np.mean(arr)) if arr.size else 0.0

        def _masked_rate(values, mask, *, predicate=None) -> float:
            arr = np.asarray(values, dtype=np.float32)
            mask_arr = np.asarray(mask, dtype=bool)
            if arr.size == 0 or mask_arr.size == 0 or arr.shape[0] != mask_arr.shape[0]:
                return 0.0
            selected = arr[mask_arr]
            if selected.size == 0:
                return 0.0
            if predicate is not None:
                selected = predicate(selected)
            return float(np.mean(np.asarray(selected, dtype=np.float32)))

        if infos:
            t_before = np.stack([obs.contact_semantic for obs in batch["obs"]], axis=0)
            t_after = np.stack([obs.contact_semantic for obs in batch["next_obs"]], axis=0)
            prob_before = np.asarray([info.calibrated_stability_before for info in infos], dtype=np.float32)
            prob_after = np.asarray([info.calibrated_stability_after for info in infos], dtype=np.float32)
            prob_delta = prob_after - prob_before
            reward_drop = np.asarray([info.reward_drop for info in infos], dtype=np.float32)
            reward_stability = np.asarray([info.reward_stability for info in infos], dtype=np.float32)
            reward_contact = np.asarray([info.reward_contact for info in infos], dtype=np.float32)
            drop_success = np.asarray([info.drop_success for info in infos], dtype=np.float32)
            dataset_before = np.asarray(
                [info.extra.get("legacy_drop_success_before", np.nan) for info in infos],
                dtype=np.float32,
            )
            raw_logit_before = np.asarray(batch["raw_logit_before"], dtype=np.float32)
            raw_logit_after = np.asarray(batch["raw_logit_after"], dtype=np.float32)
            posterior_trace_snapshot = np.asarray([info.posterior_trace for info in infos], dtype=np.float32)
        else:
            t_before = np.zeros((0, 2), dtype=np.float32)
            t_after = np.zeros((0, 2), dtype=np.float32)
            prob_before = np.zeros((0,), dtype=np.float32)
            prob_after = np.zeros((0,), dtype=np.float32)
            prob_delta = np.zeros((0,), dtype=np.float32)
            reward_drop = np.zeros((0,), dtype=np.float32)
            reward_stability = np.zeros((0,), dtype=np.float32)
            reward_contact = np.zeros((0,), dtype=np.float32)
            drop_success = np.zeros((0,), dtype=np.float32)
            dataset_before = np.zeros((0,), dtype=np.float32)
            raw_logit_before = np.zeros((0,), dtype=np.float32)
            raw_logit_after = np.zeros((0,), dtype=np.float32)
            posterior_trace_snapshot = np.zeros((0,), dtype=np.float32)

        dataset_positive_mask = np.isfinite(dataset_before) & (dataset_before >= 0.5)
        dataset_negative_mask = np.isfinite(dataset_before) & (dataset_before < 0.5)
        dataset_positive_count = int(np.sum(dataset_positive_mask))
        dataset_negative_count = int(np.sum(dataset_negative_mask))

        total_attempts = max(int(collection_report.get("attempts_total", 0)), 1)
        status_counts: dict[str, int] = {}
        system_invalid_count = 0
        policy_forward_values = []
        for summary in attempt_summaries:
            status = str(summary.get("trial_status", "unknown"))
            status_counts[status] = status_counts.get(status, 0) + 1
            if status.startswith("system_"):
                system_invalid_count += 1
            policy_forward_values.append(float(summary.get("policy_forward_s", 0.0)))

        eps = 1e-6
        clipped_prob_after = np.clip(prob_after, eps, 1.0 - eps)
        brier = np.mean((prob_after - drop_success) ** 2) if prob_after.size else 0.0
        bce = np.mean(-(drop_success * np.log(clipped_prob_after) + (1.0 - drop_success) * np.log(1.0 - clipped_prob_after))) if prob_after.size else 0.0

        raw_stats = {
            "collection/attempts_total": float(collection_report.get("attempts_total", 0)),
            "collection/valid_episodes": float(collection_report.get("valid_episodes", 0)),
            "collection/valid_rate": float(collection_report.get("valid_episodes", 0)) / float(total_attempts),
            "collection/scene_rebuild_performed": float(collection_report.get("scene_rebuild_performed", 0)),
            "collection/scene_rebuild_workers": float(collection_report.get("scene_rebuild_workers", 0)),
            "collection/worker_recycle_performed": float(collection_report.get("worker_recycle_performed", 0)),
            "collection/worker_recycle_slots": float(collection_report.get("worker_recycle_slots", 0)),
            "collection/worker_recycle_prefetched": float(collection_report.get("worker_recycle_prefetched", 0)),
            "collection/worker_recycle_prefetch_ready": float(
                collection_report.get("worker_recycle_prefetch_ready", 0)
            ),
            "outcome/success_rate_live_after": _rate(drop_success),
            "outcome/success_rate_dataset_before": _finite_mean(dataset_before),
            "outcome/success_lift_vs_dataset": _rate(drop_success) - _finite_mean(dataset_before),
            "outcome/drop_rate_after_given_dataset_positive": _masked_rate(
                drop_success,
                dataset_positive_mask,
                predicate=lambda arr: arr < 0.5,
            ),
            "outcome/hold_rate_after_given_dataset_negative": _masked_rate(
                drop_success,
                dataset_negative_mask,
                predicate=lambda arr: arr >= 0.5,
            ),
            "outcome/dataset_positive_count": float(dataset_positive_count),
            "outcome/dataset_negative_count": float(dataset_negative_count),
            "outcome/system_invalid_rate": float(system_invalid_count) / float(total_attempts),
            "reward/total_mean": _mean(rewards),
            "reward/total_std": _std(rewards),
            "reward/drop_mean": _mean(reward_drop),
            "reward/stability_mean": _mean(reward_stability),
            "reward/contact_mean": _mean(reward_contact),
            "contact/t_cover_before_mean": _mean(t_before[:, 0]) if t_before.size else 0.0,
            "contact/t_cover_before_std": _std(t_before[:, 0]) if t_before.size else 0.0,
            "contact/t_cover_after_mean": _mean(t_after[:, 0]) if t_after.size else 0.0,
            "contact/t_cover_after_std": _std(t_after[:, 0]) if t_after.size else 0.0,
            "contact/t_cover_delta_mean": _mean(t_after[:, 0] - t_before[:, 0]) if t_before.size else 0.0,
            "contact/t_edge_before_mean": _mean(t_before[:, 1]) if t_before.size else 0.0,
            "contact/t_edge_before_std": _std(t_before[:, 1]) if t_before.size else 0.0,
            "contact/t_edge_after_mean": _mean(t_after[:, 1]) if t_after.size else 0.0,
            "contact/t_edge_after_std": _std(t_after[:, 1]) if t_after.size else 0.0,
            "contact/t_edge_delta_mean": _mean(t_after[:, 1] - t_before[:, 1]) if t_before.size else 0.0,
            "calibrator/raw_logit_before_mean": _mean(raw_logit_before),
            "calibrator/raw_logit_after_mean": _mean(raw_logit_after),
            "calibrator/prob_before_mean": _mean(prob_before),
            "calibrator/prob_after_mean": _mean(prob_after),
            "calibrator/prob_delta_mean": _mean(prob_delta),
            "calibrator/prob_delta_std": _std(prob_delta),
            "calibrator/prob_delta_positive_rate": _rate(prob_delta > 0.0),
            "calibrator/posterior_trace_snapshot": _mean(posterior_trace_snapshot),
            "calibrator/posterior_trace_post_update": float(np.trace(np.asarray(calibrator_post_state["posterior_cov"]))),
            "calibrator/param_a": float(calibrator_post_state["a"]),
            "calibrator/param_b": float(calibrator_post_state["b"]),
            "calibrator/after_brier": float(brier),
            "calibrator/after_bce": float(bce),
            "action/abs_mean": _mean(np.abs(actions)),
            "action/l2_mean": _mean(np.linalg.norm(actions, axis=1)) if actions.size else 0.0,
            "action/saturation_rate": _rate(np.abs(actions) >= 0.999) if actions.size else 0.0,
            "timing/policy_forward_s_mean": _mean(policy_forward_values),
            "timing/scene_rebuild_wall_s": float(collection_report.get("scene_rebuild_wall_s", 0.0)),
            "timing/worker_recycle_wall_s": float(collection_report.get("worker_recycle_wall_s", 0.0)),
            "timing/worker_recycle_wait_ready_wall_s": float(
                collection_report.get("worker_recycle_wait_ready_wall_s", 0.0)
            ),
            **timing_stats,
        }
        for status, count in sorted(status_counts.items()):
            raw_stats[f"outcome/trial_status_{status}_rate"] = float(count) / float(total_attempts)
        if not prefix:
            return raw_stats
        return {f"{prefix}{key}": value for key, value in raw_stats.items()}
