from __future__ import annotations

import argparse
from copy import deepcopy
from typing import Any

import torch

from _common import build_actor_critic, build_env, load_experiment_bundle, snapshot_experiment_configs
from src.calibration.online_logit_calibrator import OnlineLogitCalibrator
from src.rl.ppo_agent import PPOAgent
from src.rl.rollout_buffer import RolloutBuffer
from src.rl.subproc_async_rollout_collector import SubprocAsyncRolloutCollector
from src.rl.trainer import Trainer
from src.runtime.object_splits import resolve_object_split
from src.runtime.render_env import configure_render_environment
from src.runtime.train_state import restore_training_state
from src.utils.checkpoint import save_checkpoint
from src.utils.logger import Logger, resolve_experiment_artifact_path
from src.utils.seed import set_seed
from src.utils.system_diagnostics import collect_system_metrics


def _build_checkpoint_payload(
    *,
    actor_critic,
    optimizer,
    calibrator,
    history: list[dict[str, Any]],
    experiment_cfg: dict[str, Any],
    best_metric_name: str | None = None,
    best_metric_mode: str | None = None,
    best_metric_value: float | None = None,
    best_iteration: int | None = None,
    object_split: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "actor_critic": actor_critic.state_dict(),
        "optimizer": optimizer.state_dict(),
        "history": history,
        "completed_iterations": len(history),
        "calibrator": calibrator.get_state(),
        "experiment_cfg": experiment_cfg,
        "best_metric_name": best_metric_name,
        "best_metric_mode": best_metric_mode,
        "best_metric_value": best_metric_value,
        "best_iteration": best_iteration,
        "object_split": deepcopy(object_split) if object_split is not None else None,
    }


def _metric_is_better(candidate: float, incumbent: float | None, mode: str) -> bool:
    if incumbent is None:
        return True
    if mode == "min":
        return candidate < incumbent
    return candidate > incumbent


def main():
    parser = argparse.ArgumentParser(description="Train the v1 single-step grasp refinement pipeline.")
    parser.add_argument(
        "--experiment",
        default="configs/experiment/exp_debug.yaml",
        help="Path to the experiment config.",
    )
    parser.add_argument(
        "--resume-from",
        default=None,
        help="Optional checkpoint path to restore actor/optimizer/calibrator/history from before continuing training.",
    )
    args = parser.parse_args()

    experiment_cfg, config_bundle = load_experiment_bundle(args.experiment)
    env_cfg = deepcopy(config_bundle["env"])
    perception_cfg = config_bundle["perception"]
    calibration_cfg = config_bundle["calibration"]
    rl_cfg = config_bundle["rl"]
    actor_critic_cfg = config_bundle["actor_critic"]
    validation_cfg = dict(experiment_cfg.get("validation", {}))
    object_split_cfg_present = any(
        experiment_cfg.get(key) is not None
        for key in ("train_object_id_range", "holdout_object_id_range", "val_object_count", "val_object_ratio")
    )
    resolved_object_split = resolve_object_split(experiment_cfg).as_dict() if object_split_cfg_present else None
    if resolved_object_split is not None:
        dataset_cfg = env_cfg.setdefault("dataset", {})
        dataset_cfg["include_object_ids"] = list(resolved_object_split["train_ids"])
    configure_render_environment(env_cfg.get("scene", {}))

    seed = int(experiment_cfg.get("seed", 0))
    set_seed(seed)
    logging_cfg = dict(experiment_cfg.get("logging", {}))
    logger = Logger(logging_cfg)
    num_envs = int(rl_cfg.get("num_envs", 1))
    collector = None
    validation_env = None
    validation_collector = None

    if resolved_object_split is not None:
        logger.info(
            "Resolved object split: "
            f"train={resolved_object_split['train_ids']}, "
            f"val={resolved_object_split['val_ids']}, "
            f"test={resolved_object_split['test_ids']} "
            f"(split_seed={resolved_object_split['split_seed']})."
        )
    validation_enabled = bool(validation_cfg.get("enabled", False))
    if validation_enabled:
        if resolved_object_split is None:
            raise RuntimeError("Validation requires object split config so train/val object pools are well-defined.")
        if not resolved_object_split["val_ids"]:
            raise RuntimeError("Validation is enabled but resolved val_ids is empty. Adjust val_object_count/ratio.")

    if num_envs > 1:
        calibrator = OnlineLogitCalibrator(calibration_cfg)
        actor_critic = build_actor_critic(perception_cfg, actor_critic_cfg)
        collector = SubprocAsyncRolloutCollector(
            env_cfg=env_cfg,
            perception_cfg=perception_cfg,
            calibration_cfg=calibration_cfg,
            actor_critic_cfg=actor_critic_cfg,
            rl_cfg=rl_cfg,
            num_workers=num_envs,
            observation_spec=getattr(actor_critic, "observation_spec", None),
        )
        env = None
    else:
        env, calibrator = build_env(env_cfg, perception_cfg, calibration_cfg)
        actor_critic = build_actor_critic(perception_cfg, actor_critic_cfg)

    if validation_enabled:
        validation_env_cfg = deepcopy(config_bundle["env"])
        validation_dataset_cfg = validation_env_cfg.setdefault("dataset", {})
        validation_dataset_cfg["include_object_ids"] = list(resolved_object_split["val_ids"])
        validation_dataset_cfg["fixed_sample_sequence"] = bool(validation_cfg.get("fixed_sample_sequence", True))
        validation_dataset_cfg["fixed_sample_sequence_seed"] = int(
            validation_cfg.get("fixed_sample_sequence_seed", resolved_object_split["split_seed"])
        )
        validation_rl_cfg = deepcopy(rl_cfg)
        validation_num_workers = max(int(validation_cfg.get("num_workers", 1)), 1)
        validation_rl_cfg["num_envs"] = validation_num_workers
        validation_rl_cfg["scene_rebuild_every_n_iterations"] = max(
            int(validation_cfg.get("scene_rebuild_every_n_iterations", 0)),
            0,
        )
        validation_rl_cfg["worker_recycle_every_n_iterations"] = max(
            int(validation_cfg.get("worker_recycle_every_n_iterations", 0)),
            0,
        )
        validation_rl_cfg["worker_recycle_slots_per_event"] = max(
            int(validation_cfg.get("worker_recycle_slots_per_event", 1)),
            1,
        )
        validation_rl_cfg["worker_recycle_enable_standby_prefetch"] = bool(
            validation_cfg.get("worker_recycle_enable_standby_prefetch", False)
        )
        validation_rl_cfg["worker_recycle_prefetch_count"] = max(
            int(validation_cfg.get("worker_recycle_prefetch_count", 0)),
            0,
        )
        if validation_num_workers > 1:
            validation_collector = SubprocAsyncRolloutCollector(
                env_cfg=validation_env_cfg,
                perception_cfg=perception_cfg,
                calibration_cfg=calibration_cfg,
                actor_critic_cfg=actor_critic_cfg,
                rl_cfg=validation_rl_cfg,
                num_workers=validation_num_workers,
                observation_spec=getattr(actor_critic, "observation_spec", None),
            )
        else:
            validation_env, _ = build_env(validation_env_cfg, perception_cfg, calibration_cfg)

    optimizer = torch.optim.Adam(actor_critic.parameters(), lr=float(rl_cfg.get("learning_rate", 3e-4)))
    agent = PPOAgent(actor_critic=actor_critic, optimizer=optimizer, cfg=rl_cfg)
    buffer = RolloutBuffer()
    initial_history: list[dict] = []
    start_iteration = 0
    best_metric_name: str | None = None
    best_metric_mode: str = "max"
    best_metric_value: float | None = None
    best_iteration: int | None = None
    if args.resume_from:
        restored = restore_training_state(
            checkpoint_path=args.resume_from,
            actor_critic=actor_critic,
            optimizer=optimizer,
            calibrator=calibrator,
            device=agent.device,
        )
        initial_history = list(restored["history"])
        start_iteration = int(restored["completed_iterations"])
        checkpoint_state = restored["checkpoint"]
        best_metric_name = checkpoint_state.get("best_metric_name")
        best_metric_mode = str(checkpoint_state.get("best_metric_mode", "max"))
        if checkpoint_state.get("best_metric_value") is not None:
            best_metric_value = float(checkpoint_state["best_metric_value"])
        if checkpoint_state.get("best_iteration") is not None:
            best_iteration = int(checkpoint_state["best_iteration"])
        logger.info(f"Resuming training from {args.resume_from} at iteration {start_iteration}.")
    trainer = Trainer(
        env=env,
        actor_critic=actor_critic,
        agent=agent,
        buffer=buffer,
        calibrator=calibrator,
        logger=logger,
        cfg=rl_cfg,
        collector=collector,
        validation_env=validation_env,
        validation_collector=validation_collector,
        validation_cfg=validation_cfg,
    )

    try:
        checkpoint_dir = resolve_experiment_artifact_path(
            logging_cfg.get("checkpoint_dir", logger.log_dir / "checkpoints"),
            logger.experiment_name,
        )
        config_snapshot_dir = logger.log_dir / "configs"
        copied_configs = snapshot_experiment_configs(args.experiment, config_snapshot_dir)
        logger.info(
            f"Saved config snapshot to {config_snapshot_dir} "
            f"({len(copied_configs)} files)."
        )
        best_cfg = dict(experiment_cfg.get("logging", {}).get("best_checkpoint", {}))
        best_enabled = bool(best_cfg.get("enabled", True))
        default_best_metric = (
            "validation/outcome/success_lift_vs_dataset"
            if validation_enabled
            else "outcome/success_lift_vs_dataset"
        )
        best_metric_name = str(best_cfg.get("metric", best_metric_name or default_best_metric))
        best_metric_mode = str(best_cfg.get("mode", best_metric_mode or "max")).lower()
        best_filename = str(best_cfg.get("filename", "best.pt"))

        def _save_best_if_needed(*, iteration: int, stats: dict[str, Any], history_snapshot: list[dict[str, Any]]) -> None:
            nonlocal best_metric_value, best_iteration
            if not best_enabled:
                return
            metric_value_raw = stats.get(best_metric_name)
            if metric_value_raw is None:
                return
            metric_value = float(metric_value_raw)
            if not _metric_is_better(metric_value, best_metric_value, best_metric_mode):
                return
            best_metric_value = metric_value
            best_iteration = int(iteration)
            save_checkpoint(
                checkpoint_dir / best_filename,
                _build_checkpoint_payload(
                    actor_critic=actor_critic,
                    optimizer=optimizer,
                    calibrator=calibrator,
                    history=history_snapshot,
                    experiment_cfg=experiment_cfg,
                    best_metric_name=best_metric_name,
                    best_metric_mode=best_metric_mode,
                    best_metric_value=best_metric_value,
                    best_iteration=best_iteration,
                    object_split=resolved_object_split,
                ),
            )
            logger.info(
                f"Saved best checkpoint to {checkpoint_dir / best_filename} "
                f"at iteration {iteration} with {best_metric_name}={metric_value:.6f}."
            )

        try:
            new_history = trainer.train(
                num_iterations=int(experiment_cfg.get("num_iterations", 1)),
                start_iteration=start_iteration,
                iteration_callback=lambda iteration, stats, history_snapshot: _save_best_if_needed(
                    iteration=iteration,
                    stats=stats,
                    history_snapshot=initial_history + history_snapshot,
                ),
            )
        except Exception:
            if logger.diagnostics_enabled:
                worker_process_states = (
                    collector.get_worker_process_states()
                    if collector is not None and hasattr(collector, "get_worker_process_states")
                    else None
                )
                crash_system_metrics = collect_system_metrics(
                    main_device=agent.device,
                    worker_process_states=worker_process_states,
                )
                logger.info(f"Crash diagnostics/system: {logger.format_payload(crash_system_metrics)}")
                if worker_process_states:
                    logger.info(f"Crash diagnostics/workers: {logger.format_payload(worker_process_states)}")
            raise
        history = initial_history + new_history
        save_checkpoint(
            checkpoint_dir / "final.pt",
            _build_checkpoint_payload(
                actor_critic=actor_critic,
                optimizer=optimizer,
                calibrator=calibrator,
                history=history,
                experiment_cfg=experiment_cfg,
                best_metric_name=best_metric_name if best_enabled else None,
                best_metric_mode=best_metric_mode if best_enabled else None,
                best_metric_value=best_metric_value if best_enabled else None,
                best_iteration=best_iteration if best_enabled else None,
                object_split=resolved_object_split,
            ),
        )
    finally:
        if collector is not None:
            collector.close()
        if validation_collector is not None:
            validation_collector.close()
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            close_fn()
        validation_close_fn = getattr(validation_env, "close", None)
        if callable(validation_close_fn):
            validation_close_fn()


if __name__ == "__main__":
    main()
