from __future__ import annotations

from copy import deepcopy
from typing import Any


def apply_experiment_overrides(experiment_cfg: dict[str, Any], bundle: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Promote experiment-level settings into sub-configs that should share them."""

    experiment_cfg_local = deepcopy(experiment_cfg)
    bundle_local = deepcopy(bundle)

    seed = experiment_cfg_local.get("seed")
    if seed is not None:
        seed = int(seed)
        env_cfg = bundle_local.get("env")
        if isinstance(env_cfg, dict):
            env_cfg["seed"] = seed
            dataset_cfg = env_cfg.get("dataset")
            if isinstance(dataset_cfg, dict):
                dataset_cfg["seed"] = seed
                dataset_cfg.pop("worker_id", None)
                dataset_cfg.pop("num_workers", None)

        perception_cfg = bundle_local.get("perception")
        if isinstance(perception_cfg, dict):
            runtime_cfg = perception_cfg.get("sga_gsn", {}).get("runtime")
            if isinstance(runtime_cfg, dict):
                runtime_cfg["seed"] = seed

    logging_cfg = experiment_cfg_local.setdefault("logging", {})
    experiment_name = str(experiment_cfg_local.get("name", "")).strip()
    if experiment_name:
        logging_cfg.setdefault("experiment_name", experiment_name)

    scene_rebuild_every_n_iterations = experiment_cfg_local.get("scene_rebuild_every_n_iterations")
    if scene_rebuild_every_n_iterations is not None:
        rl_cfg = bundle_local.get("rl")
        if isinstance(rl_cfg, dict):
            rl_cfg["scene_rebuild_every_n_iterations"] = max(int(scene_rebuild_every_n_iterations), 0)

    recycle_override_keys = (
        "worker_recycle_every_n_iterations",
        "worker_recycle_slots_per_event",
        "worker_recycle_enable_standby_prefetch",
        "worker_recycle_prefetch_count",
    )
    if any(key in experiment_cfg_local for key in recycle_override_keys):
        rl_cfg = bundle_local.get("rl")
        if isinstance(rl_cfg, dict):
            if "worker_recycle_every_n_iterations" in experiment_cfg_local:
                rl_cfg["worker_recycle_every_n_iterations"] = max(
                    int(experiment_cfg_local["worker_recycle_every_n_iterations"]),
                    0,
                )
            if "worker_recycle_slots_per_event" in experiment_cfg_local:
                rl_cfg["worker_recycle_slots_per_event"] = max(
                    int(experiment_cfg_local["worker_recycle_slots_per_event"]),
                    1,
                )
            if "worker_recycle_enable_standby_prefetch" in experiment_cfg_local:
                rl_cfg["worker_recycle_enable_standby_prefetch"] = bool(
                    experiment_cfg_local["worker_recycle_enable_standby_prefetch"]
                )
            if "worker_recycle_prefetch_count" in experiment_cfg_local:
                rl_cfg["worker_recycle_prefetch_count"] = max(
                    int(experiment_cfg_local["worker_recycle_prefetch_count"]),
                    0,
                )

    return experiment_cfg_local, bundle_local
