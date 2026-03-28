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

    return experiment_cfg_local, bundle_local
