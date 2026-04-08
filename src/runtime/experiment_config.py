from __future__ import annotations

from copy import deepcopy
from typing import Any

from src.rl.observation_spec import resolve_policy_observation_components

_SUPPORTED_ABLATION_IDS = frozenset(
    {
        "baseline",
        "wo-tac-rwd",
        "wo-stb-rwd",
        "wo-onl-cal",
        "wo-tac-sem-n-rwd",
    }
)


def _ensure_bundle_defaults(bundle: dict[str, Any]) -> None:
    env_cfg = bundle.get("env")
    if isinstance(env_cfg, dict):
        reward_cfg = env_cfg.setdefault("reward", {})
        if isinstance(reward_cfg, dict):
            reward_cfg.setdefault("drop_weight", 1.0)
            reward_cfg.setdefault("stability_weight", 1.0)
            reward_cfg.setdefault("contact_weight", 1.0)

    calibration_cfg = bundle.get("calibration")
    if isinstance(calibration_cfg, dict):
        calibration_cfg.setdefault("online_update_enabled", True)
        calibration_cfg.setdefault("signal_mode", "calibrated_probability")
        calibration_cfg.setdefault("uncertainty_discount_enabled", True)


def _apply_reward_weight(bundle: dict[str, Any], key: str, value: float) -> None:
    env_cfg = bundle.get("env")
    if not isinstance(env_cfg, dict):
        raise ValueError("Ablation overrides require an env config bundle.")
    reward_cfg = env_cfg.setdefault("reward", {})
    if not isinstance(reward_cfg, dict):
        raise ValueError("Ablation overrides require env.reward to be a mapping.")
    reward_cfg[key] = float(value)


def _set_online_calibration_enabled(bundle: dict[str, Any], enabled: bool) -> None:
    calibration_cfg = bundle.get("calibration")
    if not isinstance(calibration_cfg, dict):
        raise ValueError("Ablation overrides require a calibration config bundle.")
    calibration_cfg["online_update_enabled"] = bool(enabled)


def _set_calibration_signal_mode(bundle: dict[str, Any], mode: str) -> None:
    calibration_cfg = bundle.get("calibration")
    if not isinstance(calibration_cfg, dict):
        raise ValueError("Ablation overrides require a calibration config bundle.")
    calibration_cfg["signal_mode"] = str(mode)


def _set_uncertainty_discount_enabled(bundle: dict[str, Any], enabled: bool) -> None:
    calibration_cfg = bundle.get("calibration")
    if not isinstance(calibration_cfg, dict):
        raise ValueError("Ablation overrides require a calibration config bundle.")
    calibration_cfg["uncertainty_discount_enabled"] = bool(enabled)


def _remove_policy_observation_component(bundle: dict[str, Any], component: str) -> None:
    actor_critic_cfg = bundle.get("actor_critic")
    if not isinstance(actor_critic_cfg, dict):
        raise ValueError("Ablation overrides require an actor_critic config bundle.")
    policy_obs_cfg = dict(actor_critic_cfg.get("policy_observation", {}))
    components, _ = resolve_policy_observation_components(policy_obs_cfg)
    filtered_components = tuple(name for name in components if name != component)
    if not filtered_components:
        raise ValueError(
            f"Ablation removing '{component}' would leave policy_observation.components empty."
        )
    policy_obs_cfg["preset"] = "custom"
    policy_obs_cfg["components"] = list(filtered_components)
    actor_critic_cfg["policy_observation"] = policy_obs_cfg


def _apply_ablation_overrides(experiment_cfg: dict[str, Any], bundle: dict[str, Any]) -> None:
    ablation_cfg = dict(experiment_cfg.get("ablation", {}))
    ablation_id = str(ablation_cfg.get("id", "baseline") or "baseline").strip()
    if ablation_id not in _SUPPORTED_ABLATION_IDS:
        raise ValueError(
            f"Unknown ablation.id '{ablation_id}'. Expected one of {sorted(_SUPPORTED_ABLATION_IDS)}."
        )
    ablation_cfg["id"] = ablation_id
    experiment_cfg["ablation"] = ablation_cfg

    if ablation_id == "baseline":
        return
    if ablation_id == "wo-tac-rwd":
        _apply_reward_weight(bundle, "contact_weight", 0.0)
        return
    if ablation_id == "wo-stb-rwd":
        _apply_reward_weight(bundle, "stability_weight", 0.0)
        return
    if ablation_id == "wo-onl-cal":
        _set_online_calibration_enabled(bundle, False)
        _set_calibration_signal_mode(bundle, "identity_probability")
        _set_uncertainty_discount_enabled(bundle, False)
        return
    if ablation_id == "wo-tac-sem-n-rwd":
        _remove_policy_observation_component(bundle, "contact_semantic")
        _apply_reward_weight(bundle, "contact_weight", 0.0)
        return


def apply_experiment_overrides(experiment_cfg: dict[str, Any], bundle: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Promote experiment-level settings into sub-configs that should share them."""

    experiment_cfg_local = deepcopy(experiment_cfg)
    bundle_local = deepcopy(bundle)
    _ensure_bundle_defaults(bundle_local)
    _apply_ablation_overrides(experiment_cfg_local, bundle_local)

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
