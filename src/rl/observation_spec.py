from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.perception.factory import infer_perception_feature_dim
from src.structures.observation import Observation

POLICY_OBSERVATION_COMPONENTS = (
    "latent_feature",
    "contact_semantic",
    "grasp_position",
    "grasp_rotation",
    "raw_stability_logit",
)

_PRESET_COMPONENTS = {
    "current": POLICY_OBSERVATION_COMPONENTS,
    "paper": ("latent_feature", "contact_semantic"),
    "no_pose": ("latent_feature", "contact_semantic", "raw_stability_logit"),
    "no_logit": ("latent_feature", "contact_semantic", "grasp_position", "grasp_rotation"),
}

_STATIC_COMPONENT_DIMS = {
    "contact_semantic": 2,
    "grasp_position": 3,
    "grasp_rotation": 3,
    "raw_stability_logit": 1,
}


@dataclass(frozen=True)
class PolicyObservationSpec:
    latent_dim: int
    components: tuple[str, ...]
    preset: str = "custom"

    @property
    def obs_dim(self) -> int:
        return infer_obs_dim_from_spec(self)


def resolve_policy_observation_spec(perception_cfg: dict, actor_critic_cfg: dict) -> PolicyObservationSpec:
    latent_dim = int(infer_perception_feature_dim(perception_cfg))
    policy_obs_cfg = dict(actor_critic_cfg.get("policy_observation", {}))
    preset = str(policy_obs_cfg.get("preset", "current"))
    components_cfg = policy_obs_cfg.get("components")

    if components_cfg is None:
        if preset not in _PRESET_COMPONENTS:
            raise ValueError(
                f"Unknown policy_observation preset '{preset}'. Expected one of {sorted(_PRESET_COMPONENTS)}."
            )
        components = tuple(_PRESET_COMPONENTS[preset])
        resolved_preset = preset
    else:
        requested = set(components_cfg)
        unknown = requested.difference(POLICY_OBSERVATION_COMPONENTS)
        if unknown:
            raise ValueError(
                f"Unknown policy_observation components: {sorted(unknown)}. "
                f"Expected subset of {POLICY_OBSERVATION_COMPONENTS}."
            )
        components = tuple(component for component in POLICY_OBSERVATION_COMPONENTS if component in requested)
        if not components:
            raise ValueError("policy_observation.components cannot be empty.")
        resolved_preset = "custom"

    return PolicyObservationSpec(
        latent_dim=latent_dim,
        components=components,
        preset=resolved_preset,
    )


def infer_obs_dim_from_spec(spec: PolicyObservationSpec) -> int:
    obs_dim = 0
    for component in spec.components:
        if component == "latent_feature":
            obs_dim += int(spec.latent_dim)
            continue
        obs_dim += int(_STATIC_COMPONENT_DIMS[component])
    return obs_dim


def flatten_single_observation(obs: Observation, spec: PolicyObservationSpec) -> np.ndarray:
    components: list[np.ndarray] = []
    for component in spec.components:
        if component == "latent_feature":
            components.append(obs.latent_feature.astype(np.float32).reshape(-1))
        elif component == "contact_semantic":
            components.append(obs.contact_semantic.astype(np.float32).reshape(-1))
        elif component == "grasp_position":
            components.append(obs.grasp_pose.position.astype(np.float32).reshape(-1))
        elif component == "grasp_rotation":
            components.append(obs.grasp_pose.rotation.astype(np.float32).reshape(-1))
        elif component == "raw_stability_logit":
            components.append(np.asarray([obs.raw_stability_logit], dtype=np.float32))
        else:
            raise ValueError(f"Unsupported policy observation component: {component}")

    return np.concatenate(components, axis=0)

