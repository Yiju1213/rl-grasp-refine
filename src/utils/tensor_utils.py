from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch

from src.rl.observation_spec import PolicyObservationSpec, flatten_single_observation
from src.structures.observation import Observation


def _default_observation_spec(obs: Observation) -> PolicyObservationSpec:
    return PolicyObservationSpec(
        latent_dim=int(obs.latent_feature.shape[0]),
        components=(
            "latent_feature",
            "contact_semantic",
            "grasp_position",
            "grasp_rotation",
            "raw_stability_logit",
        ),
        preset="current",
    )


def observation_to_tensor(
    obs: Observation | Iterable[Observation],
    spec: PolicyObservationSpec | None = None,
) -> torch.Tensor:
    """Flatten structured observations into policy/value input tensors."""

    if isinstance(obs, Observation):
        resolved_spec = spec or _default_observation_spec(obs)
        return torch.from_numpy(flatten_single_observation(obs, resolved_spec)).float().unsqueeze(0)

    if not isinstance(obs, Iterable):
        raise TypeError("observation_to_tensor expects an Observation or iterable of Observation.")

    obs_list = list(obs)
    if not obs_list:
        raise ValueError("Cannot convert an empty observation batch to a tensor.")
    resolved_spec = spec or _default_observation_spec(obs_list[0])
    flattened = [flatten_single_observation(item, resolved_spec) for item in obs_list]
    if not flattened:
        raise ValueError("Cannot convert an empty observation batch to a tensor.")
    return torch.from_numpy(np.stack(flattened, axis=0)).float()


def action_tensor_to_numpy(action_tensor: torch.Tensor) -> np.ndarray:
    """Detach an action tensor and return a numpy array."""

    return action_tensor.detach().cpu().numpy().astype(np.float32)
