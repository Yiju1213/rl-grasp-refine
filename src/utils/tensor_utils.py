from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch

from src.structures.observation import Observation


def _flatten_single_observation(obs: Observation) -> np.ndarray:
    return np.concatenate(
        [
            obs.latent_feature.astype(np.float32).reshape(-1),
            obs.contact_semantic.astype(np.float32).reshape(-1),
            obs.grasp_pose.position.astype(np.float32).reshape(-1),
            obs.grasp_pose.rotation.astype(np.float32).reshape(-1),
            np.asarray([obs.raw_stability_logit], dtype=np.float32),
        ],
        axis=0,
    )


def observation_to_tensor(obs: Observation | Iterable[Observation]) -> torch.Tensor:
    """Flatten structured observations into policy/value input tensors."""

    if isinstance(obs, Observation):
        return torch.from_numpy(_flatten_single_observation(obs)).float().unsqueeze(0)

    if not isinstance(obs, Iterable):
        raise TypeError("observation_to_tensor expects an Observation or iterable of Observation.")

    flattened = [_flatten_single_observation(item) for item in obs]
    if not flattened:
        raise ValueError("Cannot convert an empty observation batch to a tensor.")
    return torch.from_numpy(np.stack(flattened, axis=0)).float()


def action_tensor_to_numpy(action_tensor: torch.Tensor) -> np.ndarray:
    """Detach an action tensor and return a numpy array."""

    return action_tensor.detach().cpu().numpy().astype(np.float32)
