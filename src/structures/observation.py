from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.structures.action import GraspPose


def _default_action_axes_in_camera() -> np.ndarray:
    return np.eye(3, dtype=np.float32).reshape(-1)


def _default_hand_pose_in_camera() -> np.ndarray:
    return np.concatenate(
        [np.zeros(3, dtype=np.float32), np.eye(3, dtype=np.float32).reshape(-1)],
        axis=0,
    ).astype(np.float32)


@dataclass
class Observation:
    """Structured observation used throughout the non-RL stack."""

    latent_feature: np.ndarray
    contact_semantic: np.ndarray
    grasp_pose: GraspPose
    raw_stability_logit: float
    action_axes_in_camera: np.ndarray = field(default_factory=_default_action_axes_in_camera)
    hand_pose_in_camera: np.ndarray = field(default_factory=_default_hand_pose_in_camera)

    def __post_init__(self) -> None:
        self.latent_feature = np.asarray(self.latent_feature, dtype=np.float32).reshape(-1)
        self.contact_semantic = np.asarray(self.contact_semantic, dtype=np.float32).reshape(-1)
        self.raw_stability_logit = float(self.raw_stability_logit)
        self.action_axes_in_camera = np.asarray(self.action_axes_in_camera, dtype=np.float32).reshape(-1)
        self.hand_pose_in_camera = np.asarray(self.hand_pose_in_camera, dtype=np.float32).reshape(-1)
        if self.action_axes_in_camera.shape != (9,):
            raise ValueError(f"action_axes_in_camera must have shape (9,), got {self.action_axes_in_camera.shape}")
        if self.hand_pose_in_camera.shape != (12,):
            raise ValueError(f"hand_pose_in_camera must have shape (12,), got {self.hand_pose_in_camera.shape}")


@dataclass
class RawSensorObservation:
    """Raw observation container returned by the scene."""

    visual_data: Any
    tactile_data: Any
    grasp_metadata: dict[str, Any]
