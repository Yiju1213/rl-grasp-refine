from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class NormalizedAction:
    """Normalized policy action in [-1, 1]^6."""

    value: np.ndarray

    def __post_init__(self) -> None:
        value = np.asarray(self.value, dtype=np.float32)
        if value.shape != (6,):
            raise ValueError(f"NormalizedAction must have shape (6,), got {value.shape}")
        self.value = np.clip(value, -1.0, 1.0)


@dataclass
class PhysicalAction:
    """Physical refinement delta in workspace units."""

    delta_translation: np.ndarray
    delta_rotation: np.ndarray

    def __post_init__(self) -> None:
        translation = np.asarray(self.delta_translation, dtype=np.float32)
        rotation = np.asarray(self.delta_rotation, dtype=np.float32)
        if translation.shape != (3,):
            raise ValueError(f"delta_translation must have shape (3,), got {translation.shape}")
        if rotation.shape != (3,):
            raise ValueError(f"delta_rotation must have shape (3,), got {rotation.shape}")
        self.delta_translation = translation
        self.delta_rotation = rotation


@dataclass
class GraspPose:
    """Pose expressed as position + rotation vector."""

    position: np.ndarray
    rotation: np.ndarray

    def __post_init__(self) -> None:
        position = np.asarray(self.position, dtype=np.float32)
        rotation = np.asarray(self.rotation, dtype=np.float32)
        if position.shape != (3,):
            raise ValueError(f"position must have shape (3,), got {position.shape}")
        if rotation.shape != (3,):
            raise ValueError(f"rotation must have shape (3,), got {rotation.shape}")
        self.position = position
        self.rotation = rotation

    def as_array(self) -> np.ndarray:
        return np.concatenate([self.position, self.rotation], axis=0).astype(np.float32)
