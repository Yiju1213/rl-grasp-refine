from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.structures.action import GraspPose


@dataclass
class Observation:
    """Structured observation used throughout the non-RL stack."""

    latent_feature: np.ndarray
    contact_semantic: np.ndarray
    grasp_pose: GraspPose
    raw_stability_logit: float

    def __post_init__(self) -> None:
        self.latent_feature = np.asarray(self.latent_feature, dtype=np.float32).reshape(-1)
        self.contact_semantic = np.asarray(self.contact_semantic, dtype=np.float32).reshape(-1)
        self.raw_stability_logit = float(self.raw_stability_logit)


@dataclass
class RawSensorObservation:
    """Raw observation container returned by the scene."""

    visual_data: Any
    tactile_data: Any
    grasp_metadata: dict[str, Any]
