from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PreparedVTGInputs:
    """CPU-side inputs matching the VTG3D preprocessing contract."""

    sc_input: np.ndarray
    gs_input: np.ndarray
    zero_mean: np.ndarray
    debug_visual_world_points: np.ndarray
    debug_tactile_left_world_points: np.ndarray
    debug_tactile_right_world_points: np.ndarray
    debug_tactile_left_contact_world_points: np.ndarray
    debug_tactile_right_contact_world_points: np.ndarray
    debug_tactile_left_gel_mask: np.ndarray
    debug_tactile_right_gel_mask: np.ndarray

    def __post_init__(self) -> None:
        self.sc_input = np.asarray(self.sc_input, dtype=np.float32).reshape(-1, 3)
        self.gs_input = np.asarray(self.gs_input, dtype=np.float32).reshape(-1, 4)
        self.zero_mean = np.asarray(self.zero_mean, dtype=np.float32).reshape(3)
        self.debug_visual_world_points = np.asarray(self.debug_visual_world_points, dtype=np.float32).reshape(-1, 3)
        self.debug_tactile_left_world_points = np.asarray(
            self.debug_tactile_left_world_points, dtype=np.float32
        ).reshape(-1, 3)
        self.debug_tactile_right_world_points = np.asarray(
            self.debug_tactile_right_world_points, dtype=np.float32
        ).reshape(-1, 3)
        self.debug_tactile_left_contact_world_points = np.asarray(
            self.debug_tactile_left_contact_world_points, dtype=np.float32
        ).reshape(-1, 3)
        self.debug_tactile_right_contact_world_points = np.asarray(
            self.debug_tactile_right_contact_world_points, dtype=np.float32
        ).reshape(-1, 3)
        self.debug_tactile_left_gel_mask = np.asarray(self.debug_tactile_left_gel_mask, dtype=bool).reshape(-1)
        self.debug_tactile_right_gel_mask = np.asarray(self.debug_tactile_right_gel_mask, dtype=bool).reshape(-1)


@dataclass
class SGAGSNInferenceResult:
    """Cached result for a single raw observation."""

    prepared_inputs: PreparedVTGInputs
    body_feature: np.ndarray
    raw_logit: float

    def __post_init__(self) -> None:
        self.body_feature = np.asarray(self.body_feature, dtype=np.float32).reshape(-1)
        self.raw_logit = float(self.raw_logit)
