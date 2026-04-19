from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PreparedCNNMCAInputs:
    """CPU-side inputs matching the CNNMCA 2D VTG inference contract."""

    visual_img: np.ndarray
    tactile_img: np.ndarray

    def __post_init__(self) -> None:
        self.visual_img = self._validate_image(self.visual_img, name="visual_img")
        self.tactile_img = self._validate_image(self.tactile_img, name="tactile_img")

    @staticmethod
    def _validate_image(value, *, name: str) -> np.ndarray:
        image = np.asarray(value, dtype=np.float32)
        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError(f"{name} must have shape (3, H, W). Got {image.shape}.")
        return image


@dataclass
class CNNMCAInferenceResult:
    """Single CNNMCA runtime inference result."""

    prepared_inputs: PreparedCNNMCAInputs
    body_feature: np.ndarray
    raw_logit: float

    def __post_init__(self) -> None:
        self.body_feature = np.asarray(self.body_feature, dtype=np.float32).reshape(-1)
        self.raw_logit = float(self.raw_logit)
