from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class StepInfo:
    """Step-level diagnostics returned by the environment."""

    drop_success: int
    calibrated_stability_before: float
    calibrated_stability_after: float
    uncertainty_before: float
    uncertainty_after: float
    reward_drop: float
    reward_stability: float
    reward_contact: float
    extra: dict[str, Any]
