from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class RewardBreakdown:
    """Reward decomposition tracked for logging and debugging."""

    total: float
    drop: float
    stability: float
    contact: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)
