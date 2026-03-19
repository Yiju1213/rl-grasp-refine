from __future__ import annotations

from dataclasses import dataclass

from src.structures.action import NormalizedAction
from src.structures.info import StepInfo
from src.structures.observation import Observation


@dataclass
class Transition:
    """Single transition stored in the rollout buffer."""

    obs: Observation
    action: NormalizedAction
    reward: float
    next_obs: Observation
    done: bool
    log_prob: float
    value: float
    info: StepInfo
