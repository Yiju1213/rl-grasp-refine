from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple


class BaseCalibrator(ABC):
    """Base interface for online calibration."""

    @abstractmethod
    def predict(self, logits) -> Tuple[float, float]:
        raise NotImplementedError

    @abstractmethod
    def update(self, logits, labels) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_state(self) -> dict:
        raise NotImplementedError

    @abstractmethod
    def load_state(self, state: dict) -> None:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError
