from __future__ import annotations

from abc import ABC, abstractmethod


class BaseCalibrator(ABC):
    """Base interface for online calibration."""

    @abstractmethod
    def predict(self, logits):
        raise NotImplementedError

    @abstractmethod
    def update(self, logits, labels) -> None:
        raise NotImplementedError

    @abstractmethod
    def posterior_trace(self) -> float:
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
