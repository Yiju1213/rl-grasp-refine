from __future__ import annotations

from abc import ABC, abstractmethod

import torch.nn as nn


class BasePredictor(nn.Module, ABC):
    """Abstract predictor interface."""

    @abstractmethod
    def forward(self, **kwargs):
        raise NotImplementedError
