from __future__ import annotations

from abc import ABC, abstractmethod

import torch.nn as nn


class BaseBackbone(nn.Module, ABC):
    """Abstract backbone interface."""

    @abstractmethod
    def forward(self, **kwargs):
        raise NotImplementedError
