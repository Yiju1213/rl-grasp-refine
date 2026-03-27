from __future__ import annotations

import torch
import torch.nn as nn

from src.models.predictors.base_predictor import BasePredictor


class StabilityHead(BasePredictor):
    """Small MLP that predicts a raw stability logit."""

    def __init__(self, cfg: dict):
        super().__init__()
        latent_dim = int(cfg.get("latent_dim", 32))
        hidden_dim = int(cfg.get("hidden_dim", 64))
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, **kwargs) -> torch.Tensor:
        latent_feature = kwargs["latent_feature"]
        return self.net(latent_feature).squeeze(-1)
