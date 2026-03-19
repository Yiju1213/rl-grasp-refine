from __future__ import annotations

import torch
import torch.nn as nn

from src.models.backbones.base_backbone import BaseBackbone
from src.models.backbones.sga_gsn_encoder import _summarize_tensor


class DGCNNEncoder(BaseBackbone):
    """Stub-compatible alternative backbone."""

    def __init__(self, cfg: dict):
        super().__init__()
        hidden_dim = int(cfg.get("hidden_dim", 64))
        latent_dim = int(cfg.get("latent_dim", 32))
        self.net = nn.Sequential(
            nn.Linear(12, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, **kwargs) -> torch.Tensor:
        point_cloud = kwargs.get("point_cloud")
        tactile = kwargs.get("tactile")
        batch_size = 1
        if point_cloud is not None:
            batch_size = point_cloud.shape[0]
        elif tactile is not None:
            batch_size = tactile.shape[0]
        features = torch.cat(
            [
                _summarize_tensor(point_cloud, batch_size, 6),
                _summarize_tensor(tactile, batch_size, 6),
            ],
            dim=1,
        )
        return self.net(features)
