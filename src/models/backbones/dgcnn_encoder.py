from __future__ import annotations

import torch
import torch.nn as nn

from src.models.backbones.base_backbone import BaseBackbone


def _summarize_tensor(tensor: torch.Tensor | None, batch_size: int, target_dim: int) -> torch.Tensor:
    if tensor is None:
        return torch.zeros(batch_size, target_dim)

    if tensor.dim() == 2:
        flat = tensor
    else:
        flat = tensor.reshape(tensor.shape[0], -1)
    if flat.shape[1] == 0:
        return torch.zeros(batch_size, target_dim, dtype=flat.dtype, device=flat.device)
    mean = flat.mean(dim=1, keepdim=True)
    std = flat.std(dim=1, keepdim=True, unbiased=False)
    min_value = flat.min(dim=1, keepdim=True).values
    max_value = flat.max(dim=1, keepdim=True).values
    l2 = torch.linalg.norm(flat, dim=1, keepdim=True)
    size = torch.full_like(mean, flat.shape[1], dtype=flat.dtype)
    stats = torch.cat([mean, std, min_value, max_value, l2, size], dim=1)
    return stats[:, :target_dim]


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
