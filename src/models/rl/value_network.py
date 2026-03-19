from __future__ import annotations

import torch
import torch.nn as nn

from src.models.rl.policy_network import _build_mlp


class ValueNetwork(nn.Module):
    """Minimal value network."""

    def __init__(self, obs_dim: int, cfg: dict):
        super().__init__()
        hidden_dims = [int(dim) for dim in cfg.get("value_hidden_dims", [128, 128])]
        self.backbone = _build_mlp(obs_dim, hidden_dims, 1)

    def forward(self, obs_tensor: torch.Tensor):
        return self.backbone(obs_tensor)
