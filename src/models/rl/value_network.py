from __future__ import annotations

import torch
import torch.nn as nn

from src.models.rl.policy_network import _build_mlp, validate_late_fusion_hidden_dims


class ValueNetwork(nn.Module):
    """Minimal value network."""

    def __init__(self, obs_dim: int, cfg: dict):
        super().__init__()
        hidden_dims = [int(dim) for dim in cfg.get("value_hidden_dims", [128, 128])]
        self.backbone = _build_mlp(obs_dim, hidden_dims, 1)

    def forward(self, obs_tensor: torch.Tensor):
        return self.backbone(obs_tensor)


class LatentFirstLateFusionValueNetwork(nn.Module):
    """Value network that consumes latent features first and injects aux features after the first layer."""

    def __init__(self, latent_dim: int, aux_dim: int, cfg: dict):
        super().__init__()
        hidden_dims = [int(dim) for dim in cfg.get("value_hidden_dims", [128, 128])]
        latent_hidden_dim, trunk_hidden_dim = validate_late_fusion_hidden_dims(
            hidden_dims,
            network_name="ValueNetwork",
        )
        self.latent_dim = int(latent_dim)
        self.aux_dim = int(aux_dim)
        self.latent_layer = nn.Linear(self.latent_dim, latent_hidden_dim)
        self.trunk_layer = nn.Linear(latent_hidden_dim + self.aux_dim, trunk_hidden_dim)
        self.output_layer = nn.Linear(trunk_hidden_dim, 1)

    def forward(self, obs_tensor: torch.Tensor):
        latent = obs_tensor[:, : self.latent_dim]
        aux = obs_tensor[:, self.latent_dim :]
        latent_hidden = torch.relu(self.latent_layer(latent))
        fused = torch.cat([latent_hidden, aux], dim=-1) if self.aux_dim > 0 else latent_hidden
        return self.output_layer(torch.relu(self.trunk_layer(fused)))
