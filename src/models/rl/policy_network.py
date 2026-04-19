from __future__ import annotations

import torch
import torch.nn as nn


def _build_mlp(input_dim: int, hidden_dims: list[int], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU()])
        prev_dim = hidden_dim
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)


def resolve_actor_critic_architecture_type(cfg: dict) -> str:
    architecture_cfg = dict(cfg.get("architecture", {}))
    return str(architecture_cfg.get("type", "plain") or "plain").strip().lower()


def split_late_fusion_hidden_dims(hidden_dims: list[int], *, network_name: str) -> tuple[int, list[int]]:
    if len(hidden_dims) < 2:
        raise ValueError(
            f"{network_name} with latent_first_late_fusion requires at least two hidden dims, "
            f"got {hidden_dims}."
        )
    return int(hidden_dims[0]), [int(dim) for dim in hidden_dims[1:]]


def _build_late_fusion_trunk(input_dim: int, hidden_dims: list[int], output_dim: int) -> nn.Sequential:
    return _build_mlp(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=output_dim)


class PolicyNetwork(nn.Module):
    """Minimal Gaussian policy network."""

    def __init__(self, obs_dim: int, action_dim: int, cfg: dict):
        super().__init__()
        hidden_dims = [int(dim) for dim in cfg.get("policy_hidden_dims", [128, 128])]
        initial_log_std = float(cfg.get("initial_log_std", -0.5))
        self.backbone = _build_mlp(obs_dim, hidden_dims, action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), initial_log_std))

    def forward(self, obs_tensor: torch.Tensor):
        action_mean = torch.tanh(self.backbone(obs_tensor))
        action_log_std = self.log_std.unsqueeze(0).expand_as(action_mean)
        return action_mean, action_log_std


class LatentFirstLateFusionPolicyNetwork(nn.Module):
    """Policy network that consumes latent features first and injects aux features after the first layer."""

    def __init__(self, latent_dim: int, aux_dim: int, action_dim: int, cfg: dict):
        super().__init__()
        hidden_dims = [int(dim) for dim in cfg.get("policy_hidden_dims", [128, 128])]
        latent_hidden_dim, trunk_hidden_dims = split_late_fusion_hidden_dims(
            hidden_dims,
            network_name="PolicyNetwork",
        )
        initial_log_std = float(cfg.get("initial_log_std", -0.5))
        self.latent_dim = int(latent_dim)
        self.aux_dim = int(aux_dim)
        self.latent_layer = nn.Linear(self.latent_dim, latent_hidden_dim)
        self.trunk = _build_late_fusion_trunk(
            input_dim=latent_hidden_dim + self.aux_dim,
            hidden_dims=trunk_hidden_dims,
            output_dim=action_dim,
        )
        self.log_std = nn.Parameter(torch.full((action_dim,), initial_log_std))

    def forward(self, obs_tensor: torch.Tensor):
        latent = obs_tensor[:, : self.latent_dim]
        aux = obs_tensor[:, self.latent_dim :]
        latent_hidden = torch.relu(self.latent_layer(latent))
        fused = torch.cat([latent_hidden, aux], dim=-1) if self.aux_dim > 0 else latent_hidden
        action_mean = torch.tanh(self.trunk(fused))
        action_log_std = self.log_std.unsqueeze(0).expand_as(action_mean)
        return action_mean, action_log_std
