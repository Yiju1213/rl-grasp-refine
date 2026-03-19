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
