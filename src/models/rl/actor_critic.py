from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal


class ActorCritic(nn.Module):
    """Joint actor-critic wrapper used by PPO."""

    def __init__(self, policy_net, value_net):
        super().__init__()
        self.policy_net = policy_net
        self.value_net = value_net

    def _distribution(self, obs_tensor: torch.Tensor) -> Normal:
        action_mean, action_log_std = self.policy_net(obs_tensor)
        action_std = action_log_std.exp().clamp(min=1e-4, max=10.0)
        return Normal(action_mean, action_std)

    def act(self, obs_tensor: torch.Tensor, deterministic: bool = False):
        dist = self._distribution(obs_tensor)
        if deterministic:
            action = dist.mean.clamp(-1.0, 1.0)
        else:
            action = dist.rsample().clamp(-1.0, 1.0)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.value_net(obs_tensor).squeeze(-1)
        return action, log_prob, value, entropy

    def evaluate_actions(self, obs_tensor: torch.Tensor, action_tensor: torch.Tensor):
        dist = self._distribution(obs_tensor)
        clipped_action = action_tensor.clamp(-1.0, 1.0)
        log_prob = dist.log_prob(clipped_action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.value_net(obs_tensor).squeeze(-1)
        return log_prob, entropy, value
