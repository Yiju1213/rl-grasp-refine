from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from src.utils.tensor_utils import observation_to_tensor


class PPOAgent:
    """Minimal PPO implementation for the v1 single-step setup."""

    def __init__(self, actor_critic, optimizer, cfg: dict):
        self.actor_critic = actor_critic
        self.optimizer = optimizer
        self.clip_range = float(cfg.get("clip_range", 0.2))
        self.value_loss_coef = float(cfg.get("value_loss_coef", 0.5))
        self.entropy_coef = float(cfg.get("entropy_coef", 0.01))
        self.update_epochs = int(cfg.get("update_epochs", 4))
        self.minibatch_size = int(cfg.get("minibatch_size", 16))
        self.max_grad_norm = float(cfg.get("max_grad_norm", 0.5))
        self.normalize_advantages = bool(cfg.get("normalize_advantages", True))
        self.device = torch.device(cfg.get("device", "cpu"))
        self.actor_critic.to(self.device)

    def update(self, batch: dict) -> dict:
        obs_tensor = observation_to_tensor(batch["obs"]).to(self.device)
        action_tensor = torch.as_tensor(batch["actions"], dtype=torch.float32, device=self.device)
        old_log_probs = torch.as_tensor(batch["log_probs"], dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(batch["returns"], dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor(batch["advantages"], dtype=torch.float32, device=self.device)

        if self.normalize_advantages and advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        batch_size = obs_tensor.shape[0]
        minibatch_size = min(self.minibatch_size, batch_size)
        stats = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "total_loss": [],
            "approx_kl": [],
        }

        for _ in range(self.update_epochs):
            permutation = torch.randperm(batch_size, device=self.device)
            for start in range(0, batch_size, minibatch_size):
                indices = permutation[start : start + minibatch_size]
                obs_mb = obs_tensor[indices]
                action_mb = action_tensor[indices]
                old_log_prob_mb = old_log_probs[indices]
                returns_mb = returns[indices]
                advantages_mb = advantages[indices]

                log_prob, entropy, value = self.actor_critic.evaluate_actions(obs_mb, action_mb)
                ratios = torch.exp(log_prob - old_log_prob_mb)
                surrogate_1 = ratios * advantages_mb
                surrogate_2 = torch.clamp(
                    ratios,
                    1.0 - self.clip_range,
                    1.0 + self.clip_range,
                ) * advantages_mb
                policy_loss = -torch.min(surrogate_1, surrogate_2).mean()
                value_loss = F.mse_loss(value, returns_mb)
                entropy_mean = entropy.mean()
                total_loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_mean

                self.optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                approx_kl = (old_log_prob_mb - log_prob).mean().abs()
                stats["policy_loss"].append(float(policy_loss.detach().cpu().item()))
                stats["value_loss"].append(float(value_loss.detach().cpu().item()))
                stats["entropy"].append(float(entropy_mean.detach().cpu().item()))
                stats["total_loss"].append(float(total_loss.detach().cpu().item()))
                stats["approx_kl"].append(float(approx_kl.detach().cpu().item()))

        return {key: float(np.mean(values)) if values else 0.0 for key, values in stats.items()}
