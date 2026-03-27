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
        self.observation_spec = getattr(actor_critic, "observation_spec", None)
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
        obs_tensor = observation_to_tensor(batch["obs"], spec=self.observation_spec).to(self.device)
        action_tensor = torch.as_tensor(batch["actions"], dtype=torch.float32, device=self.device)
        old_log_probs = torch.as_tensor(batch["log_probs"], dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(batch["returns"], dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor(batch["advantages"], dtype=torch.float32, device=self.device)

        if obs_tensor.shape[0] == 0:
            return {
                "ppo/policy_loss": 0.0,
                "ppo/value_loss": 0.0,
                "ppo/entropy": 0.0,
                "ppo/total_loss": 0.0,
                "ppo/approx_kl": 0.0,
                "ppo/clip_fraction": 0.0,
                "ppo/explained_variance": 0.0,
                "ppo/grad_norm": 0.0,
                "ppo/returns_mean": 0.0,
                "ppo/returns_std": 0.0,
                "ppo/advantages_mean": 0.0,
                "ppo/advantages_std": 0.0,
                "ppo/value_pred_mean": 0.0,
                "ppo/value_pred_std": 0.0,
                "ppo/policy_log_std_mean": 0.0,
            }

        raw_advantages = advantages.detach().clone()
        value_predictions = torch.as_tensor(batch["values"], dtype=torch.float32, device=self.device)

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
            "clip_fraction": [],
            "grad_norm": [],
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
                grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                approx_kl = (old_log_prob_mb - log_prob).mean().abs()
                clip_fraction = (torch.abs(ratios - 1.0) > self.clip_range).float().mean()
                stats["policy_loss"].append(float(policy_loss.detach().cpu().item()))
                stats["value_loss"].append(float(value_loss.detach().cpu().item()))
                stats["entropy"].append(float(entropy_mean.detach().cpu().item()))
                stats["total_loss"].append(float(total_loss.detach().cpu().item()))
                stats["approx_kl"].append(float(approx_kl.detach().cpu().item()))
                stats["clip_fraction"].append(float(clip_fraction.detach().cpu().item()))
                stats["grad_norm"].append(float(torch.as_tensor(grad_norm).detach().cpu().item()))

        returns_np = returns.detach().cpu().numpy()
        raw_advantages_np = raw_advantages.detach().cpu().numpy()
        value_predictions_np = value_predictions.detach().cpu().numpy()
        residual_var = float(np.var(returns_np - value_predictions_np))
        returns_var = float(np.var(returns_np))
        explained_variance = 0.0 if returns_var <= 1e-8 else float(1.0 - residual_var / returns_var)
        policy_log_std_mean = float(self.actor_critic.policy_net.log_std.detach().mean().cpu().item())

        aggregated = {key: float(np.mean(values)) if values else 0.0 for key, values in stats.items()}
        aggregated.update(
            {
                "explained_variance": explained_variance,
                "returns_mean": float(np.mean(returns_np)),
                "returns_std": float(np.std(returns_np)),
                "advantages_mean": float(np.mean(raw_advantages_np)),
                "advantages_std": float(np.std(raw_advantages_np)),
                "value_pred_mean": float(np.mean(value_predictions_np)),
                "value_pred_std": float(np.std(value_predictions_np)),
                "policy_log_std_mean": policy_log_std_mean,
            }
        )
        return {f"ppo/{key}": float(value) for key, value in aggregated.items()}
