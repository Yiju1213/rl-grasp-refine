from __future__ import annotations

import numpy as np

from src.evaluation.metrics import (
    compute_average_reward,
    compute_average_stability_gain,
    compute_success_rate,
)
from src.models.rl.actor_critic import ActorCritic
from src.structures.action import NormalizedAction
from src.utils.tensor_utils import observation_to_tensor


class Evaluator:
    """Evaluate learned and random policies."""

    def __init__(self, env, actor_critic: ActorCritic, cfg: dict):
        self.env = env
        self.actor_critic = actor_critic
        self.cfg = cfg
        self.observation_spec = getattr(actor_critic, "observation_spec", None)

    def run(self, num_episodes: int) -> dict:
        results = []
        device = next(self.actor_critic.parameters()).device
        for _ in range(num_episodes):
            obs = self.env.reset()
            obs_tensor = observation_to_tensor(obs, spec=self.observation_spec).to(device)
            with np.errstate(all="ignore"):
                action_mean, _ = self.actor_critic.policy_net(obs_tensor)
            action = action_mean.squeeze(0).detach().cpu().numpy()
            _, reward, _, info = self.env.step(NormalizedAction(value=action))
            results.append(
                {
                    "reward": reward,
                    "drop_success": info.drop_success,
                    "calibrated_stability_before": info.calibrated_stability_before,
                    "calibrated_stability_after": info.calibrated_stability_after,
                }
            )
        return self._aggregate(results)

    def run_random_policy(self, num_episodes: int) -> dict:
        results = []
        rng = np.random.default_rng(int(self.cfg.get("seed", 0)))
        for _ in range(num_episodes):
            self.env.reset()
            action = rng.uniform(-1.0, 1.0, size=6).astype(np.float32)
            _, reward, _, info = self.env.step(NormalizedAction(value=action))
            results.append(
                {
                    "reward": reward,
                    "drop_success": info.drop_success,
                    "calibrated_stability_before": info.calibrated_stability_before,
                    "calibrated_stability_after": info.calibrated_stability_after,
                }
            )
        return self._aggregate(results)

    @staticmethod
    def _aggregate(results: list[dict]) -> dict:
        return {
            "success_rate": compute_success_rate(results),
            "average_reward": compute_average_reward(results),
            "average_stability_gain": compute_average_stability_gain(results),
        }
