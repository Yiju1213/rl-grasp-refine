from __future__ import annotations

from typing import Any

import numpy as np
import torch

from src.rl.advantage import compute_returns_and_advantages
from src.rl.vec_env_wrapper import DummyVecEnvWrapper
from src.structures.action import NormalizedAction
from src.utils.tensor_utils import action_tensor_to_numpy, observation_to_tensor


class Trainer:
    """Coordinate rollout collection, PPO updates, and calibrator updates."""

    def __init__(
        self,
        env,
        actor_critic,
        agent,
        buffer,
        calibrator,
        logger,
        cfg: dict,
    ):
        self.env = env
        self.actor_critic = actor_critic
        self.agent = agent
        self.buffer = buffer
        self.calibrator = calibrator
        self.logger = logger
        self.cfg = cfg
        self.gamma = float(cfg.get("gamma", 0.99))
        self.lam = float(cfg.get("lam", 0.95))
        self.batch_episodes = int(cfg.get("batch_episodes", 32))
        self.device = torch.device(cfg.get("device", "cpu"))
        self.iteration = 0

    def train(self, num_iterations: int):
        history: list[dict[str, Any]] = []
        for iteration in range(num_iterations):
            self.iteration = iteration
            self.collect_rollout(self.batch_episodes)
            batch = self.buffer.get_all()
            returns, advantages = compute_returns_and_advantages(
                rewards=batch["rewards"],
                values=batch["values"],
                dones=batch["dones"],
                gamma=self.gamma,
                lam=self.lam,
            )
            batch["returns"] = returns
            batch["advantages"] = advantages
            training_stats = self.agent.update(batch)
            self.update_calibrator()
            rollout_stats = self._summarize_rollout(batch)
            stats = {**training_stats, **rollout_stats}
            self.log_iteration(stats)
            history.append(stats)
            self.buffer.clear()
        return history

    def collect_rollout(self, num_episodes: int):
        self.buffer.clear()
        if isinstance(self.env, DummyVecEnvWrapper):
            self._collect_rollout_vec(num_episodes)
        else:
            self._collect_rollout_single(num_episodes)

    def _collect_rollout_single(self, num_episodes: int) -> None:
        for _ in range(num_episodes):
            obs = self.env.reset()
            obs_tensor = observation_to_tensor(obs).to(self.device)
            with torch.no_grad():
                action_tensor, log_prob, value, _ = self.actor_critic.act(obs_tensor)
            action_np = action_tensor_to_numpy(action_tensor).reshape(-1)
            next_obs, reward, done, info = self.env.step(NormalizedAction(value=action_np))
            self.buffer.add(
                obs=obs,
                action=NormalizedAction(value=action_np),
                reward=reward,
                next_obs=next_obs,
                done=done,
                log_prob=float(log_prob.squeeze(0).cpu().item()),
                value=float(value.squeeze(0).cpu().item()),
                info=info,
                raw_logit_before=info.extra["raw_logit_before"],
                raw_logit_after=info.extra["raw_logit_after"],
            )

    def _collect_rollout_vec(self, num_episodes: int) -> None:
        episodes_collected = 0
        while episodes_collected < num_episodes:
            obs_batch = self.env.reset()
            obs_tensor = observation_to_tensor(obs_batch).to(self.device)
            with torch.no_grad():
                action_tensor, log_prob, value, _ = self.actor_critic.act(obs_tensor)
            actions_np = action_tensor_to_numpy(action_tensor)
            next_obs_batch, rewards, dones, infos = self.env.step(actions_np)
            remaining = num_episodes - episodes_collected
            keep = min(remaining, len(obs_batch))
            for index in range(keep):
                action = NormalizedAction(value=actions_np[index])
                self.buffer.add(
                    obs=obs_batch[index],
                    action=action,
                    reward=float(rewards[index]),
                    next_obs=next_obs_batch[index],
                    done=bool(dones[index]),
                    log_prob=float(log_prob[index].cpu().item()),
                    value=float(value[index].cpu().item()),
                    info=infos[index],
                    raw_logit_before=infos[index].extra["raw_logit_before"],
                    raw_logit_after=infos[index].extra["raw_logit_after"],
                )
            episodes_collected += keep

    def update_calibrator(self):
        batch = self.buffer.get_all()
        logits = batch["raw_logit_after"]
        labels = np.asarray([info.drop_success for info in batch["infos"]], dtype=np.float32)
        self.calibrator.update(logits, labels)

    def log_iteration(self, stats: dict):
        self.logger.log_dict(stats, step=self.iteration)
        self.logger.info(f"Iteration {self.iteration}: {stats}")

    @staticmethod
    def _summarize_rollout(batch: dict) -> dict[str, float]:
        rewards = batch["rewards"]
        infos = batch["infos"]
        success_rate = float(np.mean([info.drop_success for info in infos])) if infos else 0.0
        return {
            "average_reward": float(np.mean(rewards)) if rewards.size else 0.0,
            "success_rate": success_rate,
            "num_episodes": float(len(infos)),
        }
