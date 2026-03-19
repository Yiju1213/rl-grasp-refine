from __future__ import annotations

import unittest

import torch

from src.rl.ppo_agent import PPOAgent
from src.rl.rollout_buffer import RolloutBuffer
from src.rl.trainer import Trainer
from src.rl.vec_env_wrapper import DummyVecEnvWrapper
from src.utils.tensor_utils import observation_to_tensor
from tests.fakes import (
    DummyLogger,
    build_test_actor_critic,
    build_test_env,
    make_rl_cfg,
)


class TestSingleStepPipeline(unittest.TestCase):
    def test_env_to_policy_to_step_runs(self):
        env, _, _, perception_cfg, _ = build_test_env()
        obs = env.reset()
        obs_dim = observation_to_tensor(obs).shape[-1]
        actor_critic, _ = build_test_actor_critic(obs_dim)
        obs_tensor = observation_to_tensor(obs)
        action_tensor, _, _, _ = actor_critic.act(obs_tensor)
        next_obs, reward, done, info = env.step(action_tensor.squeeze(0).detach().cpu().numpy())
        self.assertTrue(done)
        self.assertIsInstance(reward, float)
        self.assertIn("reward_breakdown", info.extra)
        self.assertEqual(next_obs.latent_feature.shape[0], int(perception_cfg["backbone"]["latent_dim"]))

    def test_trainer_runs_single_iteration(self):
        env, calibrator, _, _, _ = build_test_env()
        obs_dim = observation_to_tensor(env.reset()).shape[-1]
        actor_critic, _ = build_test_actor_critic(obs_dim)
        rl_cfg = make_rl_cfg()
        optimizer = torch.optim.Adam(actor_critic.parameters(), lr=float(rl_cfg["learning_rate"]))
        agent = PPOAgent(actor_critic=actor_critic, optimizer=optimizer, cfg=rl_cfg)
        trainer = Trainer(
            env=env,
            actor_critic=actor_critic,
            agent=agent,
            buffer=RolloutBuffer(),
            calibrator=calibrator,
            logger=DummyLogger(),
            cfg=rl_cfg,
        )
        history = trainer.train(num_iterations=1)
        self.assertEqual(len(history), 1)
        self.assertIn("average_reward", history[0])

    def test_dummy_vec_env_wrapper_shapes(self):
        env_fns = [lambda seed=seed: build_test_env(seed=seed)[0] for seed in (1, 2)]
        vec_env = DummyVecEnvWrapper(env_fns)
        observations = vec_env.reset()
        self.assertEqual(len(observations), 2)
        actions = [torch.zeros(6).numpy(), torch.zeros(6).numpy()]
        next_obs, rewards, dones, infos = vec_env.step(actions)
        self.assertEqual(len(next_obs), 2)
        self.assertEqual(rewards.shape, (2,))
        self.assertEqual(dones.shape, (2,))
        self.assertEqual(len(infos), 2)


if __name__ == "__main__":
    unittest.main()
