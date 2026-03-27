from __future__ import annotations

import unittest

import numpy as np
import torch

from src.calibration.online_logit_calibrator import OnlineLogitCalibrator
from src.rl.ppo_agent import PPOAgent
from src.rl.rollout_buffer import RolloutBuffer
from src.rl.trainer import Trainer
from src.rl.vec_env_wrapper import DummyVecEnvWrapper
from src.utils.tensor_utils import observation_to_tensor
from tests.fakes import (
    DummyLogger,
    build_test_actor_critic,
    build_test_env,
    make_calibration_cfg,
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

    def test_dummy_vec_env_wrapper_syncs_calibrator_state(self):
        env_fns = [lambda seed=seed: build_test_env(seed=seed)[0] for seed in (1, 2)]
        vec_env = DummyVecEnvWrapper(env_fns)
        state = {
            "a": 2.5,
            "b": -0.75,
            "posterior_cov": np.asarray([[2.0, 0.1], [0.1, 3.0]], dtype=np.float64),
        }

        vec_env.sync_calibrator(state)

        for env in vec_env.envs:
            synced = env.calibrator.get_state()
            self.assertAlmostEqual(synced["a"], state["a"], places=7)
            self.assertAlmostEqual(synced["b"], state["b"], places=7)
            self.assertTrue(np.allclose(synced["posterior_cov"], state["posterior_cov"]))

    def test_trainer_syncs_vec_env_calibrator_before_rollout(self):
        env_fns = [lambda seed=seed: build_test_env(seed=seed)[0] for seed in (1, 2)]
        vec_env = DummyVecEnvWrapper(env_fns)
        main_calibrator = OnlineLogitCalibrator(make_calibration_cfg())
        target_state = {
            "a": 1.8,
            "b": 0.4,
            "posterior_cov": np.asarray([[1.5, 0.0], [0.0, 2.0]], dtype=np.float64),
        }
        main_calibrator.load_state(target_state)

        obs_dim = observation_to_tensor(vec_env.reset()).shape[-1]
        actor_critic, _ = build_test_actor_critic(obs_dim)
        trainer = Trainer(
            env=vec_env,
            actor_critic=actor_critic,
            agent=None,
            buffer=RolloutBuffer(),
            calibrator=main_calibrator,
            logger=DummyLogger(),
            cfg={"batch_episodes": 1, "device": "cpu", "max_collect_attempt_factor": 4},
        )

        trainer.collect_rollout(1)

        for env in vec_env.envs:
            synced = env.calibrator.get_state()
            self.assertAlmostEqual(synced["a"], target_state["a"], places=7)
            self.assertAlmostEqual(synced["b"], target_state["b"], places=7)
            self.assertTrue(np.allclose(synced["posterior_cov"], target_state["posterior_cov"]))


if __name__ == "__main__":
    unittest.main()
