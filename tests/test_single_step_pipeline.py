from __future__ import annotations

import unittest

import torch

from src.rl.ppo_agent import PPOAgent
from src.rl.rollout_buffer import RolloutBuffer
from src.rl.trainer import Trainer
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
        for key in (
            "collection/attempts_total",
            "collection/valid_episodes",
            "collection/scene_rebuild_performed",
            "collection/scene_rebuild_workers",
            "outcome/success_rate_live_after",
            "reward/total_mean",
            "contact/t_cover_after_mean",
            "calibrator/prob_after_mean",
            "ppo/policy_loss",
            "ppo/clip_fraction",
            "ppo/grad_norm",
            "timing/scene_rebuild_wall_s",
            "timing/iteration_wall_s",
            "system/process_rss_mb",
            "system/process_vms_mb",
        ):
            self.assertIn(key, history[0])

    def test_collect_rollout_returns_collection_report(self):
        env, calibrator, _, _, _ = build_test_env()
        obs_dim = observation_to_tensor(env.reset()).shape[-1]
        actor_critic, _ = build_test_actor_critic(obs_dim)
        trainer = Trainer(
            env=env,
            actor_critic=actor_critic,
            agent=None,
            buffer=RolloutBuffer(),
            calibrator=calibrator,
            logger=DummyLogger(),
            cfg={"batch_episodes": 2, "device": "cpu", "max_collect_attempt_factor": 4},
        )

        report = trainer.collect_rollout(2)
        self.assertEqual(report["valid_episodes"], 2)
        self.assertEqual(len(report["attempt_summaries"]), report["attempts_total"])
        self.assertTrue(all("trial_status" in item for item in report["attempt_summaries"]))

    def test_trainer_can_continue_iteration_numbering(self):
        env, calibrator, _, _, _ = build_test_env()
        obs_dim = observation_to_tensor(env.reset()).shape[-1]
        actor_critic, _ = build_test_actor_critic(obs_dim)
        rl_cfg = make_rl_cfg()
        optimizer = torch.optim.Adam(actor_critic.parameters(), lr=float(rl_cfg["learning_rate"]))
        agent = PPOAgent(actor_critic=actor_critic, optimizer=optimizer, cfg=rl_cfg)
        logger = DummyLogger()
        trainer = Trainer(
            env=env,
            actor_critic=actor_critic,
            agent=agent,
            buffer=RolloutBuffer(),
            calibrator=calibrator,
            logger=logger,
            cfg=rl_cfg,
        )

        history = trainer.train(num_iterations=1, start_iteration=5)
        self.assertEqual(len(history), 1)
        self.assertTrue(any(step == 5 for step, _ in logger.records if isinstance(step, int)))


if __name__ == "__main__":
    unittest.main()
