from __future__ import annotations

import unittest

import numpy as np
import torch

from src.rl.ppo_agent import PPOAgent
from src.rl.rollout_buffer import RolloutBuffer
from src.rl.trainer import Trainer
from src.runtime.builders import build_actor_critic as runtime_build_actor_critic
from src.runtime.experiment_config import apply_experiment_overrides
from src.structures.action import GraspPose
from src.structures.info import StepInfo
from src.structures.observation import Observation
from src.utils.tensor_utils import observation_to_tensor
from tests.fakes import (
    DummyLogger,
    build_test_actor_critic,
    build_test_env,
    make_rl_cfg,
    make_actor_critic_cfg,
    make_calibration_cfg,
    make_env_cfg,
    make_perception_cfg,
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
            "outcome/drop_rate_after_given_dataset_positive",
            "outcome/hold_rate_after_given_dataset_negative",
            "outcome/dataset_positive_count",
            "outcome/dataset_negative_count",
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

    def test_validation_runs_without_mutating_actor_or_calibrator(self):
        env, calibrator, _, _, _ = build_test_env()
        validation_env, _, _, _, _ = build_test_env(seed=11)
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
            validation_env=validation_env,
            validation_cfg={"enabled": True, "every_n_iterations": 1, "num_episodes": 2},
        )

        trainer.iteration = 0
        actor_state_before = {key: value.detach().clone() for key, value in actor_critic.state_dict().items()}
        calibrator_state_before = calibrator.get_state()

        validation_stats, validation_wall_s = trainer.run_validation(calibrator_state=calibrator.get_state())

        self.assertGreaterEqual(validation_wall_s, 0.0)
        self.assertIn("validation/outcome/success_rate_live_after", validation_stats)
        self.assertIn("validation/reward/total_mean", validation_stats)
        self.assertIn("validation/contact/t_cover_after_mean", validation_stats)
        self.assertIn("validation/calibrator/prob_after_mean", validation_stats)
        for key, value_before in actor_state_before.items():
            self.assertTrue(torch.equal(value_before, actor_critic.state_dict()[key]))
        calibrator_state_after = calibrator.get_state()
        self.assertEqual(float(calibrator_state_before["a"]), float(calibrator_state_after["a"]))
        self.assertEqual(float(calibrator_state_before["b"]), float(calibrator_state_after["b"]))
        np.testing.assert_allclose(calibrator_state_before["posterior_cov"], calibrator_state_after["posterior_cov"])

    def test_rollout_summary_reports_dataset_conditioned_outcome_rates(self):
        obs_list = [
            Observation(
                latent_feature=np.zeros(4, dtype=np.float32),
                contact_semantic=np.asarray([0.2, 0.1], dtype=np.float32),
                grasp_pose=GraspPose(position=np.zeros(3, dtype=np.float32), rotation=np.zeros(3, dtype=np.float32)),
                raw_stability_logit=0.0,
            )
            for _ in range(4)
        ]
        next_obs_list = [
            Observation(
                latent_feature=np.zeros(4, dtype=np.float32),
                contact_semantic=np.asarray([0.3, 0.2], dtype=np.float32),
                grasp_pose=GraspPose(position=np.zeros(3, dtype=np.float32), rotation=np.zeros(3, dtype=np.float32)),
                raw_stability_logit=0.1,
            )
            for _ in range(4)
        ]
        infos = [
            StepInfo(
                drop_success=0,
                calibrated_stability_before=0.4,
                calibrated_stability_after=0.5,
                posterior_trace=2.0,
                reward_drop=-1.0,
                reward_stability=0.1,
                reward_contact=0.0,
                extra={"legacy_drop_success_before": 1.0},
            ),
            StepInfo(
                drop_success=1,
                calibrated_stability_before=0.4,
                calibrated_stability_after=0.6,
                posterior_trace=2.0,
                reward_drop=1.0,
                reward_stability=0.2,
                reward_contact=0.0,
                extra={"legacy_drop_success_before": 1.0},
            ),
            StepInfo(
                drop_success=1,
                calibrated_stability_before=0.3,
                calibrated_stability_after=0.5,
                posterior_trace=2.0,
                reward_drop=1.0,
                reward_stability=0.2,
                reward_contact=0.0,
                extra={"legacy_drop_success_before": 0.0},
            ),
            StepInfo(
                drop_success=0,
                calibrated_stability_before=0.3,
                calibrated_stability_after=0.4,
                posterior_trace=2.0,
                reward_drop=-1.0,
                reward_stability=0.1,
                reward_contact=0.0,
                extra={"legacy_drop_success_before": 0.0},
            ),
        ]
        stats = Trainer._summarize_rollout(
            batch={
                "rewards": np.asarray([-0.9, 1.2, 1.2, -0.9], dtype=np.float32),
                "infos": infos,
                "actions": np.zeros((4, 6), dtype=np.float32),
                "obs": obs_list,
                "next_obs": next_obs_list,
                "raw_logit_before": np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
                "raw_logit_after": np.asarray([0.1, 0.1, 0.1, 0.1], dtype=np.float32),
            },
            collection_report={"attempts_total": 4, "valid_episodes": 4, "attempt_summaries": []},
            calibrator_post_state={"a": 1.0, "b": 0.0, "posterior_cov": np.eye(2, dtype=np.float32)},
            timing_stats={},
        )

        self.assertEqual(stats["outcome/dataset_positive_count"], 2.0)
        self.assertEqual(stats["outcome/dataset_negative_count"], 2.0)
        self.assertAlmostEqual(stats["outcome/drop_rate_after_given_dataset_positive"], 0.5, places=7)
        self.assertAlmostEqual(stats["outcome/hold_rate_after_given_dataset_negative"], 0.5, places=7)

    def test_joint_tactile_ablation_removes_policy_input_but_keeps_contact_metrics(self):
        _, bundle = apply_experiment_overrides(
            {"ablation": {"id": "wo-tac-sem-n-rwd"}},
            {
                "env": make_env_cfg(),
                "perception": make_perception_cfg(),
                "calibration": make_calibration_cfg(),
                "actor_critic": make_actor_critic_cfg(),
                "rl": make_rl_cfg(),
            },
        )

        env, calibrator, env_cfg, perception_cfg, _ = build_test_env(
            env_cfg=bundle["env"],
            perception_cfg=bundle["perception"],
            calibration_cfg=bundle["calibration"],
        )
        actor_critic = runtime_build_actor_critic(perception_cfg, bundle["actor_critic"])
        obs = env.reset()
        obs_tensor = observation_to_tensor(obs, spec=actor_critic.observation_spec)

        self.assertEqual(actor_critic.observation_spec.components, ("latent_feature", "grasp_position", "grasp_rotation", "raw_stability_logit"))
        self.assertEqual(tuple(obs_tensor.shape), (1, 39))

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
        self.assertEqual(env_cfg["reward"]["contact_weight"], 0.0)
        self.assertEqual(history[0]["reward/contact_mean"], 0.0)
        self.assertGreater(history[0]["contact/t_cover_after_mean"], 0.0)
        self.assertGreater(history[0]["contact/t_edge_after_mean"], 0.0)


if __name__ == "__main__":
    unittest.main()
