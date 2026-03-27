from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.calibration.online_logit_calibrator import OnlineLogitCalibrator
from src.rl.ppo_agent import PPOAgent
from src.rl.rollout_buffer import RolloutBuffer
from src.rl.subproc_async_rollout_collector import SubprocAsyncRolloutCollector
from src.rl.trainer import Trainer
from src.runtime.builders import build_actor_critic
from tests.fakes import DummyLogger
from tests.runtime_smoke_utils import assert_real_sgagsn_resources, build_real_training_bundle
from tests.runtime_smoke_utils import configure_headless_render_env


class TestAsyncTrainerRealSGAGSN(unittest.TestCase):
    def test_trainer_train_runs_three_real_iterations_with_three_workers(self):
        assert_real_sgagsn_resources()
        configure_headless_render_env()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir)
            experiment_cfg, bundle = build_real_training_bundle(
                output_root=output_root,
                num_envs=3,
                batch_episodes=9,
                num_iterations=3,
                device="cuda:0",
                worker_policy_device="cuda:0",
            )

            env_cfg = bundle["env"]
            perception_cfg = bundle["perception"]
            calibration_cfg = bundle["calibration"]
            rl_cfg = bundle["rl"]
            actor_critic_cfg = bundle["actor_critic"]

            calibrator = OnlineLogitCalibrator(calibration_cfg)
            actor_critic = build_actor_critic(perception_cfg, actor_critic_cfg)
            actor_before = {key: value.detach().cpu().clone() for key, value in actor_critic.state_dict().items()}
            calibrator_before = calibrator.get_state()

            optimizer = torch.optim.Adam(actor_critic.parameters(), lr=float(rl_cfg.get("learning_rate", 3e-4)))
            agent = PPOAgent(actor_critic=actor_critic, optimizer=optimizer, cfg=rl_cfg)
            collector = SubprocAsyncRolloutCollector(
                env_cfg=env_cfg,
                perception_cfg=perception_cfg,
                calibration_cfg=calibration_cfg,
                actor_critic_cfg=actor_critic_cfg,
                rl_cfg=rl_cfg,
                num_workers=3,
                observation_spec=getattr(actor_critic, "observation_spec", None),
            )
            trainer = Trainer(
                env=None,
                actor_critic=actor_critic,
                agent=agent,
                buffer=RolloutBuffer(),
                calibrator=calibrator,
                logger=DummyLogger(),
                cfg=rl_cfg,
                collector=collector,
            )

            try:
                history = trainer.train(num_iterations=int(experiment_cfg["num_iterations"]))
                debug_states = collector.get_worker_debug_states()
            finally:
                collector.close()

            self.assertEqual(len(history), 3)
            for stats in history:
                for key in (
                    "collection/attempts_total",
                    "collection/valid_episodes",
                    "collection/valid_rate",
                    "outcome/success_rate_live_after",
                    "outcome/success_rate_dataset_before",
                    "reward/total_mean",
                    "contact/t_cover_after_mean",
                    "calibrator/posterior_trace_post_update",
                    "calibrator/after_brier",
                    "ppo/policy_loss",
                    "ppo/value_loss",
                    "ppo/entropy",
                    "ppo/total_loss",
                    "ppo/approx_kl",
                    "ppo/clip_fraction",
                    "ppo/explained_variance",
                    "ppo/grad_norm",
                    "ppo/policy_log_std_mean",
                    "timing/iteration_wall_s",
                ):
                    self.assertIn(key, stats)
                    self.assertTrue(np.isfinite(float(stats[key])), msg=f"{key} is not finite: {stats[key]}")

            actor_after = actor_critic.state_dict()
            self.assertTrue(
                any(not torch.equal(actor_before[key], actor_after[key].detach().cpu()) for key in actor_before),
                msg="Actor-critic parameters did not change after training.",
            )

            calibrator_after = calibrator.get_state()
            self.assertNotEqual(float(calibrator_before["a"]), float(calibrator_after["a"]))
            self.assertNotEqual(float(calibrator_before["b"]), float(calibrator_after["b"]))
            self.assertFalse(np.allclose(calibrator_before["posterior_cov"], calibrator_after["posterior_cov"]))

            self.assertEqual(str(next(actor_critic.parameters()).device), "cuda:0")
            self.assertEqual([state["worker_id"] for state in debug_states], [0, 1, 2])
            self.assertTrue(all(state["rollout_version"] == 2 for state in debug_states))
            self.assertTrue(all(state["device"] == "cuda:0" for state in debug_states))
            self.assertTrue(all(not process.is_alive() for process in collector._processes))


if __name__ == "__main__":
    unittest.main()
