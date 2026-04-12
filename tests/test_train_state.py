from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.calibration.online_logit_calibrator import OnlineLogitCalibrator
from src.evaluation.best_checkpoint_pipeline import restore_evaluation_state
from src.runtime.train_state import resolve_remaining_training_iterations, restore_training_state
from src.utils.checkpoint import save_checkpoint
from tests.fakes import build_test_actor_critic, make_actor_critic_cfg, make_calibration_cfg


class TestTrainState(unittest.TestCase):
    def test_resolve_remaining_training_iterations_uses_num_iterations_as_global_target(self):
        self.assertEqual(
            resolve_remaining_training_iterations(target_iterations=500, start_iteration=0),
            500,
        )
        self.assertEqual(
            resolve_remaining_training_iterations(target_iterations=500, start_iteration=120),
            380,
        )
        self.assertEqual(
            resolve_remaining_training_iterations(target_iterations=500, start_iteration=500),
            0,
        )
        self.assertEqual(
            resolve_remaining_training_iterations(target_iterations=500, start_iteration=620),
            0,
        )

    def test_resolve_remaining_training_iterations_rejects_negative_values(self):
        with self.assertRaises(ValueError):
            resolve_remaining_training_iterations(target_iterations=-1, start_iteration=0)
        with self.assertRaises(ValueError):
            resolve_remaining_training_iterations(target_iterations=1, start_iteration=-1)

    def test_restore_training_state_loads_actor_optimizer_calibrator_and_history(self):
        actor_critic, _ = build_test_actor_critic(obs_dim=41)
        optimizer = torch.optim.Adam(actor_critic.parameters(), lr=3e-4)
        calibrator = OnlineLogitCalibrator(make_calibration_cfg())

        obs = torch.randn(2, 41)
        action, log_prob, value, entropy = actor_critic.act(obs)
        loss = -(log_prob.mean() + value.mean() + entropy.mean() * 0.01)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        calibrator.update(np.asarray([0.2, 0.8], dtype=np.float32), np.asarray([0.0, 1.0], dtype=np.float32))

        history = [{"ppo/policy_loss": 1.0}, {"ppo/policy_loss": 0.5}]
        actor_state = {key: value.detach().clone() for key, value in actor_critic.state_dict().items()}
        calibrator_state = calibrator.get_state()

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "resume.pt"
            save_checkpoint(
                checkpoint_path,
                {
                    "actor_critic": actor_critic.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "history": history,
                    "completed_iterations": 2,
                    "calibrator": calibrator_state,
                    "experiment_cfg": {"seed": 7},
                },
            )

            restored_actor_critic, _ = build_test_actor_critic(obs_dim=41)
            restored_optimizer = torch.optim.Adam(restored_actor_critic.parameters(), lr=3e-4)
            restored_calibrator = OnlineLogitCalibrator(make_calibration_cfg())
            restored = restore_training_state(
                checkpoint_path=checkpoint_path,
                actor_critic=restored_actor_critic,
                optimizer=restored_optimizer,
                calibrator=restored_calibrator,
                device=torch.device("cpu"),
            )

        self.assertEqual(restored["completed_iterations"], 2)
        self.assertEqual(restored["history"], history)
        for key, value in actor_state.items():
            self.assertTrue(torch.equal(value, restored_actor_critic.state_dict()[key]))
        self.assertEqual(float(restored_calibrator.get_state()["a"]), float(calibrator_state["a"]))
        self.assertEqual(float(restored_calibrator.get_state()["b"]), float(calibrator_state["b"]))
        self.assertTrue(
            np.allclose(
                restored_calibrator.get_state()["posterior_cov"],
                calibrator_state["posterior_cov"],
            )
        )
        self.assertTrue(restored_optimizer.state)

    def test_restore_training_state_roundtrip_for_late_fusion_actor_critic(self):
        actor_critic_cfg = make_actor_critic_cfg()
        actor_critic_cfg["architecture"] = {"type": "latent_first_late_fusion"}
        actor_critic, _ = build_test_actor_critic(obs_dim=41, actor_critic_cfg=actor_critic_cfg, latent_dim=32)
        optimizer = torch.optim.Adam(actor_critic.parameters(), lr=3e-4)
        calibrator = OnlineLogitCalibrator(make_calibration_cfg())

        obs = torch.randn(2, 41)
        action, log_prob, value, entropy = actor_critic.act(obs)
        loss = -(log_prob.mean() + value.mean() + entropy.mean() * 0.01)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        actor_state = {key: value.detach().clone() for key, value in actor_critic.state_dict().items()}

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "resume_late_fusion.pt"
            save_checkpoint(
                checkpoint_path,
                {
                    "actor_critic": actor_critic.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "history": [],
                    "completed_iterations": 0,
                    "calibrator": calibrator.get_state(),
                    "experiment_cfg": {"seed": 7},
                },
            )

            restored_actor_critic, _ = build_test_actor_critic(
                obs_dim=41,
                actor_critic_cfg=actor_critic_cfg,
                latent_dim=32,
            )
            restored_optimizer = torch.optim.Adam(restored_actor_critic.parameters(), lr=3e-4)
            restored_calibrator = OnlineLogitCalibrator(make_calibration_cfg())
            restore_training_state(
                checkpoint_path=checkpoint_path,
                actor_critic=restored_actor_critic,
                optimizer=restored_optimizer,
                calibrator=restored_calibrator,
                device=torch.device("cpu"),
            )

        for key, value in actor_state.items():
            self.assertTrue(torch.equal(value, restored_actor_critic.state_dict()[key]))

    def test_restore_evaluation_state_loads_actor_and_calibrator_without_optimizer(self):
        actor_critic, _ = build_test_actor_critic(obs_dim=41)
        calibrator = OnlineLogitCalibrator(make_calibration_cfg())

        obs = torch.randn(2, 41)
        action, log_prob, value, entropy = actor_critic.act(obs)
        loss = -(log_prob.mean() + value.mean() + entropy.mean() * 0.01)
        loss.backward()
        with torch.no_grad():
            for parameter in actor_critic.parameters():
                if parameter.grad is not None:
                    parameter.add_(0.01 * parameter.grad)
        calibrator.update(np.asarray([0.2, 0.8], dtype=np.float32), np.asarray([0.0, 1.0], dtype=np.float32))

        actor_state = {key: value.detach().clone() for key, value in actor_critic.state_dict().items()}
        calibrator_state = calibrator.get_state()

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "best.pt"
            save_checkpoint(
                checkpoint_path,
                {
                    "actor_critic": actor_critic.state_dict(),
                    "calibrator": calibrator_state,
                    "experiment_cfg": {"seed": 7},
                },
            )

            restored_actor_critic, _ = build_test_actor_critic(obs_dim=41)
            restored_calibrator = OnlineLogitCalibrator(make_calibration_cfg())
            restored_checkpoint = restore_evaluation_state(
                checkpoint_path=checkpoint_path,
                actor_critic=restored_actor_critic,
                calibrator=restored_calibrator,
            )

        self.assertEqual(restored_checkpoint["experiment_cfg"], {"seed": 7})
        for key, value in actor_state.items():
            self.assertTrue(torch.equal(value, restored_actor_critic.state_dict()[key]))
        self.assertEqual(float(restored_calibrator.get_state()["a"]), float(calibrator_state["a"]))
        self.assertEqual(float(restored_calibrator.get_state()["b"]), float(calibrator_state["b"]))
        self.assertTrue(
            np.allclose(
                restored_calibrator.get_state()["posterior_cov"],
                calibrator_state["posterior_cov"],
            )
        )


if __name__ == "__main__":
    unittest.main()
