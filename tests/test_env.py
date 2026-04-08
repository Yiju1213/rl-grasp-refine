from __future__ import annotations

import unittest

import numpy as np

from src.structures.action import NormalizedAction
from src.structures.observation import Observation
from tests.fakes import build_test_env, make_env_cfg


class _DummySampleProvider:
    def sample(self):
        raise AssertionError("sample() should not be used during scene rebuild.")


class TestEnv(unittest.TestCase):
    def test_reset_returns_observation(self):
        env, _, _, _, _ = build_test_env()
        obs = env.reset()
        self.assertIsInstance(obs, Observation)

    def test_step_returns_valid_tuple(self):
        env, _, _, _, _ = build_test_env()
        env.reset()
        action = NormalizedAction(value=np.zeros(6, dtype=np.float32))
        next_obs, reward, done, info = env.step(action)
        self.assertIsInstance(next_obs, Observation)
        self.assertIsInstance(reward, float)
        self.assertTrue(done)
        self.assertIn("raw_logit_before", info.extra)
        self.assertIn("raw_logit_after", info.extra)

    def test_rebuild_scene_replaces_scene_and_clears_episode_state(self):
        env, calibrator, _, _, _ = build_test_env()
        env.reset()
        sample_provider = _DummySampleProvider()
        env.sample_provider = sample_provider

        old_scene = env.scene
        old_scene_id = old_scene.instance_id
        env.rebuild_scene()

        self.assertIsNot(env.scene, old_scene)
        self.assertNotEqual(env.scene.instance_id, old_scene_id)
        self.assertTrue(old_scene.closed)
        self.assertIs(env.calibrator, calibrator)
        self.assertIs(env.sample_provider, sample_provider)
        self.assertIsNone(env.obs_before)
        self.assertIsNone(env.grasp_pose_before)
        self.assertIsNone(env.sample_cfg)
        self.assertIsNone(env.raw_obs_before)
        self.assertIsNone(env.raw_obs_after)

    def test_wo_onl_cal_uses_identity_probability_delta_without_uncertainty_discount(self):
        env_cfg = make_env_cfg()
        env_cfg["reward"]["drop_weight"] = 0.0
        env_cfg["reward"]["contact_weight"] = 0.0
        env_cfg["reward"]["stability_weight"] = 1.0
        calibration_cfg = {
            "init_a": 1.0,
            "init_b": 0.0,
            "lambda": 1.0,
            "online_update_enabled": False,
            "signal_mode": "identity_probability",
            "uncertainty_discount_enabled": False,
        }
        env, _, _, _, _ = build_test_env(env_cfg=env_cfg, calibration_cfg=calibration_cfg)
        env.reset()

        next_obs, reward, done, info = env.step(NormalizedAction(value=np.zeros(6, dtype=np.float32)))

        raw_before = float(info.extra["raw_logit_before"])
        raw_after = float(info.extra["raw_logit_after"])
        prob_before = 1.0 / (1.0 + np.exp(-raw_before))
        prob_after = 1.0 / (1.0 + np.exp(-raw_after))
        self.assertIsInstance(next_obs, Observation)
        self.assertTrue(done)
        self.assertAlmostEqual(info.calibrated_stability_before, prob_before)
        self.assertAlmostEqual(info.calibrated_stability_after, prob_after)
        self.assertAlmostEqual(info.posterior_trace, 0.0)
        self.assertAlmostEqual(info.reward_stability, prob_after - prob_before)
        self.assertAlmostEqual(reward, prob_after - prob_before)


if __name__ == "__main__":
    unittest.main()
