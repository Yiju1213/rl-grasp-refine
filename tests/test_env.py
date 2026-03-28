from __future__ import annotations

import unittest

import numpy as np

from src.structures.action import NormalizedAction
from src.structures.observation import Observation
from tests.fakes import build_test_env


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


if __name__ == "__main__":
    unittest.main()
