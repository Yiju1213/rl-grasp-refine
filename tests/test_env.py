from __future__ import annotations

import unittest

import numpy as np

from src.structures.action import NormalizedAction
from src.structures.observation import Observation
from tests.fakes import build_test_env


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


if __name__ == "__main__":
    unittest.main()
