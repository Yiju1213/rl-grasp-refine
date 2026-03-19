from __future__ import annotations

import unittest

import numpy as np

from src.envs.reward_manager import RewardManager
from tests.fakes import make_env_cfg


class TestRewardManager(unittest.TestCase):
    def test_total_equals_components(self):
        reward_manager = RewardManager(make_env_cfg()["reward"])
        reward = reward_manager.compute(
            drop_success=1,
            calibrated_before=0.3,
            calibrated_after=0.7,
            uncertainty_before=0.1,
            uncertainty_after=0.15,
            contact_before=np.asarray([0.2, 0.4], dtype=np.float32),
            contact_after=np.asarray([0.6, 0.2], dtype=np.float32),
        )
        expected_total = (
            reward_manager.w_drop * reward.drop
            + reward_manager.w_stability * reward.stability
            + reward_manager.w_contact * reward.contact
        )
        self.assertAlmostEqual(reward.total, expected_total)


if __name__ == "__main__":
    unittest.main()
