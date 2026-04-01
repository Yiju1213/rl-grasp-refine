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
            posterior_trace=2.0,
            contact_after=np.asarray([0.1, 0.3], dtype=np.float32),
        )
        expected_stability = (0.7 - 0.3) / (1.0 + reward_manager.stability_kappa * 2.0)
        expected_contact = -(
            reward_manager.contact_lambda_cover * max(0.0, reward_manager.contact_threshold_cover - 0.1)
            + reward_manager.contact_lambda_edge * max(0.0, reward_manager.contact_threshold_edge - 0.3)
        )
        self.assertAlmostEqual(reward.stability, expected_stability)
        self.assertAlmostEqual(reward.contact, expected_contact)
        expected_total = reward.drop + reward.stability + reward.contact
        self.assertAlmostEqual(reward.total, expected_total)

    def test_contact_weight_zero_disables_contact_component(self):
        reward_cfg = make_env_cfg()["reward"]
        reward_cfg["contact_weight"] = 0.0
        reward_manager = RewardManager(reward_cfg)

        reward = reward_manager.compute(
            drop_success=1,
            calibrated_before=0.3,
            calibrated_after=0.7,
            posterior_trace=2.0,
            contact_after=np.asarray([0.0, 0.0], dtype=np.float32),
        )

        self.assertAlmostEqual(reward.contact, 0.0)
        self.assertAlmostEqual(reward.total, reward.drop + reward.stability)

    def test_stability_weight_scales_component_and_total(self):
        reward_cfg = make_env_cfg()["reward"]
        reward_cfg["stability_weight"] = 2.5
        reward_manager = RewardManager(reward_cfg)

        reward = reward_manager.compute(
            drop_success=1,
            calibrated_before=0.3,
            calibrated_after=0.7,
            posterior_trace=2.0,
            contact_after=np.asarray([0.3, 0.3], dtype=np.float32),
        )

        base_stability = (0.7 - 0.3) / (1.0 + reward_manager.stability_kappa * 2.0)
        self.assertAlmostEqual(reward.stability, 2.5 * base_stability)
        self.assertAlmostEqual(reward.total, reward.drop + reward.stability + reward.contact)

    def test_drop_weight_scales_drop_component(self):
        reward_cfg = make_env_cfg()["reward"]
        reward_cfg["drop_weight"] = 0.5
        reward_manager = RewardManager(reward_cfg)

        reward = reward_manager.compute(
            drop_success=1,
            calibrated_before=0.3,
            calibrated_after=0.3,
            posterior_trace=2.0,
            contact_after=np.asarray([0.3, 0.3], dtype=np.float32),
        )

        self.assertAlmostEqual(reward.drop, 0.5)
        self.assertAlmostEqual(reward.total, reward.drop + reward.stability + reward.contact)


if __name__ == "__main__":
    unittest.main()
