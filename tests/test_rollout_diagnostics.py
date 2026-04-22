from __future__ import annotations

import unittest

import numpy as np
import torch

from src.metrics.rollout_diagnostics import (
    action_bin_stats,
    action_distribution_stats,
    binary_auc,
    policy_latent_hidden_stats,
    reliability_stats,
    safe_pearson,
)
from src.structures.action import GraspPose
from src.structures.observation import Observation


class _DummyPolicyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.latent_dim = 3
        self.latent_layer = torch.nn.Linear(3, 2, bias=False)
        with torch.no_grad():
            self.latent_layer.weight.copy_(torch.tensor([[1.0, -1.0, 0.5], [-0.5, 1.0, 1.0]]))


class _DummyActorCritic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.policy_net = _DummyPolicyNet()
        self.observation_spec = None


class TestRolloutDiagnostics(unittest.TestCase):
    def test_action_distribution_reports_norms_and_saturation(self):
        stats = action_distribution_stats(
            np.asarray(
                [
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, -0.95, 0.0],
                ],
                dtype=np.float32,
            )
        )

        self.assertAlmostEqual(stats["action/translation_norm_mean"], 0.5, places=7)
        self.assertAlmostEqual(stats["action/rotation_norm_mean"], 0.475, places=7)
        self.assertAlmostEqual(stats["action/dim_0_saturation_rate"], 0.5, places=7)
        self.assertAlmostEqual(stats["action/dim_4_saturation_rate"], 0.5, places=7)
        self.assertAlmostEqual(stats["action/saturation_rate"], 2.0 / 12.0, places=7)

    def test_pearson_and_auc_return_none_when_undefined(self):
        self.assertIsNone(safe_pearson([1.0, 1.0], [0.0, 1.0]))
        self.assertIsNone(binary_auc([0.1, 0.2], [1.0, 1.0]))
        self.assertAlmostEqual(binary_auc([0.1, 0.3, 0.2, 0.4], [0, 1, 0, 1]), 1.0, places=7)

    def test_reliability_stats_split_recovery_and_degradation_auc(self):
        records = [
            {
                "prob_before": 0.2,
                "prob_after": 0.8,
                "raw_logit_before": -1.0,
                "raw_logit_after": 1.0,
                "prob_delta": 0.6,
                "drop_success": 1,
                "legacy_drop_success_before": 0,
                "positive_drop_event": None,
            },
            {
                "prob_before": 0.4,
                "prob_after": 0.1,
                "raw_logit_before": 0.0,
                "raw_logit_after": -1.0,
                "prob_delta": -0.3,
                "drop_success": 0,
                "legacy_drop_success_before": 0,
                "positive_drop_event": None,
            },
            {
                "prob_before": 0.9,
                "prob_after": 0.2,
                "raw_logit_before": 1.0,
                "raw_logit_after": -0.5,
                "prob_delta": -0.7,
                "drop_success": 0,
                "legacy_drop_success_before": 1,
                "positive_drop_event": 1,
            },
            {
                "prob_before": 0.8,
                "prob_after": 0.7,
                "raw_logit_before": 0.9,
                "raw_logit_after": 0.4,
                "prob_delta": -0.1,
                "drop_success": 1,
                "legacy_drop_success_before": 1,
                "positive_drop_event": 0,
            },
        ]

        stats = reliability_stats(records)

        self.assertIsNotNone(stats["calibrator/after_brier"])
        self.assertAlmostEqual(stats["calibrator/prob_delta_recovery_auc"], 1.0, places=7)
        self.assertAlmostEqual(stats["calibrator/neg_prob_delta_degradation_auc"], 1.0, places=7)

    def test_action_bin_stats_uses_four_norm_ratio_bins(self):
        records = [
            {"translation_norm": 0.0, "rotation_norm": 0.0, "prob_delta": 0.1, "success_delta": 0.0, "positive_drop_event": 0},
            {
                "translation_norm": float(np.sqrt(3.0) * 0.5),
                "rotation_norm": float(np.sqrt(3.0)),
                "prob_delta": 0.3,
                "success_delta": 1.0,
                "positive_drop_event": 1,
            },
        ]

        stats = action_bin_stats(records)

        self.assertEqual(stats["action_bin/trans_bin_0_count"], 1.0)
        self.assertEqual(stats["action_bin/trans_bin_2_count"], 1.0)
        self.assertEqual(stats["action_bin/rot_bin_0_count"], 1.0)
        self.assertEqual(stats["action_bin/rot_bin_3_count"], 1.0)
        self.assertAlmostEqual(stats["action_bin/trans_bin_2_prob_delta_mean"], 0.3, places=7)

    def test_policy_latent_hidden_stats_matches_late_fusion_first_layer(self):
        actor_critic = _DummyActorCritic()
        obs = Observation(
            latent_feature=np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
            contact_semantic=np.zeros(2, dtype=np.float32),
            grasp_pose=GraspPose(position=np.zeros(3, dtype=np.float32), rotation=np.zeros(3, dtype=np.float32)),
            raw_stability_logit=0.0,
        )

        stats = policy_latent_hidden_stats(actor_critic, obs)

        hidden = np.asarray([0.5, 4.5], dtype=np.float32)
        self.assertAlmostEqual(stats["policy_latent_hidden_before_norm"], float(np.linalg.norm(hidden)), places=6)
        self.assertAlmostEqual(stats["policy_latent_hidden_before_mean"], float(np.mean(hidden)), places=7)
        self.assertAlmostEqual(stats["policy_latent_hidden_before_std"], float(np.std(hidden)), places=7)


if __name__ == "__main__":
    unittest.main()
