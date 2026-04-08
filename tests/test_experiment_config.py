from __future__ import annotations

import unittest

from src.runtime.experiment_config import apply_experiment_overrides
from tests.fakes import make_actor_critic_cfg, make_calibration_cfg, make_env_cfg


class TestExperimentConfig(unittest.TestCase):
    @staticmethod
    def _make_bundle() -> dict:
        return {
            "env": {
                **make_env_cfg(),
                "dataset": {
                    "enabled": True,
                    "seed": 10,
                    "worker_id": 0,
                    "num_workers": 1,
                },
            },
            "perception": {
                "sga_gsn": {
                    "runtime": {
                        "seed": 3,
                    }
                }
            },
            "calibration": make_calibration_cfg(),
            "actor_critic": make_actor_critic_cfg(),
            "rl": {},
        }

    def test_apply_experiment_overrides_synchronizes_seed_and_logging_name(self):
        experiment_cfg = {
            "name": "exp_debug",
            "seed": 7,
            "scene_rebuild_every_n_iterations": 3,
            "worker_recycle_every_n_iterations": 2,
            "worker_recycle_slots_per_event": 1,
            "worker_recycle_enable_standby_prefetch": True,
            "worker_recycle_prefetch_count": 1,
            "logging": {"log_dir": "outputs/exp_debug"},
        }
        bundle = self._make_bundle()
        bundle["env"]["seed"] = 42

        experiment_cfg_out, bundle_out = apply_experiment_overrides(experiment_cfg, bundle)

        self.assertEqual(experiment_cfg_out["logging"]["experiment_name"], "exp_debug")
        self.assertEqual(experiment_cfg_out["ablation"]["id"], "baseline")
        self.assertEqual(bundle_out["env"]["seed"], 7)
        self.assertEqual(bundle_out["env"]["dataset"]["seed"], 7)
        self.assertNotIn("worker_id", bundle_out["env"]["dataset"])
        self.assertNotIn("num_workers", bundle_out["env"]["dataset"])
        self.assertEqual(bundle_out["perception"]["sga_gsn"]["runtime"]["seed"], 7)
        self.assertEqual(bundle_out["env"]["reward"]["drop_weight"], 1.0)
        self.assertEqual(bundle_out["env"]["reward"]["stability_weight"], 1.0)
        self.assertEqual(bundle_out["env"]["reward"]["contact_weight"], 1.0)
        self.assertTrue(bundle_out["calibration"]["online_update_enabled"])
        self.assertEqual(bundle_out["rl"]["scene_rebuild_every_n_iterations"], 3)
        self.assertEqual(bundle_out["rl"]["worker_recycle_every_n_iterations"], 2)
        self.assertEqual(bundle_out["rl"]["worker_recycle_slots_per_event"], 1)
        self.assertTrue(bundle_out["rl"]["worker_recycle_enable_standby_prefetch"])
        self.assertEqual(bundle_out["rl"]["worker_recycle_prefetch_count"], 1)

    def test_apply_experiment_overrides_maps_reward_and_calibration_ablations(self):
        _, tac_bundle = apply_experiment_overrides({"ablation": {"id": "wo-tac-rwd"}}, self._make_bundle())
        self.assertEqual(tac_bundle["env"]["reward"]["contact_weight"], 0.0)
        self.assertEqual(tac_bundle["env"]["reward"]["stability_weight"], 1.0)

        _, stb_bundle = apply_experiment_overrides({"ablation": {"id": "wo-stb-rwd"}}, self._make_bundle())
        self.assertEqual(stb_bundle["env"]["reward"]["stability_weight"], 0.0)
        self.assertEqual(stb_bundle["env"]["reward"]["contact_weight"], 1.0)

        _, cal_bundle = apply_experiment_overrides({"ablation": {"id": "wo-onl-cal"}}, self._make_bundle())
        self.assertFalse(cal_bundle["calibration"]["online_update_enabled"])
        self.assertEqual(cal_bundle["calibration"]["signal_mode"], "identity_probability")
        self.assertFalse(cal_bundle["calibration"]["uncertainty_discount_enabled"])

    def test_apply_experiment_overrides_removes_contact_semantic_for_joint_tactile_ablation(self):
        bundle = self._make_bundle()
        bundle["actor_critic"]["policy_observation"] = {"preset": "paper"}

        experiment_cfg_out, bundle_out = apply_experiment_overrides(
            {"ablation": {"id": "wo-tac-sem-n-rwd"}},
            bundle,
        )

        self.assertEqual(experiment_cfg_out["ablation"]["id"], "wo-tac-sem-n-rwd")
        self.assertEqual(bundle_out["env"]["reward"]["contact_weight"], 0.0)
        self.assertEqual(bundle_out["actor_critic"]["policy_observation"]["preset"], "custom")
        self.assertEqual(bundle_out["actor_critic"]["policy_observation"]["components"], ["latent_feature"])

    def test_apply_experiment_overrides_rejects_unknown_ablation_id(self):
        with self.assertRaisesRegex(ValueError, "Unknown ablation.id"):
            apply_experiment_overrides({"ablation": {"id": "not-a-real-ablation"}}, self._make_bundle())


if __name__ == "__main__":
    unittest.main()
