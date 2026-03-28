from __future__ import annotations

import unittest

from src.runtime.experiment_config import apply_experiment_overrides


class TestExperimentConfig(unittest.TestCase):
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
        bundle = {
            "env": {
                "seed": 42,
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
            "rl": {},
        }

        experiment_cfg_out, bundle_out = apply_experiment_overrides(experiment_cfg, bundle)

        self.assertEqual(experiment_cfg_out["logging"]["experiment_name"], "exp_debug")
        self.assertEqual(bundle_out["env"]["seed"], 7)
        self.assertEqual(bundle_out["env"]["dataset"]["seed"], 7)
        self.assertNotIn("worker_id", bundle_out["env"]["dataset"])
        self.assertNotIn("num_workers", bundle_out["env"]["dataset"])
        self.assertEqual(bundle_out["perception"]["sga_gsn"]["runtime"]["seed"], 7)
        self.assertEqual(bundle_out["rl"]["scene_rebuild_every_n_iterations"], 3)
        self.assertEqual(bundle_out["rl"]["worker_recycle_every_n_iterations"], 2)
        self.assertEqual(bundle_out["rl"]["worker_recycle_slots_per_event"], 1)
        self.assertTrue(bundle_out["rl"]["worker_recycle_enable_standby_prefetch"])
        self.assertEqual(bundle_out["rl"]["worker_recycle_prefetch_count"], 1)


if __name__ == "__main__":
    unittest.main()
