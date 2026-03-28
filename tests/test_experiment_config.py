from __future__ import annotations

import unittest

from src.runtime.experiment_config import apply_experiment_overrides


class TestExperimentConfig(unittest.TestCase):
    def test_apply_experiment_overrides_synchronizes_seed_and_logging_name(self):
        experiment_cfg = {
            "name": "exp_debug",
            "seed": 7,
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
        }

        experiment_cfg_out, bundle_out = apply_experiment_overrides(experiment_cfg, bundle)

        self.assertEqual(experiment_cfg_out["logging"]["experiment_name"], "exp_debug")
        self.assertEqual(bundle_out["env"]["seed"], 7)
        self.assertEqual(bundle_out["env"]["dataset"]["seed"], 7)
        self.assertNotIn("worker_id", bundle_out["env"]["dataset"])
        self.assertNotIn("num_workers", bundle_out["env"]["dataset"])
        self.assertEqual(bundle_out["perception"]["sga_gsn"]["runtime"]["seed"], 7)


if __name__ == "__main__":
    unittest.main()
