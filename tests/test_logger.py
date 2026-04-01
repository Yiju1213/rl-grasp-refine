from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path

from src.utils.logger import Logger


class TestLogger(unittest.TestCase):
    def test_log_dict_requires_module_prefixed_keys_and_writes_tensorboard(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            logger = Logger(
                {
                    "log_dir": str(root / "logs"),
                    "tensorboard": {"enabled": True, "dir": str(root / "tb")},
                }
            )

            logger.log_dict({"ppo/policy_loss": 1.0, "reward/total_mean": 0.5}, step=3)

            self.assertTrue((root / "logs" / "metrics.jsonl").exists())
            self.assertTrue((root / "tb").exists())
            self.assertTrue(any(path.name.startswith("events.out.tfevents") for path in (root / "tb").iterdir()))

            with self.assertRaisesRegex(ValueError, "Logger stats keys must use"):
                logger.log_dict({"policy_loss": 1.0}, step=4)

    def test_log_episode_samples_respects_enable_flag_and_frequency(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            logger = Logger(
                {
                    "log_dir": str(root / "logs"),
                    "tensorboard": {"enabled": False},
                    "sample_metrics": {
                        "enabled": True,
                        "path": str(root / "logs" / "episode_metrics.jsonl"),
                        "every_n_iterations": 2,
                    },
                }
            )

            logger.log_episode_samples([{"reward": {"total": 1.0}}], step=1)
            self.assertFalse((root / "logs" / "episode_metrics.jsonl").exists())

            logger.log_episode_samples([{"reward": {"total": 1.0}}], step=2)
            self.assertTrue((root / "logs" / "episode_metrics.jsonl").exists())

    def test_experiment_name_nests_artifacts_under_standard_names(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            logger = Logger(
                {
                    "log_dir": str(root / "logs"),
                    "experiment_name": "exp_debug",
                    "tensorboard": {"enabled": True, "dir": str(root / "logs" / "tensorboard")},
                    "sample_metrics": {
                        "enabled": True,
                        "path": str(root / "logs" / "episode_metrics.jsonl"),
                        "every_n_iterations": 1,
                    },
                }
            )

            logger.log_dict({"ppo/policy_loss": 1.0}, step=0)
            logger.log_episode_samples([{"reward": {"total": 1.0}}], step=0)
            logger.info("hello")

            experiment_dir = root / "logs" / "exp_debug"
            self.assertTrue((experiment_dir / "metrics.jsonl").exists())
            self.assertTrue((experiment_dir / "run.log").exists())
            self.assertTrue((experiment_dir / "episode_metrics.jsonl").exists())
            tensorboard_dir = experiment_dir / "tensorboard"
            self.assertTrue(tensorboard_dir.exists())
            self.assertTrue(any(path.name.startswith("events.out.tfevents") for path in tensorboard_dir.iterdir()))

    def test_log_dict_rounds_numeric_payload_for_jsonl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            logger = Logger(
                {
                    "log_dir": str(root / "logs"),
                    "tensorboard": {"enabled": False},
                }
            )

            logger.log_dict(
                {
                    "ppo/large_value": 1.123456789,
                    "ppo/medium_value": 0.123456789,
                    "ppo/small_value": 0.000123456789,
                },
                step=1,
            )

            payload = json.loads((root / "logs" / "metrics.jsonl").read_text(encoding="utf-8").strip())
            stats = payload["stats"]
            self.assertEqual(stats["ppo/large_value"], 1.1235)
            self.assertEqual(stats["ppo/medium_value"], 0.123457)
            self.assertEqual(stats["ppo/small_value"], 0.00012346)

    def test_paper_metric_profile_filters_debug_metrics_consistently(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            logger = Logger(
                {
                    "log_dir": str(root / "logs"),
                    "metric_profile": "paper",
                    "diagnostics": {"enabled": False},
                    "tensorboard": {"enabled": False},
                }
            )

            stats = {
                "outcome/success_lift_vs_dataset": 0.1,
                "outcome/drop_rate_after_given_dataset_positive": 0.25,
                "outcome/hold_rate_after_given_dataset_negative": 0.75,
                "outcome/dataset_positive_count": 12.0,
                "reward/total_mean": 0.2,
                "contact/t_cover_after_mean": 0.3,
                "calibrator/after_brier": 0.4,
                "ppo/approx_kl": 0.5,
                "timing/validation_wall_s": 1.0,
                "system/process_rss_mb": 123.0,
                "action/saturation_rate": 0.6,
                "collection/worker_recycle_performed": 1.0,
                "calibrator/raw_logit_before_mean": 0.7,
            }
            logger.log_dict(stats, step=0)

            payload = json.loads((root / "logs" / "metrics.jsonl").read_text(encoding="utf-8").strip())
            filtered = payload["stats"]
            self.assertIn("outcome/success_lift_vs_dataset", filtered)
            self.assertIn("outcome/drop_rate_after_given_dataset_positive", filtered)
            self.assertIn("outcome/hold_rate_after_given_dataset_negative", filtered)
            self.assertIn("reward/total_mean", filtered)
            self.assertIn("timing/validation_wall_s", filtered)
            self.assertNotIn("outcome/dataset_positive_count", filtered)
            self.assertNotIn("system/process_rss_mb", filtered)
            self.assertNotIn("action/saturation_rate", filtered)
            self.assertNotIn("collection/worker_recycle_performed", filtered)
            self.assertNotIn("calibrator/raw_logit_before_mean", filtered)

            rendered = logger.format_payload(stats)
            self.assertIn("outcome/success_lift_vs_dataset", rendered)
            self.assertNotIn("system/process_rss_mb", rendered)


if __name__ == "__main__":
    unittest.main()
