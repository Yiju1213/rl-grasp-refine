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


if __name__ == "__main__":
    unittest.main()
