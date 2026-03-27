from __future__ import annotations

import tempfile
import unittest
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


if __name__ == "__main__":
    unittest.main()
