from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.utils.checkpoint import load_checkpoint
from tests.runtime_smoke_utils import (
    assert_real_sgagsn_resources,
    configure_headless_render_env,
    write_temp_experiment_bundle,
)


class TestTrainScriptRealSGAGSN(unittest.TestCase):
    def test_train_script_runs_real_three_worker_smoke(self):
        assert_real_sgagsn_resources()
        headless_env = configure_headless_render_env()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir)
            experiment_path = write_temp_experiment_bundle(
                output_root,
                num_envs=3,
                batch_episodes=9,
                num_iterations=1,
                device="cuda:0",
                worker_policy_device="cuda:0",
            )
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            env.update(headless_env)
            completed = subprocess.run(
                [sys.executable, "scripts/train.py", "--experiment", str(experiment_path)],
                cwd=str(Path(__file__).resolve().parents[1]),
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            if completed.returncode != 0:
                raise AssertionError(
                    "train.py smoke test failed.\n"
                    f"stdout:\n{completed.stdout}\n\nstderr:\n{completed.stderr}"
                )

            final_checkpoint = output_root / "checkpoints" / "final.pt"
            self.assertTrue(final_checkpoint.exists(), msg=f"Missing checkpoint: {final_checkpoint}")
            checkpoint = load_checkpoint(final_checkpoint)
            for key in ("actor_critic", "optimizer", "history", "calibrator", "experiment_cfg"):
                self.assertIn(key, checkpoint)
            self.assertEqual(len(checkpoint["history"]), 1)

            self.assertTrue((output_root / "logs" / "metrics.jsonl").exists())
            self.assertTrue((output_root / "logs" / "run.log").exists())
            tensorboard_dir = output_root / "logs" / "tensorboard"
            self.assertTrue(tensorboard_dir.exists())
            self.assertTrue(any(path.name.startswith("events.out.tfevents") for path in tensorboard_dir.iterdir()))


if __name__ == "__main__":
    unittest.main()
