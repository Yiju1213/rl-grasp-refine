from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.utils.checkpoint import load_checkpoint
from src.utils.config import load_config
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

            experiment_cfg = load_config(experiment_path)
            experiment_dir = output_root / "logs" / experiment_cfg["name"]
            final_checkpoint = experiment_dir / "checkpoints" / "final.pt"
            best_checkpoint = experiment_dir / "checkpoints" / "best.pt"
            last_checkpoint = experiment_dir / "checkpoints" / "last.pt"
            self.assertTrue(final_checkpoint.exists(), msg=f"Missing checkpoint: {final_checkpoint}")
            self.assertTrue(best_checkpoint.exists(), msg=f"Missing checkpoint: {best_checkpoint}")
            self.assertTrue(last_checkpoint.exists(), msg=f"Missing checkpoint: {last_checkpoint}")
            checkpoint = load_checkpoint(final_checkpoint)
            last_checkpoint_payload = load_checkpoint(last_checkpoint)
            for key in ("actor_critic", "optimizer", "history", "calibrator", "experiment_cfg"):
                self.assertIn(key, checkpoint)
                self.assertIn(key, last_checkpoint_payload)
            self.assertEqual(len(checkpoint["history"]), 1)
            self.assertEqual(len(last_checkpoint_payload["history"]), 1)
            self.assertEqual(last_checkpoint_payload["completed_iterations"], 1)
            self.assertEqual(checkpoint["best_metric_name"], "outcome/success_lift_vs_dataset")

            self.assertTrue((experiment_dir / "metrics.jsonl").exists())
            self.assertTrue((experiment_dir / "run.log").exists())
            tensorboard_dir = experiment_dir / "tensorboard"
            self.assertTrue(tensorboard_dir.exists())
            self.assertTrue(any(path.name.startswith("events.out.tfevents") for path in tensorboard_dir.iterdir()))
            config_snapshot_dir = experiment_dir / "configs"
            for filename in ("experiment.yaml", "env.yaml", "perception.yaml", "calibration.yaml", "rl.yaml", "actor_critic.yaml"):
                self.assertTrue((config_snapshot_dir / filename).exists(), msg=f"Missing config snapshot: {filename}")


if __name__ == "__main__":
    unittest.main()
