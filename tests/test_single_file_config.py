from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from src.utils.config import load_config
from src.utils.single_file_config import (
    build_single_file_config,
    discover_experiment_config,
    dump_single_file_config,
)


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _make_nested_bundle(root: Path) -> Path:
    configs_root = root / "configs"
    experiment_path = configs_root / "experiment" / "exp_debug.yaml"
    _write_yaml(
        experiment_path,
        {
            "name": "demo_exp",
            "seed": 7,
            "scene_rebuild_every_n_iterations": 3,
            "worker_recycle_every_n_iterations": 2,
            "worker_recycle_slots_per_event": 1,
            "worker_recycle_enable_standby_prefetch": True,
            "worker_recycle_prefetch_count": 4,
            "validation": {"enabled": True, "num_episodes": 16},
            "logging": {"log_dir": "outputs/demo"},
            "configs": {
                "env": "configs/env/grasp_refine_env.yaml",
                "perception": "configs/perception/perception.yaml",
                "calibration": "configs/calibration/online_calibrator.yaml",
                "rl": "configs/rl/ppo.yaml",
                "actor_critic": "configs/model/actor_critic.yaml",
            },
        },
    )
    _write_yaml(
        configs_root / "env" / "grasp_refine_env.yaml",
        {
            "seed": 42,
            "dataset": {"seed": 10, "worker_id": 0, "num_workers": 3},
            "scene": {"use_gui": False},
        },
    )
    _write_yaml(
        configs_root / "perception" / "perception.yaml",
        {
            "adapter_type": "sga_gsn",
            "sga_gsn": {"runtime": {"seed": 3}},
        },
    )
    _write_yaml(configs_root / "calibration" / "online_calibrator.yaml", {"init_a": 1.0, "init_b": 0.0})
    _write_yaml(configs_root / "rl" / "ppo.yaml", {"num_envs": 1})
    _write_yaml(configs_root / "model" / "actor_critic.yaml", {"policy_observation": {"preset": "paper"}})
    return experiment_path


def _make_flat_snapshot_bundle(root: Path) -> Path:
    experiment_path = root / "experiment.yaml"
    _write_yaml(
        experiment_path,
        {
            "name": "flat_exp",
            "seed": 11,
            "logging": {"log_dir": "outputs/flat"},
            "configs": {
                "env": "env.yaml",
                "perception": "perception.yaml",
                "calibration": "calibration.yaml",
                "rl": "rl.yaml",
                "actor_critic": "actor_critic.yaml",
            },
        },
    )
    _write_yaml(root / "env.yaml", {"seed": 0, "dataset": {"seed": 1}})
    _write_yaml(root / "perception.yaml", {"adapter_type": "sga_gsn", "sga_gsn": {"runtime": {"seed": 2}}})
    _write_yaml(root / "calibration.yaml", {"init_a": 1.2})
    _write_yaml(root / "rl.yaml", {"num_envs": 2})
    _write_yaml(root / "actor_critic.yaml", {"policy_observation": {"preset": "current"}})
    return experiment_path


class TestSingleFileConfig(unittest.TestCase):
    def test_build_single_file_config_resolves_effective_bundle_from_nested_layout(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            experiment_path = _make_nested_bundle(Path(tmpdir))

            payload = build_single_file_config(experiment_path)

            self.assertEqual(set(payload.keys()), {"experiment", "env", "perception", "calibration", "rl", "actor_critic"})
            self.assertNotIn("configs", payload["experiment"])
            self.assertEqual(payload["experiment"]["name"], "demo_exp")
            self.assertEqual(payload["experiment"]["logging"]["experiment_name"], "demo_exp")
            self.assertEqual(payload["env"]["seed"], 7)
            self.assertEqual(payload["env"]["dataset"]["seed"], 7)
            self.assertNotIn("worker_id", payload["env"]["dataset"])
            self.assertNotIn("num_workers", payload["env"]["dataset"])
            self.assertEqual(payload["perception"]["sga_gsn"]["runtime"]["seed"], 7)
            self.assertEqual(payload["rl"]["scene_rebuild_every_n_iterations"], 3)
            self.assertEqual(payload["rl"]["worker_recycle_every_n_iterations"], 2)
            self.assertEqual(payload["rl"]["worker_recycle_slots_per_event"], 1)
            self.assertTrue(payload["rl"]["worker_recycle_enable_standby_prefetch"])
            self.assertEqual(payload["rl"]["worker_recycle_prefetch_count"], 4)
            self.assertEqual(payload["calibration"]["init_a"], 1.0)

    def test_discover_experiment_config_accepts_directory_and_module_file_inputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            experiment_path = _make_nested_bundle(Path(tmpdir))
            configs_root = experiment_path.parents[1]
            env_cfg_path = configs_root / "env" / "grasp_refine_env.yaml"

            self.assertEqual(discover_experiment_config(configs_root), experiment_path.resolve())
            self.assertEqual(discover_experiment_config(env_cfg_path), experiment_path.resolve())

    def test_dump_single_file_config_supports_flat_snapshot_layout(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot_root = Path(tmpdir) / "snapshot"
            _make_flat_snapshot_bundle(snapshot_root)
            output_path = snapshot_root / "merged.yaml"

            written_path = dump_single_file_config(snapshot_root, output_path)
            payload = load_config(written_path)

            self.assertEqual(written_path, output_path.resolve())
            self.assertEqual(payload["experiment"]["name"], "flat_exp")
            self.assertEqual(payload["experiment"]["logging"]["experiment_name"], "flat_exp")
            self.assertEqual(payload["env"]["seed"], 11)
            self.assertEqual(payload["env"]["dataset"]["seed"], 11)
            self.assertEqual(payload["perception"]["sga_gsn"]["runtime"]["seed"], 11)
            self.assertEqual(payload["calibration"]["init_a"], 1.2)


if __name__ == "__main__":
    unittest.main()
