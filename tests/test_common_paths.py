from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import yaml

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from _common import load_experiment_bundle, resolve_experiment_source_paths, snapshot_experiment_configs


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


class TestCommonPathResolution(unittest.TestCase):
    def test_load_experiment_bundle_resolves_configs_within_same_bundle_root(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_root = Path(tmpdir) / "artifact_bundle"
            experiment_path = bundle_root / "configs" / "experiment" / "exp_debug.yaml"
            env_path = bundle_root / "configs" / "env" / "grasp_refine_env.yaml"
            rl_path = bundle_root / "configs" / "rl" / "ppo.yaml"

            _write_yaml(
                experiment_path,
                {
                    "name": "snapshot-exp",
                    "seed": 13,
                    "configs": {
                        "env": "configs/env/grasp_refine_env.yaml",
                        "rl": "configs/rl/ppo.yaml",
                    },
                },
            )
            _write_yaml(env_path, {"seed": 999, "marker": "bundle-env", "dataset": {"enabled": False}})
            _write_yaml(rl_path, {"marker": "bundle-rl"})

            experiment_cfg, bundle = load_experiment_bundle(experiment_path)

            self.assertEqual(experiment_cfg["name"], "snapshot-exp")
            self.assertEqual(bundle["env"]["marker"], "bundle-env")
            self.assertEqual(bundle["rl"]["marker"], "bundle-rl")
            self.assertEqual(bundle["env"]["seed"], 13)

    def test_resolve_and_snapshot_preserve_external_bundle_layout(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_root = Path(tmpdir) / "artifact_bundle"
            experiment_path = bundle_root / "configs" / "experiment" / "exp_debug.yaml"
            env_path = bundle_root / "configs" / "env" / "grasp_refine_env.yaml"
            rl_path = bundle_root / "configs" / "rl" / "ppo.yaml"

            _write_yaml(
                experiment_path,
                {
                    "name": "snapshot-exp",
                    "configs": {
                        "env": "configs/env/grasp_refine_env.yaml",
                        "rl": "configs/rl/ppo.yaml",
                    },
                },
            )
            _write_yaml(env_path, {"marker": "bundle-env"})
            _write_yaml(rl_path, {"marker": "bundle-rl"})

            source_paths = resolve_experiment_source_paths(experiment_path)
            self.assertEqual(source_paths["env"], env_path.resolve())
            self.assertEqual(source_paths["rl"], rl_path.resolve())

            snapshot_dir = bundle_root / "snapshot_copy"
            copied_paths = snapshot_experiment_configs(experiment_path, snapshot_dir)

            copied_paths_set = {path.resolve() for path in copied_paths}
            self.assertIn((snapshot_dir / "experiment" / "exp_debug.yaml").resolve(), copied_paths_set)
            self.assertIn((snapshot_dir / "env" / "grasp_refine_env.yaml").resolve(), copied_paths_set)
            self.assertIn((snapshot_dir / "rl" / "ppo.yaml").resolve(), copied_paths_set)

    def test_snapshot_to_existing_bundle_root_skips_same_file_copy(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_root = Path(tmpdir) / "artifact_bundle"
            configs_root = bundle_root / "configs"
            experiment_path = configs_root / "experiment" / "exp_debug.yaml"
            env_path = configs_root / "env" / "grasp_refine_env.yaml"
            rl_path = configs_root / "rl" / "ppo.yaml"

            _write_yaml(
                experiment_path,
                {
                    "name": "snapshot-exp",
                    "configs": {
                        "env": "configs/env/grasp_refine_env.yaml",
                        "rl": "configs/rl/ppo.yaml",
                    },
                },
            )
            _write_yaml(env_path, {"marker": "bundle-env"})
            _write_yaml(rl_path, {"marker": "bundle-rl"})

            copied_paths = snapshot_experiment_configs(experiment_path, configs_root)

            self.assertEqual(
                {path.resolve() for path in copied_paths},
                {
                    experiment_path.resolve(),
                    env_path.resolve(),
                    rl_path.resolve(),
                },
            )


if __name__ == "__main__":
    unittest.main()
