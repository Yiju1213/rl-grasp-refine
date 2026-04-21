from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from contextlib import redirect_stderr

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import visualize_full_inference as viz
from src.utils.config import load_config


def _raw_obs(rgb_value: int):
    return SimpleNamespace(
        visual_data={
            "rgb": np.full((32, 32, 3), rgb_value, dtype=np.uint8),
            "view_matrix": np.eye(4, dtype=np.float32),
            "proj_matrix": np.eye(4, dtype=np.float32),
        },
        tactile_data={
            "rgb": np.full((2, 24, 16, 3), rgb_value, dtype=np.uint8),
        },
        grasp_metadata={"observation_valid": True},
    )


class TestVisualizeFullInferenceHelpers(unittest.TestCase):
    def test_build_inference_panel_uses_two_stage_columns(self):
        panel = viz.build_inference_panel(_raw_obs(32), _raw_obs(224), panel_width=48)
        self.assertEqual(panel.dtype, np.uint8)
        self.assertEqual(panel.shape[1], 96)
        self.assertGreater(panel.shape[0], 48)

    def test_write_inference_panel_preserves_rgb_channel_order(self):
        panel = np.zeros((4, 4, 3), dtype=np.uint8)
        panel[:, :] = [255, 0, 0]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "panel.png"
            viz._write_inference_panel(path, panel)
            decoded_rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        np.testing.assert_array_equal(decoded_rgb, panel)

    def test_parse_object_ids_accepts_space_and_comma_forms(self):
        self.assertEqual(viz.parse_object_ids(["75", "77,76", "75"]), [75, 76, 77])
        self.assertIsNone(viz.parse_object_ids(None))

    def test_resolve_seed8_object_sets(self):
        cfg = load_config(REPO_ROOT / "configs/experiment/exp_debug_stb5x_latefus_128_epi_seed8.yaml")
        self.assertEqual(viz.resolve_object_ids(cfg, "val", None), [78, 82, 85, 87])
        self.assertEqual(viz.resolve_object_ids(cfg, "test", None), [75, 76, 77, 79, 80, 81, 83, 84, 86])
        self.assertEqual(viz.resolve_object_ids(cfg, "holdout", None), list(range(75, 88)))
        self.assertEqual(viz.resolve_object_ids(cfg, "train", [3, 1, 3]), [1, 3])

    def test_missing_checkpoint_has_clear_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            missing = Path(tmpdir) / "missing.pt"
            with self.assertRaisesRegex(FileNotFoundError, "Checkpoint does not exist"):
                viz.validate_checkpoint_path(missing)

    def test_table_override_resolution_and_env_cfg_application(self):
        self.assertIsNone(viz._resolve_table_override(False, False))
        self.assertTrue(viz._resolve_table_override(True, False))
        self.assertFalse(viz._resolve_table_override(False, True))
        with self.assertRaisesRegex(ValueError, "cannot be used together"):
            viz._resolve_table_override(True, True)

        env_cfg = {"scene": {"table": {"enabled": False, "urdf_path": "table.urdf"}}}
        table_cfg = viz._apply_table_override(env_cfg, True)
        self.assertTrue(table_cfg["enabled"])
        self.assertTrue(env_cfg["scene"]["table"]["enabled"])
        self.assertEqual(env_cfg["scene"]["table"]["urdf_path"], "table.urdf")

        env_cfg_without_table = {"scene": {}}
        table_cfg = viz._apply_table_override(env_cfg_without_table, False)
        self.assertFalse(table_cfg["enabled"])
        self.assertFalse(env_cfg_without_table["scene"]["table"]["enabled"])

    def test_table_cli_flags_are_mutually_exclusive(self):
        parser = viz.build_parser()
        self.assertTrue(parser.parse_args(["--enable-table"]).enable_table)
        self.assertTrue(parser.parse_args(["--disable-table"]).disable_table)
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args(["--enable-table", "--disable-table"])

    def test_upgrade_legacy_latefus_actor_state_renames_expected_keys(self):
        legacy = {
            "policy_net.latent_layer.weight": torch.ones(1),
            "policy_net.trunk_layer.weight": torch.ones(2),
            "policy_net.trunk_layer.bias": torch.ones(3),
            "policy_net.output_layer.weight": torch.ones(4),
            "policy_net.output_layer.bias": torch.ones(5),
            "value_net.trunk_layer.weight": torch.ones(6),
            "value_net.output_layer.bias": torch.ones(7),
        }

        upgraded, changed = viz._upgrade_legacy_latefus_actor_state(legacy)

        self.assertTrue(changed)
        self.assertIn("policy_net.latent_layer.weight", upgraded)
        self.assertIn("policy_net.trunk.0.weight", upgraded)
        self.assertIn("policy_net.trunk.0.bias", upgraded)
        self.assertIn("policy_net.trunk.2.weight", upgraded)
        self.assertIn("policy_net.trunk.2.bias", upgraded)
        self.assertIn("value_net.trunk.0.weight", upgraded)
        self.assertIn("value_net.trunk.2.bias", upgraded)
        self.assertNotIn("policy_net.trunk_layer.weight", upgraded)
        self.assertNotIn("policy_net.output_layer.weight", upgraded)

    def test_upgrade_legacy_latefus_actor_state_leaves_current_keys_unchanged(self):
        current = {
            "policy_net.trunk.0.weight": torch.ones(1),
            "policy_net.trunk.2.bias": torch.ones(2),
            "value_net.trunk.0.weight": torch.ones(3),
        }

        upgraded, changed = viz._upgrade_legacy_latefus_actor_state(current)

        self.assertFalse(changed)
        self.assertEqual(set(upgraded), set(current))
        for key, value in current.items():
            self.assertIs(upgraded[key], value)


if __name__ == "__main__":
    unittest.main()
