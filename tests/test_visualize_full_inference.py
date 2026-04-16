from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest

import numpy as np

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


if __name__ == "__main__":
    unittest.main()
