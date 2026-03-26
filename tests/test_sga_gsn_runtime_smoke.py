from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import torch

from src.perception.sga_gsn_runtime import get_shared_sga_gsn_runtime, infer_sga_gsn_body_feature_dim
from src.perception.sga_gsn_types import PreparedVTGInputs
from src.utils.config import load_config


@unittest.skipUnless(Path("/AdaPoinTr").exists(), "AdaPoinTr source tree is required for runtime smoke tests.")
@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for runtime smoke tests.")
class TestSGAGSNRuntimeSmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        perception_cfg = load_config(Path("configs/perception/perception.yaml"))
        cls.runtime_cfg = perception_cfg["sga_gsn"]["runtime"]

    def test_runtime_loads_and_runs_synthetic_forward(self):
        runtime = get_shared_sga_gsn_runtime(self.runtime_cfg)
        self.assertEqual(runtime.body_feature_dim, infer_sga_gsn_body_feature_dim(self.runtime_cfg))

        rng = np.random.default_rng(7)
        sc_points = int(self.runtime_cfg["sc_input_points"])
        tac_points = int(self.runtime_cfg["tac_points_per_side"]) * 2
        prepared = PreparedVTGInputs(
            sc_input=rng.normal(size=(sc_points, 3)).astype(np.float32),
            gs_input=rng.normal(size=(tac_points, 4)).astype(np.float32),
            zero_mean=np.zeros(3, dtype=np.float32),
            debug_visual_world_points=np.zeros((0, 3), dtype=np.float32),
            debug_tactile_left_world_points=np.zeros((0, 3), dtype=np.float32),
            debug_tactile_right_world_points=np.zeros((0, 3), dtype=np.float32),
            debug_tactile_left_contact_world_points=np.zeros((0, 3), dtype=np.float32),
            debug_tactile_right_contact_world_points=np.zeros((0, 3), dtype=np.float32),
            debug_tactile_left_gel_mask=np.zeros((0,), dtype=bool),
            debug_tactile_right_gel_mask=np.zeros((0,), dtype=bool),
        )

        result = runtime.run_prepared(prepared)
        self.assertEqual(tuple(result.body_feature.shape), (runtime.body_feature_dim,))
        self.assertTrue(np.all(np.isfinite(result.body_feature)))
        self.assertTrue(np.isfinite(result.raw_logit))
        self.assertIs(get_shared_sga_gsn_runtime(self.runtime_cfg), runtime)


if __name__ == "__main__":
    unittest.main()
