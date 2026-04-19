from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import torch

from src.perception.cnnmca_runtime import get_shared_cnnmca_runtime, infer_cnnmca_body_feature_dim
from src.perception.cnnmca_types import PreparedCNNMCAInputs
from src.utils.config import load_config


def _cnnmca_resources_available() -> bool:
    if not Path("/AdaPoinTr").exists():
        return False
    cfg = load_config(Path("configs/perception/perception_cnnmca.yaml"))
    runtime_cfg = cfg["cnnmca"]["runtime"]
    return Path(runtime_cfg["config_path"]).exists() and Path(runtime_cfg["checkpoint"]).exists()


@unittest.skipUnless(_cnnmca_resources_available(), "AdaPoinTr CNNMCA resources are required for runtime smoke tests.")
@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for CNNMCA runtime smoke tests.")
class TestCNNMCARuntimeSmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        perception_cfg = load_config(Path("configs/perception/perception_cnnmca.yaml"))
        cls.runtime_cfg = perception_cfg["cnnmca"]["runtime"]

    def test_runtime_loads_and_runs_synthetic_forward(self):
        runtime = get_shared_cnnmca_runtime(self.runtime_cfg)
        self.assertEqual(runtime.body_feature_dim, infer_cnnmca_body_feature_dim(self.runtime_cfg))
        self.assertGreater(runtime.ignored_checkpoint_keys_count, 0)

        rng = np.random.default_rng(7)
        image_size = int(self.runtime_cfg["image_size"])
        prepared = PreparedCNNMCAInputs(
            visual_img=rng.normal(size=(3, image_size, image_size)).astype(np.float32),
            tactile_img=rng.normal(size=(3, image_size, image_size)).astype(np.float32),
        )

        result = runtime.run_prepared(prepared)
        self.assertEqual(tuple(result.body_feature.shape), (runtime.body_feature_dim,))
        self.assertTrue(np.all(np.isfinite(result.body_feature)))
        self.assertTrue(np.isfinite(result.raw_logit))
        self.assertIs(get_shared_cnnmca_runtime(self.runtime_cfg), runtime)


if __name__ == "__main__":
    unittest.main()
