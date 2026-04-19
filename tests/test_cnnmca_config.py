from __future__ import annotations

import unittest
from pathlib import Path

import torch

from scripts._common import load_experiment_bundle
from src.perception.factory import build_perception_stack, infer_perception_feature_dim
from src.utils.config import load_config


class TestCNNMCAConfig(unittest.TestCase):
    def test_cnnmca_feature_dim_infers_from_config(self):
        perception_cfg = load_config(Path("configs/perception/perception_cnnmca.yaml"))

        self.assertEqual(infer_perception_feature_dim(perception_cfg), 512)

    def test_cnnmca_experiment_only_replaces_perception_config(self):
        base_exp, base_bundle = load_experiment_bundle("configs/experiment/exp_debug_stb5x_latefus_128_epi.yaml")
        cnnmca_exp, cnnmca_bundle = load_experiment_bundle(
            "configs/experiment/exp_debug_stb5x_latefus_128_epi_cnnmca.yaml"
        )

        self.assertNotEqual(base_exp["name"], cnnmca_exp["name"])
        self.assertEqual(base_bundle["env"], cnnmca_bundle["env"])
        self.assertEqual(base_bundle["calibration"], cnnmca_bundle["calibration"])
        self.assertEqual(base_bundle["rl"], cnnmca_bundle["rl"])
        self.assertEqual(base_bundle["actor_critic"], cnnmca_bundle["actor_critic"])
        self.assertEqual(cnnmca_bundle["perception"]["adapter_type"], "cnnmca")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required to instantiate CNNMCA runtime.")
    def test_build_perception_stack_accepts_cnnmca_adapter(self):
        perception_cfg = load_config(Path("configs/perception/perception_cnnmca.yaml"))
        runtime_cfg = perception_cfg["cnnmca"]["runtime"]
        if not Path(runtime_cfg["config_path"]).exists() or not Path(runtime_cfg["checkpoint"]).exists():
            self.skipTest("CNNMCA config or checkpoint is missing.")

        feature_extractor, contact_semantics_extractor, stability_predictor = build_perception_stack(perception_cfg)

        self.assertEqual(feature_extractor.runtime.body_feature_dim, 512)
        self.assertIsNotNone(contact_semantics_extractor)
        self.assertIsNone(stability_predictor.predictor_model)


if __name__ == "__main__":
    unittest.main()
