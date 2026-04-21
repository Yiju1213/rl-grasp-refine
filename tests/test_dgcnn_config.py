from __future__ import annotations

import unittest
from pathlib import Path

import torch

from scripts._common import load_experiment_bundle
from src.perception.factory import build_perception_stack, infer_perception_feature_dim
from src.utils.config import load_config


class TestDGCNNConfig(unittest.TestCase):
    def test_dgcnn_feature_dim_infers_from_config(self):
        perception_cfg = load_config(Path("configs/perception/perception_dgcnn.yaml"))

        self.assertEqual(infer_perception_feature_dim(perception_cfg), 8192)

    def test_dgcnn_experiment_only_replaces_perception_config(self):
        base_exp, base_bundle = load_experiment_bundle("configs/experiment/exp_debug_stb5x_latefus_128_epi.yaml")
        dgcnn_exp, dgcnn_bundle = load_experiment_bundle(
            "configs/experiment/exp_debug_stb5x_latefus_128_epi_dgcnn.yaml"
        )

        self.assertNotEqual(base_exp["name"], dgcnn_exp["name"])
        self.assertEqual(base_bundle["env"], dgcnn_bundle["env"])
        self.assertEqual(base_bundle["calibration"], dgcnn_bundle["calibration"])
        self.assertEqual(base_bundle["rl"], dgcnn_bundle["rl"])
        self.assertEqual(base_bundle["actor_critic"], dgcnn_bundle["actor_critic"])
        self.assertEqual(dgcnn_bundle["perception"]["adapter_type"], "dgcnn")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required to instantiate DGCNN runtime.")
    def test_build_perception_stack_accepts_dgcnn_adapter(self):
        perception_cfg = load_config(Path("configs/perception/perception_dgcnn.yaml"))
        runtime_cfg = perception_cfg["dgcnn"]["runtime"]
        required_paths = [
            runtime_cfg["config_path"],
            runtime_cfg["shape_checkpoint"],
            runtime_cfg["grasp_checkpoint"],
        ]
        if not all(Path(path).exists() for path in required_paths):
            self.skipTest("DGCNN config or checkpoint is missing.")

        feature_extractor, contact_semantics_extractor, stability_predictor = build_perception_stack(perception_cfg)

        self.assertEqual(feature_extractor.runtime.body_feature_dim, 8192)
        self.assertIsNotNone(contact_semantics_extractor)
        self.assertIsNone(stability_predictor.predictor_model)


if __name__ == "__main__":
    unittest.main()
