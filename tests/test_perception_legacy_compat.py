from __future__ import annotations

import unittest

import numpy as np

from src.perception.adapters import SGAGSNAdapter
from src.perception.contact_semantics import ContactSemanticsExtractor
from src.structures.observation import RawSensorObservation


class TestPerceptionLegacyCompat(unittest.TestCase):
    def test_adapter_builds_dummy_point_cloud_when_env_does_not_provide_one(self):
        adapter = SGAGSNAdapter()
        raw_obs = RawSensorObservation(
            visual_data={
                "rgb": np.zeros((4, 4, 3), dtype=np.uint8),
                "depth": np.zeros((4, 4), dtype=np.float32),
                "seg": np.zeros((4, 4), dtype=np.int16),
                "view_matrix": np.eye(4, dtype=np.float32),
                "proj_matrix": np.eye(4, dtype=np.float32),
            },
            tactile_data={"contact_map": np.zeros((2, 4, 4), dtype=np.float32)},
            grasp_metadata={},
        )

        model_inputs = adapter.adapt_feature_input(raw_obs)

        self.assertIn("point_cloud", model_inputs)
        self.assertEqual(tuple(model_inputs["point_cloud"].shape), (1, 3))
        self.assertTrue(np.allclose(model_inputs["point_cloud"].cpu().numpy(), 0.0))

    def test_contact_semantics_use_tactile_only_placeholder(self):
        extractor = ContactSemanticsExtractor({"tactile_threshold": 0.2})
        raw_obs = RawSensorObservation(
            visual_data={
                "distance_to_edge": 999.0,
            },
            tactile_data={
                "contact_map": np.asarray(
                    [
                        [[1.0, 1.0], [1.0, 1.0]],
                        [[0.0, 0.0], [0.0, 0.0]],
                    ],
                    dtype=np.float32,
                ),
            },
            grasp_metadata={"distance_to_edge": 999.0},
        )

        semantic = extractor.extract(raw_obs)

        self.assertEqual(tuple(semantic.shape), (2,))
        self.assertAlmostEqual(float(semantic[0]), 0.5, places=6)
        self.assertAlmostEqual(float(semantic[1]), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
