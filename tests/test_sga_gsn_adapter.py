from __future__ import annotations

import unittest

import numpy as np

from src.perception.adapters import SGAGSNAdapter
from src.structures.observation import RawSensorObservation


class TestSGAGSNAdapter(unittest.TestCase):
    def test_prepare_inputs_returns_expected_shapes(self):
        adapter = SGAGSNAdapter(
            {
                "sga_gsn": {
                    "runtime": {
                        "vis_points": 32,
                        "tac_points_per_side": 20,
                        "sc_input_points": 64,
                        "tactile_step": 16,
                        "camera_distance_to_gel_m": 0.02315,
                        "seed": 7,
                    }
                }
            }
        )
        visual_depth = np.full((16, 16), 0.2, dtype=np.float32)
        visual_seg = np.ones((16, 16), dtype=np.int16)
        tactile_depth = np.zeros((2, 320, 240), dtype=np.float32)
        tactile_depth[0, 64:160, 40:120] = 0.001
        tactile_depth[1, 160:224, 120:200] = 0.0005

        raw_obs = RawSensorObservation(
            visual_data={
                "rgb": np.zeros((16, 16, 3), dtype=np.uint8),
                "depth": visual_depth,
                "seg": visual_seg,
                "view_matrix": np.eye(4, dtype=np.float32),
                "proj_matrix": np.eye(4, dtype=np.float32),
            },
            tactile_data={
                "rgb": np.zeros((2, 320, 240, 3), dtype=np.uint8),
                "depth": tactile_depth,
                "proj_matrix": np.eye(4, dtype=np.float32),
                "camera_distance_to_gel_m": 0.02315,
            },
            grasp_metadata={
                "segmentation_ids": {"object": 1},
                "source_global_id": 11,
                "observation_stage": "before",
                "gel_pose_world": {
                    "left": {"position": [0.0, 0.0, 0.0], "quaternion": [0.0, 0.0, 0.0, 1.0]},
                    "right": {"position": [0.05, 0.0, 0.0], "quaternion": [0.0, 0.0, 0.0, 1.0]},
                },
            },
        )

        prepared = adapter.prepare_inputs(raw_obs)

        self.assertEqual(tuple(prepared.sc_input.shape), (64, 3))
        self.assertEqual(tuple(prepared.gs_input.shape), (40, 4))
        self.assertEqual(tuple(prepared.zero_mean.shape), (3,))
        self.assertTrue(np.all(np.isfinite(prepared.sc_input)))
        self.assertTrue(np.all(np.isfinite(prepared.gs_input)))
        self.assertEqual(prepared.debug_tactile_left_gel_mask.shape[0], 20)
        self.assertEqual(prepared.debug_tactile_right_gel_mask.shape[0], 20)


if __name__ == "__main__":
    unittest.main()
