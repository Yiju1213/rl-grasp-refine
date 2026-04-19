from __future__ import annotations

import unittest

import numpy as np

from src.perception.adapters import CNNMCAAdapter
from src.structures.observation import RawSensorObservation


class TestCNNMCAAdapter(unittest.TestCase):
    @staticmethod
    def _raw_obs() -> RawSensorObservation:
        visual_rgb = np.zeros((448, 448, 3), dtype=np.uint8)
        tactile_rgb = np.zeros((2, 320, 240, 3), dtype=np.uint8)
        tactile_rgb[0, ..., 0] = 255
        tactile_rgb[1, ..., 1] = 255
        return RawSensorObservation(
            visual_data={"rgb": visual_rgb},
            tactile_data={"rgb": tactile_rgb},
            grasp_metadata={"source_global_id": 1, "observation_stage": "before"},
        )

    def test_prepare_inputs_converts_env_rgb_to_cnnmca_tensors(self):
        adapter = CNNMCAAdapter({"cnnmca": {"runtime": {"image_size": 224, "concat_two_tactile": True}}})

        prepared = adapter.prepare_inputs(self._raw_obs())

        self.assertEqual(tuple(prepared.visual_img.shape), (3, 224, 224))
        self.assertEqual(tuple(prepared.tactile_img.shape), (3, 224, 224))
        self.assertEqual(prepared.visual_img.dtype, np.float32)
        self.assertEqual(prepared.tactile_img.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(prepared.visual_img)))
        self.assertTrue(np.all(np.isfinite(prepared.tactile_img)))

        model_inputs = adapter.adapt_feature_input(self._raw_obs())
        self.assertEqual(tuple(model_inputs["visual_img"].shape), (1, 3, 224, 224))
        self.assertEqual(tuple(model_inputs["tactile_img"].shape), (1, 3, 224, 224))

    def test_prepare_inputs_rejects_missing_rgb(self):
        adapter = CNNMCAAdapter({"cnnmca": {"runtime": {"image_size": 224, "concat_two_tactile": True}}})
        raw_obs = RawSensorObservation(visual_data={}, tactile_data={"rgb": np.zeros((2, 8, 8, 3))}, grasp_metadata={})

        with self.assertRaisesRegex(ValueError, "visual_data"):
            adapter.prepare_inputs(raw_obs)

    def test_prepare_inputs_rejects_non_concatenated_tactile_mode(self):
        adapter = CNNMCAAdapter({"cnnmca": {"runtime": {"image_size": 224, "concat_two_tactile": False}}})

        with self.assertRaisesRegex(NotImplementedError, "concat_two_tactile=true"):
            adapter.prepare_inputs(self._raw_obs())


if __name__ == "__main__":
    unittest.main()
