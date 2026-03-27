from __future__ import annotations

import unittest

import numpy as np

from src.envs.observation_builder import ObservationBuilder
from src.perception.contact_semantics import ContactSemanticsExtractor
from src.perception.feature_extractor import FeatureExtractor
from src.perception.sga_gsn_types import PreparedVTGInputs, SGAGSNInferenceResult
from src.perception.stability_predictor import StabilityPredictor
from src.structures.action import GraspPose
from src.structures.observation import RawSensorObservation


class _DummyRuntime:
    def __init__(self):
        self.calls = 0

    def infer(self, raw_obs, adapter):
        self.calls += 1
        return SGAGSNInferenceResult(
            prepared_inputs=PreparedVTGInputs(
                sc_input=np.zeros((4, 3), dtype=np.float32),
                gs_input=np.zeros((8, 4), dtype=np.float32),
                zero_mean=np.zeros(3, dtype=np.float32),
                debug_visual_world_points=np.zeros((0, 3), dtype=np.float32),
                debug_tactile_left_world_points=np.zeros((0, 3), dtype=np.float32),
                debug_tactile_right_world_points=np.zeros((0, 3), dtype=np.float32),
                debug_tactile_left_contact_world_points=np.zeros((0, 3), dtype=np.float32),
                debug_tactile_right_contact_world_points=np.zeros((0, 3), dtype=np.float32),
                debug_tactile_left_gel_mask=np.zeros((0,), dtype=bool),
                debug_tactile_right_gel_mask=np.zeros((0,), dtype=bool),
            ),
            body_feature=np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
            raw_logit=0.75,
        )


class TestObservationBuilderRuntimeCache(unittest.TestCase):
    def test_builder_uses_runtime_once_without_mutating_raw_obs_metadata(self):
        runtime = _DummyRuntime()
        feature_extractor = FeatureExtractor(backbone_model=None, adapter=object(), runtime=runtime)
        stability_predictor = StabilityPredictor(predictor_model=None)
        builder = ObservationBuilder(
            feature_extractor=feature_extractor,
            contact_semantics_extractor=ContactSemanticsExtractor({"tactile_threshold": 0.2}),
            stability_predictor=stability_predictor,
        )

        raw_obs = RawSensorObservation(
            visual_data={"point_cloud": np.zeros((8, 3), dtype=np.float32)},
            tactile_data={"contact_map": np.zeros((2, 4, 4), dtype=np.float32)},
            grasp_metadata={"grasp_pose": "original_pose", "observation_stage": "before"},
        )
        metadata_keys_before = set(raw_obs.grasp_metadata.keys())
        grasp_pose = GraspPose(position=np.zeros(3, dtype=np.float32), rotation=np.zeros(3, dtype=np.float32))

        obs = builder.build(raw_obs, grasp_pose)

        self.assertEqual(runtime.calls, 1)
        self.assertEqual(tuple(obs.latent_feature.shape), (4,))
        self.assertAlmostEqual(obs.raw_stability_logit, 0.75, places=6)
        self.assertEqual(set(raw_obs.grasp_metadata.keys()), metadata_keys_before)
        self.assertNotIn("contact_semantic", raw_obs.grasp_metadata)


if __name__ == "__main__":
    unittest.main()
