from __future__ import annotations

import os
import unittest
from pathlib import Path

import numpy as np

from src.envs.dataset_sample_provider import DatasetSampleProvider
from src.envs.pybullet_scene import PyBulletScene
from src.structures.action import GraspPose


class TestPyBulletSceneSmoke(unittest.TestCase):
    def test_scene_can_reset_and_refine_from_real_dataset_sample(self):
        dataset_root = Path("/Datasets/GraspNet-1billion/tactile-extended")
        if not dataset_root.exists():
            self.skipTest(f"Dataset root not available: {dataset_root}")
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

        provider = DatasetSampleProvider(
            {
                "dataset_root": str(dataset_root),
                "seed": 1,
                "metadata_cache_size": 1,
                "runtime_defaults": {
                    "time_step": 0.005,
                    "close_timeout_s": 0.1,
                    "effort_timeout_s": 0.1,
                    "grip_force": 10.0,
                    "release_duration_s": 0.2,
                    "release_check_interval_steps": 2,
                    "post_refine_settle_steps": 2,
                },
            }
        )
        metadata_0 = provider._load_object_metadata(0)
        global_id_0 = int(next(iter(metadata_0.keys())))
        sample_0 = provider._entry_to_sample_cfg(object_id=0, global_id=global_id_0, entry=metadata_0[str(global_id_0)])
        metadata_1 = provider._load_object_metadata(1)
        global_id_1 = int(next(iter(metadata_1.keys())))
        sample_1 = provider._entry_to_sample_cfg(object_id=1, global_id=global_id_1, entry=metadata_1[str(global_id_1)])
        scene = PyBulletScene(
            {
                "use_gui": False,
                "time_step": 0.005,
                "tacto_width": 240,
                "tacto_height": 320,
                "visual_width": 64,
                "visual_height": 64,
                "visual_near": 0.01,
                "visual_far": 2.0,
                "visualize_tacto_gui": False,
                "visual_point_cloud_max_points": 128,
            }
        )

        try:
            scene.reset_scene(sample_0)
            before = scene.get_raw_observation()
            initial_debug = scene.get_debug_snapshot()
            first_refined_pose = GraspPose(
                position=np.asarray(sample_0["initial_grasp_pose"]["position"], dtype=np.float32),
                rotation=np.asarray(sample_0["initial_grasp_pose"]["rotation"], dtype=np.float32),
            )

            original_reset = scene.hand.reset
            reset_calls = {"count": 0}

            def counted_reset():
                reset_calls["count"] += 1
                return original_reset()

            scene.hand.reset = counted_reset
            scene.apply_refinement(first_refined_pose)
            after_first = scene.get_raw_observation()
            first_trial = scene.run_grasp_trial()

            scene.reset_scene(sample_0)
            same_object_debug = scene.get_debug_snapshot()

            scene.reset_scene(sample_1)
            swapped_debug = scene.get_debug_snapshot()
            second_refined_pose = GraspPose(
                position=np.asarray(sample_1["initial_grasp_pose"]["position"], dtype=np.float32),
                rotation=np.asarray(sample_1["initial_grasp_pose"]["rotation"], dtype=np.float32),
            )
            scene.apply_refinement(second_refined_pose)
            after_second = scene.get_raw_observation()
            second_trial = scene.run_grasp_trial()
            final_debug = scene.get_debug_snapshot()
        finally:
            scene.close()

        self.assertEqual(before.grasp_metadata["observation_stage"], "before")
        self.assertEqual(after_first.grasp_metadata["observation_stage"], "after")
        self.assertEqual(after_second.grasp_metadata["observation_stage"], "after")
        self.assertIn("valid_for_learning", first_trial["trial_metadata"])
        self.assertIn("valid_for_learning", second_trial["trial_metadata"])
        self.assertGreaterEqual(reset_calls["count"], 1)
        self.assertIn("refine_close_steps", final_debug["runtime_counters"])
        self.assertIn("refine_effort_steps", final_debug["runtime_counters"])
        self.assertEqual(final_debug["refine_debug"]["stage"], "refine")
        self.assertEqual(initial_debug["source_object_id"], 0)
        self.assertEqual(swapped_debug["source_object_id"], 1)

        self.assertEqual(initial_debug["reset_debug"]["object_action"], "spawn")
        self.assertEqual(initial_debug["reset_debug"]["tacto_action"], "created")
        self.assertEqual(same_object_debug["reset_debug"]["object_action"], "reuse")
        self.assertEqual(same_object_debug["reset_debug"]["tacto_action"], "reuse")
        self.assertEqual(swapped_debug["reset_debug"]["object_action"], "swap")
        self.assertEqual(swapped_debug["reset_debug"]["tacto_action"], "reuse")

        self.assertEqual(initial_debug["hand_body_id"], same_object_debug["hand_body_id"])
        self.assertEqual(initial_debug["hand_body_id"], swapped_debug["hand_body_id"])
        self.assertEqual(initial_debug["current_object_body_id"], same_object_debug["current_object_body_id"])


if __name__ == "__main__":
    unittest.main()
