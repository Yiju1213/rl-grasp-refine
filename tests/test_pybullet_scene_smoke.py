from __future__ import annotations

import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from src.envs.dataset_sample_provider import DatasetSampleProvider
from src.envs.pybullet_scene import PyBulletScene
from src.structures.action import GraspPose


class TestPyBulletSceneSmoke(unittest.TestCase):
    def test_table_disabled_by_default_is_not_spawned(self):
        fake_hand = SimpleNamespace(id=1, gsmini_joint_ids=[0])
        with (
            patch.object(PyBulletScene, "_connect"),
            patch("src.envs.pybullet_scene.spawn_hand", return_value=fake_hand),
            patch("src.envs.pybullet_scene.create_tacto_sensor", return_value=SimpleNamespace()),
            patch("src.envs.pybullet_scene.spawn_table") as table_mock,
        ):
            scene = PyBulletScene({"use_gui": False})
            scene._ensure_static_scene_assets()

        table_mock.assert_not_called()
        self.assertIsNone(scene.table_body)
        snapshot = scene.get_debug_snapshot()
        self.assertFalse(snapshot["table"]["enabled"])
        self.assertIsNone(snapshot["table"]["body_id"])

    def test_table_enabled_spawns_once_and_is_reported_in_debug_snapshot(self):
        fake_hand = SimpleNamespace(id=1, gsmini_joint_ids=[0])
        fake_table = SimpleNamespace(id=42)
        expected_urdf = (Path(__file__).resolve().parents[1] / "src/envs/object_model/table/table.urdf").resolve()
        table_cfg = {
            "enabled": True,
            "urdf_path": "src/envs/object_model/table/table.urdf",
            "base_position": [0, 0, 0],
            "base_orientation": [0, 0, 0, 1],
            "use_fixed_base": True,
        }
        with (
            patch.object(PyBulletScene, "_connect"),
            patch("src.envs.pybullet_scene.spawn_hand", return_value=fake_hand),
            patch("src.envs.pybullet_scene.create_tacto_sensor", return_value=SimpleNamespace()),
            patch("src.envs.pybullet_scene.spawn_table", return_value=fake_table) as table_mock,
        ):
            scene = PyBulletScene({"use_gui": False, "table": table_cfg})
            scene._ensure_static_scene_assets()
            scene._ensure_static_scene_assets()

        table_mock.assert_called_once_with(table_cfg)
        self.assertIs(scene.table_body, fake_table)
        snapshot = scene.get_debug_snapshot()
        self.assertTrue(snapshot["table"]["enabled"])
        self.assertEqual(snapshot["table"]["body_id"], 42)
        self.assertEqual(Path(snapshot["table"]["urdf_path"]).resolve(), expected_urdf)

    def test_post_refine_settle_toggle_resolution(self):
        resolve = PyBulletScene._resolve_post_refine_settle_steps

        self.assertEqual(resolve({"post_refine_settle_steps": 8}, "reset"), 0)
        self.assertEqual(resolve({}, "refine"), 8)
        self.assertEqual(resolve({"post_refine_settle_steps": 2}, "refine"), 2)
        self.assertEqual(
            resolve({"post_refine_settle_enabled": False, "post_refine_settle_steps": 8}, "refine"),
            0,
        )
        self.assertEqual(
            resolve({"post_refine_settle_enabled": True, "post_refine_settle_steps": -1}, "refine"),
            0,
        )

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
        for key in sample_0["source"]["before_paths"]:
            sample_0["source"]["before_paths"][key] = f"/nonexistent/{key}"
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
        self.assertEqual(before.visual_data["rgb"].shape, after_first.visual_data["rgb"].shape)
        self.assertEqual(before.visual_data["depth"].shape, after_first.visual_data["depth"].shape)
        self.assertEqual(before.tactile_data["rgb"].shape, after_first.tactile_data["rgb"].shape)
        self.assertEqual(before.tactile_data["depth"].shape, after_first.tactile_data["depth"].shape)
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
