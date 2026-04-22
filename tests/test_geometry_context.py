from __future__ import annotations

import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from src.envs.geometry_context import (
    action_axes_in_camera_from_view,
    camera_geometry_context,
    hand_pose_in_camera_from_view,
)
from src.structures.action import GraspPose
from src.structures.observation import RawSensorObservation
from src.utils.geometry import pose_to_matrix, rotvec_to_quaternion


class TestGeometryContext(unittest.TestCase):
    def test_identity_view_and_hand_pose(self):
        view = np.eye(4, dtype=np.float32)
        grasp_pose = GraspPose(position=np.zeros(3, dtype=np.float32), rotation=np.zeros(3, dtype=np.float32))

        action_axes = action_axes_in_camera_from_view(view)
        hand_pose = hand_pose_in_camera_from_view(view, grasp_pose)

        np.testing.assert_allclose(action_axes, np.eye(3, dtype=np.float32).reshape(-1))
        np.testing.assert_allclose(hand_pose[:3], np.zeros(3, dtype=np.float32))
        np.testing.assert_allclose(hand_pose[3:], np.eye(3, dtype=np.float32).reshape(-1))
        self.assertEqual(action_axes.dtype, np.float32)
        self.assertEqual(hand_pose.dtype, np.float32)

    def test_nontrivial_camera_hand_transform_matches_matrix_product(self):
        view = np.eye(4, dtype=np.float32)
        view[:3, :3] = Rotation.from_euler("zyx", [20.0, -10.0, 5.0], degrees=True).as_matrix().astype(np.float32)
        view[:3, 3] = np.asarray([0.2, -0.3, 0.4], dtype=np.float32)
        grasp_pose = GraspPose(
            position=np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
            rotation=Rotation.from_euler("xyz", [7.0, -3.0, 11.0], degrees=True).as_rotvec().astype(np.float32),
        )

        expected = view @ pose_to_matrix(grasp_pose.position, rotvec_to_quaternion(grasp_pose.rotation))
        hand_pose = hand_pose_in_camera_from_view(view, grasp_pose)

        np.testing.assert_allclose(hand_pose[:3], expected[:3, 3], rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(hand_pose[3:], expected[:3, :3].reshape(-1), rtol=1e-6, atol=1e-6)

    def test_camera_geometry_context_requires_view_matrix(self):
        raw_obs = RawSensorObservation(visual_data={}, tactile_data={}, grasp_metadata={})
        grasp_pose = GraspPose(position=np.zeros(3, dtype=np.float32), rotation=np.zeros(3, dtype=np.float32))

        with self.assertRaisesRegex(ValueError, "view_matrix"):
            camera_geometry_context(raw_obs, grasp_pose)


if __name__ == "__main__":
    unittest.main()
