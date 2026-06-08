from __future__ import annotations

import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from src.envs.geometry_context import (
    action_axes_in_camera_from_view,
    camera_geometry_context,
    finger_geometry_in_camera_from_view,
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

    def test_finger_geometry_identity_view_with_symmetric_gels(self):
        view = np.eye(4, dtype=np.float32)
        grasp_pose = GraspPose(position=np.zeros(3, dtype=np.float32), rotation=np.zeros(3, dtype=np.float32))

        finger_geom = finger_geometry_in_camera_from_view(
            view,
            grasp_pose,
            left_gel_pose_world={"position": [0.0, -0.02, 0.0], "quaternion": [0.0, 0.0, 0.0, 1.0]},
            right_gel_pose_world={"position": [0.0, 0.02, 0.0], "quaternion": [0.0, 0.0, 0.0, 1.0]},
        )

        self.assertEqual(finger_geom.shape, (9,))
        self.assertEqual(finger_geom.dtype, np.float32)
        np.testing.assert_allclose(finger_geom[:3], np.zeros(3, dtype=np.float32), atol=1e-7)
        np.testing.assert_allclose(finger_geom[3:6], np.asarray([0.0, 0.04, 0.0], dtype=np.float32), atol=1e-7)
        np.testing.assert_allclose(finger_geom[6:9], np.asarray([0.0, 0.0, 1.0], dtype=np.float32), atol=1e-7)

    def test_finger_geometry_transforms_gel_positions_to_camera_frame(self):
        view = np.eye(4, dtype=np.float32)
        view[:3, :3] = Rotation.from_euler("z", 90.0, degrees=True).as_matrix().astype(np.float32)
        view[:3, 3] = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
        grasp_pose = GraspPose(
            position=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
            rotation=np.zeros(3, dtype=np.float32),
        )

        finger_geom = finger_geometry_in_camera_from_view(
            view,
            grasp_pose,
            left_gel_pose_world={"position": [1.0, 0.0, 0.0], "quaternion": [0.0, 0.0, 0.0, 1.0]},
            right_gel_pose_world={"position": [1.0, 1.0, 0.0], "quaternion": [0.0, 0.0, 0.0, 1.0]},
        )

        left_camera = (view @ np.asarray([1.0, 0.0, 0.0, 1.0], dtype=np.float32))[:3]
        right_camera = (view @ np.asarray([1.0, 1.0, 0.0, 1.0], dtype=np.float32))[:3]
        expected_center = 0.5 * (left_camera + right_camera)
        expected_baseline = right_camera - left_camera

        np.testing.assert_allclose(finger_geom[:3], expected_center, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(finger_geom[3:6], expected_baseline, rtol=1e-6, atol=1e-6)
        self.assertAlmostEqual(float(np.linalg.norm(finger_geom[6:9])), 1.0, places=6)

    def test_finger_geometry_missing_gels_uses_finite_fallback(self):
        view = np.eye(4, dtype=np.float32)
        view[:3, 3] = np.asarray([0.1, -0.2, 0.3], dtype=np.float32)
        grasp_pose = GraspPose(
            position=np.asarray([0.4, 0.5, 0.6], dtype=np.float32),
            rotation=Rotation.from_euler("x", 30.0, degrees=True).as_rotvec().astype(np.float32),
        )

        finger_geom = finger_geometry_in_camera_from_view(
            view,
            grasp_pose,
            left_gel_pose_world=None,
            right_gel_pose_world=None,
        )

        expected_center = hand_pose_in_camera_from_view(view, grasp_pose)[:3]
        self.assertEqual(finger_geom.shape, (9,))
        self.assertTrue(np.all(np.isfinite(finger_geom)))
        np.testing.assert_allclose(finger_geom[:3], expected_center, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(finger_geom[3:6], np.zeros(3, dtype=np.float32), atol=1e-7)
        self.assertAlmostEqual(float(np.linalg.norm(finger_geom[6:9])), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
