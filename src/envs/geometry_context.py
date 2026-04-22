from __future__ import annotations

import numpy as np

from src.structures.action import GraspPose
from src.structures.observation import RawSensorObservation
from src.utils.geometry import pose_to_matrix, rotvec_to_quaternion


def _as_view_matrix(view_matrix) -> np.ndarray:
    matrix = np.asarray(view_matrix, dtype=np.float32).reshape(4, 4)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("visual view_matrix must contain only finite values.")
    return matrix


def action_axes_in_camera_from_view(view_matrix) -> np.ndarray:
    """Return world/action axes expressed in the current camera frame."""
    matrix = _as_view_matrix(view_matrix)
    return matrix[:3, :3].astype(np.float32).reshape(-1)


def hand_pose_in_camera_from_view(view_matrix, grasp_pose: GraspPose) -> np.ndarray:
    """Return current hand pose in camera frame as position + 3x3 rotation."""
    matrix = _as_view_matrix(view_matrix)
    hand_world = pose_to_matrix(grasp_pose.position, rotvec_to_quaternion(grasp_pose.rotation))
    camera_hand = matrix @ hand_world
    return np.concatenate(
        [
            camera_hand[:3, 3].astype(np.float32).reshape(3),
            camera_hand[:3, :3].astype(np.float32).reshape(-1),
        ],
        axis=0,
    ).astype(np.float32)


def camera_geometry_context(raw_obs: RawSensorObservation, grasp_pose: GraspPose) -> tuple[np.ndarray, np.ndarray]:
    visual = raw_obs.visual_data or {}
    if "view_matrix" not in visual:
        raise ValueError("RawSensorObservation.visual_data must include 'view_matrix' for camera geometry features.")
    view_matrix = visual["view_matrix"]
    return (
        action_axes_in_camera_from_view(view_matrix),
        hand_pose_in_camera_from_view(view_matrix, grasp_pose),
    )
