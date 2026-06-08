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


def _hand_world_matrix(grasp_pose: GraspPose) -> np.ndarray:
    return pose_to_matrix(grasp_pose.position, rotvec_to_quaternion(grasp_pose.rotation))


def _pose_position(pose_data) -> np.ndarray | None:
    if pose_data is None:
        return None
    if not isinstance(pose_data, dict) or "position" not in pose_data:
        return None
    position = np.asarray(pose_data["position"], dtype=np.float32).reshape(3)
    if not np.all(np.isfinite(position)):
        return None
    return position


def _gel_pose_source(raw_obs: RawSensorObservation, side: str):
    grasp_metadata = raw_obs.grasp_metadata or {}
    gel_pose_world = grasp_metadata.get("gel_pose_world", {})
    if isinstance(gel_pose_world, dict):
        pose_data = gel_pose_world.get(side)
        if pose_data is not None:
            return pose_data

    tactile_data = raw_obs.tactile_data or {}
    sensor_poses_world = tactile_data.get("sensor_poses_world", {})
    if isinstance(sensor_poses_world, dict):
        return sensor_poses_world.get(side)
    return None


def _world_point_to_camera(view_matrix: np.ndarray, world_point: np.ndarray) -> np.ndarray:
    hom = np.concatenate([np.asarray(world_point, dtype=np.float32).reshape(3), np.ones(1, dtype=np.float32)])
    return (view_matrix @ hom)[:3].astype(np.float32)


def _safe_unit(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float32).reshape(3)
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm) or norm <= 1e-8:
        return np.asarray(fallback, dtype=np.float32).reshape(3)
    return (value / norm).astype(np.float32)


def finger_geometry_in_camera_from_view(
    view_matrix,
    grasp_pose: GraspPose,
    *,
    left_gel_pose_world=None,
    right_gel_pose_world=None,
) -> np.ndarray:
    """Return compact finger/contact-frame geometry in camera coordinates.

    Layout: gripper center (3), right-left gel baseline vector (3), hand +z approach axis (3).
    The baseline keeps physical length; only the approach axis is normalized.
    """

    matrix = _as_view_matrix(view_matrix)
    hand_world = _hand_world_matrix(grasp_pose)
    hand_pose = hand_pose_in_camera_from_view(matrix, grasp_pose)
    fallback_center = hand_pose[:3].astype(np.float32)

    left_position = _pose_position(left_gel_pose_world)
    right_position = _pose_position(right_gel_pose_world)
    if left_position is not None and right_position is not None:
        left_camera = _world_point_to_camera(matrix, left_position)
        right_camera = _world_point_to_camera(matrix, right_position)
        center = (0.5 * (left_camera + right_camera)).astype(np.float32)
        baseline = (right_camera - left_camera).astype(np.float32)
    else:
        center = fallback_center
        baseline = np.zeros(3, dtype=np.float32)

    approach_world = hand_world[:3, 2].astype(np.float32)
    approach_camera = _safe_unit(
        matrix[:3, :3] @ approach_world,
        fallback=np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
    )
    return np.concatenate([center, baseline, approach_camera], axis=0).astype(np.float32)


def finger_geometry_context(raw_obs: RawSensorObservation, grasp_pose: GraspPose) -> np.ndarray:
    visual = raw_obs.visual_data or {}
    if "view_matrix" not in visual:
        raise ValueError("RawSensorObservation.visual_data must include 'view_matrix' for finger geometry features.")
    return finger_geometry_in_camera_from_view(
        visual["view_matrix"],
        grasp_pose,
        left_gel_pose_world=_gel_pose_source(raw_obs, "left"),
        right_gel_pose_world=_gel_pose_source(raw_obs, "right"),
    )


def camera_geometry_context(raw_obs: RawSensorObservation, grasp_pose: GraspPose) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    visual = raw_obs.visual_data or {}
    if "view_matrix" not in visual:
        raise ValueError("RawSensorObservation.visual_data must include 'view_matrix' for camera geometry features.")
    view_matrix = visual["view_matrix"]
    return (
        action_axes_in_camera_from_view(view_matrix),
        hand_pose_in_camera_from_view(view_matrix, grasp_pose),
        finger_geometry_context(raw_obs, grasp_pose),
    )
