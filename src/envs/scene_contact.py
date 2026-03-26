from __future__ import annotations

from typing import Any

import numpy as np
import pybullet as pb

from src.utils.geometry import Pose


def check_contact(gripper_id: int, object_id: int, desired_contact_links: list[int]) -> tuple[bool, int]:
    contact_points = pb.getContactPoints(bodyA=gripper_id, bodyB=object_id)
    if not contact_points:
        return False, 0

    contact_links = set(contact[3] for contact in contact_points)
    desired_links = set(desired_contact_links)
    non_desired_links = contact_links - desired_links
    if len(contact_links & desired_links) == len(desired_links) or non_desired_links:
        return True, len(non_desired_links)
    return False, 0


def has_any_contact(gripper_id: int, object_id: int) -> bool:
    return bool(pb.getContactPoints(bodyA=gripper_id, bodyB=object_id))


def check_target_force(gripper_id: int, joint_ids: list[int], target_force: float, tolerance: float = 5.0) -> bool:
    for joint_id in joint_ids:
        current_force = float(pb.getJointState(gripper_id, joint_id)[3])
        if abs(current_force - target_force) > tolerance:
            return False
    return True


def get_link_poses(body_id: int, link_ids: list[int]) -> list[Pose]:
    poses: list[Pose] = []
    for link_id in link_ids:
        state = pb.getLinkState(body_id, link_id, computeLinkVelocity=False, computeForwardKinematics=False)
        poses.append(Pose(position=state[4], quaternion=state[5]))
    return poses


def matrix_from_metadata(flat_values: Any) -> np.ndarray:
    return np.asarray(flat_values, dtype=np.float32).reshape(4, 4)


def matrix_to_pybullet_list(matrix: np.ndarray) -> list[float]:
    return np.asarray(matrix, dtype=np.float32).reshape(4, 4).T.reshape(-1).tolist()
