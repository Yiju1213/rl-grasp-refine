from __future__ import annotations

from typing import Any

import numpy as np
import pybullet as pb

from src.envs.scene_contact import get_link_poses, matrix_from_metadata, matrix_to_pybullet_list
from src.structures.observation import RawSensorObservation
from src.utils.geometry import DEFAULT_TACTILE_CAMERA_TO_GEL_M, normalize_tactile_depth, pose_to_matrix


def capture_scene_observation(
    sample_cfg: dict,
    scene_cfg: dict,
    tacto_sensor,
    hand,
    object_body,
    client_id: int,
    current_grasp_pose,
    stage: str,
) -> RawSensorObservation:
    gels_color, gels_depth = tacto_sensor.render()
    tactile_rgb = np.stack([np.asarray(color, dtype=np.uint8) for color in gels_color], axis=0)
    tactile_depth = np.stack([np.asarray(depth, dtype=np.float32) for depth in gels_depth], axis=0)

    view_matrix = current_visual_view_matrix(sample_cfg=sample_cfg, hand=hand, client_id=client_id)
    visual_proj_matrix = matrix_from_metadata(sample_cfg["camera"]["visual_proj_matrix"])
    width = int(scene_cfg.get("visual_width", 448))
    height = int(scene_cfg.get("visual_height", 448))
    renderer = pb.ER_BULLET_HARDWARE_OPENGL if bool(scene_cfg.get("use_gui", False)) else pb.ER_TINY_RENDERER
    _, _, rgb_buffer, depth_ndc, segmentation = pb.getCameraImage(
        width=width,
        height=height,
        viewMatrix=matrix_to_pybullet_list(view_matrix),
        projectionMatrix=matrix_to_pybullet_list(visual_proj_matrix),
        renderer=renderer,
        physicsClientId=client_id,
    )
    # PyBullet returns an RGBA buffer here even for RGB rendering; drop alpha explicitly.
    visual_rgb = np.asarray(rgb_buffer, dtype=np.uint8)[..., :3].copy()
    depth_ndc = np.asarray(depth_ndc, dtype=np.float32)
    near_val = float(scene_cfg.get("visual_near", 0.01))
    far_val = float(scene_cfg.get("visual_far", 2.0))
    visual_depth = far_val * near_val / (far_val - (far_val - near_val) * depth_ndc)
    visual_seg = np.asarray(segmentation, dtype=np.int16)

    tactile_proj_matrix = current_tactile_proj_matrix(tacto_sensor, scene_cfg)
    left_pose, right_pose = get_link_poses(hand.id, hand.gsmini_gel_ids)
    contact_map = normalize_tactile_depth(tactile_depth)
    object_position, object_quaternion = pb.getBasePositionAndOrientation(
        object_body.id, physicsClientId=client_id
    )

    observation_valid = observation_arrays_valid(
        visual_rgb=visual_rgb,
        visual_depth=visual_depth,
        visual_seg=visual_seg,
        tactile_rgb=tactile_rgb,
        tactile_depth=tactile_depth,
    )

    return RawSensorObservation(
        visual_data={
            "rgb": visual_rgb,
            "depth": visual_depth.astype(np.float32),
            "seg": visual_seg,
            "view_matrix": view_matrix,
            "proj_matrix": visual_proj_matrix,
        },
        tactile_data={
            "rgb": tactile_rgb,
            "depth": tactile_depth.astype(np.float32),
            "proj_matrix": tactile_proj_matrix,
            "sensor_poses_world": {
                "left": {"position": left_pose.position.tolist(), "quaternion": left_pose.quaternion.tolist()},
                "right": {"position": right_pose.position.tolist(), "quaternion": right_pose.quaternion.tolist()},
            },
            "camera_distance_to_gel_m": DEFAULT_TACTILE_CAMERA_TO_GEL_M,
            "contact_map": contact_map,
            "contact_force": float(np.mean(contact_map)),
        },
        grasp_metadata={
            "grasp_pose": current_grasp_pose,
            "object_pose_world": {
                "position": list(object_position),
                "quaternion": list(object_quaternion),
            },
            "source_object_id": int(sample_cfg["source"]["object_id"]),
            "source_global_id": int(sample_cfg["source"]["global_id"]),
            "observation_stage": str(stage),
            "segmentation_ids": {"object": int(object_body.id), "hand": int(hand.id)},
            "gel_pose_world": {
                "left": {"position": left_pose.position.tolist(), "quaternion": left_pose.quaternion.tolist()},
                "right": {"position": right_pose.position.tolist(), "quaternion": right_pose.quaternion.tolist()},
            },
            "observation_valid": observation_valid,
        },
    )


def make_invalid_after_observation(sample_cfg: dict, scene_cfg: dict, current_grasp_pose) -> RawSensorObservation:
    view_matrix = matrix_from_metadata(sample_cfg["camera"]["view_matrix"])
    visual_proj_matrix = matrix_from_metadata(sample_cfg["camera"]["visual_proj_matrix"])
    tactile_proj_matrix = matrix_from_metadata(sample_cfg["camera"]["tactile_proj_matrix"])
    tactile_height = int(scene_cfg.get("tacto_height", 320))
    tactile_width = int(scene_cfg.get("tacto_width", 240))
    visual_height = int(scene_cfg.get("visual_height", 448))
    visual_width = int(scene_cfg.get("visual_width", 448))

    return RawSensorObservation(
        visual_data={
            "rgb": np.zeros((visual_height, visual_width, 3), dtype=np.uint8),
            "depth": np.zeros((visual_height, visual_width), dtype=np.float32),
            "seg": -np.ones((visual_height, visual_width), dtype=np.int16),
            "view_matrix": view_matrix,
            "proj_matrix": visual_proj_matrix,
        },
        tactile_data={
            "rgb": np.zeros((2, tactile_height, tactile_width, 3), dtype=np.uint8),
            "depth": np.zeros((2, tactile_height, tactile_width), dtype=np.float32),
            "proj_matrix": tactile_proj_matrix,
            "sensor_poses_world": {"left": None, "right": None},
            "camera_distance_to_gel_m": DEFAULT_TACTILE_CAMERA_TO_GEL_M,
            "contact_map": np.zeros((2, tactile_height, tactile_width), dtype=np.float32),
            "contact_force": 0.0,
        },
        grasp_metadata={
            "grasp_pose": current_grasp_pose,
            "object_pose_world": sample_cfg["grasping"]["object_pose_world"],
            "source_object_id": int(sample_cfg["source"]["object_id"]),
            "source_global_id": int(sample_cfg["source"]["global_id"]),
            "observation_stage": "after",
            "segmentation_ids": sample_cfg["source"].get("segmentation_ids", {"object": 1, "hand": 3}),
            "gel_pose_world": {"left": None, "right": None},
            "observation_valid": False,
        },
    )


def current_tactile_proj_matrix(tacto_sensor, scene_cfg: dict) -> np.ndarray:
    node = tacto_sensor.renderer.camera_nodes[0]
    camera = node.camera
    camera.aspectRatio = float(scene_cfg.get("tacto_width", 240)) / max(float(scene_cfg.get("tacto_height", 320)), 1.0)
    return np.asarray(camera.get_projection_matrix(), dtype=np.float32).reshape(4, 4)


def current_visual_view_matrix(sample_cfg: dict, hand, client_id: int) -> np.ndarray:
    sample_view_matrix = matrix_from_metadata(sample_cfg["camera"]["view_matrix"])
    sample_camera_pose = np.linalg.inv(sample_view_matrix).astype(np.float32)

    sample_hand_pose = pose_to_matrix(
        sample_cfg["pre_grasp"]["hand_pose_world"]["position"],
        sample_cfg["pre_grasp"]["hand_pose_world"]["quaternion"],
    )
    hand_to_camera = np.linalg.inv(sample_hand_pose).astype(np.float32) @ sample_camera_pose

    current_hand_position, current_hand_quaternion = pb.getBasePositionAndOrientation(hand.id, physicsClientId=client_id)
    current_hand_pose = pose_to_matrix(current_hand_position, current_hand_quaternion)
    current_camera_pose = current_hand_pose @ hand_to_camera
    return np.linalg.inv(current_camera_pose).astype(np.float32)


def observation_arrays_valid(**arrays: Any) -> bool:
    for value in arrays.values():
        arr = np.asarray(value)
        if arr.size == 0 or not np.all(np.isfinite(arr)):
            return False
    return True
