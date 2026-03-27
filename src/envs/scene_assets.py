from __future__ import annotations

import cv2
import pybullet as pb
import pybulletX as px
import tacto

from src.envs.asset_paths import SceneAssetPaths, resolve_object_urdf
from src.envs.hand.gsmini_panda_hand import GSminiPandaHand


def spawn_object(asset_paths: SceneAssetPaths, object_id: int, object_pose_world: dict):
    object_urdf = resolve_object_urdf(object_id)
    if not object_urdf.exists():
        raise FileNotFoundError(f"Missing object URDF: {object_urdf}")
    # TODO: Object URDF load/spawn remains a known reset hotspot. Optimize only
    # after the scene lifecycle refactor is stable and measured end-to-end.
    return px.Body(
        urdf_path=str(object_urdf),
        base_position=object_pose_world["position"],
        base_orientation=object_pose_world["quaternion"],
        use_fixed_base=False,
    )


def spawn_hand(asset_paths: SceneAssetPaths, hand_pose_world: dict):
    if not asset_paths.hand_urdf.exists():
        raise FileNotFoundError(f"Missing hand URDF: {asset_paths.hand_urdf}")
    hand = GSminiPandaHand(
        robot_params={
            "urdf_path": str(asset_paths.hand_urdf),
            "base_position": hand_pose_world["position"],
            "base_orientation": hand_pose_world["quaternion"],
            "use_fixed_base": True,
        }
    )
    hand.reset()
    return hand


def create_tacto_sensor(asset_paths: SceneAssetPaths, cfg: dict, hand, client_id: int):
    background = cv2.imread(str(asset_paths.tacto_background))
    if background is None:
        raise FileNotFoundError(f"Missing or unreadable TACTO background: {asset_paths.tacto_background}")
    if not asset_paths.tacto_config.exists():
        raise FileNotFoundError(f"Missing TACTO config: {asset_paths.tacto_config}")
    tacto_sensor = tacto.Sensor(
        width=int(cfg.get("tacto_width", 240)),
        height=int(cfg.get("tacto_height", 320)),
        background=background,
        config_path=str(asset_paths.tacto_config),
        visualize_gui=bool(cfg.get("visualize_tacto_gui", False)),
        show_depth=False,
        cid=client_id,
    )
    tacto_sensor.add_camera(hand.id, hand.gsmini_joint_ids)
    return tacto_sensor


def attach_object_to_tacto_sensor(tacto_sensor, object_body) -> None:
    tacto_sensor.add_body(object_body)


def remove_object_from_tacto_sensor(tacto_sensor, object_body_id: int) -> None:
    object_prefix = f"{int(object_body_id)}_"
    object_names = [name for name in list(tacto_sensor.objects.keys()) if name.startswith(object_prefix)]
    if not object_names:
        return

    removed_node_ids: set[int] = set()
    for obj_name in object_names:
        tacto_sensor.objects.pop(obj_name, None)
        tacto_sensor.object_poses.pop(obj_name, None)

        current_node = tacto_sensor.renderer.current_object_nodes.pop(obj_name, None)
        if current_node is not None and id(current_node) not in removed_node_ids:
            tacto_sensor.renderer.scene.remove_node(current_node)
            removed_node_ids.add(id(current_node))

        object_node = tacto_sensor.renderer.object_nodes.pop(obj_name, None)
        if object_node is not None and id(object_node) not in removed_node_ids:
            tacto_sensor.renderer.scene.remove_node(object_node)
            removed_node_ids.add(id(object_node))

    for normal_forces in tacto_sensor.normal_forces.values():
        for obj_name in object_names:
            normal_forces.pop(obj_name, None)


def remove_object_body(object_body_id: int, client_id: int) -> None:
    pb.removeBody(int(object_body_id), physicsClientId=int(client_id))


def set_object_body_collision_enabled(object_body_id: int, client_id: int, enabled: bool) -> None:
    collision_filter_group = 1 if enabled else 0
    collision_filter_mask = -1 if enabled else 0
    num_joints = pb.getNumJoints(int(object_body_id), physicsClientId=int(client_id))
    for link_index in range(-1, num_joints):
        pb.setCollisionFilterGroupMask(
            int(object_body_id),
            int(link_index),
            collision_filter_group,
            collision_filter_mask,
            physicsClientId=int(client_id),
        )
