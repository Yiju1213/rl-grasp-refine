from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation as R


DEFAULT_TACTILE_CAMERA_TO_GEL_M = 0.0235
DEFAULT_GEL_MAX_DEPTH_M = 0.00425


@dataclass(frozen=True)
class Pose:
    position: np.ndarray
    quaternion: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "position", np.asarray(self.position, dtype=np.float32).reshape(3))
        object.__setattr__(self, "quaternion", np.asarray(self.quaternion, dtype=np.float32).reshape(4))


def quaternion_to_rotvec(quaternion) -> np.ndarray:
    return R.from_quat(np.asarray(quaternion, dtype=np.float32).reshape(4)).as_rotvec().astype(np.float32)


def rotvec_to_quaternion(rotvec) -> np.ndarray:
    return R.from_rotvec(np.asarray(rotvec, dtype=np.float32).reshape(3)).as_quat().astype(np.float32)


def pose_to_matrix(position, quaternion) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = R.from_quat(np.asarray(quaternion, dtype=np.float32).reshape(4)).as_matrix().astype(np.float32)
    matrix[:3, 3] = np.asarray(position, dtype=np.float32).reshape(3)
    return matrix


def matrix_to_pose(matrix: np.ndarray) -> Pose:
    matrix = np.asarray(matrix, dtype=np.float32).reshape(4, 4)
    quaternion = R.from_matrix(matrix[:3, :3]).as_quat().astype(np.float32)
    position = matrix[:3, 3].astype(np.float32)
    return Pose(position=position, quaternion=quaternion)


def projection_matrix_to_intrinsics(proj_matrix: np.ndarray, width: int, height: int) -> tuple[float, float, float, float]:
    proj_matrix = np.asarray(proj_matrix, dtype=np.float32).reshape(4, 4)
    fx = float(proj_matrix[0, 0] * width / 2.0)
    fy = float(proj_matrix[1, 1] * height / 2.0)
    cx = float(width / 2.0)
    cy = float(height / 2.0)
    return fx, fy, cx, cy


def depth_to_camera_points(
    depth_m: np.ndarray,
    proj_matrix: np.ndarray,
    mask: np.ndarray | None = None,
    max_points: int | None = None,
) -> np.ndarray:
    depth_m = np.asarray(depth_m, dtype=np.float32)
    if depth_m.ndim != 2:
        raise ValueError(f"depth_to_camera_points expects HxW depth, got shape {depth_m.shape}")

    valid = np.isfinite(depth_m) & (depth_m > 0.0)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float32)

    height, width = depth_m.shape
    fx, fy, cx, cy = projection_matrix_to_intrinsics(proj_matrix, width=width, height=height)

    v_coords, u_coords = np.nonzero(valid)
    z = depth_m[v_coords, u_coords]
    x = (u_coords.astype(np.float32) - cx) * z / max(fx, 1e-6)
    y = (v_coords.astype(np.float32) - cy) * z / max(fy, 1e-6)

    camera_points = np.stack([x, -y, -z], axis=1).astype(np.float32)
    if max_points is not None and camera_points.shape[0] > max_points:
        indices = np.linspace(0, camera_points.shape[0] - 1, max_points, dtype=np.int32)
        camera_points = camera_points[indices]
    return camera_points


def camera_points_to_world(camera_points: np.ndarray, view_matrix: np.ndarray) -> np.ndarray:
    camera_points = np.asarray(camera_points, dtype=np.float32).reshape(-1, 3)
    if camera_points.size == 0:
        return np.zeros((0, 3), dtype=np.float32)

    view_matrix = np.asarray(view_matrix, dtype=np.float32).reshape(4, 4)
    camera_to_world = np.linalg.inv(view_matrix).astype(np.float32)
    hom = np.concatenate([camera_points, np.ones((camera_points.shape[0], 1), dtype=np.float32)], axis=1)
    world = hom @ camera_to_world.T
    return world[:, :3].astype(np.float32)


def depth_to_world_points(
    depth_m: np.ndarray,
    proj_matrix: np.ndarray,
    view_matrix: np.ndarray,
    mask: np.ndarray | None = None,
    max_points: int | None = None,
) -> np.ndarray:
    camera_points = depth_to_camera_points(depth_m=depth_m, proj_matrix=proj_matrix, mask=mask, max_points=max_points)
    return camera_points_to_world(camera_points, view_matrix=view_matrix)


def tactile_depth_to_gel_points(
    depth_m: np.ndarray,
    proj_matrix: np.ndarray,
    camera_distance_to_gel_m: float = DEFAULT_TACTILE_CAMERA_TO_GEL_M,
    remove_surface: bool = True,
    max_points: int | None = None,
) -> np.ndarray:
    depth_m = np.asarray(depth_m, dtype=np.float32)
    if depth_m.ndim != 2:
        raise ValueError(f"tactile_depth_to_gel_points expects HxW depth, got shape {depth_m.shape}")

    restored_depth = np.full_like(depth_m, float(camera_distance_to_gel_m), dtype=np.float32) - depth_m
    mask = np.isfinite(restored_depth) & (restored_depth > 0.0)
    if remove_surface:
        mask &= restored_depth < (float(camera_distance_to_gel_m) - 2e-4)

    camera_points = depth_to_camera_points(
        depth_m=restored_depth,
        proj_matrix=proj_matrix,
        mask=mask,
        max_points=max_points,
    )
    if camera_points.size == 0:
        return np.zeros((0, 3), dtype=np.float32)

    transform = np.asarray(
        [
            [0.0, 0.0, -1.0, -float(camera_distance_to_gel_m)],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    hom = np.concatenate([camera_points, np.ones((camera_points.shape[0], 1), dtype=np.float32)], axis=1)
    gel_points = hom @ transform.T
    return gel_points[:, :3].astype(np.float32)


def gel_points_to_world(gel_points: np.ndarray, gel_position, gel_quaternion) -> np.ndarray:
    gel_points = np.asarray(gel_points, dtype=np.float32).reshape(-1, 3)
    if gel_points.size == 0:
        return np.zeros((0, 3), dtype=np.float32)

    gel_to_world = pose_to_matrix(gel_position, gel_quaternion)
    hom = np.concatenate([gel_points, np.ones((gel_points.shape[0], 1), dtype=np.float32)], axis=1)
    world = hom @ gel_to_world.T
    return world[:, :3].astype(np.float32)


def compute_distance_to_edge(segmentation: np.ndarray, object_id: int) -> float:
    segmentation = np.asarray(segmentation)
    mask = segmentation == int(object_id)
    if not np.any(mask):
        return 1.0

    rows, cols = np.nonzero(mask)
    height, width = segmentation.shape[:2]
    border_distance = np.min(
        np.stack(
            [
                rows.astype(np.float32),
                cols.astype(np.float32),
                (height - 1 - rows).astype(np.float32),
                (width - 1 - cols).astype(np.float32),
            ],
            axis=1,
        ),
        axis=1,
    )
    return float(np.min(border_distance) / max(min(height, width), 1))


def normalize_tactile_depth(depth_m: np.ndarray, gel_max_depth_m: float = DEFAULT_GEL_MAX_DEPTH_M) -> np.ndarray:
    depth_m = np.asarray(depth_m, dtype=np.float32)
    if gel_max_depth_m <= 0:
        return np.zeros_like(depth_m, dtype=np.float32)
    return np.clip(depth_m / float(gel_max_depth_m), 0.0, 1.0).astype(np.float32)


def estimate_box_half_extents_from_points(
    world_points: np.ndarray,
    object_position,
    object_quaternion,
    min_half_extent: float = 0.01,
    max_half_extent: float = 0.08,
) -> np.ndarray:
    world_points = np.asarray(world_points, dtype=np.float32).reshape(-1, 3)
    if world_points.size == 0:
        return np.asarray([0.02, 0.02, 0.02], dtype=np.float32)

    object_to_world = pose_to_matrix(object_position, object_quaternion)
    world_to_object = np.linalg.inv(object_to_world).astype(np.float32)
    hom = np.concatenate([world_points, np.ones((world_points.shape[0], 1), dtype=np.float32)], axis=1)
    object_points = hom @ world_to_object.T
    object_points = object_points[:, :3]
    extents = np.max(np.abs(object_points), axis=0)
    extents = np.clip(extents, min_half_extent, max_half_extent)
    return extents.astype(np.float32)
