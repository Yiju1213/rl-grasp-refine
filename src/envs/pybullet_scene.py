from __future__ import annotations

import math
from typing import Any

import numpy as np

from src.structures.action import GraspPose
from src.structures.observation import RawSensorObservation

try:
    import pybullet as pb
except ImportError:  # pragma: no cover - handled at runtime if missing.
    pb = None


class PyBulletScene:
    """Lightweight self-contained scene wrapper with heuristic grasp outcomes."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.client_id: int | None = None
        self.object_id: int | None = None
        self.sample_cfg: dict[str, Any] | None = None
        self.current_grasp_pose: GraspPose | None = None
        self.initial_grasp_pose: GraspPose | None = None
        self.target_grasp_pose: GraspPose | None = None
        self._connect()

    def _connect(self) -> None:
        if pb is None:
            return
        if self.client_id is not None:
            return
        mode = pb.GUI if self.cfg.get("use_gui", False) else pb.DIRECT
        self.client_id = pb.connect(mode)
        gravity = self.cfg.get("gravity", [0.0, 0.0, -9.81])
        pb.setGravity(*gravity, physicsClientId=self.client_id)
        time_step = float(self.cfg.get("time_step", 1.0 / 240.0))
        pb.setTimeStep(time_step, physicsClientId=self.client_id)

    @staticmethod
    def _dict_to_pose(pose_dict: dict) -> GraspPose:
        return GraspPose(position=pose_dict["position"], rotation=pose_dict["rotation"])

    def reset_scene(self, sample_cfg: dict) -> None:
        self.sample_cfg = sample_cfg
        self.target_grasp_pose = self._dict_to_pose(sample_cfg["target_grasp_pose"])
        self.initial_grasp_pose = None
        self.current_grasp_pose = None

        if pb is None or self.client_id is None:
            self.object_id = None
            return

        pb.resetSimulation(physicsClientId=self.client_id)
        gravity = self.cfg.get("gravity", [0.0, 0.0, -9.81])
        pb.setGravity(*gravity, physicsClientId=self.client_id)
        pb.setTimeStep(float(self.cfg.get("time_step", 1.0 / 240.0)), physicsClientId=self.client_id)

        object_pose = sample_cfg.get("object_pose", {})
        position = object_pose.get("position", [0.0, 0.0, 0.04])
        half_extents = sample_cfg.get("object_half_extents", [0.02, 0.02, 0.02])
        collision_shape = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=half_extents, physicsClientId=self.client_id)
        visual_shape = pb.createVisualShape(
            pb.GEOM_BOX,
            halfExtents=half_extents,
            rgbaColor=[0.5, 0.5, 0.7, 1.0],
            physicsClientId=self.client_id,
        )
        self.object_id = pb.createMultiBody(
            baseMass=0.1,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=position,
            physicsClientId=self.client_id,
        )

    def set_initial_grasp(self, grasp_pose) -> None:
        self.initial_grasp_pose = grasp_pose
        self.current_grasp_pose = grasp_pose

    def _compute_errors(self) -> tuple[float, float]:
        if self.current_grasp_pose is None or self.target_grasp_pose is None:
            return math.inf, math.inf
        position_error = float(np.linalg.norm(self.current_grasp_pose.position - self.target_grasp_pose.position))
        rotation_error = float(np.linalg.norm(self.current_grasp_pose.rotation - self.target_grasp_pose.rotation))
        return position_error, rotation_error

    def _make_point_cloud(self) -> np.ndarray:
        if self.current_grasp_pose is None:
            center = np.zeros(3, dtype=np.float32)
        else:
            center = self.current_grasp_pose.position.astype(np.float32)
        offsets = np.asarray(
            [
                [-1, -1, -1],
                [-1, -1, 1],
                [-1, 1, -1],
                [-1, 1, 1],
                [1, -1, -1],
                [1, -1, 1],
                [1, 1, -1],
                [1, 1, 1],
            ],
            dtype=np.float32,
        )
        return center[None, :] + 0.01 * offsets

    def _make_tactile(self, position_error: float, rotation_error: float) -> np.ndarray:
        confidence = float(np.exp(-(position_error * 30.0 + rotation_error * 2.0)))
        return np.asarray(
            [
                confidence,
                confidence * 0.9,
                confidence * 0.8,
                max(confidence - 0.1, 0.0),
            ],
            dtype=np.float32,
        )

    def get_raw_observation(self):
        position_error, rotation_error = self._compute_errors()
        tactile = self._make_tactile(position_error, rotation_error)
        target_position = (
            self.target_grasp_pose.position if self.target_grasp_pose is not None else np.zeros(3, dtype=np.float32)
        )
        current_position = (
            self.current_grasp_pose.position if self.current_grasp_pose is not None else np.zeros(3, dtype=np.float32)
        )
        distance_to_edge = float(np.clip(np.linalg.norm(current_position[:2] - target_position[:2]), 0.0, 1.0))
        metadata = {
            "grasp_pose": self.current_grasp_pose,
            "target_grasp_pose": self.target_grasp_pose,
            "distance_to_edge": distance_to_edge,
            "position_error": position_error,
            "rotation_error": rotation_error,
        }
        visual_data = {
            "point_cloud": self._make_point_cloud(),
            "distance_to_edge": distance_to_edge,
        }
        tactile_data = {
            "contact_map": tactile,
            "contact_force": float(np.mean(tactile)),
        }
        return RawSensorObservation(
            visual_data=visual_data,
            tactile_data=tactile_data,
            grasp_metadata=metadata,
        )

    def apply_refinement(self, refined_pose) -> None:
        self.current_grasp_pose = refined_pose

    def run_grasp_trial(self) -> dict:
        position_error, rotation_error = self._compute_errors()
        trial_cfg = (self.sample_cfg or {}).get("trial", {})
        max_position_error = float(trial_cfg.get("max_position_error", 0.015))
        max_rotation_error = float(trial_cfg.get("max_rotation_error", 0.2))
        drop_success = int(position_error <= max_position_error and rotation_error <= max_rotation_error)
        return {
            "drop_success": drop_success,
            "trial_metadata": {
                "position_error": position_error,
                "rotation_error": rotation_error,
                "max_position_error": max_position_error,
                "max_rotation_error": max_rotation_error,
            },
        }

    def close(self) -> None:
        if pb is not None and self.client_id is not None:
            pb.disconnect(self.client_id)
        self.client_id = None
