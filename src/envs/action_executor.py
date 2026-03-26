from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from src.structures.action import GraspPose, NormalizedAction, PhysicalAction

# TODO 名称可以优化，比如叫PoseDeltaDecoder，PoseRefiner等，或者直接放在GraspRefineEnv里作为一个方法，因为它的功能比较简单，单独成文件可能有点过于冗余了
class ActionExecutor:
    """Convert normalized policy outputs into physical pose deltas."""

    def __init__(self, cfg: dict):
        self.translation_bound = np.asarray(cfg.get("translation_bound", [0.01, 0.01, 0.01]), dtype=np.float32)
        self.rotation_bound = np.asarray(cfg.get("rotation_bound", [0.1, 0.1, 0.1]), dtype=np.float32)
        if self.translation_bound.shape != (3,) or self.rotation_bound.shape != (3,):
            raise ValueError("Action bounds must each have shape (3,).")

    def decode(self, action: NormalizedAction) -> PhysicalAction:
        normalized = np.asarray(action.value, dtype=np.float32)
        delta_translation = normalized[:3] * self.translation_bound
        delta_rotation = normalized[3:] * self.rotation_bound
        return PhysicalAction(delta_translation=delta_translation, delta_rotation=delta_rotation)

    def apply_to_pose(self, grasp_pose: GraspPose, action: PhysicalAction) -> GraspPose:
        base_rotation = Rotation.from_rotvec(grasp_pose.rotation)
        delta_rotation = Rotation.from_rotvec(action.delta_rotation)
        refined_rotation = (delta_rotation * base_rotation).as_rotvec().astype(np.float32)
        refined_position = (grasp_pose.position + action.delta_translation).astype(np.float32)
        return GraspPose(position=refined_position, rotation=refined_rotation)
