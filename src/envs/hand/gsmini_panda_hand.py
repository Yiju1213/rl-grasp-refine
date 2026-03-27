from __future__ import annotations

import pybullet as p
import pybulletX as px


class GSminiPandaHand(px.Robot):
    GRIPPER_MAX_FORCE = 70
    gripper_joint_names = ["panda_finger_joint_left", "panda_finger_joint_right"]
    gsmini_joint_names = ["finger_gsmini_joint_left", "finger_gsmini_joint_right"]
    gel_link_names = ["gsmini_gel_left", "gsmini_gel_right"]

    def __init__(self, robot_params):
        super().__init__(**robot_params)
        self.reset()

    def finger_position_control(self, gripper_width, max_force=GRIPPER_MAX_FORCE):
        half_width = gripper_width / 2
        p.setJointMotorControl2(
            self.id,
            self.gripper_joint_ids[0],
            p.POSITION_CONTROL,
            targetPosition=half_width,
            force=max_force,
            **self._client_kwargs,
        )
        p.setJointMotorControl2(
            self.id,
            self.gripper_joint_ids[1],
            p.POSITION_CONTROL,
            targetPosition=half_width,
            force=max_force,
            **self._client_kwargs,
        )

    @property
    def gripper_joint_ids(self):
        return [self.get_joint_index_by_name(name) for name in self.gripper_joint_names]

    @property
    def gsmini_joint_ids(self):
        return [self.get_joint_index_by_name(name) for name in self.gsmini_joint_names]

    @property
    def gsmini_gel_ids(self):
        return [self.get_joint_index_by_name(name) for name in self.gel_link_names]
