import copy
import pybullet as p
import numpy as np
import functools
from gym import spaces
import pybulletX as px  # 假设使用了 pybulletX 库
from pybulletX.utils.space_dict import SpaceDict

class GSminiPandaHand(px.Robot):  # 继承 px.Robot
    GRIPPER_MAX_FORCE = 70  # 定义最大夹爪力
    gripper_joint_names = ["panda_finger_joint_left", "panda_finger_joint_right"]
    gsmini_joint_names = ["finger_gsmini_joint_left", "finger_gsmini_joint_right"]
    gel_link_names = ["gsmini_gel_left", "gsmini_gel_right"]

    def __init__(self, robot_params):
        super().__init__(**robot_params)  # 调用父类的构造函数
        self.reset()
        # self.hold()
    
    def hold(self):
        p.setJointMotorControlArray(
            self.id,
            list(range(self.num_joints)),
            p.POSITION_CONTROL,
            targetPositions=[0] * self.num_joints,
            forces=[1000] * self.num_joints,
            **self._client_kwargs,
        )

    def joint_pos(self):
        """ current joint positions of this robot """
        joints = list(range(self.num_joints))
        joint_states = p.getJointStates(self.id, joints, **self._client_kwargs)
        pos = [joint_states[joint][0] for joint in joints]
        for i in range(self.num_joints):
            print(f"joint {i} pos: {pos[i]}")
        return np.array(pos)
    
    def get_states(self):
        joint_states = self.get_joint_states()
        states = self.state_space.new()
        states.dummy_x = joint_states[0][0]
        states.dummy_y = joint_states[1][0]
        states.dummy_z = joint_states[2][0]
        states.gripper_width = 2 * np.abs(joint_states[3][0])
        return states

    def finger_position_control(self, gripper_width, max_force=GRIPPER_MAX_FORCE):
        # 控制手指的开合，gripper_width 表示夹爪的目标开合度
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
        
    def finger_effort_control(self, gripper_effort):
        if gripper_effort < 0:
            gripper_effort = 0  # 如果力小于 0，设置为 0
        elif gripper_effort > self.GRIPPER_MAX_FORCE:
            gripper_effort = self.GRIPPER_MAX_FORCE  # 如果力大于最大力，设置为最大力
        p.setJointMotorControl2(
            self.id,
            self.gripper_joint_ids[0],
            p.TORQUE_CONTROL,
            force=gripper_effort,
            **self._client_kwargs,
        )
        p.setJointMotorControl2(
            self.id,
            self.gripper_joint_ids[1],
            p.TORQUE_CONTROL,
            force=gripper_effort,
            **self._client_kwargs,
        )

    def set_actions(self, actions:dict): # actions from action_space
        # 设置夹爪的动作和力
        gripper_width = actions.get("gripper_width", 0.0)
        dummy_x = actions.get("dummy_x", 0.0)
        dummy_y = actions.get("dummy_y", 0.0)
        dummy_z = actions.get("dummy_z", 0.0)
            
        for joint_id in range(self.num_joints):
            if joint_id in self.gripper_joint_ids:
                p.setJointMotorControl2(
                    self.id,
                    joint_id,
                    p.POSITION_CONTROL,
                    targetPosition=gripper_width / 2,
                    force=self.GRIPPER_MAX_FORCE,
                    **self._client_kwargs,
                )
            elif joint_id == 0:
                p.setJointMotorControl2(
                    self.id,
                    joint_id,
                    p.POSITION_CONTROL,
                    targetPosition=dummy_x,
                    force=1000,
                    **self._client_kwargs,
                )
            elif joint_id == 1:
                p.setJointMotorControl2(
                    self.id,
                    joint_id,
                    p.POSITION_CONTROL,
                    targetPosition=dummy_y,
                    force=1000,
                    **self._client_kwargs,
                )
            elif joint_id == 2:
                p.setJointMotorControl2(
                    self.id,
                    joint_id,
                    p.POSITION_CONTROL,
                    targetPosition=dummy_z,
                    force=1000,
                    **self._client_kwargs,
                )
            else:
                continue

    def grasp(self, width, grip_force=20):
        # 抓取物体，width是目标宽度，grip_force是施加的夹爪力
        self.set_actions(width, grip_force)
        
    @property
    def gripper_joint_ids(self):
        return [self.get_joint_index_by_name(name) for name in self.gripper_joint_names]
    
    @property
    def gsmini_joint_ids(self):
        return [self.get_joint_index_by_name(name) for name in self.gsmini_joint_names]
    
    @property
    def gsmini_gel_ids(self):
        return [self.get_joint_index_by_name(name) for name in self.gel_link_names]
      
    @property
    @functools.lru_cache(maxsize=None)
    def state_space(self):
        return SpaceDict(
            {
                "dummy_x": spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
                "dummy_y": spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
                "dummy_z": spaces.Box(low=0, high=0.5, shape=(1,), dtype=np.float32),
                "gripper_width": spaces.Box(low=0, high=0.08, shape=(1,), dtype=np.float32)
            }
        )
    @property
    @functools.lru_cache(maxsize=None)
    def action_space(self):
        action_space = copy.deepcopy(self.state_space)
        action_space["gripper_force"] = spaces.Box(
            low=0, high=self.GRIPPER_MAX_FORCE, shape=(1,)
        )
        return action_space
