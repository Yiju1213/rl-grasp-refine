from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch

from src.calibration.online_logit_calibrator import OnlineLogitCalibrator
from src.envs.action_executor import ActionExecutor
from src.envs.grasp_refine_env import GraspRefineEnv
from src.envs.observation_builder import ObservationBuilder
from src.envs.reward_manager import RewardManager
from src.envs.termination import SingleStepTermination
from src.models.rl.actor_critic import ActorCritic
from src.models.rl.policy_network import PolicyNetwork
from src.models.rl.value_network import ValueNetwork
from src.perception.factory import build_perception_stack
from src.structures.action import GraspPose
from src.structures.observation import RawSensorObservation


def make_env_cfg(seed: int = 7) -> dict:
    return {
        "seed": seed,
        "action": {
            "translation_bound": [0.01, 0.01, 0.01],
            "rotation_bound": [0.1, 0.1, 0.1],
        },
        "reward": {
            "weights": {"drop": 1.0, "stability": 0.5, "contact": 0.1},
            "stability_alpha": 0.1,
            "stability_clip": [-1.0, 1.0],
            "contact_beta": 0.5,
            "contact_clip": [-0.25, 0.25],
            "drop_success_reward": 1.0,
            "drop_failure_reward": -1.0,
        },
        "sampling": {
            "position_noise": [0.01, 0.01, 0.01],
            "rotation_noise": [0.1, 0.1, 0.1],
        },
        "default_sample_cfg": {
            "object_name": "fake_cube",
            "object_pose": {
                "position": [0.0, 0.0, 0.04],
                "rotation": [0.0, 0.0, 0.0],
            },
            "target_grasp_pose": {
                "position": [0.0, 0.0, 0.08],
                "rotation": [0.0, 0.0, 0.0],
            },
            "trial": {
                "max_position_error": 0.02,
                "max_rotation_error": 0.25,
            },
        },
    }


def make_perception_cfg() -> dict:
    return {
        "adapter_type": "dgcnn",
        "feature_extractor": {"freeze": True},
        "backbone": {"type": "dgcnn", "latent_dim": 32, "hidden_dim": 64},
        "predictor": {"type": "stability_head", "hidden_dim": 64},
        "contact_semantics": {"tactile_threshold": 0.2, "edge_scale": 0.05},
    }


def make_calibration_cfg() -> dict:
    return {
        "init_a": 1.0,
        "init_b": 0.0,
        "learning_rate": 0.05,
        "l2_reg": 0.001,
        "prior_var": 1.0,
        "uncertainty_base": 0.05,
    }


def make_rl_cfg() -> dict:
    return {
        "device": "cpu",
        "batch_episodes": 4,
        "gamma": 0.99,
        "lam": 0.95,
        "learning_rate": 3e-4,
        "clip_range": 0.2,
        "value_loss_coef": 0.5,
        "entropy_coef": 0.01,
        "update_epochs": 2,
        "minibatch_size": 2,
        "max_grad_norm": 0.5,
        "normalize_advantages": True,
        "num_envs": 1,
    }


def make_actor_critic_cfg() -> dict:
    return {
        "policy_hidden_dims": [64, 64],
        "value_hidden_dims": [64, 64],
        "initial_log_std": -0.5,
    }


class FakeScene:
    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or {}
        self.sample_cfg = None
        self.current_pose = None
        self.target_pose = None

    def reset_scene(self, sample_cfg: dict) -> None:
        self.sample_cfg = deepcopy(sample_cfg)
        target = sample_cfg["target_grasp_pose"]
        self.target_pose = GraspPose(position=target["position"], rotation=target["rotation"])
        self.current_pose = None

    def set_initial_grasp(self, grasp_pose) -> None:
        self.current_pose = grasp_pose

    def get_raw_observation(self):
        position_error = float(np.linalg.norm(self.current_pose.position - self.target_pose.position))
        rotation_error = float(np.linalg.norm(self.current_pose.rotation - self.target_pose.rotation))
        confidence = float(np.exp(-(position_error * 25.0 + rotation_error * 2.0)))
        tactile = np.asarray([confidence, confidence * 0.8, confidence * 0.6, max(confidence - 0.1, 0.0)])
        point_cloud = self.current_pose.position[None, :] + 0.01 * np.asarray(
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
        metadata = {
            "grasp_pose": self.current_pose,
            "target_grasp_pose": self.target_pose,
            "distance_to_edge": float(np.clip(np.linalg.norm(self.current_pose.position[:2]), 0.0, 1.0)),
        }
        return RawSensorObservation(
            visual_data={"point_cloud": point_cloud, "distance_to_edge": metadata["distance_to_edge"]},
            tactile_data={"contact_map": tactile},
            grasp_metadata=metadata,
        )

    def apply_refinement(self, refined_pose) -> None:
        self.current_pose = refined_pose

    def run_grasp_trial(self) -> dict:
        position_error = float(np.linalg.norm(self.current_pose.position - self.target_pose.position))
        rotation_error = float(np.linalg.norm(self.current_pose.rotation - self.target_pose.rotation))
        trial = self.sample_cfg["trial"]
        drop_success = int(
            position_error <= float(trial["max_position_error"])
            and rotation_error <= float(trial["max_rotation_error"])
        )
        return {
            "drop_success": drop_success,
            "trial_metadata": {
                "position_error": position_error,
                "rotation_error": rotation_error,
            },
        }

    def close(self) -> None:
        return None


def build_test_env(seed: int = 7):
    env_cfg = make_env_cfg(seed)
    perception_cfg = make_perception_cfg()
    calibration_cfg = make_calibration_cfg()
    feature_extractor, contact_semantics_extractor, stability_predictor = build_perception_stack(perception_cfg)
    calibrator = OnlineLogitCalibrator(calibration_cfg)
    env = GraspRefineEnv(
        cfg=env_cfg,
        scene=FakeScene(),
        action_executor=ActionExecutor(env_cfg["action"]),
        observation_builder=ObservationBuilder(
            feature_extractor=feature_extractor,
            contact_semantics_extractor=contact_semantics_extractor,
            stability_predictor=stability_predictor,
        ),
        reward_manager=RewardManager(env_cfg["reward"]),
        calibrator=calibrator,
        termination=SingleStepTermination(env_cfg),
    )
    return env, calibrator, env_cfg, perception_cfg, calibration_cfg


def build_test_actor_critic(obs_dim: int):
    actor_critic_cfg = make_actor_critic_cfg()
    policy_net = PolicyNetwork(obs_dim=obs_dim, action_dim=6, cfg=actor_critic_cfg)
    value_net = ValueNetwork(obs_dim=obs_dim, cfg=actor_critic_cfg)
    return ActorCritic(policy_net=policy_net, value_net=value_net), actor_critic_cfg


class DummyLogger:
    def __init__(self):
        self.records = []

    def log_scalar(self, name: str, value: float, step: int):
        self.records.append((step, {name: value}))

    def log_dict(self, stats: dict, step: int):
        self.records.append((step, stats))

    def info(self, msg: str):
        self.records.append(("info", msg))


def cpu_device():
    return torch.device("cpu")
