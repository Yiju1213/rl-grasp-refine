from __future__ import annotations

from copy import deepcopy
import time

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
from src.structures.action import GraspPose, NormalizedAction
from src.structures.info import StepInfo
from src.structures.observation import Observation
from src.structures.observation import RawSensorObservation


def make_env_cfg(seed: int = 7) -> dict:
    return {
        "seed": seed,
        "action": {
            "translation_bound": [0.01, 0.01, 0.01],
            "rotation_bound": [0.1, 0.1, 0.1],
        },
        "reward": {
            "stability_kappa": 1.0,
            "contact_lambda_cover": 0.1,
            "contact_lambda_edge": 0.1,
            "contact_threshold_cover": 0.2,
            "contact_threshold_edge": 0.2,
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
        "contact_semantics": {"tactile_threshold": 0.2},
    }


def make_calibration_cfg() -> dict:
    return {
        "init_a": 1.0,
        "init_b": 0.0,
        "lambda": 1.0,
    }


def make_rl_cfg() -> dict:
    return {
        "device": "cpu",
        "worker_policy_device": "cpu",
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
        "max_collect_attempt_factor": 10,
        "num_envs": 1,
    }


def make_actor_critic_cfg() -> dict:
    return {
        "policy_hidden_dims": [64, 64],
        "value_hidden_dims": [64, 64],
        "initial_log_std": -0.5,
        "policy_observation": {"preset": "current"},
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


def build_test_env_for_worker(
    env_cfg: dict,
    perception_cfg: dict,
    calibration_cfg: dict,
    worker_id: int | None = None,
    num_workers: int | None = None,
    worker_seed: int | None = None,
):
    del env_cfg, perception_cfg, calibration_cfg, worker_id, num_workers
    env, _, _, _, _ = build_test_env(seed=int(worker_seed or 7))
    return env


class AsyncDelayEnv:
    def __init__(
        self,
        worker_id: int,
        calibration_cfg: dict,
        delay_schedules: dict[int, list[float]] | None = None,
        invalid_attempts: dict[int, int] | None = None,
    ):
        self.worker_id = int(worker_id)
        self.calibrator = OnlineLogitCalibrator(calibration_cfg)
        self.delay_schedules = delay_schedules or {}
        self.invalid_attempts = dict(invalid_attempts or {})
        self.episode_index = 0

    def reset(self):
        base = 0.1 + 0.05 * self.worker_id + 0.01 * self.episode_index
        position = np.asarray([base, 0.0, 0.0], dtype=np.float32)
        return Observation(
            latent_feature=np.full(32, fill_value=base, dtype=np.float32),
            contact_semantic=np.asarray([base, base / 2.0], dtype=np.float32),
            grasp_pose=GraspPose(position=position, rotation=np.zeros(3, dtype=np.float32)),
            raw_stability_logit=float(base),
        )

    def step(self, action: NormalizedAction):
        if not isinstance(action, NormalizedAction):
            action = NormalizedAction(value=np.asarray(action, dtype=np.float32))
        delays = self.delay_schedules.get(self.worker_id, [])
        delay = delays[self.episode_index] if self.episode_index < len(delays) else 0.0
        if delay > 0.0:
            time.sleep(delay)

        invalid_budget = int(self.invalid_attempts.get(self.worker_id, 0))
        is_valid = self.episode_index >= invalid_budget
        logit_before = 0.1 + 0.05 * self.worker_id + 0.01 * self.episode_index
        logit_after = logit_before + 0.05
        next_obs = Observation(
            latent_feature=np.full(32, fill_value=logit_after, dtype=np.float32),
            contact_semantic=np.asarray([logit_after, logit_after / 2.0], dtype=np.float32),
            grasp_pose=GraspPose(position=np.asarray([logit_after, 0.0, 0.0]), rotation=np.zeros(3, dtype=np.float32)),
            raw_stability_logit=float(logit_after),
        )
        info = StepInfo(
            drop_success=1 if is_valid else 0,
            calibrated_stability_before=0.5,
            calibrated_stability_after=0.6,
            posterior_trace=2.0,
            reward_drop=1.0 if is_valid else -1.0,
            reward_stability=0.1,
            reward_contact=0.0,
            extra={
                "reward_breakdown": None,
                "raw_logit_before": logit_before,
                "raw_logit_after": logit_after,
                "legacy_drop_success_before": float(self.worker_id % 2),
                "source_object_id": self.worker_id,
                "source_global_id": self.episode_index,
                "trial_metadata": {
                    "valid_for_learning": bool(is_valid),
                    "worker_id": self.worker_id,
                    "episode_index": self.episode_index,
                    "trial_status": "success" if is_valid else "system_invalid_observation",
                    "failure_reason": None if is_valid else "synthetic_invalid_attempt",
                },
            },
        )
        self.episode_index += 1
        return next_obs, float(1.0 if is_valid else -1.0), True, info

    def sync_calibrator(self, state: dict) -> None:
        self.calibrator.load_state(state)

    def close(self) -> None:
        return None


def build_async_delay_env_for_worker(
    env_cfg: dict,
    perception_cfg: dict,
    calibration_cfg: dict,
    worker_id: int | None = None,
    num_workers: int | None = None,
    worker_seed: int | None = None,
):
    del perception_cfg, num_workers, worker_seed
    return AsyncDelayEnv(
        worker_id=int(worker_id or 0),
        calibration_cfg=calibration_cfg,
        delay_schedules=deepcopy(env_cfg.get("delay_schedules", {})),
        invalid_attempts=deepcopy(env_cfg.get("invalid_attempts", {})),
    )


def build_test_actor_critic(obs_dim: int):
    actor_critic_cfg = make_actor_critic_cfg()
    policy_net = PolicyNetwork(obs_dim=obs_dim, action_dim=6, cfg=actor_critic_cfg)
    value_net = ValueNetwork(obs_dim=obs_dim, cfg=actor_critic_cfg)
    return ActorCritic(policy_net=policy_net, value_net=value_net), actor_critic_cfg


class DummyLogger:
    def __init__(self):
        self.records = []
        self.episode_records = []

    def log_scalar(self, name: str, value: float, step: int):
        self.records.append((step, {name: value}))

    def log_dict(self, stats: dict, step: int):
        self.records.append((step, stats))

    def log_episode_samples(self, samples: list[dict], step: int):
        self.episode_records.append((step, samples))

    def info(self, msg: str):
        self.records.append(("info", msg))


def cpu_device():
    return torch.device("cpu")
