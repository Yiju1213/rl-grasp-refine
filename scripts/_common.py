from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.calibration.online_logit_calibrator import OnlineLogitCalibrator
from src.envs.action_executor import ActionExecutor
from src.envs.dataset_sample_provider import DatasetSampleProvider
from src.envs.grasp_refine_env import GraspRefineEnv
from src.envs.observation_builder import ObservationBuilder
from src.envs.pybullet_scene import PyBulletScene
from src.envs.reward_manager import RewardManager
from src.envs.termination import SingleStepTermination
from src.models.rl.actor_critic import ActorCritic
from src.models.rl.policy_network import PolicyNetwork
from src.models.rl.value_network import ValueNetwork
from src.perception.factory import build_perception_stack
from src.utils.config import load_config


def resolve_path(path_str: str | Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def load_experiment_bundle(experiment_path: str | Path) -> tuple[dict, dict]:
    experiment_cfg = load_config(resolve_path(experiment_path))
    bundle = {}
    for key, relative_path in experiment_cfg.get("configs", {}).items():
        bundle[key] = load_config(resolve_path(relative_path))
    return experiment_cfg, bundle


def infer_obs_dim(perception_cfg: dict) -> int:
    latent_dim = int(perception_cfg.get("backbone", {}).get("latent_dim", 32))
    return latent_dim + 2 + 6 + 1


def build_env(
    env_cfg: dict,
    perception_cfg: dict,
    calibration_cfg: dict,
    calibrator=None,
):
    feature_extractor, contact_semantics_extractor, stability_predictor = build_perception_stack(perception_cfg)
    calibrator = calibrator or OnlineLogitCalibrator(calibration_cfg)
    scene = PyBulletScene(env_cfg.get("scene", {}))
    sample_provider = None
    dataset_cfg = env_cfg.get("dataset", {})
    if dataset_cfg.get("enabled", False):
        # TODO: DatasetSampleProvider still performs eager metadata indexing at
        # startup. Keep the current behavior for now and optimize only after the
        # scene reset lifecycle changes are validated.
        sample_provider = DatasetSampleProvider(dataset_cfg)
    action_executor = ActionExecutor(env_cfg.get("action", {}))
    #TODO 分析ObservationBuilder是否需要改成工厂模式，或者直接放在GraspRefineEnv里，避免每次构建环境都要构建一遍感知模块
    # 因为感知模块的构建可能比较慢，尤其是特征提取器，如果每次环境重置或是并行环境构建都要重新构建一遍，可能会导致性能问题
    # 但是保持工厂模式又需要考虑如何在多个环境实例之间共享感知模块
    # 因为多个环境难以在同一个时间步给出raw_obs，所以尝试使用batch的inference可能有难度
    # 可以尝试构建batch_size=queue，队列收满就做一次inference，或者直接在每个环境里维护一个感知模块的实例，虽然会有资源浪费，但实现起来可能更简单
    observation_builder = ObservationBuilder(
        feature_extractor=feature_extractor,
        contact_semantics_extractor=contact_semantics_extractor,
        stability_predictor=stability_predictor,
    )
    reward_manager = RewardManager(env_cfg.get("reward", {}))
    termination = SingleStepTermination(env_cfg)
    env = GraspRefineEnv(
        cfg=env_cfg,
        scene=scene,
        action_executor=action_executor,
        observation_builder=observation_builder,
        reward_manager=reward_manager,
        calibrator=calibrator,
        termination=termination,
        sample_provider=sample_provider,
    )
    return env, calibrator


def build_actor_critic(perception_cfg: dict, actor_critic_cfg: dict) -> ActorCritic:
    obs_dim = infer_obs_dim(perception_cfg)
    action_dim = 6
    policy_net = PolicyNetwork(obs_dim=obs_dim, action_dim=action_dim, cfg=actor_critic_cfg)
    value_net = ValueNetwork(obs_dim=obs_dim, cfg=actor_critic_cfg)
    return ActorCritic(policy_net=policy_net, value_net=value_net)


def maybe_load_actor_critic(actor_critic: ActorCritic, checkpoint_path: str | Path | None) -> ActorCritic:
    if checkpoint_path is None:
        return actor_critic
    checkpoint = torch.load(resolve_path(checkpoint_path), map_location="cpu")
    actor_critic.load_state_dict(checkpoint["actor_critic"])
    return actor_critic
