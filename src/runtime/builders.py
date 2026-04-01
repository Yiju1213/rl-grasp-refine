from __future__ import annotations

from copy import deepcopy

from src.calibration.online_logit_calibrator import OnlineLogitCalibrator
from src.envs.action_executor import ActionExecutor
from src.envs.dataset_sample_provider import DatasetSampleProvider
from src.envs.grasp_refine_env import GraspRefineEnv
from src.envs.observation_builder import ObservationBuilder
from src.envs.pybullet_scene import PyBulletScene
from src.envs.reward_manager import RewardManager
from src.envs.termination import SingleStepTermination
from src.models.rl.actor_critic import ActorCritic
from src.models.rl.policy_network import (
    LatentFirstLateFusionPolicyNetwork,
    PolicyNetwork,
    resolve_actor_critic_architecture_type,
)
from src.models.rl.value_network import LatentFirstLateFusionValueNetwork, ValueNetwork
from src.perception.factory import build_perception_stack
from src.rl.observation_spec import (
    PolicyObservationSpec,
    infer_obs_dim_from_spec,
    resolve_policy_observation_spec,
)
from src.runtime.render_env import configure_render_environment

def build_env(
    env_cfg: dict,
    perception_cfg: dict,
    calibration_cfg: dict,
    calibrator=None,
    worker_id: int | None = None,
    num_workers: int | None = None,
    worker_seed: int | None = None,
    worker_generation: int | None = None,
):
    env_cfg_local = deepcopy(env_cfg)
    if worker_seed is not None:
        env_cfg_local["seed"] = int(worker_seed)
    configure_render_environment(env_cfg_local.get("scene", {}))
    scene_cfg = deepcopy(env_cfg_local.get("scene", {}))

    feature_extractor, contact_semantics_extractor, stability_predictor = build_perception_stack(perception_cfg)
    calibrator = calibrator or OnlineLogitCalibrator(calibration_cfg)

    def scene_factory() -> PyBulletScene:
        return PyBulletScene(deepcopy(scene_cfg))

    scene = scene_factory()

    sample_provider = None
    dataset_cfg = deepcopy(env_cfg_local.get("dataset", {}))
    if dataset_cfg.get("enabled", False):
        if worker_id is not None:
            dataset_cfg["worker_id"] = int(worker_id)
        if num_workers is not None:
            dataset_cfg["num_workers"] = int(num_workers)
        if worker_generation is not None:
            dataset_cfg["worker_generation"] = int(worker_generation)
        sample_provider = DatasetSampleProvider(dataset_cfg)

    action_executor = ActionExecutor(env_cfg_local.get("action", {}))
    observation_builder = ObservationBuilder(
        feature_extractor=feature_extractor,
        contact_semantics_extractor=contact_semantics_extractor,
        stability_predictor=stability_predictor,
    )
    reward_manager = RewardManager(env_cfg_local.get("reward", {}))
    termination = SingleStepTermination(env_cfg_local)
    env = GraspRefineEnv(
        cfg=env_cfg_local,
        scene=scene,
        scene_factory=scene_factory,
        action_executor=action_executor,
        observation_builder=observation_builder,
        reward_manager=reward_manager,
        calibrator=calibrator,
        termination=termination,
        sample_provider=sample_provider,
    )
    return env, calibrator


def build_actor_critic(
    perception_cfg: dict,
    actor_critic_cfg: dict,
    observation_spec: PolicyObservationSpec | None = None,
) -> ActorCritic:
    spec = observation_spec or resolve_policy_observation_spec(perception_cfg, actor_critic_cfg)
    obs_dim = infer_obs_dim_from_spec(spec)
    action_dim = 6
    architecture_type = resolve_actor_critic_architecture_type(actor_critic_cfg)
    if architecture_type == "plain":
        policy_net = PolicyNetwork(obs_dim=obs_dim, action_dim=action_dim, cfg=actor_critic_cfg)
        value_net = ValueNetwork(obs_dim=obs_dim, cfg=actor_critic_cfg)
    elif architecture_type == "latent_first_late_fusion":
        if "latent_feature" not in spec.components:
            raise ValueError("latent_first_late_fusion requires policy_observation to include latent_feature.")
        latent_dim = int(spec.latent_dim)
        aux_dim = int(obs_dim - latent_dim)
        policy_net = LatentFirstLateFusionPolicyNetwork(
            latent_dim=latent_dim,
            aux_dim=aux_dim,
            action_dim=action_dim,
            cfg=actor_critic_cfg,
        )
        value_net = LatentFirstLateFusionValueNetwork(
            latent_dim=latent_dim,
            aux_dim=aux_dim,
            cfg=actor_critic_cfg,
        )
    else:
        raise ValueError(
            f"Unknown actor_critic architecture.type '{architecture_type}'. Expected 'plain' or "
            f"'latent_first_late_fusion'."
        )
    actor_critic = ActorCritic(policy_net=policy_net, value_net=value_net)
    actor_critic.observation_spec = spec
    actor_critic.architecture_type = architecture_type
    return actor_critic
