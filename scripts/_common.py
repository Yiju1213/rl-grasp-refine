from __future__ import annotations

import shutil
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.calibration.online_logit_calibrator import OnlineLogitCalibrator
from src.models.rl.actor_critic import ActorCritic
from src.rl.observation_spec import PolicyObservationSpec, resolve_policy_observation_spec
from src.runtime.builders import build_actor_critic as runtime_build_actor_critic
from src.runtime.builders import build_env as runtime_build_env
from src.runtime.experiment_config import apply_experiment_overrides
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
    return apply_experiment_overrides(experiment_cfg, bundle)


def resolve_experiment_source_paths(experiment_path: str | Path) -> dict[str, Path]:
    experiment_path_resolved = resolve_path(experiment_path)
    experiment_cfg = load_config(experiment_path_resolved)
    source_paths: dict[str, Path] = {"experiment": experiment_path_resolved}
    for key, relative_path in experiment_cfg.get("configs", {}).items():
        source_paths[key] = resolve_path(relative_path)
    return source_paths


def snapshot_experiment_configs(experiment_path: str | Path, snapshot_dir: str | Path) -> list[Path]:
    snapshot_root = resolve_path(snapshot_dir)
    snapshot_root.mkdir(parents=True, exist_ok=True)
    repo_configs_root = resolve_path("configs")
    copied_paths: list[Path] = []

    for source_path in resolve_experiment_source_paths(experiment_path).values():
        try:
            relative_path = source_path.relative_to(repo_configs_root)
        except ValueError:
            relative_path = Path(source_path.name)
        destination_path = snapshot_root / relative_path
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination_path)
        copied_paths.append(destination_path)
    return copied_paths


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
    calibrator = calibrator or OnlineLogitCalibrator(calibration_cfg)
    return runtime_build_env(
        env_cfg=env_cfg,
        perception_cfg=perception_cfg,
        calibration_cfg=calibration_cfg,
        calibrator=calibrator,
        worker_id=worker_id,
        num_workers=num_workers,
        worker_seed=worker_seed,
        worker_generation=worker_generation,
    )


def build_actor_critic(
    perception_cfg: dict,
    actor_critic_cfg: dict,
    observation_spec: PolicyObservationSpec | None = None,
) -> ActorCritic:
    return runtime_build_actor_critic(
        perception_cfg=perception_cfg,
        actor_critic_cfg=actor_critic_cfg,
        observation_spec=observation_spec or resolve_policy_observation_spec(perception_cfg, actor_critic_cfg),
    )


def maybe_load_actor_critic(actor_critic: ActorCritic, checkpoint_path: str | Path | None) -> ActorCritic:
    if checkpoint_path is None:
        return actor_critic
    checkpoint = torch.load(resolve_path(checkpoint_path), map_location="cpu")
    actor_critic.load_state_dict(checkpoint["actor_critic"])
    return actor_critic
