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


def resolve_path(path_str: str | Path, *, base_dir: str | Path | None = None) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path.resolve()
    base = ROOT if base_dir is None else Path(base_dir).resolve()
    return (base / path).resolve()


def _find_enclosing_configs_root(path: Path) -> Path | None:
    current = path if path.is_dir() else path.parent
    for candidate in (current, *current.parents):
        if candidate.name == "configs":
            return candidate
    return None


def _resolve_config_reference(path_str: str | Path, *, experiment_path: Path, bundle_base: Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path.resolve()
    if path.parts and path.parts[0] == "configs":
        return resolve_path(path, base_dir=bundle_base)
    return resolve_path(path, base_dir=experiment_path.parent)


def load_experiment_bundle(experiment_path: str | Path) -> tuple[dict, dict]:
    experiment_path_resolved = resolve_path(experiment_path)
    experiment_cfg = load_config(experiment_path_resolved)
    configs_root = _find_enclosing_configs_root(experiment_path_resolved)
    bundle_base = configs_root.parent if configs_root is not None else ROOT
    bundle = {}
    for key, relative_path in experiment_cfg.get("configs", {}).items():
        bundle[key] = load_config(
            _resolve_config_reference(relative_path, experiment_path=experiment_path_resolved, bundle_base=bundle_base)
        )
    return apply_experiment_overrides(experiment_cfg, bundle)


def resolve_experiment_source_paths(experiment_path: str | Path) -> dict[str, Path]:
    experiment_path_resolved = resolve_path(experiment_path)
    experiment_cfg = load_config(experiment_path_resolved)
    configs_root = _find_enclosing_configs_root(experiment_path_resolved)
    bundle_base = configs_root.parent if configs_root is not None else ROOT
    source_paths: dict[str, Path] = {"experiment": experiment_path_resolved}
    for key, relative_path in experiment_cfg.get("configs", {}).items():
        source_paths[key] = _resolve_config_reference(
            relative_path,
            experiment_path=experiment_path_resolved,
            bundle_base=bundle_base,
        )
    return source_paths


def snapshot_experiment_configs(experiment_path: str | Path, snapshot_dir: str | Path) -> list[Path]:
    snapshot_root = resolve_path(snapshot_dir)
    snapshot_root.mkdir(parents=True, exist_ok=True)
    repo_configs_root = resolve_path("configs")
    experiment_path_resolved = resolve_path(experiment_path)
    source_configs_root = _find_enclosing_configs_root(experiment_path_resolved)
    copied_paths: list[Path] = []

    for source_path in resolve_experiment_source_paths(experiment_path).values():
        if source_configs_root is not None:
            try:
                relative_path = source_path.relative_to(source_configs_root)
                destination_path = snapshot_root / relative_path
                destination_path.parent.mkdir(parents=True, exist_ok=True)
                if source_path.resolve() != destination_path.resolve():
                    shutil.copy2(source_path, destination_path)
                copied_paths.append(destination_path)
                continue
            except ValueError:
                pass
        try:
            relative_path = source_path.relative_to(repo_configs_root)
        except ValueError:
            relative_path = Path(source_path.name)
        destination_path = snapshot_root / relative_path
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        if source_path.resolve() != destination_path.resolve():
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
