from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.runtime.object_splits import resolve_object_split
from src.structures.action import NormalizedAction
from src.utils.seed import set_seed
from src.utils.tensor_utils import observation_to_tensor

DEFAULT_EXPERIMENT = "configs/experiment/exp_debug_stb5x_latefus_128_epi_seed8.yaml"
DEFAULT_CHECKPOINT = (
    "outputs/exp_debug/rwd-grp-b_stb-rwd-5x_128-epi_paper-spec_latefus_seed8/checkpoints/best.pt"
)
DEFAULT_OUTPUT_DIR = "outputs/inference_visualization/full_seed8"
OBJECT_SET_CHOICES = ("train", "val", "test", "holdout", "all")
SETTLE_POLICY_CHOICES = ("before_match_after", "train_exact", "no_after_settle")


def resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate.resolve()
    return (ROOT / candidate).resolve()


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().tolist()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if hasattr(value, "as_dict") and callable(value.as_dict):
        return _json_ready(value.as_dict())
    return value


def validate_checkpoint_path(checkpoint_path: str | Path) -> Path:
    path = resolve_repo_path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    return path


def parse_object_ids(raw_values: list[str] | None) -> list[int] | None:
    if not raw_values:
        return None
    parsed: list[int] = []
    for value in raw_values:
        for item in str(value).split(","):
            item = item.strip()
            if item:
                parsed.append(int(item))
    return sorted(set(parsed))


def resolve_object_ids(experiment_cfg: dict[str, Any], object_set: str, object_ids: list[int] | None) -> list[int]:
    if object_ids is not None:
        if not object_ids:
            raise ValueError("--object-ids was provided but no object ids were parsed.")
        return sorted(set(int(object_id) for object_id in object_ids))

    split = resolve_object_split(experiment_cfg).as_dict()
    if object_set == "train":
        return list(split["train_ids"])
    if object_set == "val":
        return list(split["val_ids"])
    if object_set == "test":
        return list(split["test_ids"])
    if object_set == "holdout":
        return sorted(set(split["val_ids"]) | set(split["test_ids"]))
    if object_set == "all":
        return sorted(set(split["train_ids"]) | set(split["val_ids"]) | set(split["test_ids"]))
    raise ValueError(f"Unknown object set {object_set!r}. Expected one of {OBJECT_SET_CHOICES}.")


def _write_rgb(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR))


def _label_panel(image: np.ndarray, label: str) -> np.ndarray:
    panel = np.asarray(image, dtype=np.uint8).copy()
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 28), (20, 20, 20), thickness=-1)
    cv2.putText(
        panel,
        str(label),
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    return panel


def _resize_exact(image: np.ndarray, *, width: int, height: int) -> np.ndarray:
    return cv2.resize(np.asarray(image, dtype=np.uint8), (int(width), int(height)), interpolation=cv2.INTER_AREA)


def _tactile_pair_rgb(raw_obs) -> np.ndarray:
    tactile_rgb = np.asarray(raw_obs.tactile_data["rgb"], dtype=np.uint8)
    return np.concatenate([tactile_rgb[0], tactile_rgb[1]], axis=1)


def build_inference_panel(raw_before, raw_after, *, panel_width: int = 480) -> np.ndarray:
    visual_height = int(panel_width)
    tactile_height = max(int(round(panel_width * 2.0 / 3.0)), 1)

    def _stage_column(stage: str, raw_obs) -> np.ndarray:
        visual = _resize_exact(raw_obs.visual_data["rgb"], width=panel_width, height=visual_height)
        tactile = _resize_exact(_tactile_pair_rgb(raw_obs), width=panel_width, height=tactile_height)
        return np.concatenate(
            [
                _label_panel(visual, f"{stage} visual RGB"),
                _label_panel(tactile, f"{stage} tactile RGB pair"),
            ],
            axis=0,
        )

    before_col = _stage_column("Before", raw_before)
    after_col = _stage_column("After", raw_after)
    return np.concatenate([before_col, after_col], axis=1)


def _export_stage(stage_dir: Path, raw_obs) -> dict[str, str]:
    stage_dir.mkdir(parents=True, exist_ok=True)
    _write_rgb(stage_dir / "visual_rgb.png", raw_obs.visual_data["rgb"])
    tactile_rgb = np.asarray(raw_obs.tactile_data["rgb"], dtype=np.uint8)
    _write_rgb(stage_dir / "tactile_left_rgb.png", tactile_rgb[0])
    _write_rgb(stage_dir / "tactile_right_rgb.png", tactile_rgb[1])
    return {
        "visual_rgb": str((stage_dir / "visual_rgb.png").resolve()),
        "tactile_left_rgb": str((stage_dir / "tactile_left_rgb.png").resolve()),
        "tactile_right_rgb": str((stage_dir / "tactile_right_rgb.png").resolve()),
    }


def _resolve_policy_device(bundle: dict[str, Any], policy_device: str | None) -> torch.device:
    if policy_device:
        return torch.device(policy_device)
    rl_cfg = bundle.get("rl", {})
    return torch.device(rl_cfg.get("worker_policy_device", rl_cfg.get("device", "cpu")))


def _load_checkpoint(actor_critic, calibrator, checkpoint_path: Path) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    actor_state = checkpoint.get("actor_critic")
    calibrator_state = checkpoint.get("calibrator")
    if actor_state is None:
        raise KeyError(f"Checkpoint {checkpoint_path} does not contain 'actor_critic'.")
    if calibrator_state is None:
        raise KeyError(f"Checkpoint {checkpoint_path} does not contain 'calibrator'.")
    actor_critic.load_state_dict(actor_state)
    load_state = getattr(calibrator, "load_state", None)
    if not callable(load_state):
        raise TypeError("Calibrator does not support load_state().")
    load_state(calibrator_state)
    return checkpoint


def _build_env_for_visualization(
    experiment_path: str | Path,
    *,
    object_ids: list[int],
    sample_seed: int,
):
    from _common import build_env, load_experiment_bundle

    experiment_cfg, bundle = load_experiment_bundle(experiment_path)
    set_seed(int(experiment_cfg.get("seed", 0)))
    env_cfg = deepcopy(bundle["env"])
    env_cfg.setdefault("scene", {})
    env_cfg["scene"]["use_gui"] = False
    env_cfg["scene"]["visualize_tacto_gui"] = False
    env_cfg["scene"]["visualize_debug_windows"] = False
    dataset_cfg = env_cfg.setdefault("dataset", {})
    dataset_cfg["include_object_ids"] = list(object_ids)
    dataset_cfg["fixed_sample_sequence"] = True
    dataset_cfg["fixed_sample_sequence_seed"] = int(sample_seed)
    dataset_cfg["worker_id"] = 0
    dataset_cfg["num_workers"] = 1
    env, calibrator = build_env(env_cfg, bundle["perception"], bundle["calibration"])
    return experiment_cfg, bundle, env, calibrator


def _post_refine_settle_steps(env) -> int:
    sample_runtime = dict((env.sample_cfg or {}).get("runtime", {}))
    runtime_defaults = dict(env.cfg.get("dataset", {}).get("runtime_defaults", {}))
    return int(sample_runtime.get("post_refine_settle_steps", runtime_defaults.get("post_refine_settle_steps", 8)))


def _step_simulation(env, steps: int) -> None:
    import pybullet as pb

    for _ in range(max(int(steps), 0)):
        pb.stepSimulation(physicsClientId=int(env.scene.client_id))


def _recapture_before_observation(env):
    from src.envs.scene_observation import capture_scene_observation

    if env.sample_cfg is None or env.grasp_pose_before is None:
        raise RuntimeError("Environment must be reset before recapturing before observation.")
    raw_before = capture_scene_observation(
        sample_cfg=env.sample_cfg,
        scene_cfg=env.scene.cfg,
        tacto_sensor=env.scene.tacto_sensor,
        hand=env.scene.hand,
        object_body=env.scene.object_body,
        client_id=env.scene.client_id,
        current_grasp_pose=env.grasp_pose_before,
        stage="before",
    )
    env.raw_obs_before = raw_before
    env.scene.before_raw_obs = raw_before
    env.obs_before = env.observation_builder.build(raw_before, env.grasp_pose_before)
    return env.obs_before, raw_before


def _prepare_before_observation(env, settle_policy: str) -> tuple[Any, Any, int]:
    obs_before = env.reset()
    raw_before = env.raw_obs_before
    before_settle_steps = 0
    if settle_policy == "before_match_after":
        before_settle_steps = _post_refine_settle_steps(env)
        _step_simulation(env, before_settle_steps)
        obs_before, raw_before = _recapture_before_observation(env)
    elif settle_policy in {"train_exact", "no_after_settle"}:
        before_settle_steps = 0
    else:
        raise ValueError(f"Unknown settle policy: {settle_policy}")
    return obs_before, raw_before, before_settle_steps


@contextmanager
def _temporary_after_settle_steps(env, settle_policy: str):
    if settle_policy != "no_after_settle":
        yield _post_refine_settle_steps(env)
        return

    runtime_cfg = env.sample_cfg.setdefault("runtime", {})
    had_key = "post_refine_settle_steps" in runtime_cfg
    original_value = runtime_cfg.get("post_refine_settle_steps")
    runtime_cfg["post_refine_settle_steps"] = 0
    try:
        yield 0
    finally:
        if had_key:
            runtime_cfg["post_refine_settle_steps"] = original_value
        else:
            runtime_cfg.pop("post_refine_settle_steps", None)


def _act(actor_critic, obs_before, *, device: torch.device):
    obs_tensor = observation_to_tensor(obs_before, spec=getattr(actor_critic, "observation_spec", None)).to(device)
    with torch.no_grad():
        action_tensor, log_prob, value, entropy = actor_critic.act(obs_tensor, deterministic=True)
    return {
        "action": action_tensor.squeeze(0).detach().cpu().numpy().astype(np.float32),
        "log_prob": float(log_prob.squeeze(0).detach().cpu().item()),
        "value": float(value.squeeze(0).detach().cpu().item()),
        "entropy": float(entropy.squeeze(0).detach().cpu().item()),
    }


def _observation_payload(obs, raw_obs, calibrated_probability: float) -> dict[str, Any]:
    return {
        "raw_stability_logit": float(obs.raw_stability_logit),
        "calibrated_probability": float(calibrated_probability),
        "contact_semantic": np.asarray(obs.contact_semantic, dtype=np.float32).tolist(),
        "grasp_pose": obs.grasp_pose.as_array().tolist(),
        "observation_valid": bool(raw_obs.grasp_metadata.get("observation_valid", True)),
    }


def _camera_payload(raw_before, raw_after) -> dict[str, Any]:
    before_view = np.asarray(raw_before.visual_data["view_matrix"], dtype=np.float32)
    after_view = np.asarray(raw_after.visual_data["view_matrix"], dtype=np.float32)
    before_proj = np.asarray(raw_before.visual_data["proj_matrix"], dtype=np.float32)
    after_proj = np.asarray(raw_after.visual_data["proj_matrix"], dtype=np.float32)
    return {
        "before_visual_view_matrix": before_view.tolist(),
        "after_visual_view_matrix": after_view.tolist(),
        "before_visual_proj_matrix": before_proj.tolist(),
        "after_visual_proj_matrix": after_proj.tolist(),
        "view_matrix_frobenius_diff": float(np.linalg.norm(after_view - before_view)),
        "projection_matrix_frobenius_diff": float(np.linalg.norm(after_proj - before_proj)),
        "note": (
            "The visual camera is hand-relative. Projection stays fixed, while the view matrix follows the current "
            "hand pose rather than a fixed world camera."
        ),
    }


def _round_payload(
    *,
    round_index: int,
    round_dir: Path,
    experiment_path: Path,
    checkpoint_path: Path,
    sample_cfg: dict[str, Any],
    object_set: str,
    selected_object_ids: list[int],
    sample_seed: int,
    settle_policy: str,
    before_settle_steps: int,
    after_settle_steps: int,
    obs_before,
    raw_before,
    obs_after,
    raw_after,
    action_payload: dict[str, Any],
    physical_action,
    reward: float,
    done: bool,
    info,
    elapsed_s: float,
) -> dict[str, Any]:
    source_cfg = dict(sample_cfg.get("source", {}))
    reward_breakdown = info.extra.get("reward_breakdown")
    before_prob = float(info.calibrated_stability_before)
    after_prob = float(info.calibrated_stability_after)
    before_contact = np.asarray(obs_before.contact_semantic, dtype=np.float32)
    after_contact = np.asarray(obs_after.contact_semantic, dtype=np.float32)
    before_logit = float(obs_before.raw_stability_logit)
    after_logit = float(obs_after.raw_stability_logit)
    trial_metadata = dict(info.extra.get("trial_metadata", {}))

    return {
        "round_index": int(round_index),
        "round_dir": str(round_dir.resolve()),
        "elapsed_s": float(elapsed_s),
        "sample": {
            "object_id": int(source_cfg.get("object_id")),
            "global_id": int(source_cfg.get("global_id")),
            "legacy_drop_success": bool(source_cfg.get("legacy_drop_success")),
            "graspnet_score": float(source_cfg.get("graspnet_score", 0.0)),
        },
        "policy": {
            "experiment": str(experiment_path),
            "checkpoint": str(checkpoint_path),
            "deterministic": True,
            "normalized_action": np.asarray(action_payload["action"], dtype=np.float32).tolist(),
            "decoded_translation": np.asarray(physical_action.delta_translation, dtype=np.float32).tolist(),
            "decoded_rotation": np.asarray(physical_action.delta_rotation, dtype=np.float32).tolist(),
            "log_prob": float(action_payload["log_prob"]),
            "value": float(action_payload["value"]),
            "entropy": float(action_payload["entropy"]),
        },
        "before": _observation_payload(obs_before, raw_before, before_prob),
        "after": _observation_payload(obs_after, raw_after, after_prob),
        "delta": {
            "prob_delta": float(after_prob - before_prob),
            "raw_logit_delta": float(after_logit - before_logit),
            "t_cover_delta": float(after_contact[0] - before_contact[0]) if before_contact.size > 0 else None,
            "t_edge_delta": float(after_contact[1] - before_contact[1]) if before_contact.size > 1 else None,
        },
        "outcome": {
            "reward": float(reward),
            "reward_breakdown": _json_ready(reward_breakdown),
            "drop_success": int(info.drop_success),
            "done": bool(done),
            "trial_metadata": _json_ready(trial_metadata),
            "failure_reason": trial_metadata.get("failure_reason"),
        },
        "camera": _camera_payload(raw_before, raw_after),
        "settle": {
            "settle_policy": str(settle_policy),
            "before_settle_steps": int(before_settle_steps),
            "after_settle_steps": int(after_settle_steps),
        },
        "selection": {
            "object_set": str(object_set),
            "object_ids": list(selected_object_ids),
            "sample_seed": int(sample_seed),
        },
        "outputs": {
            "before": _export_stage(round_dir / "before", raw_before),
            "after": _export_stage(round_dir / "after", raw_after),
            "inference_panel": str((round_dir / "inference_panel.png").resolve()),
            "info_json": str((round_dir / "info.json").resolve()),
        },
    }


def run_one_round(
    *,
    env,
    actor_critic,
    policy_device: torch.device,
    round_index: int,
    output_dir: Path,
    experiment_path: Path,
    checkpoint_path: Path,
    object_set: str,
    selected_object_ids: list[int],
    sample_seed: int,
    settle_policy: str,
) -> dict[str, Any]:
    round_start = time.perf_counter()
    round_dir = output_dir / f"round_{round_index:06d}"
    round_dir.mkdir(parents=True, exist_ok=True)

    obs_before, raw_before, before_settle_steps = _prepare_before_observation(env, settle_policy)
    sample_cfg = deepcopy(env.sample_cfg or {})
    action_payload = _act(actor_critic, obs_before, device=policy_device)
    normalized_action = NormalizedAction(action_payload["action"])
    physical_action = env.action_executor.decode(normalized_action)

    with _temporary_after_settle_steps(env, settle_policy) as after_settle_steps:
        obs_after, reward, done, info = env.step(normalized_action)
    raw_after = env.raw_obs_after

    panel = build_inference_panel(raw_before, raw_after)
    cv2.imwrite(str(round_dir / "inference_panel.png"), panel)

    payload = _round_payload(
        round_index=round_index,
        round_dir=round_dir,
        experiment_path=experiment_path,
        checkpoint_path=checkpoint_path,
        sample_cfg=sample_cfg,
        object_set=object_set,
        selected_object_ids=selected_object_ids,
        sample_seed=sample_seed,
        settle_policy=settle_policy,
        before_settle_steps=before_settle_steps,
        after_settle_steps=after_settle_steps,
        obs_before=obs_before,
        raw_before=raw_before,
        obs_after=obs_after,
        raw_after=raw_after,
        action_payload=action_payload,
        physical_action=physical_action,
        reward=float(reward),
        done=bool(done),
        info=info,
        elapsed_s=time.perf_counter() - round_start,
    )
    with (round_dir / "info.json").open("w", encoding="utf-8") as handle:
        json.dump(_json_ready(payload), handle, indent=2, sort_keys=True, ensure_ascii=False)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize one-step Full seed8 inference in the real DIRECT env.")
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--object-set", default="test", choices=OBJECT_SET_CHOICES)
    parser.add_argument("--object-ids", nargs="+", help="Explicit object ids, e.g. --object-ids 75 76 or 75,76.")
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means loop until Ctrl+C.")
    parser.add_argument("--settle-policy", default="before_match_after", choices=SETTLE_POLICY_CHOICES)
    parser.add_argument("--policy-device", default=None, help="Defaults to rl.worker_policy_device/device from config.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from _common import build_actor_critic, load_experiment_bundle

    experiment_path = resolve_repo_path(args.experiment)
    checkpoint_path = validate_checkpoint_path(args.checkpoint)
    output_dir = resolve_repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    experiment_cfg, _ = load_experiment_bundle(experiment_path)
    selected_object_ids = resolve_object_ids(
        experiment_cfg,
        object_set=args.object_set,
        object_ids=parse_object_ids(args.object_ids),
    )
    experiment_cfg, bundle, env, calibrator = _build_env_for_visualization(
        experiment_path,
        object_ids=selected_object_ids,
        sample_seed=int(args.sample_seed),
    )
    actor_critic = build_actor_critic(bundle["perception"], bundle["actor_critic"])
    checkpoint = _load_checkpoint(actor_critic, calibrator, checkpoint_path)
    policy_device = _resolve_policy_device(bundle, args.policy_device)
    actor_critic.to(policy_device)
    actor_critic.eval()

    print(
        json.dumps(
            {
                "experiment": str(experiment_path),
                "checkpoint": str(checkpoint_path),
                "checkpoint_best_iteration": checkpoint.get("best_iteration"),
                "checkpoint_best_metric_name": checkpoint.get("best_metric_name"),
                "checkpoint_best_metric_value": checkpoint.get("best_metric_value"),
                "object_set": args.object_set,
                "object_ids": selected_object_ids,
                "sample_seed": int(args.sample_seed),
                "settle_policy": args.settle_policy,
                "policy_device": str(policy_device),
                "output_dir": str(output_dir),
            },
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
    )

    round_index = 0
    try:
        while int(args.max_samples) <= 0 or round_index < int(args.max_samples):
            payload = run_one_round(
                env=env,
                actor_critic=actor_critic,
                policy_device=policy_device,
                round_index=round_index,
                output_dir=output_dir,
                experiment_path=experiment_path,
                checkpoint_path=checkpoint_path,
                object_set=args.object_set,
                selected_object_ids=selected_object_ids,
                sample_seed=int(args.sample_seed),
                settle_policy=args.settle_policy,
            )
            print(
                json.dumps(
                    {
                        "round_index": payload["round_index"],
                        "sample": payload["sample"],
                        "delta": payload["delta"],
                        "outcome": payload["outcome"],
                        "inference_panel": payload["outputs"]["inference_panel"],
                        "info_json": payload["outputs"]["info_json"],
                    },
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=False,
                )
            )
            round_index += 1
    except KeyboardInterrupt:
        print(f"Interrupted after {round_index} completed samples.")
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
