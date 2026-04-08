from __future__ import annotations

import csv
import json
import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from src.calibration.online_logit_calibrator import OnlineLogitCalibrator
from src.envs.dataset_sample_provider import DatasetSampleProvider
from src.rl.subproc_async_rollout_collector import SubprocAsyncRolloutCollector
from src.runtime.builders import build_actor_critic as runtime_build_actor_critic
from src.runtime.builders import build_env as runtime_build_env
from src.utils.checkpoint import load_checkpoint
from src.utils.config import load_config
from src.utils.single_file_config import load_experiment_bundle_from_input

_SUMMARY_MEAN_CI_METRICS = (
    ("macro_success_lift", "macro_success_lift", "success_lift_vs_dataset"),
    ("pos_drop", "pos_drop_rate", "positive_drop_rate"),
    ("neg_hold", "neg_hold_rate", "negative_hold_rate"),
    ("t_cover_after", "t_cover_after_mean", "t_cover_after_mean"),
    ("t_edge_after", "t_edge_after_mean", "t_edge_after_mean"),
    ("prob_delta_mean", "prob_delta_mean", "prob_delta_mean"),
)

_PER_OBJECT_FIELDS = (
    "experiment_name",
    "test_seed",
    "object_id",
    "success_lift_vs_dataset",
    "positive_drop_rate",
    "negative_hold_rate",
    "t_cover_after_mean",
    "t_edge_after_mean",
    "prob_delta_mean",
    "num_episodes",
    "positive_count",
    "negative_count",
    "positive_drop_count",
    "negative_hold_count",
)

_PER_RUN_FIELDS = (
    "experiment_name",
    "test_seed",
    "macro_success_lift",
    "pos_drop_rate",
    "neg_hold_rate",
    "across_object_lift_std",
    "across_object_lift_iqr",
    "t_cover_after_mean",
    "t_edge_after_mean",
    "prob_delta_mean",
    "num_objects",
    "total_episodes",
)

_SUMMARY_FIELDS = (
    "experiment_name",
    "evaluation_wall_minutes",
    "macro_success_lift_mean",
    "macro_success_lift_std",
    "macro_success_lift_ci95_low",
    "macro_success_lift_ci95_high",
    "pos_drop_mean",
    "pos_drop_std",
    "pos_drop_ci95_low",
    "pos_drop_ci95_high",
    "neg_hold_mean",
    "neg_hold_std",
    "neg_hold_ci95_low",
    "neg_hold_ci95_high",
    "across_object_lift_std_mean",
    "across_object_lift_std_std",
    "across_object_lift_iqr_mean",
    "across_object_lift_iqr_std",
    "t_cover_after_mean",
    "t_cover_after_std",
    "t_cover_after_ci95_low",
    "t_cover_after_ci95_high",
    "t_edge_after_mean",
    "t_edge_after_std",
    "t_edge_after_ci95_low",
    "t_edge_after_ci95_high",
    "prob_delta_mean_mean",
    "prob_delta_mean_std",
    "prob_delta_mean_ci95_low",
    "prob_delta_mean_ci95_high",
)


@dataclass(frozen=True)
class EvaluationExperiment:
    label: str
    experiment_dir: Path


@dataclass(frozen=True)
class EvaluationProtocol:
    test_object_ids: tuple[int, ...]
    test_seeds: tuple[int, ...]
    episodes_per_object: int
    bootstrap_iterations: int
    confidence_level: float


@dataclass(frozen=True)
class CollectorOverrides:
    num_workers: int = 1
    batch_episodes: int | None = None
    worker_policy_device: str | None = None
    scene_rebuild_every_n_iterations: int | None = None
    worker_recycle_every_n_iterations: int | None = None
    worker_recycle_slots_per_event: int | None = None
    worker_recycle_enable_standby_prefetch: bool | None = None
    worker_recycle_prefetch_count: int | None = None


@dataclass(frozen=True)
class EvaluationManifest:
    manifest_path: Path
    output_dir: Path
    experiments: tuple[EvaluationExperiment, ...]
    protocol: EvaluationProtocol
    collector: CollectorOverrides


@dataclass(frozen=True)
class ObjectEvaluationBudget:
    available_samples: int
    effective_num_workers: int
    per_worker_dispatch_limits: dict[int, int]
    total_block_count: int


def _validate_output_label(label: str) -> None:
    path = Path(label)
    if label in {"", ".", ".."} or path.name != label:
        raise ValueError(
            f"Manifest experiment label {label!r} is not a valid output directory name. "
            "Labels must not contain path separators or parent-directory references."
        )


def _resolve_path(path_str: str | Path, *, base_dir: Path) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    scalar = float(value)
    if not np.isfinite(scalar):
        return None
    return scalar


def _mean_or_none(values: list[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and np.isfinite(float(value))]
    if not finite:
        return None
    return float(np.mean(np.asarray(finite, dtype=np.float64)))


def _std(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    return float(np.std(arr))


def _iqr(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    return float(np.percentile(arr, 75.0) - np.percentile(arr, 25.0))


def load_evaluation_manifest(path: str | Path) -> EvaluationManifest:
    manifest_path = Path(path).expanduser().resolve()
    payload = load_config(manifest_path)
    base_dir = manifest_path.parent

    raw_experiments = payload.get("experiments")
    if not isinstance(raw_experiments, list) or not raw_experiments:
        raise ValueError("Evaluation manifest must define a non-empty 'experiments' list.")

    experiments: list[EvaluationExperiment] = []
    seen_labels: set[str] = set()
    for item in raw_experiments:
        if not isinstance(item, dict):
            raise ValueError("Each manifest experiment entry must be a mapping.")
        label = str(item.get("label", "")).strip()
        if not label:
            raise ValueError("Each manifest experiment entry must define a non-empty 'label'.")
        _validate_output_label(label)
        if label in seen_labels:
            raise ValueError(f"Duplicate experiment label in manifest: {label!r}")
        seen_labels.add(label)
        experiment_dir_raw = item.get("experiment_dir")
        if experiment_dir_raw is None:
            raise ValueError(f"Manifest experiment {label!r} is missing 'experiment_dir'.")
        experiments.append(
            EvaluationExperiment(
                label=label,
                experiment_dir=_resolve_path(experiment_dir_raw, base_dir=base_dir),
            )
        )

    protocol_payload = payload.get("protocol")
    if not isinstance(protocol_payload, dict):
        raise ValueError("Evaluation manifest must define a 'protocol' mapping.")
    raw_test_object_ids = protocol_payload.get("test_object_ids")
    if not isinstance(raw_test_object_ids, list) or not raw_test_object_ids:
        raise ValueError("'protocol.test_object_ids' must be a non-empty list.")
    test_object_ids = tuple(int(object_id) for object_id in raw_test_object_ids)
    if len(set(test_object_ids)) != len(test_object_ids):
        raise ValueError("'protocol.test_object_ids' must not contain duplicates.")

    raw_test_seeds = protocol_payload.get("test_seeds")
    if not isinstance(raw_test_seeds, list) or len(raw_test_seeds) != 3:
        raise ValueError("'protocol.test_seeds' must contain exactly 3 seeds.")
    test_seeds = tuple(int(seed) for seed in raw_test_seeds)
    if len(set(test_seeds)) != len(test_seeds):
        raise ValueError("'protocol.test_seeds' must not contain duplicates.")

    episodes_per_object = int(protocol_payload.get("episodes_per_object", 0))
    if episodes_per_object <= 0:
        raise ValueError("'protocol.episodes_per_object' must be a positive integer.")
    bootstrap_iterations = int(protocol_payload.get("bootstrap_iterations", 10000))
    if bootstrap_iterations <= 0:
        raise ValueError("'protocol.bootstrap_iterations' must be a positive integer.")
    confidence_level = float(protocol_payload.get("confidence_level", 0.95))
    if not (0.0 < confidence_level < 1.0):
        raise ValueError("'protocol.confidence_level' must be in (0, 1).")

    collector_payload = payload.get("collector", {})
    if collector_payload is None:
        collector_payload = {}
    if not isinstance(collector_payload, dict):
        raise ValueError("'collector' must be a mapping when provided.")
    collector = CollectorOverrides(
        num_workers=max(int(collector_payload.get("num_workers", 1)), 1),
        batch_episodes=(
            None if collector_payload.get("batch_episodes") is None else max(int(collector_payload["batch_episodes"]), 1)
        ),
        worker_policy_device=(
            None
            if collector_payload.get("worker_policy_device") in (None, "")
            else str(collector_payload["worker_policy_device"])
        ),
        scene_rebuild_every_n_iterations=(
            None
            if collector_payload.get("scene_rebuild_every_n_iterations") is None
            else max(int(collector_payload["scene_rebuild_every_n_iterations"]), 0)
        ),
        worker_recycle_every_n_iterations=(
            None
            if collector_payload.get("worker_recycle_every_n_iterations") is None
            else max(int(collector_payload["worker_recycle_every_n_iterations"]), 0)
        ),
        worker_recycle_slots_per_event=(
            None
            if collector_payload.get("worker_recycle_slots_per_event") is None
            else max(int(collector_payload["worker_recycle_slots_per_event"]), 1)
        ),
        worker_recycle_enable_standby_prefetch=(
            None
            if collector_payload.get("worker_recycle_enable_standby_prefetch") is None
            else bool(collector_payload["worker_recycle_enable_standby_prefetch"])
        ),
        worker_recycle_prefetch_count=(
            None
            if collector_payload.get("worker_recycle_prefetch_count") is None
            else max(int(collector_payload["worker_recycle_prefetch_count"]), 0)
        ),
    )

    output_dir_raw = payload.get("output_dir")
    if output_dir_raw is None:
        raise ValueError("Evaluation manifest must define 'output_dir'.")
    output_dir = _resolve_path(output_dir_raw, base_dir=base_dir)

    return EvaluationManifest(
        manifest_path=manifest_path,
        output_dir=output_dir,
        experiments=tuple(experiments),
        protocol=EvaluationProtocol(
            test_object_ids=test_object_ids,
            test_seeds=test_seeds,
            episodes_per_object=episodes_per_object,
            bootstrap_iterations=bootstrap_iterations,
            confidence_level=confidence_level,
        ),
        collector=collector,
    )


def restore_evaluation_state(*, checkpoint_path: str | Path, actor_critic, calibrator) -> dict[str, Any]:
    checkpoint = load_checkpoint(checkpoint_path)
    actor_state = checkpoint.get("actor_critic")
    calibrator_state = checkpoint.get("calibrator")
    if actor_state is None:
        raise KeyError(f"Checkpoint is missing 'actor_critic': {checkpoint_path}")
    if calibrator_state is None:
        raise KeyError(f"Checkpoint is missing 'calibrator': {checkpoint_path}")
    actor_critic.load_state_dict(actor_state)
    load_state = getattr(calibrator, "load_state", None)
    if not callable(load_state):
        raise TypeError("Calibrator does not support load_state().")
    load_state(calibrator_state)
    return checkpoint


def resolve_object_evaluation_budget(
    *,
    dataset_cfg: dict[str, Any],
    object_id: int,
    test_seed: int,
    requested_num_workers: int,
) -> ObjectEvaluationBudget:
    if not bool(dataset_cfg.get("enabled", False)):
        raise RuntimeError("Evaluation protocol requires a dataset-backed env config.")

    base_cfg = deepcopy(dataset_cfg)
    base_cfg["include_object_ids"] = [int(object_id)]
    base_cfg["fixed_sample_sequence"] = True
    base_cfg["fixed_sample_sequence_seed"] = int(test_seed)
    base_cfg["worker_generation"] = 0

    counting_cfg = deepcopy(base_cfg)
    counting_cfg["worker_id"] = 0
    counting_cfg["num_workers"] = 1
    counting_provider = DatasetSampleProvider(counting_cfg)
    available_samples = int(counting_provider.sequence_length())
    total_block_count = int(counting_provider.sequence_block_count())
    if available_samples <= 0 or total_block_count <= 0:
        raise RuntimeError(
            f"Object {object_id} has no available fixed-sequence samples for test seed {test_seed}."
        )

    effective_num_workers = max(1, min(int(requested_num_workers), available_samples, total_block_count))
    while effective_num_workers > 0:
        per_worker_dispatch_limits: dict[int, int] = {}
        for worker_id in range(effective_num_workers):
            worker_cfg = deepcopy(base_cfg)
            worker_cfg["worker_id"] = int(worker_id)
            worker_cfg["num_workers"] = int(effective_num_workers)
            provider = DatasetSampleProvider(worker_cfg)
            per_worker_dispatch_limits[int(worker_id)] = int(provider.sequence_length())
        if all(limit > 0 for limit in per_worker_dispatch_limits.values()):
            available_samples = int(sum(per_worker_dispatch_limits.values()))
            return ObjectEvaluationBudget(
                available_samples=available_samples,
                effective_num_workers=int(effective_num_workers),
                per_worker_dispatch_limits=per_worker_dispatch_limits,
                total_block_count=total_block_count,
            )
        effective_num_workers -= 1

    raise RuntimeError(f"Unable to allocate a non-empty worker set for object {object_id}.")


def _resolve_experiment_context(experiment_dir: Path) -> tuple[dict[str, Any], dict[str, Any], Path, Path]:
    experiment_dir = Path(experiment_dir).expanduser().resolve()
    checkpoint_path = experiment_dir / "checkpoints" / "best.pt"
    configs_dir = experiment_dir / "configs"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing best checkpoint: {checkpoint_path}")
    if not configs_dir.exists():
        raise FileNotFoundError(f"Missing config snapshot directory: {configs_dir}")
    experiment_cfg, bundle, _ = load_experiment_bundle_from_input(configs_dir)
    return experiment_cfg, bundle, checkpoint_path, configs_dir


def _build_actor_critic(actor_critic_factory, *, perception_cfg: dict, actor_critic_cfg: dict):
    actor_critic = actor_critic_factory(
        perception_cfg=perception_cfg,
        actor_critic_cfg=actor_critic_cfg,
    )
    actor_critic.eval()
    return actor_critic


def _object_episode_record_from_transition(
    *,
    experiment_name: str,
    test_seed: int,
    object_id: int,
    transition: dict[str, Any],
) -> dict[str, Any]:
    info = transition["info"]
    source_object_id = info.extra.get("source_object_id")
    if source_object_id is None or int(source_object_id) != int(object_id):
        raise RuntimeError(
            f"Object-restricted evaluation expected object_id={object_id}, got source_object_id={source_object_id}."
        )
    legacy_before = info.extra.get("legacy_drop_success_before")
    if legacy_before is None or not np.isfinite(float(legacy_before)):
        raise RuntimeError(f"Missing or invalid legacy_drop_success_before for object {object_id}.")
    next_obs = transition["next_obs"]
    contact_semantic = np.asarray(next_obs.contact_semantic, dtype=np.float64).reshape(-1)
    if contact_semantic.size < 2:
        raise RuntimeError(f"Object {object_id} produced an invalid contact_semantic payload.")

    return {
        "experiment_name": str(experiment_name),
        "test_seed": int(test_seed),
        "object_id": int(object_id),
        "drop_success": int(info.drop_success),
        "legacy_drop_success_before": float(legacy_before),
        "t_cover_after": float(contact_semantic[0]),
        "t_edge_after": float(contact_semantic[1]),
        "prob_delta": float(info.calibrated_stability_after - info.calibrated_stability_before),
    }


def aggregate_object_episode_records(
    *,
    experiment_name: str,
    test_seed: int,
    object_id: int,
    episode_records: list[dict[str, Any]],
) -> dict[str, Any]:
    if not episode_records:
        raise ValueError(f"Cannot aggregate empty episode records for object {object_id}.")

    drop_success = np.asarray([record["drop_success"] for record in episode_records], dtype=np.float64)
    legacy_before = np.asarray([record["legacy_drop_success_before"] for record in episode_records], dtype=np.float64)
    t_cover_after = np.asarray([record["t_cover_after"] for record in episode_records], dtype=np.float64)
    t_edge_after = np.asarray([record["t_edge_after"] for record in episode_records], dtype=np.float64)
    prob_delta = np.asarray([record["prob_delta"] for record in episode_records], dtype=np.float64)

    positive_mask = legacy_before >= 0.5
    negative_mask = legacy_before < 0.5
    positive_count = int(np.sum(positive_mask))
    negative_count = int(np.sum(negative_mask))
    positive_drop_count = int(np.sum(drop_success[positive_mask] < 0.5)) if positive_count else 0
    negative_hold_count = int(np.sum(drop_success[negative_mask] >= 0.5)) if negative_count else 0

    return {
        "experiment_name": str(experiment_name),
        "test_seed": int(test_seed),
        "object_id": int(object_id),
        "success_lift_vs_dataset": float(np.mean(drop_success) - np.mean(legacy_before)),
        "positive_drop_rate": (
            None if positive_count == 0 else float(positive_drop_count / positive_count)
        ),
        "negative_hold_rate": (
            None if negative_count == 0 else float(negative_hold_count / negative_count)
        ),
        "t_cover_after_mean": float(np.mean(t_cover_after)),
        "t_edge_after_mean": float(np.mean(t_edge_after)),
        "prob_delta_mean": float(np.mean(prob_delta)),
        "num_episodes": int(len(episode_records)),
        "positive_count": int(positive_count),
        "negative_count": int(negative_count),
        "positive_drop_count": int(positive_drop_count),
        "negative_hold_count": int(negative_hold_count),
    }


def aggregate_run_object_rows(
    *,
    experiment_name: str,
    test_seed: int,
    object_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if not object_rows:
        raise ValueError(f"Cannot aggregate empty object rows for run seed {test_seed}.")

    success_lift_values = [float(row["success_lift_vs_dataset"]) for row in object_rows]
    pos_drop_macro = _mean_or_none([_float_or_none(row["positive_drop_rate"]) for row in object_rows])
    neg_hold_macro = _mean_or_none([_float_or_none(row["negative_hold_rate"]) for row in object_rows])
    if pos_drop_macro is None:
        raise RuntimeError(
            f"Run {experiment_name}[seed={test_seed}] has no defined positive_drop_rate across objects."
        )
    if neg_hold_macro is None:
        raise RuntimeError(
            f"Run {experiment_name}[seed={test_seed}] has no defined negative_hold_rate across objects."
        )

    return {
        "experiment_name": str(experiment_name),
        "test_seed": int(test_seed),
        "macro_success_lift": float(np.mean(np.asarray(success_lift_values, dtype=np.float64))),
        "pos_drop_rate": float(pos_drop_macro),
        "neg_hold_rate": float(neg_hold_macro),
        "across_object_lift_std": _std(success_lift_values),
        "across_object_lift_iqr": _iqr(success_lift_values),
        "t_cover_after_mean": float(np.mean([float(row["t_cover_after_mean"]) for row in object_rows])),
        "t_edge_after_mean": float(np.mean([float(row["t_edge_after_mean"]) for row in object_rows])),
        "prob_delta_mean": float(np.mean([float(row["prob_delta_mean"]) for row in object_rows])),
        "num_objects": int(len(object_rows)),
        "total_episodes": int(sum(int(row["num_episodes"]) for row in object_rows)),
    }


def _bootstrap_metric_ci(
    *,
    per_object_rows: list[dict[str, Any]],
    metric_key: str,
    bootstrap_iterations: int,
    confidence_level: float,
    bootstrap_seed: int = 0,
) -> tuple[float, float]:
    grouped_rows: dict[int, list[float]] = {}
    for row in per_object_rows:
        raw_value = row[metric_key]
        value = _float_or_none(raw_value)
        if value is None:
            continue
        grouped_rows.setdefault(int(row["test_seed"]), []).append(float(value))
    if not grouped_rows:
        raise RuntimeError(f"Cannot bootstrap metric {metric_key!r} with no finite object-level values.")

    run_arrays = {
        int(test_seed): np.asarray(values, dtype=np.float64)
        for test_seed, values in grouped_rows.items()
    }
    rng = np.random.default_rng(int(bootstrap_seed))
    bootstrap_values = np.zeros(int(bootstrap_iterations), dtype=np.float64)
    for index in range(int(bootstrap_iterations)):
        run_means = []
        for arr in run_arrays.values():
            sampled = arr[rng.integers(0, arr.size, size=arr.size)]
            run_means.append(float(np.mean(sampled)))
        bootstrap_values[index] = float(np.mean(np.asarray(run_means, dtype=np.float64)))

    alpha = (1.0 - float(confidence_level)) / 2.0
    low = float(np.quantile(bootstrap_values, alpha))
    high = float(np.quantile(bootstrap_values, 1.0 - alpha))
    return low, high


def summarize_experiment_rows(
    *,
    experiment_name: str,
    run_rows: list[dict[str, Any]],
    object_rows: list[dict[str, Any]],
    bootstrap_iterations: int,
    confidence_level: float,
) -> dict[str, Any]:
    if len(run_rows) != 3:
        raise ValueError(f"Experiment {experiment_name!r} must provide exactly 3 run rows.")

    summary: dict[str, Any] = {"experiment_name": str(experiment_name)}
    for output_prefix, run_metric_key, object_metric_key in _SUMMARY_MEAN_CI_METRICS:
        metric_values = [float(row[run_metric_key]) for row in run_rows]
        ci_low, ci_high = _bootstrap_metric_ci(
            per_object_rows=object_rows,
            metric_key=object_metric_key,
            bootstrap_iterations=bootstrap_iterations,
            confidence_level=confidence_level,
        )
        summary[f"{output_prefix}_mean"] = float(np.mean(np.asarray(metric_values, dtype=np.float64)))
        summary[f"{output_prefix}_std"] = _std(metric_values)
        summary[f"{output_prefix}_ci95_low"] = float(ci_low)
        summary[f"{output_prefix}_ci95_high"] = float(ci_high)

    across_object_std_values = [float(row["across_object_lift_std"]) for row in run_rows]
    across_object_iqr_values = [float(row["across_object_lift_iqr"]) for row in run_rows]
    summary["across_object_lift_std_mean"] = float(np.mean(np.asarray(across_object_std_values, dtype=np.float64)))
    summary["across_object_lift_std_std"] = _std(across_object_std_values)
    summary["across_object_lift_iqr_mean"] = float(np.mean(np.asarray(across_object_iqr_values, dtype=np.float64)))
    summary["across_object_lift_iqr_std"] = _std(across_object_iqr_values)
    return summary


def _write_csv(path: Path, *, fieldnames: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _resolve_object_rl_cfg(
    *,
    experiment_rl_cfg: dict[str, Any],
    collector_overrides: CollectorOverrides,
    protocol: EvaluationProtocol,
    effective_num_workers: int,
) -> dict[str, Any]:
    rl_cfg = deepcopy(experiment_rl_cfg)
    default_batch = min(
        int(protocol.episodes_per_object),
        max(int(experiment_rl_cfg.get("batch_episodes", protocol.episodes_per_object)), 1),
    )
    rl_cfg["batch_episodes"] = (
        int(default_batch)
        if collector_overrides.batch_episodes is None
        else int(collector_overrides.batch_episodes)
    )
    rl_cfg["num_envs"] = int(effective_num_workers)
    if collector_overrides.worker_policy_device is not None:
        rl_cfg["worker_policy_device"] = str(collector_overrides.worker_policy_device)
    if collector_overrides.scene_rebuild_every_n_iterations is not None:
        rl_cfg["scene_rebuild_every_n_iterations"] = int(collector_overrides.scene_rebuild_every_n_iterations)
    if collector_overrides.worker_recycle_every_n_iterations is not None:
        rl_cfg["worker_recycle_every_n_iterations"] = int(collector_overrides.worker_recycle_every_n_iterations)
    if collector_overrides.worker_recycle_slots_per_event is not None:
        rl_cfg["worker_recycle_slots_per_event"] = int(collector_overrides.worker_recycle_slots_per_event)
    if collector_overrides.worker_recycle_enable_standby_prefetch is not None:
        rl_cfg["worker_recycle_enable_standby_prefetch"] = bool(
            collector_overrides.worker_recycle_enable_standby_prefetch
        )
    if collector_overrides.worker_recycle_prefetch_count is not None:
        rl_cfg["worker_recycle_prefetch_count"] = int(collector_overrides.worker_recycle_prefetch_count)
    return rl_cfg


def run_best_checkpoint_evaluation(
    manifest_path: str | Path,
    *,
    env_factory=runtime_build_env,
    actor_critic_factory=runtime_build_actor_critic,
    collector_factory=SubprocAsyncRolloutCollector,
    object_budget_resolver: Callable[..., ObjectEvaluationBudget] = resolve_object_evaluation_budget,
) -> dict[str, Any]:
    manifest = load_evaluation_manifest(manifest_path)
    output_dir = manifest.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths_by_experiment: dict[str, dict[str, Path]] = {}
    failed_experiments: dict[str, dict[str, Any]] = {}

    for experiment in manifest.experiments:
        experiment_start = time.perf_counter()
        try:
            experiment_cfg, bundle, checkpoint_path, configs_dir = _resolve_experiment_context(experiment.experiment_dir)
            env_cfg = deepcopy(bundle["env"])
            perception_cfg = deepcopy(bundle["perception"])
            calibration_cfg = deepcopy(bundle["calibration"])
            actor_critic_cfg = deepcopy(bundle["actor_critic"])
            experiment_rl_cfg = deepcopy(bundle["rl"])
            actor_critic = _build_actor_critic(
                actor_critic_factory,
                perception_cfg=perception_cfg,
                actor_critic_cfg=actor_critic_cfg,
            )
            calibrator = OnlineLogitCalibrator(calibration_cfg)
            checkpoint = restore_evaluation_state(
                checkpoint_path=checkpoint_path,
                actor_critic=actor_critic,
                calibrator=calibrator,
            )
            actor_state = {
                key: value.detach().cpu().clone()
                for key, value in actor_critic.state_dict().items()
            }
            calibrator_state = calibrator.get_state()

            experiment_object_rows: list[dict[str, Any]] = []
            experiment_run_rows: list[dict[str, Any]] = []
            protocol_notes = {
                "action_mode": "deterministic_mean",
                "episode_records_persisted": False,
                "episode_records_storage": "memory_only",
                "per_object_budget_mode": "equal_budget_with_truncation",
                "risk_metric_na_policy": "object_level_na_run_macro_defined_domain_only",
                "ci_method": "object_bootstrap",
                "std_ddof": 0,
                "async_collector_note": (
                    "num_workers>1 may introduce small sample-membership differences due to async dispatch/drain order; "
                    "num_workers=1 is the strongest determinism setting."
                ),
            }
            experiment_metadata = {
                "manifest_path": manifest.manifest_path,
                "experiment_name": experiment.label,
                "experiment_dir": experiment.experiment_dir,
                "checkpoint_path": checkpoint_path,
                "configs_dir": configs_dir,
                "experiment_cfg": experiment_cfg,
                "protocol": asdict(manifest.protocol),
                "collector": asdict(manifest.collector),
                "protocol_notes": protocol_notes,
                "checkpoint_best_metric_name": checkpoint.get("best_metric_name"),
                "resolved_objects": [],
            }

            dataset_cfg = deepcopy(env_cfg.get("dataset", {}))
            for test_seed in manifest.protocol.test_seeds:
                run_object_rows: list[dict[str, Any]] = []
                for object_id in manifest.protocol.test_object_ids:
                    budget = object_budget_resolver(
                        dataset_cfg=dataset_cfg,
                        object_id=int(object_id),
                        test_seed=int(test_seed),
                        requested_num_workers=int(manifest.collector.num_workers),
                    )
                    target_episodes = min(int(manifest.protocol.episodes_per_object), int(budget.available_samples))
                    if target_episodes <= 0:
                        raise RuntimeError(
                            f"Object {object_id} has no available samples for experiment {experiment.label!r}."
                        )

                    object_env_cfg = deepcopy(env_cfg)
                    object_dataset_cfg = object_env_cfg.setdefault("dataset", {})
                    object_dataset_cfg["include_object_ids"] = [int(object_id)]
                    object_dataset_cfg["fixed_sample_sequence"] = True
                    object_dataset_cfg["fixed_sample_sequence_seed"] = int(test_seed)

                    object_rl_cfg = _resolve_object_rl_cfg(
                        experiment_rl_cfg=experiment_rl_cfg,
                        collector_overrides=manifest.collector,
                        protocol=manifest.protocol,
                        effective_num_workers=int(budget.effective_num_workers),
                    )
                    object_rl_cfg["batch_episodes"] = min(int(object_rl_cfg["batch_episodes"]), int(target_episodes))

                    collector = collector_factory(
                        env_cfg=object_env_cfg,
                        perception_cfg=perception_cfg,
                        calibration_cfg=calibration_cfg,
                        actor_critic_cfg=actor_critic_cfg,
                        rl_cfg=object_rl_cfg,
                        num_workers=int(budget.effective_num_workers),
                        observation_spec=getattr(actor_critic, "observation_spec", None),
                        env_factory=env_factory,
                        actor_critic_factory=actor_critic_factory,
                    )
                    object_records: list[dict[str, Any]] = []
                    overflow_transitions: list[dict[str, Any]] = []
                    remaining_dispatch_limits = dict(budget.per_worker_dispatch_limits)
                    rollout_version = 0
                    try:
                        while len(object_records) < target_episodes:
                            if overflow_transitions:
                                take_count = min(len(overflow_transitions), target_episodes - len(object_records))
                                retained = overflow_transitions[:take_count]
                                overflow_transitions = overflow_transitions[take_count:]
                                for transition in retained:
                                    object_records.append(
                                        _object_episode_record_from_transition(
                                            experiment_name=experiment.label,
                                            test_seed=int(test_seed),
                                            object_id=int(object_id),
                                            transition=transition,
                                        )
                                    )
                                if len(object_records) >= target_episodes:
                                    break

                            remaining = target_episodes - len(object_records)
                            chunk_target = min(int(object_rl_cfg["batch_episodes"]), int(remaining))
                            payload = collector.collect_batch(
                                target_valid_episodes=int(chunk_target),
                                actor_state=actor_state,
                                calibrator_state=calibrator_state,
                                obs_spec=getattr(actor_critic, "observation_spec", None),
                                rollout_version=int(rollout_version),
                                deterministic_policy=True,
                                return_overflow_transitions=True,
                                per_worker_dispatch_limits=remaining_dispatch_limits,
                            )
                            dispatched_counts_by_worker: dict[int, int] = {}
                            for summary in payload.get("attempt_summaries", []):
                                worker_id = int(summary.get("worker_id", -1))
                                if worker_id < 0:
                                    continue
                                dispatched_counts_by_worker[worker_id] = (
                                    dispatched_counts_by_worker.get(worker_id, 0) + 1
                                )
                            for worker_id, dispatched_count in dispatched_counts_by_worker.items():
                                remaining_dispatch_limits[worker_id] = max(
                                    int(remaining_dispatch_limits.get(worker_id, 0)) - int(dispatched_count),
                                    0,
                                )
                            rollout_version += 1
                            for transition in payload["transitions"]:
                                object_records.append(
                                    _object_episode_record_from_transition(
                                        experiment_name=experiment.label,
                                        test_seed=int(test_seed),
                                        object_id=int(object_id),
                                        transition=transition,
                                    )
                                )
                            overflow_transitions.extend(list(payload.get("overflow_transitions", [])))
                    finally:
                        collector.close()

                    if len(object_records) != target_episodes:
                        raise RuntimeError(
                            f"Object {object_id} for experiment {experiment.label!r} collected "
                            f"{len(object_records)} episodes, expected {target_episodes}."
                        )
                    object_row = aggregate_object_episode_records(
                        experiment_name=experiment.label,
                        test_seed=int(test_seed),
                        object_id=int(object_id),
                        episode_records=object_records,
                    )
                    run_object_rows.append(object_row)
                    experiment_object_rows.append(object_row)
                    experiment_metadata["resolved_objects"].append(
                        {
                            "test_seed": int(test_seed),
                            "object_id": int(object_id),
                            "available_samples": int(budget.available_samples),
                            "effective_num_workers": int(budget.effective_num_workers),
                            "per_worker_dispatch_limits": dict(budget.per_worker_dispatch_limits),
                            "target_episodes": int(target_episodes),
                            "collector_batch_episodes": int(object_rl_cfg["batch_episodes"]),
                        }
                    )

                run_row = aggregate_run_object_rows(
                    experiment_name=experiment.label,
                    test_seed=int(test_seed),
                    object_rows=run_object_rows,
                )
                experiment_run_rows.append(run_row)

            summary_row = summarize_experiment_rows(
                experiment_name=experiment.label,
                run_rows=experiment_run_rows,
                object_rows=experiment_object_rows,
                bootstrap_iterations=int(manifest.protocol.bootstrap_iterations),
                confidence_level=float(manifest.protocol.confidence_level),
            )
            evaluation_wall_minutes = float((time.perf_counter() - experiment_start) / 60.0)
            summary_row["evaluation_wall_minutes"] = evaluation_wall_minutes
            experiment_metadata["evaluation_wall_minutes"] = evaluation_wall_minutes
            experiment_output_dir = output_dir / experiment.label
            experiment_output_dir.mkdir(parents=True, exist_ok=True)
            per_object_path = experiment_output_dir / "per_object_summary.csv"
            per_run_path = experiment_output_dir / "per_run_summary.csv"
            summary_path = experiment_output_dir / "summary.csv"
            metadata_path = experiment_output_dir / "metadata.json"

            _write_csv(per_object_path, fieldnames=_PER_OBJECT_FIELDS, rows=experiment_object_rows)
            _write_csv(per_run_path, fieldnames=_PER_RUN_FIELDS, rows=experiment_run_rows)
            _write_csv(summary_path, fieldnames=_SUMMARY_FIELDS, rows=[summary_row])
            with metadata_path.open("w", encoding="utf-8") as handle:
                json.dump(_json_ready(experiment_metadata), handle, indent=2, sort_keys=True, ensure_ascii=False)
                handle.write("\n")

            output_paths_by_experiment[experiment.label] = {
                "directory": experiment_output_dir,
                "per_object_summary": per_object_path,
                "per_run_summary": per_run_path,
                "summary": summary_path,
                "metadata": metadata_path,
            }
        except Exception as exc:
            failed_experiments[experiment.label] = {
                "experiment_dir": experiment.experiment_dir,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }

    if not output_paths_by_experiment:
        failure_labels = ", ".join(sorted(failed_experiments))
        raise RuntimeError(f"All experiments failed during evaluation: {failure_labels}")

    return {
        "output_dir": output_dir,
        "experiments": output_paths_by_experiment,
        "failed_experiments": failed_experiments,
    }
