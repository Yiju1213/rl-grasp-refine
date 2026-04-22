from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import torch

from src.structures.observation import Observation
from src.utils.tensor_utils import observation_to_tensor

ACTION_DIM = 6
ACTION_SATURATION_THRESHOLD = 0.9
ACTION_NORM_RATIO_BINS = (
    (0.0, 0.25),
    (0.25, 0.50),
    (0.50, 0.75),
    (0.75, 1.00),
)

ACTION_DISTRIBUTION_KEYS = (
    "action/translation_norm_mean",
    "action/translation_norm_std",
    "action/rotation_norm_mean",
    "action/rotation_norm_std",
    *(f"action/dim_{idx}_mean" for idx in range(ACTION_DIM)),
    *(f"action/dim_{idx}_std" for idx in range(ACTION_DIM)),
    *(f"action/dim_{idx}_saturation_rate" for idx in range(ACTION_DIM)),
    "action/saturation_rate",
)

ACTION_OUTCOME_CORRELATION_KEYS = (
    "corr/translation_norm_prob_delta",
    "corr/rotation_norm_prob_delta",
    "corr/translation_norm_success_delta",
    "corr/rotation_norm_success_delta",
    *(f"corr/dim_{idx}_prob_delta" for idx in range(ACTION_DIM)),
    *(f"corr/dim_{idx}_positive_drop" for idx in range(ACTION_DIM)),
    *(f"corr/dim_{idx}_negative_recovery" for idx in range(ACTION_DIM)),
)

ACTION_BIN_KEYS = tuple(
    f"action_bin/{prefix}_bin_{bin_idx}_{suffix}"
    for prefix in ("trans", "rot")
    for bin_idx in range(len(ACTION_NORM_RATIO_BINS))
    for suffix in ("count", "prob_delta_mean", "success_delta_mean", "drop_rate")
)

RELIABILITY_KEYS = (
    "calibrator/after_brier",
    "calibrator/before_brier_vs_legacy",
    "calibrator/prob_after_auc",
    "calibrator/raw_logit_after_auc",
    "calibrator/prob_before_auc_vs_legacy",
    "calibrator/raw_logit_before_auc_vs_legacy",
    "calibrator/prob_delta_recovery_auc",
    "calibrator/neg_prob_delta_degradation_auc",
)

FORMAL_DIAGNOSTIC_KEYS = (
    *ACTION_DISTRIBUTION_KEYS,
    *ACTION_OUTCOME_CORRELATION_KEYS,
    *ACTION_BIN_KEYS,
    *RELIABILITY_KEYS,
)

EPISODE_RECORD_EXTRA_FIELDS = (
    "source_global_id",
    "source_object_id",
    *(f"action_{idx}" for idx in range(ACTION_DIM)),
    "translation_norm",
    "rotation_norm",
    "saturation_rate",
    "success_delta",
    "positive_drop_event",
    "negative_recovery_event",
    "raw_logit_before",
    "raw_logit_after",
    "prob_before",
    "prob_after",
    "t_cover_before",
    "t_cover_after",
    "t_edge_before",
    "t_edge_after",
    "latent_before_norm",
    "latent_before_mean",
    "latent_before_std",
    "latent_after_norm",
    "latent_after_mean",
    "latent_after_std",
    "policy_latent_hidden_before_norm",
    "policy_latent_hidden_before_mean",
    "policy_latent_hidden_before_std",
    "trial_status",
    "failure_reason",
)


def diagnostic_key_to_csv_field(key: str) -> str:
    return str(key).replace("/", "_")


FORMAL_DIAGNOSTIC_FIELDS = tuple(diagnostic_key_to_csv_field(key) for key in FORMAL_DIAGNOSTIC_KEYS)


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(scalar):
        return None
    return scalar


def _finite_array(values: Iterable[Any]) -> np.ndarray:
    scalars = [_finite_float(value) for value in values]
    finite = [float(value) for value in scalars if value is not None]
    return np.asarray(finite, dtype=np.float64)


def _mean_or_none(values: Iterable[Any]) -> float | None:
    arr = _finite_array(values)
    if arr.size == 0:
        return None
    return float(np.mean(arr))


def _std_or_none(values: Iterable[Any]) -> float | None:
    arr = _finite_array(values)
    if arr.size == 0:
        return None
    return float(np.std(arr))


def safe_pearson(x_values: Iterable[Any], y_values: Iterable[Any]) -> float | None:
    pairs: list[tuple[float, float]] = []
    for x_raw, y_raw in zip(x_values, y_values):
        x = _finite_float(x_raw)
        y = _finite_float(y_raw)
        if x is not None and y is not None:
            pairs.append((x, y))
    if len(pairs) < 2:
        return None
    x = np.asarray([item[0] for item in pairs], dtype=np.float64)
    y = np.asarray([item[1] for item in pairs], dtype=np.float64)
    if float(np.std(x)) <= 0.0 or float(np.std(y)) <= 0.0:
        return None
    corr = float(np.corrcoef(x, y)[0, 1])
    return corr if np.isfinite(corr) else None


def binary_auc(scores: Iterable[Any], labels: Iterable[Any]) -> float | None:
    pairs: list[tuple[float, int]] = []
    for score_raw, label_raw in zip(scores, labels):
        score = _finite_float(score_raw)
        label = _finite_float(label_raw)
        if score is None or label is None:
            continue
        pairs.append((score, int(label >= 0.5)))
    if not pairs:
        return None
    labels_arr = np.asarray([item[1] for item in pairs], dtype=np.int32)
    pos_count = int(np.sum(labels_arr == 1))
    neg_count = int(np.sum(labels_arr == 0))
    if pos_count == 0 or neg_count == 0:
        return None

    scores_arr = np.asarray([item[0] for item in pairs], dtype=np.float64)
    order = np.argsort(scores_arr, kind="mergesort")
    ranks = np.empty_like(scores_arr, dtype=np.float64)
    sorted_scores = scores_arr[order]
    start = 0
    while start < sorted_scores.size:
        end = start + 1
        while end < sorted_scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        avg_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = avg_rank
        start = end

    pos_rank_sum = float(np.sum(ranks[labels_arr == 1]))
    auc = (pos_rank_sum - pos_count * (pos_count + 1) / 2.0) / float(pos_count * neg_count)
    return float(auc) if np.isfinite(auc) else None


def _coerce_actions(actions: Any) -> np.ndarray:
    arr = np.asarray(actions, dtype=np.float64)
    if arr.size == 0:
        return np.zeros((0, ACTION_DIM), dtype=np.float64)
    arr = arr.reshape((-1, ACTION_DIM))
    return arr


def action_distribution_stats(actions: Any) -> dict[str, float | None]:
    action_arr = _coerce_actions(actions)
    stats: dict[str, float | None] = {key: None for key in ACTION_DISTRIBUTION_KEYS}
    if action_arr.shape[0] == 0:
        return stats

    trans_norm = np.linalg.norm(action_arr[:, :3], axis=1)
    rot_norm = np.linalg.norm(action_arr[:, 3:6], axis=1)
    stats["action/translation_norm_mean"] = float(np.mean(trans_norm))
    stats["action/translation_norm_std"] = float(np.std(trans_norm))
    stats["action/rotation_norm_mean"] = float(np.mean(rot_norm))
    stats["action/rotation_norm_std"] = float(np.std(rot_norm))
    for idx in range(ACTION_DIM):
        dim_values = action_arr[:, idx]
        stats[f"action/dim_{idx}_mean"] = float(np.mean(dim_values))
        stats[f"action/dim_{idx}_std"] = float(np.std(dim_values))
        stats[f"action/dim_{idx}_saturation_rate"] = float(
            np.mean(np.abs(dim_values) > ACTION_SATURATION_THRESHOLD)
        )
    stats["action/saturation_rate"] = float(np.mean(np.abs(action_arr) > ACTION_SATURATION_THRESHOLD))
    return stats


def _record_value(record: Mapping[str, Any], key: str) -> float | None:
    return _finite_float(record.get(key))


def _action_values(records: Sequence[Mapping[str, Any]], idx: int) -> list[float | None]:
    return [_record_value(record, f"action_{idx}") for record in records]


def action_outcome_correlation_stats(records: Sequence[Mapping[str, Any]]) -> dict[str, float | None]:
    stats: dict[str, float | None] = {key: None for key in ACTION_OUTCOME_CORRELATION_KEYS}
    if not records:
        return stats

    trans_norm = [_record_value(record, "translation_norm") for record in records]
    rot_norm = [_record_value(record, "rotation_norm") for record in records]
    prob_delta = [_record_value(record, "prob_delta") for record in records]
    success_delta = [_record_value(record, "success_delta") for record in records]
    stats["corr/translation_norm_prob_delta"] = safe_pearson(trans_norm, prob_delta)
    stats["corr/rotation_norm_prob_delta"] = safe_pearson(rot_norm, prob_delta)
    stats["corr/translation_norm_success_delta"] = safe_pearson(trans_norm, success_delta)
    stats["corr/rotation_norm_success_delta"] = safe_pearson(rot_norm, success_delta)

    positive_records = [
        record for record in records if (_record_value(record, "legacy_drop_success_before") or 0.0) >= 0.5
    ]
    negative_records = [
        record for record in records if (_record_value(record, "legacy_drop_success_before") or 0.0) < 0.5
    ]
    for idx in range(ACTION_DIM):
        stats[f"corr/dim_{idx}_prob_delta"] = safe_pearson(_action_values(records, idx), prob_delta)
        stats[f"corr/dim_{idx}_positive_drop"] = safe_pearson(
            _action_values(positive_records, idx),
            [_record_value(record, "positive_drop_event") for record in positive_records],
        )
        stats[f"corr/dim_{idx}_negative_recovery"] = safe_pearson(
            _action_values(negative_records, idx),
            [_record_value(record, "negative_recovery_event") for record in negative_records],
        )
    return stats


def _norm_ratio(value: Any) -> float | None:
    norm = _finite_float(value)
    if norm is None:
        return None
    return float(np.clip(norm / np.sqrt(3.0), 0.0, 1.0))


def action_bin_stats(records: Sequence[Mapping[str, Any]]) -> dict[str, float | None]:
    stats: dict[str, float | None] = {key: None for key in ACTION_BIN_KEYS}
    for prefix, norm_key in (("trans", "translation_norm"), ("rot", "rotation_norm")):
        ratios = [_norm_ratio(record.get(norm_key)) for record in records]
        for bin_idx, (low, high) in enumerate(ACTION_NORM_RATIO_BINS):
            if bin_idx == len(ACTION_NORM_RATIO_BINS) - 1:
                selected = [
                    record for ratio, record in zip(ratios, records) if ratio is not None and low <= ratio <= high
                ]
            else:
                selected = [
                    record for ratio, record in zip(ratios, records) if ratio is not None and low <= ratio < high
                ]
            base = f"action_bin/{prefix}_bin_{bin_idx}"
            stats[f"{base}_count"] = float(len(selected))
            stats[f"{base}_prob_delta_mean"] = _mean_or_none(record.get("prob_delta") for record in selected)
            stats[f"{base}_success_delta_mean"] = _mean_or_none(record.get("success_delta") for record in selected)
            stats[f"{base}_drop_rate"] = _mean_or_none(record.get("positive_drop_event") for record in selected)
    return stats


def reliability_stats(records: Sequence[Mapping[str, Any]]) -> dict[str, float | None]:
    stats: dict[str, float | None] = {key: None for key in RELIABILITY_KEYS}
    if not records:
        return stats

    prob_after = [_record_value(record, "prob_after") for record in records]
    prob_before = [_record_value(record, "prob_before") for record in records]
    raw_after = [_record_value(record, "raw_logit_after") for record in records]
    raw_before = [_record_value(record, "raw_logit_before") for record in records]
    drop_success = [_record_value(record, "drop_success") for record in records]
    legacy = [_record_value(record, "legacy_drop_success_before") for record in records]
    prob_delta = [_record_value(record, "prob_delta") for record in records]

    after_pairs = [(p, y) for p, y in zip(prob_after, drop_success) if p is not None and y is not None]
    if after_pairs:
        stats["calibrator/after_brier"] = float(np.mean([(p - y) ** 2 for p, y in after_pairs]))
    before_pairs = [(p, y) for p, y in zip(prob_before, legacy) if p is not None and y is not None]
    if before_pairs:
        stats["calibrator/before_brier_vs_legacy"] = float(np.mean([(p - y) ** 2 for p, y in before_pairs]))

    stats["calibrator/prob_after_auc"] = binary_auc(prob_after, drop_success)
    stats["calibrator/raw_logit_after_auc"] = binary_auc(raw_after, drop_success)
    stats["calibrator/prob_before_auc_vs_legacy"] = binary_auc(prob_before, legacy)
    stats["calibrator/raw_logit_before_auc_vs_legacy"] = binary_auc(raw_before, legacy)

    recovery_records = [
        record for record in records if (_record_value(record, "legacy_drop_success_before") or 0.0) < 0.5
    ]
    degradation_records = [
        record for record in records if (_record_value(record, "legacy_drop_success_before") or 0.0) >= 0.5
    ]
    stats["calibrator/prob_delta_recovery_auc"] = binary_auc(
        [record.get("prob_delta") for record in recovery_records],
        [record.get("drop_success") for record in recovery_records],
    )
    stats["calibrator/neg_prob_delta_degradation_auc"] = binary_auc(
        [
            None if (value := _finite_float(record.get("prob_delta"))) is None else -value
            for record in degradation_records
        ],
        [record.get("positive_drop_event") for record in degradation_records],
    )
    return stats


def flatten_formal_stats(stats: Mapping[str, Any]) -> dict[str, Any]:
    return {diagnostic_key_to_csv_field(key): value for key, value in stats.items()}


def formal_diagnostic_stats(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    actions = []
    for record in records:
        dims = [_finite_float(record.get(f"action_{idx}")) for idx in range(ACTION_DIM)]
        if all(value is not None for value in dims):
            actions.append([float(value) for value in dims if value is not None])
    stats: dict[str, Any] = {}
    stats.update(action_distribution_stats(np.asarray(actions, dtype=np.float64)))
    stats.update(action_outcome_correlation_stats(records))
    stats.update(action_bin_stats(records))
    stats.update(reliability_stats(records))
    return flatten_formal_stats(stats)


def records_from_rollout_batch(batch: Mapping[str, Any]) -> list[dict[str, Any]]:
    infos = list(batch.get("infos", []))
    actions = _coerce_actions(batch.get("actions", np.zeros((0, ACTION_DIM), dtype=np.float64)))
    obs_list = list(batch.get("obs", []))
    next_obs_list = list(batch.get("next_obs", []))
    raw_before = np.asarray(batch.get("raw_logit_before", []), dtype=np.float64).reshape(-1)
    raw_after = np.asarray(batch.get("raw_logit_after", []), dtype=np.float64).reshape(-1)
    records: list[dict[str, Any]] = []
    count = min(len(infos), actions.shape[0], len(obs_list), len(next_obs_list))
    for idx in range(count):
        info = infos[idx]
        obs = obs_list[idx]
        next_obs = next_obs_list[idx]
        action = actions[idx]
        legacy = _finite_float(getattr(info, "extra", {}).get("legacy_drop_success_before"))
        drop_success = _finite_float(getattr(info, "drop_success", None))
        prob_before = _finite_float(getattr(info, "calibrated_stability_before", None))
        prob_after = _finite_float(getattr(info, "calibrated_stability_after", None))
        record = {
            f"action_{dim_idx}": float(action[dim_idx])
            for dim_idx in range(ACTION_DIM)
        }
        record.update(
            {
                "translation_norm": float(np.linalg.norm(action[:3])),
                "rotation_norm": float(np.linalg.norm(action[3:6])),
                "saturation_rate": float(np.mean(np.abs(action) > ACTION_SATURATION_THRESHOLD)),
                "drop_success": drop_success,
                "legacy_drop_success_before": legacy,
                "success_delta": (
                    None if drop_success is None or legacy is None else float(drop_success - legacy)
                ),
                "positive_drop_event": (
                    None
                    if drop_success is None or legacy is None or legacy < 0.5
                    else float(drop_success < 0.5)
                ),
                "negative_recovery_event": (
                    None
                    if drop_success is None or legacy is None or legacy >= 0.5
                    else float(drop_success >= 0.5)
                ),
                "raw_logit_before": float(raw_before[idx]) if idx < raw_before.size else getattr(obs, "raw_stability_logit", None),
                "raw_logit_after": float(raw_after[idx]) if idx < raw_after.size else getattr(next_obs, "raw_stability_logit", None),
                "prob_before": prob_before,
                "prob_after": prob_after,
                "prob_delta": None if prob_before is None or prob_after is None else float(prob_after - prob_before),
            }
        )
        records.append(record)
    return records


def latent_summary(feature: Any, *, prefix: str) -> dict[str, float | None]:
    arr = np.asarray(feature, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return {f"{prefix}_norm": None, f"{prefix}_mean": None, f"{prefix}_std": None}
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {f"{prefix}_norm": None, f"{prefix}_mean": None, f"{prefix}_std": None}
    return {
        f"{prefix}_norm": float(np.linalg.norm(finite)),
        f"{prefix}_mean": float(np.mean(finite)),
        f"{prefix}_std": float(np.std(finite)),
    }


def policy_latent_hidden_stats(
    actor_critic: Any,
    obs: Observation,
    *,
    prefix: str = "policy_latent_hidden_before",
) -> dict[str, float | None]:
    policy_net = getattr(actor_critic, "policy_net", None)
    latent_layer = getattr(policy_net, "latent_layer", None)
    latent_dim = getattr(policy_net, "latent_dim", None)
    if latent_layer is None or latent_dim is None:
        return {f"{prefix}_norm": None, f"{prefix}_mean": None, f"{prefix}_std": None}

    try:
        device = next(latent_layer.parameters()).device
        obs_spec = getattr(actor_critic, "observation_spec", None)
        obs_tensor = observation_to_tensor(obs, spec=obs_spec).to(device)
        latent = obs_tensor[:, : int(latent_dim)]
        with torch.no_grad():
            hidden = torch.relu(latent_layer(latent)).detach().cpu().numpy().reshape(-1)
    except Exception:
        return {f"{prefix}_norm": None, f"{prefix}_mean": None, f"{prefix}_std": None}
    return latent_summary(hidden, prefix=prefix)
