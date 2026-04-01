from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from numbers import Number
from pathlib import Path
from typing import Any

from torch.utils.tensorboard import SummaryWriter


def _sanitize_experiment_name(name: str | None) -> str:
    raw = str(name or "").strip()
    if not raw:
        return ""
    sanitized = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in raw)
    return sanitized.strip("._")


def _nest_path_under_experiment(path: Path, experiment_name: str) -> Path:
    if not experiment_name:
        return path
    if path.parent.name == experiment_name or path.name == experiment_name:
        return path
    return path.parent / experiment_name / path.name


def resolve_experiment_artifact_path(path: str | Path, experiment_name: str | None) -> Path:
    return _nest_path_under_experiment(Path(path).resolve(), _sanitize_experiment_name(experiment_name))


class Logger:
    """Experiment logger with JSONL, TensorBoard, and optional sample dumps."""

    def __init__(self, cfg: dict[str, Any]):
        self.experiment_name = _sanitize_experiment_name(cfg.get("experiment_name"))
        log_root = Path(cfg.get("log_dir", "outputs/default")).resolve()
        self.log_dir = (log_root / self.experiment_name).resolve() if self.experiment_name else log_root
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.log_dir / "metrics.jsonl"
        self.info_path = self.log_dir / "run.log"
        self.metric_profile = str(cfg.get("metric_profile", "full") or "full").strip().lower()
        diagnostics_cfg = dict(cfg.get("diagnostics", {}))
        self.diagnostics_enabled = bool(diagnostics_cfg.get("enabled", True))
        tensorboard_cfg = dict(cfg.get("tensorboard", {}))
        self.tensorboard_enabled = bool(tensorboard_cfg.get("enabled", True))
        tensorboard_dir = tensorboard_cfg.get("dir", log_root / "tensorboard")
        self.tensorboard_dir = resolve_experiment_artifact_path(tensorboard_dir, self.experiment_name)
        self.writer = SummaryWriter(log_dir=str(self.tensorboard_dir)) if self.tensorboard_enabled else None
        sample_cfg = dict(cfg.get("sample_metrics", {}))
        self.sample_metrics_enabled = bool(sample_cfg.get("enabled", False))
        self.sample_metrics_every_n_iterations = max(int(sample_cfg.get("every_n_iterations", 10)), 1)
        episode_metrics_path = sample_cfg.get("path", log_root / "episode_metrics.jsonl")
        self.episode_metrics_path = resolve_experiment_artifact_path(episode_metrics_path, self.experiment_name)

    def log_scalar(self, name: str, value: float, step: int) -> None:
        self.log_dict({name: value}, step)

    def log_dict(self, stats: dict[str, Any], step: int) -> None:
        self._validate_stat_keys(stats)
        filtered_stats = self._filter_stats(stats)
        rounded_stats = self._round_payload(filtered_stats)
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "step": step,
            "stats": rounded_stats,
        }
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
        if self.writer is not None:
            for key, value in filtered_stats.items():
                if isinstance(value, Number):
                    self.writer.add_scalar(key, float(value), step)
            self.writer.flush()

    def log_episode_samples(self, samples: list[dict[str, Any]], step: int) -> None:
        if not self.sample_metrics_enabled or (step % self.sample_metrics_every_n_iterations) != 0:
            return
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "step": step,
            "samples": self._round_payload(samples),
        }
        self.episode_metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with self.episode_metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")

    def info(self, msg: str) -> None:
        timestamped = f"[{datetime.now(timezone.utc).isoformat()}] {msg}"
        print(timestamped)
        with self.info_path.open("a", encoding="utf-8") as handle:
            handle.write(timestamped + "\n")

    @staticmethod
    def _validate_stat_keys(stats: dict[str, Any]) -> None:
        invalid = [key for key in stats if "/" not in str(key).strip("/")]
        if invalid:
            raise ValueError(f"Logger stats keys must use '<module>/<metric>' format. Invalid keys: {invalid}")

    @staticmethod
    def _round_number(value: Number) -> Number:
        if isinstance(value, bool):
            return value
        float_value = float(value)
        if not math.isfinite(float_value):
            return float_value
        abs_value = abs(float_value)
        if abs_value == 0.0 or abs_value >= 1.0:
            digits = 4
        elif abs_value >= 1e-3:
            digits = 6
        else:
            digits = 8
        return round(float_value, digits)

    @classmethod
    def _round_payload(cls, payload: Any) -> Any:
        if isinstance(payload, dict):
            return {key: cls._round_payload(value) for key, value in payload.items()}
        if isinstance(payload, list):
            return [cls._round_payload(value) for value in payload]
        if isinstance(payload, tuple):
            return [cls._round_payload(value) for value in payload]
        if isinstance(payload, Number):
            return cls._round_number(payload)
        return payload

    def format_payload(self, payload: Any) -> str:
        if isinstance(payload, dict):
            payload = self._filter_stats(payload)
        return json.dumps(self._round_payload(payload), ensure_ascii=True)

    def _filter_stats(self, stats: dict[str, Any]) -> dict[str, Any]:
        if self.metric_profile == "full" and self.diagnostics_enabled:
            return dict(stats)
        return {key: value for key, value in stats.items() if self._should_keep_metric(str(key))}

    def _should_keep_metric(self, key: str) -> bool:
        if self.metric_profile == "paper":
            return self._is_paper_metric(key)
        if not self.diagnostics_enabled and self._is_diagnostic_metric(key):
            return False
        return True

    @staticmethod
    def _is_diagnostic_metric(key: str) -> bool:
        return key.startswith("system/")

    @classmethod
    def _is_paper_metric(cls, key: str) -> bool:
        if key == "timing/validation_wall_s":
            return True
        if key.startswith("validation/"):
            return cls._is_paper_metric(key[len("validation/") :])
        if key in {
            "collection/attempts_total",
            "collection/valid_episodes",
            "collection/valid_rate",
            "outcome/success_rate_live_after",
            "outcome/success_rate_dataset_before",
            "outcome/success_lift_vs_dataset",
            "outcome/drop_rate_after_given_dataset_positive",
            "outcome/hold_rate_after_given_dataset_negative",
            "reward/total_mean",
            "reward/drop_mean",
            "reward/stability_mean",
            "reward/contact_mean",
            "calibrator/prob_before_mean",
            "calibrator/prob_after_mean",
            "calibrator/prob_delta_mean",
            "calibrator/prob_delta_positive_rate",
            "calibrator/posterior_trace_snapshot",
            "calibrator/posterior_trace_post_update",
            "calibrator/after_brier",
            "ppo/policy_loss",
            "ppo/value_loss",
            "ppo/entropy",
            "ppo/total_loss",
            "ppo/approx_kl",
            "ppo/clip_fraction",
            "ppo/explained_variance",
            "timing/collect_wall_s",
            "timing/update_wall_s",
            "timing/iteration_wall_s",
        }:
            return True
        if key.startswith("outcome/trial_status_") and not key.startswith("outcome/trial_status_system_"):
            return True
        if key.startswith("contact/") and key.endswith("_mean"):
            return True
        return False
