from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from numbers import Number
from typing import Any

from torch.utils.tensorboard import SummaryWriter


class Logger:
    """Experiment logger with JSONL, TensorBoard, and optional sample dumps."""

    def __init__(self, cfg: dict[str, Any]):
        log_dir = Path(cfg.get("log_dir", "outputs/default")).resolve()
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = log_dir
        self.metrics_path = log_dir / "metrics.jsonl"
        self.info_path = log_dir / "run.log"
        tensorboard_cfg = dict(cfg.get("tensorboard", {}))
        self.tensorboard_enabled = bool(tensorboard_cfg.get("enabled", True))
        self.tensorboard_dir = Path(tensorboard_cfg.get("dir", log_dir / "tensorboard")).resolve()
        self.writer = SummaryWriter(log_dir=str(self.tensorboard_dir)) if self.tensorboard_enabled else None
        sample_cfg = dict(cfg.get("sample_metrics", {}))
        self.sample_metrics_enabled = bool(sample_cfg.get("enabled", False))
        self.sample_metrics_every_n_iterations = max(int(sample_cfg.get("every_n_iterations", 10)), 1)
        self.episode_metrics_path = Path(sample_cfg.get("path", log_dir / "episode_metrics.jsonl")).resolve()

    def log_scalar(self, name: str, value: float, step: int) -> None:
        self.log_dict({name: value}, step)

    def log_dict(self, stats: dict[str, Any], step: int) -> None:
        self._validate_stat_keys(stats)
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "step": step,
            "stats": stats,
        }
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
        if self.writer is not None:
            for key, value in stats.items():
                if isinstance(value, Number):
                    self.writer.add_scalar(key, float(value), step)
            self.writer.flush()

    def log_episode_samples(self, samples: list[dict[str, Any]], step: int) -> None:
        if not self.sample_metrics_enabled or (step % self.sample_metrics_every_n_iterations) != 0:
            return
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "step": step,
            "samples": samples,
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
