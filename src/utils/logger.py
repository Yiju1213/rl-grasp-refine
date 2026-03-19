from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class Logger:
    """Small JSONL + stdout logger for experiments."""

    def __init__(self, cfg: dict[str, Any]):
        log_dir = Path(cfg.get("log_dir", "outputs/default")).resolve()
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = log_dir
        self.metrics_path = log_dir / "metrics.jsonl"
        self.info_path = log_dir / "run.log"

    def log_scalar(self, name: str, value: float, step: int) -> None:
        self.log_dict({name: value}, step)

    def log_dict(self, stats: dict[str, Any], step: int) -> None:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "step": step,
            "stats": stats,
        }
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")

    def info(self, msg: str) -> None:
        timestamped = f"[{datetime.now(timezone.utc).isoformat()}] {msg}"
        print(timestamped)
        with self.info_path.open("a", encoding="utf-8") as handle:
            handle.write(timestamped + "\n")
