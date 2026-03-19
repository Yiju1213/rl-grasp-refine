from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(path: str | Path, state: dict[str, Any]) -> None:
    """Save a checkpoint dictionary with torch serialization."""

    resolved_path = Path(path).expanduser().resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, resolved_path)


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load a checkpoint dictionary."""

    resolved_path = Path(path).expanduser().resolve()
    return torch.load(resolved_path, map_location="cpu")
