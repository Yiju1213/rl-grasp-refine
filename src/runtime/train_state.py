from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from src.utils.checkpoint import load_checkpoint


def resolve_remaining_training_iterations(*, target_iterations: int, start_iteration: int) -> int:
    """Return how many iterations remain before reaching the configured global target."""
    target_iterations = int(target_iterations)
    start_iteration = int(start_iteration)
    if target_iterations < 0:
        raise ValueError("'num_iterations' must be non-negative.")
    if start_iteration < 0:
        raise ValueError("'start_iteration' must be non-negative.")
    return max(target_iterations - start_iteration, 0)


def move_optimizer_state_to_device(optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def restore_training_state(
    *,
    checkpoint_path: str | Path,
    actor_critic,
    optimizer,
    calibrator,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint = load_checkpoint(checkpoint_path)
    actor_critic.load_state_dict(checkpoint["actor_critic"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    move_optimizer_state_to_device(optimizer, device)

    calibrator_state = checkpoint.get("calibrator")
    load_state = getattr(calibrator, "load_state", None)
    if calibrator_state is not None and callable(load_state):
        load_state(calibrator_state)

    history = list(checkpoint.get("history", []))
    completed_iterations = int(checkpoint.get("completed_iterations", len(history)))
    return {
        "checkpoint": checkpoint,
        "history": history,
        "completed_iterations": completed_iterations,
    }
