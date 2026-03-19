from __future__ import annotations

import numpy as np


def compute_returns_and_advantages(rewards, values, dones, gamma: float, lam: float):
    """Compute discounted returns and GAE advantages."""

    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    dones = np.asarray(dones, dtype=np.float32)

    returns = np.zeros_like(rewards, dtype=np.float32)
    advantages = np.zeros_like(rewards, dtype=np.float32)
    gae = 0.0
    next_value = 0.0

    for index in range(len(rewards) - 1, -1, -1):
        mask = 1.0 - dones[index]
        delta = rewards[index] + gamma * next_value * mask - values[index]
        gae = delta + gamma * lam * mask * gae
        advantages[index] = gae
        returns[index] = advantages[index] + values[index]
        next_value = values[index]

    return returns, advantages
