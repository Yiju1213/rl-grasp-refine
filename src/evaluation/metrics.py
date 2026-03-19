from __future__ import annotations

import numpy as np


def compute_success_rate(results):
    return float(np.mean([result["drop_success"] for result in results])) if results else 0.0


def compute_average_reward(results):
    return float(np.mean([result["reward"] for result in results])) if results else 0.0


def compute_average_stability_gain(results):
    if not results:
        return 0.0
    gains = [
        result["calibrated_stability_after"] - result["calibrated_stability_before"]
        for result in results
    ]
    return float(np.mean(gains))
