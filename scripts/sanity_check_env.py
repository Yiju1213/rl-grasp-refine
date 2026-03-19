from __future__ import annotations

import argparse
import json

import numpy as np

from _common import build_env, load_experiment_bundle
from src.structures.action import NormalizedAction


def main():
    parser = argparse.ArgumentParser(description="Run a single random action through the environment.")
    parser.add_argument("--experiment", default="configs/experiment/exp_debug.yaml")
    args = parser.parse_args()

    _, config_bundle = load_experiment_bundle(args.experiment)
    env_cfg = config_bundle["env"]
    perception_cfg = config_bundle["perception"]
    calibration_cfg = config_bundle["calibration"]
    env, _ = build_env(env_cfg, perception_cfg, calibration_cfg)

    obs = env.reset()
    action = NormalizedAction(value=np.random.uniform(-1.0, 1.0, size=6).astype(np.float32))
    next_obs, reward, done, info = env.step(action)
    payload = {
        "obs_before_logit": obs.raw_stability_logit,
        "obs_after_logit": next_obs.raw_stability_logit,
        "reward": reward,
        "done": done,
        "drop_success": info.drop_success,
        "reward_breakdown": info.extra["reward_breakdown"].as_dict(),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    env.scene.close()


if __name__ == "__main__":
    main()
