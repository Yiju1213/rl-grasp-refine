from __future__ import annotations

import argparse
import json

from _common import build_actor_critic, build_env, load_experiment_bundle
from src.utils.tensor_utils import observation_to_tensor


def main():
    parser = argparse.ArgumentParser(description="Debug a single policy rollout end-to-end.")
    parser.add_argument("--experiment", default="configs/experiment/exp_debug.yaml")
    args = parser.parse_args()

    _, config_bundle = load_experiment_bundle(args.experiment)
    env_cfg = config_bundle["env"]
    perception_cfg = config_bundle["perception"]
    calibration_cfg = config_bundle["calibration"]
    actor_critic_cfg = config_bundle["actor_critic"]

    env, calibrator = build_env(env_cfg, perception_cfg, calibration_cfg)
    actor_critic = build_actor_critic(perception_cfg, actor_critic_cfg)

    obs_before = env.reset()
    obs_tensor = observation_to_tensor(obs_before, spec=getattr(actor_critic, "observation_spec", None))
    action_tensor, _, _, _ = actor_critic.act(obs_tensor, deterministic=True)
    action = action_tensor.squeeze(0).detach().cpu().numpy()
    obs_after, reward, done, info = env.step(action)
    payload = {
        "before_logit": obs_before.raw_stability_logit,
        "after_logit": obs_after.raw_stability_logit,
        "calibrated_before": info.calibrated_stability_before,
        "calibrated_after": info.calibrated_stability_after,
        "posterior_trace": info.posterior_trace,
        "reward": reward,
        "reward_breakdown": info.extra["reward_breakdown"].as_dict(),
        "drop_success": info.drop_success,
        "calibrator_state": calibrator.get_state(),
        "done": done,
    }
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    env.scene.close()


if __name__ == "__main__":
    main()
