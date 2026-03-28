from __future__ import annotations

import argparse
import json

from _common import build_actor_critic, build_env, load_experiment_bundle, maybe_load_actor_critic
from src.evaluation.evaluator import Evaluator
from src.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(description="Evaluate the v1 grasp refinement policy.")
    parser.add_argument("--experiment", default="configs/experiment/exp_debug.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--episodes", type=int, default=20)
    args = parser.parse_args()

    experiment_cfg, config_bundle = load_experiment_bundle(args.experiment)
    env_cfg = config_bundle["env"]
    perception_cfg = config_bundle["perception"]
    calibration_cfg = config_bundle["calibration"]
    actor_critic_cfg = config_bundle["actor_critic"]
    set_seed(int(experiment_cfg.get("seed", 0)))

    env, _ = build_env(env_cfg, perception_cfg, calibration_cfg)
    actor_critic = build_actor_critic(perception_cfg, actor_critic_cfg)
    actor_critic = maybe_load_actor_critic(actor_critic, args.checkpoint)
    evaluator = Evaluator(env=env, actor_critic=actor_critic, cfg=experiment_cfg)
    results = evaluator.run(num_episodes=args.episodes)
    print(json.dumps(results, indent=2, sort_keys=True))
    env.scene.close()


if __name__ == "__main__":
    main()
