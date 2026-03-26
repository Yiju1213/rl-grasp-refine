from __future__ import annotations

import argparse

import torch

from _common import build_actor_critic, build_env, load_experiment_bundle, resolve_path
from src.rl.ppo_agent import PPOAgent
from src.rl.rollout_buffer import RolloutBuffer
from src.rl.trainer import Trainer
from src.rl.vec_env_wrapper import DummyVecEnvWrapper
from src.utils.checkpoint import save_checkpoint
from src.utils.logger import Logger
from src.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(description="Train the v1 single-step grasp refinement pipeline.")
    parser.add_argument(
        "--experiment",
        default="configs/experiment/exp_debug.yaml",
        help="Path to the experiment config.",
    )
    args = parser.parse_args()

    experiment_cfg, config_bundle = load_experiment_bundle(args.experiment)
    env_cfg = config_bundle["env"]
    perception_cfg = config_bundle["perception"]
    calibration_cfg = config_bundle["calibration"]
    rl_cfg = config_bundle["rl"]
    actor_critic_cfg = config_bundle["actor_critic"]

    seed = int(experiment_cfg.get("seed", env_cfg.get("seed", 0)))
    set_seed(seed)
    logger = Logger(experiment_cfg.get("logging", {}))
    shared_env, calibrator = build_env(env_cfg, perception_cfg, calibration_cfg)

    if int(rl_cfg.get("num_envs", 1)) > 1:
        shared_env.scene.close()
        num_envs = int(rl_cfg["num_envs"])
        env_fns = [
            lambda calibrator=calibrator: build_env(env_cfg, perception_cfg, calibration_cfg, calibrator=calibrator)[0]
            for _ in range(num_envs)
        ]
        env = DummyVecEnvWrapper(env_fns)
    else:
        env = shared_env

    actor_critic = build_actor_critic(perception_cfg, actor_critic_cfg)
    optimizer = torch.optim.Adam(actor_critic.parameters(), lr=float(rl_cfg.get("learning_rate", 3e-4)))
    agent = PPOAgent(actor_critic=actor_critic, optimizer=optimizer, cfg=rl_cfg)
    buffer = RolloutBuffer()
    trainer = Trainer(
        env=env,
        actor_critic=actor_critic,
        agent=agent,
        buffer=buffer,
        calibrator=calibrator,
        logger=logger,
        cfg=rl_cfg,
    )

    history = trainer.train(num_iterations=int(experiment_cfg.get("num_iterations", 1)))
    checkpoint_dir = resolve_path(experiment_cfg.get("logging", {}).get("checkpoint_dir", "outputs/default/checkpoints"))
    save_checkpoint(
        checkpoint_dir / "final.pt",
        {
            "actor_critic": actor_critic.state_dict(),
            "optimizer": optimizer.state_dict(),
            "history": history,
            "calibrator": calibrator.get_state(),
            "experiment_cfg": experiment_cfg,
        },
    )
    close_fn = getattr(env, "close", None)
    if callable(close_fn):
        close_fn()


if __name__ == "__main__":
    main()
