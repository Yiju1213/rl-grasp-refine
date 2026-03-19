from __future__ import annotations

import numpy as np


class DummyVecEnvWrapper:
    """Synchronous vector environment wrapper reserved for future parallelism."""

    def __init__(self, env_fns):
        self.envs = [env_fn() for env_fn in env_fns]
        self.num_envs = len(self.envs)
        if self.num_envs == 0:
            raise ValueError("DummyVecEnvWrapper requires at least one env.")

    def reset(self):
        return [env.reset() for env in self.envs]

    def step(self, actions):
        if len(actions) != self.num_envs:
            raise ValueError(f"Expected {self.num_envs} actions, got {len(actions)}.")
        results = [env.step(action) for env, action in zip(self.envs, actions)]
        observations, rewards, dones, infos = zip(*results)
        return list(observations), np.asarray(rewards, dtype=np.float32), np.asarray(dones, dtype=bool), list(infos)

    def close(self) -> None:
        for env in self.envs:
            close_fn = getattr(env, "close", None)
            if callable(close_fn):
                close_fn()
