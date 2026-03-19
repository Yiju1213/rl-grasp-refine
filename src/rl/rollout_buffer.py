from __future__ import annotations

import numpy as np

from src.structures.action import NormalizedAction
from src.structures.transition import Transition


class RolloutBuffer:
    """Simple rollout storage for PPO and calibrator updates."""

    def __init__(self):
        self.clear()

    def add(
        self,
        obs,
        action,
        reward,
        next_obs,
        done,
        log_prob,
        value,
        info,
        raw_logit_before,
        raw_logit_after,
    ):
        if not isinstance(action, NormalizedAction):
            action = NormalizedAction(value=np.asarray(action, dtype=np.float32))
        transition = Transition(
            obs=obs,
            action=action,
            reward=float(reward),
            next_obs=next_obs,
            done=bool(done),
            log_prob=float(log_prob),
            value=float(value),
            info=info,
        )
        self.transitions.append(transition)
        self.raw_logit_before.append(float(raw_logit_before))
        self.raw_logit_after.append(float(raw_logit_after))

    def get_all(self) -> dict:
        return {
            "transitions": list(self.transitions),
            "obs": [transition.obs for transition in self.transitions],
            "actions": np.stack([transition.action.value for transition in self.transitions], axis=0)
            if self.transitions
            else np.zeros((0, 6), dtype=np.float32),
            "rewards": np.asarray([transition.reward for transition in self.transitions], dtype=np.float32),
            "next_obs": [transition.next_obs for transition in self.transitions],
            "dones": np.asarray([transition.done for transition in self.transitions], dtype=np.float32),
            "log_probs": np.asarray([transition.log_prob for transition in self.transitions], dtype=np.float32),
            "values": np.asarray([transition.value for transition in self.transitions], dtype=np.float32),
            "infos": [transition.info for transition in self.transitions],
            "raw_logit_before": np.asarray(self.raw_logit_before, dtype=np.float32),
            "raw_logit_after": np.asarray(self.raw_logit_after, dtype=np.float32),
        }

    def clear(self):
        self.transitions: list[Transition] = []
        self.raw_logit_before: list[float] = []
        self.raw_logit_after: list[float] = []
