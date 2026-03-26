from __future__ import annotations

import unittest

import numpy as np
import torch

from src.rl.rollout_buffer import RolloutBuffer
from src.rl.trainer import Trainer
from src.structures.action import GraspPose, NormalizedAction
from src.structures.info import StepInfo
from src.structures.observation import Observation


class _StubActorCritic:
    def act(self, obs_tensor):
        batch = obs_tensor.shape[0]
        action = torch.zeros(batch, 6)
        log_prob = torch.zeros(batch)
        value = torch.zeros(batch)
        return action, log_prob, value, {}


class _StubAgent:
    def update(self, batch):
        return {"loss": 0.0}


class _StubCalibrator:
    def __init__(self):
        self.updated = False

    def update(self, logits, labels):
        self.updated = True


class _StubLogger:
    def log_dict(self, stats: dict, step: int):
        return None

    def info(self, msg: str):
        return None


class _InvalidThenValidEnv:
    def __init__(self):
        self.reset_calls = 0
        self.step_calls = 0

    def reset(self):
        self.reset_calls += 1
        return Observation(
            latent_feature=np.zeros(32, dtype=np.float32),
            contact_semantic=np.zeros(2, dtype=np.float32),
            grasp_pose=GraspPose(position=np.zeros(3), rotation=np.zeros(3)),
            raw_stability_logit=0.0,
        )

    def step(self, action: NormalizedAction):
        self.step_calls += 1
        trial_metadata = {
            "trial_status": "system_invalid_observation" if self.step_calls == 1 else "success",
            "release_executed": False,
            "valid_for_learning": self.step_calls != 1,
            "failure_reason": None,
            "runtime_counters": {},
        }
        info = StepInfo(
            drop_success=1 if self.step_calls != 1 else 0,
            calibrated_stability_before=0.5,
            calibrated_stability_after=0.5,
            uncertainty_before=0.1,
            uncertainty_after=0.1,
            reward_drop=1.0,
            reward_stability=0.0,
            reward_contact=0.0,
            extra={
                "reward_breakdown": None,
                "raw_logit_before": 0.0,
                "raw_logit_after": 0.0,
                "trial_metadata": trial_metadata,
            },
        )
        next_obs = Observation(
            latent_feature=np.zeros(32, dtype=np.float32),
            contact_semantic=np.zeros(2, dtype=np.float32),
            grasp_pose=GraspPose(position=np.zeros(3), rotation=np.zeros(3)),
            raw_stability_logit=0.0,
        )
        return next_obs, 1.0, True, info


class TestTrainerInvalidTransition(unittest.TestCase):
    def test_collect_rollout_skips_invalid_learning_samples(self):
        trainer = Trainer(
            env=_InvalidThenValidEnv(),
            actor_critic=_StubActorCritic(),
            agent=_StubAgent(),
            buffer=RolloutBuffer(),
            calibrator=_StubCalibrator(),
            logger=_StubLogger(),
            cfg={"batch_episodes": 1, "device": "cpu", "max_collect_attempt_factor": 4},
        )

        trainer.collect_rollout(1)
        batch = trainer.buffer.get_all()

        self.assertEqual(len(batch["infos"]), 1)
        self.assertEqual(trainer.env.step_calls, 2)


if __name__ == "__main__":
    unittest.main()
