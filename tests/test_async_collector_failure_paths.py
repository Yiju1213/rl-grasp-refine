from __future__ import annotations

import unittest

import numpy as np

from src.calibration.online_logit_calibrator import OnlineLogitCalibrator
from src.rl.observation_spec import resolve_policy_observation_spec
from src.rl.subproc_async_rollout_collector import SubprocAsyncRolloutCollector
from src.structures.action import GraspPose, NormalizedAction
from src.structures.observation import Observation
from tests.fakes import make_actor_critic_cfg, make_calibration_cfg, make_perception_cfg, make_rl_cfg


class _ExplodingEnv:
    def __init__(self, calibration_cfg: dict):
        self.calibrator = OnlineLogitCalibrator(calibration_cfg)

    def reset(self):
        return Observation(
            latent_feature=np.zeros(32, dtype=np.float32),
            contact_semantic=np.zeros(2, dtype=np.float32),
            grasp_pose=GraspPose(position=np.zeros(3, dtype=np.float32), rotation=np.zeros(3, dtype=np.float32)),
            raw_stability_logit=0.0,
        )

    def step(self, action: NormalizedAction):
        raise RuntimeError("intentional worker failure")

    def sync_calibrator(self, state: dict) -> None:
        self.calibrator.load_state(state)

    def close(self) -> None:
        return None


def build_exploding_env_for_worker(
    env_cfg: dict,
    perception_cfg: dict,
    calibration_cfg: dict,
    worker_id: int | None = None,
    num_workers: int | None = None,
    worker_seed: int | None = None,
):
    del env_cfg, perception_cfg, worker_id, num_workers, worker_seed
    return _ExplodingEnv(calibration_cfg)


class TestAsyncCollectorFailurePaths(unittest.TestCase):
    def test_worker_failures_surface_and_processes_close(self):
        env_cfg = {"seed": 1}
        perception_cfg = make_perception_cfg()
        calibration_cfg = make_calibration_cfg()
        actor_critic_cfg = make_actor_critic_cfg()
        rl_cfg = make_rl_cfg()
        rl_cfg["num_envs"] = 3
        spec = resolve_policy_observation_spec(perception_cfg, actor_critic_cfg)

        collector = SubprocAsyncRolloutCollector(
            env_cfg=env_cfg,
            perception_cfg=perception_cfg,
            calibration_cfg=calibration_cfg,
            actor_critic_cfg=actor_critic_cfg,
            rl_cfg=rl_cfg,
            num_workers=3,
            observation_spec=spec,
            env_factory=build_exploding_env_for_worker,
        )
        try:
            from src.runtime.builders import build_actor_critic

            actor_critic = build_actor_critic(perception_cfg, actor_critic_cfg, observation_spec=spec)
            actor_state = {key: value.detach().cpu() for key, value in actor_critic.state_dict().items()}
            calibrator = OnlineLogitCalibrator(calibration_cfg)
            with self.assertRaisesRegex(RuntimeError, "intentional worker failure"):
                collector.collect_batch(
                    target_valid_episodes=1,
                    actor_state=actor_state,
                    calibrator_state=calibrator.get_state(),
                    obs_spec=spec,
                    rollout_version=0,
                )
        finally:
            collector.close()

        self.assertTrue(all(not process.is_alive() for process in collector._processes))


if __name__ == "__main__":
    unittest.main()
