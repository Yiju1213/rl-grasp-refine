from __future__ import annotations

import unittest

from src.rl.observation_spec import resolve_policy_observation_spec
from src.rl.rollout_buffer import RolloutBuffer
from src.rl.subproc_async_rollout_collector import SubprocAsyncRolloutCollector
from src.rl.trainer import Trainer
from src.utils.tensor_utils import observation_to_tensor
from tests.fakes import (
    DummyLogger,
    build_async_delay_env_for_worker,
    build_test_actor_critic,
    build_test_env,
    make_actor_critic_cfg,
    make_calibration_cfg,
    make_env_cfg,
    make_perception_cfg,
    make_rl_cfg,
)


class TestTrainerValidation(unittest.TestCase):
    def test_run_validation_with_async_collector_emits_prefixed_metrics(self):
        env, calibrator, _, _, _ = build_test_env()
        obs_dim = observation_to_tensor(env.reset()).shape[-1]
        actor_critic, _ = build_test_actor_critic(obs_dim)

        env_cfg = make_env_cfg()
        perception_cfg = make_perception_cfg()
        calibration_cfg = make_calibration_cfg()
        actor_critic_cfg = make_actor_critic_cfg()
        rl_cfg = make_rl_cfg()
        rl_cfg["num_envs"] = 2
        spec = resolve_policy_observation_spec(perception_cfg, actor_critic_cfg)
        validation_collector = SubprocAsyncRolloutCollector(
            env_cfg=env_cfg,
            perception_cfg=perception_cfg,
            calibration_cfg=calibration_cfg,
            actor_critic_cfg=actor_critic_cfg,
            rl_cfg=rl_cfg,
            num_workers=2,
            observation_spec=spec,
            env_factory=build_async_delay_env_for_worker,
        )
        trainer = Trainer(
            env=env,
            actor_critic=actor_critic,
            agent=None,
            buffer=RolloutBuffer(),
            calibrator=calibrator,
            logger=DummyLogger(),
            cfg={"batch_episodes": 2, "device": "cpu", "max_collect_attempt_factor": 4},
            validation_collector=validation_collector,
            validation_cfg={"enabled": True, "every_n_iterations": 1, "num_episodes": 2},
        )

        try:
            trainer.iteration = 0
            validation_stats, validation_wall_s = trainer.run_validation(calibrator_state=calibrator.get_state())
            self.assertGreaterEqual(validation_wall_s, 0.0)
            self.assertIn("validation/collection/attempts_total", validation_stats)
            self.assertIn("validation/outcome/success_lift_vs_dataset", validation_stats)
            self.assertIn("validation/reward/total_mean", validation_stats)
            self.assertIn("validation/timing/collect_wall_s", validation_stats)
        finally:
            validation_collector.close()

    def test_run_validation_with_async_collector_supports_worker_recycle(self):
        env, calibrator, _, _, _ = build_test_env()
        obs_dim = observation_to_tensor(env.reset()).shape[-1]
        actor_critic, _ = build_test_actor_critic(obs_dim)

        env_cfg = make_env_cfg()
        perception_cfg = make_perception_cfg()
        calibration_cfg = make_calibration_cfg()
        actor_critic_cfg = make_actor_critic_cfg()
        rl_cfg = make_rl_cfg()
        rl_cfg["num_envs"] = 3
        rl_cfg["worker_recycle_every_n_iterations"] = 1
        rl_cfg["worker_recycle_slots_per_event"] = 1
        rl_cfg["worker_recycle_enable_standby_prefetch"] = True
        rl_cfg["worker_recycle_prefetch_count"] = 1
        spec = resolve_policy_observation_spec(perception_cfg, actor_critic_cfg)
        validation_collector = SubprocAsyncRolloutCollector(
            env_cfg=env_cfg,
            perception_cfg=perception_cfg,
            calibration_cfg=calibration_cfg,
            actor_critic_cfg=actor_critic_cfg,
            rl_cfg=rl_cfg,
            num_workers=3,
            observation_spec=spec,
            env_factory=build_async_delay_env_for_worker,
        )
        trainer = Trainer(
            env=env,
            actor_critic=actor_critic,
            agent=None,
            buffer=RolloutBuffer(),
            calibrator=calibrator,
            logger=DummyLogger(),
            cfg={"batch_episodes": 3, "device": "cpu", "max_collect_attempt_factor": 4},
            validation_collector=validation_collector,
            validation_cfg={"enabled": True, "every_n_iterations": 1, "num_episodes": 3},
        )

        try:
            trainer.iteration = 0
            validation_stats_0, _ = trainer.run_validation(calibrator_state=calibrator.get_state())
            self.assertEqual(validation_stats_0["validation/collection/worker_recycle_performed"], 0.0)
            self.assertEqual(validation_stats_0["validation/collection/worker_recycle_prefetched"], 1.0)
            process_states_0 = validation_collector.get_worker_process_states()
            self.assertEqual(len(process_states_0), 4)
            self.assertEqual(sum(1 for state in process_states_0 if state["role"] == "active"), 3)
            self.assertEqual(sum(1 for state in process_states_0 if state["role"] == "standby"), 1)

            trainer.iteration = 1
            validation_stats_1, _ = trainer.run_validation(calibrator_state=calibrator.get_state())
            self.assertEqual(validation_stats_1["validation/collection/worker_recycle_performed"], 1.0)
            self.assertEqual(validation_stats_1["validation/collection/worker_recycle_slots"], 1.0)
            self.assertEqual(validation_stats_1["validation/collection/worker_recycle_prefetched"], 1.0)
            self.assertEqual(
                validation_stats_0["validation/outcome/success_lift_vs_dataset"],
                validation_stats_1["validation/outcome/success_lift_vs_dataset"],
            )
            self.assertEqual(
                validation_stats_0["validation/reward/total_mean"],
                validation_stats_1["validation/reward/total_mean"],
            )
            process_states_1 = validation_collector.get_worker_process_states()
            self.assertEqual(len(process_states_1), 4)
            self.assertEqual(sum(1 for state in process_states_1 if state["role"] == "active"), 3)
            self.assertEqual(sum(1 for state in process_states_1 if state["role"] == "standby"), 1)
        finally:
            validation_collector.close()


if __name__ == "__main__":
    unittest.main()
