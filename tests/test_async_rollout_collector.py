from __future__ import annotations

import unittest

import numpy as np

from src.calibration.online_logit_calibrator import OnlineLogitCalibrator
from src.rl.observation_spec import resolve_policy_observation_spec
from src.rl.subproc_async_rollout_collector import SubprocAsyncRolloutCollector
from tests.fakes import (
    build_async_delay_env_for_worker,
    build_test_env_for_worker,
    make_actor_critic_cfg,
    make_calibration_cfg,
    make_env_cfg,
    make_perception_cfg,
    make_rl_cfg,
)


class TestAsyncRolloutCollector(unittest.TestCase):
    def test_collect_batch_smoke_with_late_fusion_actor_critic(self):
        env_cfg = make_env_cfg()
        perception_cfg = make_perception_cfg()
        calibration_cfg = make_calibration_cfg()
        actor_critic_cfg = make_actor_critic_cfg()
        actor_critic_cfg["architecture"] = {"type": "latent_first_late_fusion"}
        actor_critic_cfg["policy_observation"] = {"preset": "paper"}
        rl_cfg = make_rl_cfg()
        rl_cfg["num_envs"] = 2
        spec = resolve_policy_observation_spec(perception_cfg, actor_critic_cfg)

        collector = SubprocAsyncRolloutCollector(
            env_cfg=env_cfg,
            perception_cfg=perception_cfg,
            calibration_cfg=calibration_cfg,
            actor_critic_cfg=actor_critic_cfg,
            rl_cfg=rl_cfg,
            num_workers=2,
            observation_spec=spec,
            env_factory=build_test_env_for_worker,
        )
        calibrator = OnlineLogitCalibrator(calibration_cfg)
        try:
            from src.runtime.builders import build_actor_critic

            actor_critic = build_actor_critic(perception_cfg, actor_critic_cfg, observation_spec=spec)
            actor_state = {key: value.detach().cpu() for key, value in actor_critic.state_dict().items()}
            payload = collector.collect_batch(
                target_valid_episodes=3,
                actor_state=actor_state,
                calibrator_state=calibrator.get_state(),
                obs_spec=spec,
                rollout_version=7,
            )
            self.assertEqual(len(payload["transitions"]), 3)
            self.assertEqual(payload["valid_episodes"], 3)
            self.assertEqual(payload["rollout_version"], 7)
        finally:
            collector.close()

    def test_collect_batch_smoke_with_three_workers(self):
        env_cfg = make_env_cfg()
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
            env_factory=build_test_env_for_worker,
        )
        calibrator = OnlineLogitCalibrator(calibration_cfg)
        try:
            actor_state = {}
            from src.runtime.builders import build_actor_critic

            actor_critic = build_actor_critic(perception_cfg, actor_critic_cfg, observation_spec=spec)
            actor_state = {key: value.detach().cpu() for key, value in actor_critic.state_dict().items()}
            payload = collector.collect_batch(
                target_valid_episodes=4,
                actor_state=actor_state,
                calibrator_state=calibrator.get_state(),
                obs_spec=spec,
                rollout_version=3,
            )
            self.assertEqual(len(payload["transitions"]), 4)
            self.assertEqual(payload["valid_episodes"], 4)
            self.assertEqual(payload["rollout_version"], 3)
            self.assertTrue({item["worker_id"] for item in payload["transitions"]}.issubset({0, 1, 2}))
            self.assertTrue(all(item["rollout_version"] == 3 for item in payload["transitions"]))
            self.assertEqual(len(payload["attempt_summaries"]), payload["attempts_total"])
            self.assertTrue(all("trial_status" in item for item in payload["attempt_summaries"]))
        finally:
            collector.close()

    def test_collect_batch_is_async_and_syncs_versions_for_three_workers(self):
        env_cfg = make_env_cfg()
        env_cfg["delay_schedules"] = {
            0: [0.0, 0.0, 0.0],
            1: [0.02, 0.02, 0.02],
            2: [0.15, 0.15, 0.15],
        }
        env_cfg["invalid_attempts"] = {0: 1}
        perception_cfg = make_perception_cfg()
        calibration_cfg = make_calibration_cfg()
        actor_critic_cfg = make_actor_critic_cfg()
        rl_cfg = make_rl_cfg()
        rl_cfg["num_envs"] = 3
        rl_cfg["max_collect_attempt_factor"] = 6
        spec = resolve_policy_observation_spec(perception_cfg, actor_critic_cfg)

        collector = SubprocAsyncRolloutCollector(
            env_cfg=env_cfg,
            perception_cfg=perception_cfg,
            calibration_cfg=calibration_cfg,
            actor_critic_cfg=actor_critic_cfg,
            rl_cfg=rl_cfg,
            num_workers=3,
            observation_spec=spec,
            env_factory=build_async_delay_env_for_worker,
        )
        try:
            from src.runtime.builders import build_actor_critic

            actor_critic = build_actor_critic(perception_cfg, actor_critic_cfg, observation_spec=spec)
            actor_state = {key: value.detach().cpu() for key, value in actor_critic.state_dict().items()}
            calibrator = OnlineLogitCalibrator(calibration_cfg)
            calibrator.load_state(
                {
                    "a": 1.7,
                    "b": -0.2,
                    "posterior_cov": np.asarray([[1.2, 0.0], [0.0, 0.8]], dtype=np.float64),
                }
            )

            payload = collector.collect_batch(
                target_valid_episodes=4,
                actor_state=actor_state,
                calibrator_state=calibrator.get_state(),
                obs_spec=spec,
                rollout_version=5,
            )
            worker_ids = [item["worker_id"] for item in payload["transitions"]]
            self.assertEqual(len(payload["transitions"]), 4)
            self.assertEqual(payload["valid_episodes"], 4)
            self.assertGreaterEqual(worker_ids.count(0), 2)
            self.assertTrue(all(item["worker_device"] == "cpu" for item in payload["transitions"]))
            self.assertTrue(all(item["rollout_version"] == 5 for item in payload["transitions"]))
            self.assertGreaterEqual(payload["attempts_total"], 4)
            self.assertEqual(len(payload["attempt_summaries"]), payload["attempts_total"])
            self.assertTrue(any(not item["valid_for_learning"] for item in payload["attempt_summaries"]))

            debug_states = collector.get_worker_debug_states()
            self.assertEqual([item["worker_id"] for item in debug_states], [0, 1, 2])
            for state in debug_states:
                self.assertEqual(state["rollout_version"], 5)
                self.assertAlmostEqual(state["calibrator_state"]["a"], 1.7, places=7)
                self.assertAlmostEqual(state["calibrator_state"]["b"], -0.2, places=7)
        finally:
            collector.close()

    def test_collect_batch_rebuilds_worker_scenes_from_second_batch(self):
        env_cfg = make_env_cfg()
        perception_cfg = make_perception_cfg()
        calibration_cfg = make_calibration_cfg()
        actor_critic_cfg = make_actor_critic_cfg()
        rl_cfg = make_rl_cfg()
        rl_cfg["num_envs"] = 2
        rl_cfg["scene_rebuild_every_n_iterations"] = 1
        spec = resolve_policy_observation_spec(perception_cfg, actor_critic_cfg)

        collector = SubprocAsyncRolloutCollector(
            env_cfg=env_cfg,
            perception_cfg=perception_cfg,
            calibration_cfg=calibration_cfg,
            actor_critic_cfg=actor_critic_cfg,
            rl_cfg=rl_cfg,
            num_workers=2,
            observation_spec=spec,
            env_factory=build_async_delay_env_for_worker,
        )
        try:
            from src.runtime.builders import build_actor_critic

            actor_critic = build_actor_critic(perception_cfg, actor_critic_cfg, observation_spec=spec)
            actor_state = {key: value.detach().cpu() for key, value in actor_critic.state_dict().items()}
            calibrator = OnlineLogitCalibrator(calibration_cfg)

            first_payload = collector.collect_batch(
                target_valid_episodes=2,
                actor_state=actor_state,
                calibrator_state=calibrator.get_state(),
                obs_spec=spec,
                rollout_version=0,
            )
            second_payload = collector.collect_batch(
                target_valid_episodes=2,
                actor_state=actor_state,
                calibrator_state=calibrator.get_state(),
                obs_spec=spec,
                rollout_version=1,
            )
            self.assertEqual(first_payload["scene_rebuild_performed"], 0)
            self.assertEqual(first_payload["scene_rebuild_workers"], 0)
            self.assertEqual(first_payload["scene_rebuild_wall_s"], 0.0)
            self.assertEqual(second_payload["scene_rebuild_performed"], 1)
            self.assertEqual(second_payload["scene_rebuild_workers"], 2)
            self.assertGreaterEqual(second_payload["scene_rebuild_wall_s"], 0.0)

            debug_states = collector.get_worker_debug_states()
            for state in debug_states:
                self.assertEqual(state["debug_snapshot"]["scene_rebuild_count"], 1)
                self.assertEqual(state["debug_snapshot"]["scene_generation"], 1)
                self.assertEqual(state["rollout_version"], 1)
        finally:
            collector.close()

    def test_collect_batch_rebuilds_every_n_iterations(self):
        env_cfg = make_env_cfg()
        perception_cfg = make_perception_cfg()
        calibration_cfg = make_calibration_cfg()
        actor_critic_cfg = make_actor_critic_cfg()
        rl_cfg = make_rl_cfg()
        rl_cfg["num_envs"] = 2
        rl_cfg["scene_rebuild_every_n_iterations"] = 2
        spec = resolve_policy_observation_spec(perception_cfg, actor_critic_cfg)

        collector = SubprocAsyncRolloutCollector(
            env_cfg=env_cfg,
            perception_cfg=perception_cfg,
            calibration_cfg=calibration_cfg,
            actor_critic_cfg=actor_critic_cfg,
            rl_cfg=rl_cfg,
            num_workers=2,
            observation_spec=spec,
            env_factory=build_async_delay_env_for_worker,
        )
        try:
            from src.runtime.builders import build_actor_critic

            actor_critic = build_actor_critic(perception_cfg, actor_critic_cfg, observation_spec=spec)
            actor_state = {key: value.detach().cpu() for key, value in actor_critic.state_dict().items()}
            calibrator = OnlineLogitCalibrator(calibration_cfg)

            payload_0 = collector.collect_batch(
                target_valid_episodes=2,
                actor_state=actor_state,
                calibrator_state=calibrator.get_state(),
                obs_spec=spec,
                rollout_version=0,
            )
            payload_1 = collector.collect_batch(
                target_valid_episodes=2,
                actor_state=actor_state,
                calibrator_state=calibrator.get_state(),
                obs_spec=spec,
                rollout_version=1,
            )
            debug_states_after_first_recycle = {
                int(item["worker_id"]): int(item["worker_generation"])
                for item in collector.get_worker_debug_states()
            }
            payload_2 = collector.collect_batch(
                target_valid_episodes=2,
                actor_state=actor_state,
                calibrator_state=calibrator.get_state(),
                obs_spec=spec,
                rollout_version=2,
            )

            self.assertEqual(payload_0["scene_rebuild_performed"], 0)
            self.assertEqual(payload_1["scene_rebuild_performed"], 0)
            self.assertEqual(payload_2["scene_rebuild_performed"], 1)

            debug_states = collector.get_worker_debug_states()
            for state in debug_states:
                self.assertEqual(state["debug_snapshot"]["scene_rebuild_count"], 1)
                self.assertEqual(state["debug_snapshot"]["scene_generation"], 1)
                self.assertEqual(state["rollout_version"], 2)
        finally:
            collector.close()

    def test_collect_batch_recycles_oldest_worker_slots_fifo_with_prefetch(self):
        env_cfg = make_env_cfg()
        perception_cfg = make_perception_cfg()
        calibration_cfg = make_calibration_cfg()
        actor_critic_cfg = make_actor_critic_cfg()
        rl_cfg = make_rl_cfg()
        rl_cfg["num_envs"] = 2
        rl_cfg["worker_recycle_every_n_iterations"] = 1
        rl_cfg["worker_recycle_slots_per_event"] = 1
        rl_cfg["worker_recycle_enable_standby_prefetch"] = True
        rl_cfg["worker_recycle_prefetch_count"] = 1
        spec = resolve_policy_observation_spec(perception_cfg, actor_critic_cfg)

        collector = SubprocAsyncRolloutCollector(
            env_cfg=env_cfg,
            perception_cfg=perception_cfg,
            calibration_cfg=calibration_cfg,
            actor_critic_cfg=actor_critic_cfg,
            rl_cfg=rl_cfg,
            num_workers=2,
            observation_spec=spec,
            env_factory=build_async_delay_env_for_worker,
        )
        try:
            from src.runtime.builders import build_actor_critic

            actor_critic = build_actor_critic(perception_cfg, actor_critic_cfg, observation_spec=spec)
            actor_state = {key: value.detach().cpu() for key, value in actor_critic.state_dict().items()}
            calibrator = OnlineLogitCalibrator(calibration_cfg)

            payload_0 = collector.collect_batch(
                target_valid_episodes=2,
                actor_state=actor_state,
                calibrator_state=calibrator.get_state(),
                obs_spec=spec,
                rollout_version=0,
            )
            payload_1 = collector.collect_batch(
                target_valid_episodes=2,
                actor_state=actor_state,
                calibrator_state=calibrator.get_state(),
                obs_spec=spec,
                rollout_version=1,
            )
            debug_states_after_first_recycle = {
                int(item["worker_id"]): int(item["worker_generation"])
                for item in collector.get_worker_debug_states()
            }
            payload_2 = collector.collect_batch(
                target_valid_episodes=2,
                actor_state=actor_state,
                calibrator_state=calibrator.get_state(),
                obs_spec=spec,
                rollout_version=2,
            )

            self.assertEqual(payload_0["worker_recycle_performed"], 0)
            self.assertEqual(payload_0["worker_recycle_prefetched"], 1)
            self.assertEqual(payload_1["worker_recycle_performed"], 1)
            self.assertEqual(payload_1["worker_recycle_slots"], 1)
            self.assertEqual(payload_1["worker_recycle_prefetched"], 1)
            self.assertEqual(debug_states_after_first_recycle[0], 1)
            self.assertEqual(debug_states_after_first_recycle[1], 0)
            self.assertEqual(payload_2["worker_recycle_performed"], 1)
            self.assertEqual(payload_2["worker_recycle_slots"], 1)

            debug_states_after_second_recycle = {
                int(item["worker_id"]): int(item["worker_generation"])
                for item in collector.get_worker_debug_states()
            }
            self.assertEqual(debug_states_after_second_recycle[0], 1)
            self.assertEqual(debug_states_after_second_recycle[1], 1)
        finally:
            collector.close()

    def test_collect_batch_waits_for_prefetched_worker_to_be_ready(self):
        env_cfg = make_env_cfg()
        env_cfg["startup_delays"] = {"0:1": 0.5}
        perception_cfg = make_perception_cfg()
        calibration_cfg = make_calibration_cfg()
        actor_critic_cfg = make_actor_critic_cfg()
        rl_cfg = make_rl_cfg()
        rl_cfg["num_envs"] = 2
        rl_cfg["worker_recycle_every_n_iterations"] = 1
        rl_cfg["worker_recycle_slots_per_event"] = 1
        rl_cfg["worker_recycle_enable_standby_prefetch"] = True
        rl_cfg["worker_recycle_prefetch_count"] = 1
        spec = resolve_policy_observation_spec(perception_cfg, actor_critic_cfg)

        collector = SubprocAsyncRolloutCollector(
            env_cfg=env_cfg,
            perception_cfg=perception_cfg,
            calibration_cfg=calibration_cfg,
            actor_critic_cfg=actor_critic_cfg,
            rl_cfg=rl_cfg,
            num_workers=2,
            observation_spec=spec,
            env_factory=build_async_delay_env_for_worker,
        )
        try:
            from src.runtime.builders import build_actor_critic

            actor_critic = build_actor_critic(perception_cfg, actor_critic_cfg, observation_spec=spec)
            actor_state = {key: value.detach().cpu() for key, value in actor_critic.state_dict().items()}
            calibrator = OnlineLogitCalibrator(calibration_cfg)

            payload_0 = collector.collect_batch(
                target_valid_episodes=1,
                actor_state=actor_state,
                calibrator_state=calibrator.get_state(),
                obs_spec=spec,
                rollout_version=0,
            )
            payload_1 = collector.collect_batch(
                target_valid_episodes=1,
                actor_state=actor_state,
                calibrator_state=calibrator.get_state(),
                obs_spec=spec,
                rollout_version=1,
            )

            self.assertEqual(payload_0["worker_recycle_prefetched"], 1)
            self.assertEqual(payload_0["worker_recycle_prefetch_ready"], 0)
            self.assertEqual(payload_1["worker_recycle_performed"], 1)
            self.assertGreater(payload_1["worker_recycle_wait_ready_wall_s"], 0.1)
        finally:
            collector.close()

    def test_collect_batch_recycles_before_scene_rebuild(self):
        env_cfg = make_env_cfg()
        perception_cfg = make_perception_cfg()
        calibration_cfg = make_calibration_cfg()
        actor_critic_cfg = make_actor_critic_cfg()
        rl_cfg = make_rl_cfg()
        rl_cfg["num_envs"] = 2
        rl_cfg["scene_rebuild_every_n_iterations"] = 1
        rl_cfg["worker_recycle_every_n_iterations"] = 1
        rl_cfg["worker_recycle_slots_per_event"] = 1
        rl_cfg["worker_recycle_enable_standby_prefetch"] = True
        rl_cfg["worker_recycle_prefetch_count"] = 1
        spec = resolve_policy_observation_spec(perception_cfg, actor_critic_cfg)

        collector = SubprocAsyncRolloutCollector(
            env_cfg=env_cfg,
            perception_cfg=perception_cfg,
            calibration_cfg=calibration_cfg,
            actor_critic_cfg=actor_critic_cfg,
            rl_cfg=rl_cfg,
            num_workers=2,
            observation_spec=spec,
            env_factory=build_async_delay_env_for_worker,
        )
        try:
            from src.runtime.builders import build_actor_critic

            actor_critic = build_actor_critic(perception_cfg, actor_critic_cfg, observation_spec=spec)
            actor_state = {key: value.detach().cpu() for key, value in actor_critic.state_dict().items()}
            calibrator = OnlineLogitCalibrator(calibration_cfg)

            collector.collect_batch(
                target_valid_episodes=2,
                actor_state=actor_state,
                calibrator_state=calibrator.get_state(),
                obs_spec=spec,
                rollout_version=0,
            )
            payload_1 = collector.collect_batch(
                target_valid_episodes=2,
                actor_state=actor_state,
                calibrator_state=calibrator.get_state(),
                obs_spec=spec,
                rollout_version=1,
            )

            self.assertEqual(payload_1["worker_recycle_performed"], 1)
            self.assertEqual(payload_1["scene_rebuild_performed"], 1)
            self.assertEqual(payload_1["scene_rebuild_workers"], 1)

            debug_states = {int(item["worker_id"]): item for item in collector.get_worker_debug_states()}
            self.assertEqual(int(debug_states[0]["worker_generation"]), 1)
            self.assertEqual(int(debug_states[0]["debug_snapshot"]["scene_rebuild_count"]), 0)
            self.assertEqual(int(debug_states[1]["worker_generation"]), 0)
            self.assertEqual(int(debug_states[1]["debug_snapshot"]["scene_rebuild_count"]), 1)
        finally:
            collector.close()


if __name__ == "__main__":
    unittest.main()
