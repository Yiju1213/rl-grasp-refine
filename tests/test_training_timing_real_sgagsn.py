from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.calibration.online_logit_calibrator import OnlineLogitCalibrator
from src.rl.advantage import compute_returns_and_advantages
from src.rl.ppo_agent import PPOAgent
from src.rl.rollout_buffer import RolloutBuffer
from src.rl.subproc_async_rollout_collector import SubprocAsyncRolloutCollector
from src.rl.trainer import Trainer
from src.runtime.builders import build_actor_critic
from tests.fakes import DummyLogger
from tests.runtime_smoke_utils import (
    TIMING_KEYS,
    assert_real_sgagsn_resources,
    build_instrumented_real_env_for_worker,
    build_real_training_bundle,
    configure_headless_render_env,
)


def _perf_counter() -> float:
    import time

    return time.perf_counter()


class TestTrainingTimingRealSGAGSN(unittest.TestCase):
    def test_training_records_real_three_worker_module_timings(self):
        assert_real_sgagsn_resources()
        configure_headless_render_env()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir)
            experiment_cfg, bundle = build_real_training_bundle(
                output_root=output_root,
                num_envs=3,
                batch_episodes=9,
                num_iterations=3,
                device="cuda:0",
                worker_policy_device="cuda:0",
            )

            env_cfg = bundle["env"]
            perception_cfg = bundle["perception"]
            calibration_cfg = bundle["calibration"]
            rl_cfg = bundle["rl"]
            actor_critic_cfg = bundle["actor_critic"]

            calibrator = OnlineLogitCalibrator(calibration_cfg)
            actor_critic = build_actor_critic(perception_cfg, actor_critic_cfg)
            optimizer = torch.optim.Adam(actor_critic.parameters(), lr=float(rl_cfg.get("learning_rate", 3e-4)))
            agent = PPOAgent(actor_critic=actor_critic, optimizer=optimizer, cfg=rl_cfg)
            collector = SubprocAsyncRolloutCollector(
                env_cfg=env_cfg,
                perception_cfg=perception_cfg,
                calibration_cfg=calibration_cfg,
                actor_critic_cfg=actor_critic_cfg,
                rl_cfg=rl_cfg,
                num_workers=3,
                observation_spec=getattr(actor_critic, "observation_spec", None),
                env_factory=build_instrumented_real_env_for_worker,
            )
            trainer = Trainer(
                env=None,
                actor_critic=actor_critic,
                agent=agent,
                buffer=RolloutBuffer(),
                calibrator=calibrator,
                logger=DummyLogger(),
                cfg=rl_cfg,
                collector=collector,
            )

            main_timings = {
                "collect_batch_total_s": [],
                "ppo_update_s": [],
                "calibrator_update_s": [],
                "iteration_total_s": [],
            }
            history = []
            try:
                for iteration in range(int(experiment_cfg["num_iterations"])):
                    trainer.iteration = iteration
                    iteration_start = _perf_counter()

                    collect_start = _perf_counter()
                    collection_report = trainer.collect_rollout(trainer.batch_episodes)
                    main_timings["collect_batch_total_s"].append(_perf_counter() - collect_start)

                    batch = trainer.buffer.get_all()
                    returns, advantages = compute_returns_and_advantages(
                        rewards=batch["rewards"],
                        values=batch["values"],
                        dones=batch["dones"],
                        gamma=trainer.gamma,
                        lam=trainer.lam,
                    )
                    batch["returns"] = returns
                    batch["advantages"] = advantages

                    update_start = _perf_counter()
                    training_stats = trainer.agent.update(batch)
                    main_timings["ppo_update_s"].append(_perf_counter() - update_start)

                    calibrator_start = _perf_counter()
                    calibrator_state = trainer.update_calibrator(batch)
                    main_timings["calibrator_update_s"].append(_perf_counter() - calibrator_start)

                    rollout_stats = trainer._summarize_rollout(
                        batch=batch,
                        collection_report=collection_report,
                        calibrator_post_state=calibrator_state,
                        timing_stats={
                            "timing/collect_wall_s": float(main_timings["collect_batch_total_s"][-1]),
                            "timing/update_wall_s": float(
                                main_timings["ppo_update_s"][-1] + main_timings["calibrator_update_s"][-1]
                            ),
                            "timing/iteration_wall_s": float(_perf_counter() - iteration_start),
                        },
                    )
                    trainer.buffer.clear()
                    history.append({**training_stats, **rollout_stats})
                    main_timings["iteration_total_s"].append(_perf_counter() - iteration_start)

                debug_states = collector.get_worker_debug_states()
            finally:
                collector.close()

            timing_payload = {
                "main": {
                    key: [float(value) for value in values]
                    for key, values in main_timings.items()
                },
                "workers": {
                    str(state["worker_id"]): state["debug_snapshot"]["timings"]
                    for state in debug_states
                },
                "history": history,
            }
            timing_path = output_root / "timings.json"
            timing_path.write_text(json.dumps(timing_payload, indent=2, sort_keys=True), encoding="utf-8")

            self.assertTrue(timing_path.exists())
            self.assertEqual(len(main_timings["iteration_total_s"]), 3)
            self.assertGreaterEqual(sum(int(item["collection/valid_episodes"]) for item in history), 27)

            for values in main_timings.values():
                self.assertEqual(len(values), 3)
                for value in values:
                    self.assertTrue(np.isfinite(float(value)))
                    self.assertGreaterEqual(float(value), 0.0)

            self.assertEqual(sorted(timing_payload["workers"].keys()), ["0", "1", "2"])
            for worker_summary in timing_payload["workers"].values():
                for key in TIMING_KEYS:
                    self.assertIn(key, worker_summary)
                    for metric_name in ("count", "total_s", "mean_s", "max_s"):
                        value = float(worker_summary[key][metric_name])
                        self.assertTrue(np.isfinite(value))
                        self.assertGreaterEqual(value, 0.0)


if __name__ == "__main__":
    unittest.main()
