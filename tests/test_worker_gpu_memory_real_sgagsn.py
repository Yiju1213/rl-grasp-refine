from __future__ import annotations

import json
import os
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
    GpuMemorySampler,
    assert_real_sgagsn_resources,
    build_real_training_bundle,
    configure_headless_render_env,
)


def _worker_count_series() -> tuple[int, ...]:
    raw_value = os.environ.get("RL_GRASP_GPU_PROFILE_WORKERS", "1,2,3,4")
    counts = tuple(int(item.strip()) for item in raw_value.split(",") if item.strip())
    if not counts:
        raise AssertionError("RL_GRASP_GPU_PROFILE_WORKERS resolved to an empty worker-count list.")
    return counts


class TestWorkerGpuMemoryRealSGAGSN(unittest.TestCase):
    def test_peak_gpu_memory_across_real_worker_counts(self):
        assert_real_sgagsn_resources()
        configure_headless_render_env()
        worker_counts = _worker_count_series()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir)
            profile_results = []

            for num_workers in worker_counts:
                experiment_cfg, bundle = build_real_training_bundle(
                    output_root=output_root / f"workers_{num_workers}",
                    num_envs=num_workers,
                    batch_episodes=9,
                    num_iterations=1,
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
                actor_critic.to(torch.device("cuda:0"))
                optimizer = torch.optim.Adam(actor_critic.parameters(), lr=float(rl_cfg.get("learning_rate", 3e-4)))
                agent = PPOAgent(actor_critic=actor_critic, optimizer=optimizer, cfg=rl_cfg)

                collector_holder: dict[str, SubprocAsyncRolloutCollector | None] = {"collector": None}

                def pid_provider():
                    pids = {os.getpid()}
                    collector = collector_holder["collector"]
                    if collector is not None:
                        pids.update(process.pid for process in collector._processes if process.pid is not None)
                    return pids

                collector = None
                trainer = None
                try:
                    with GpuMemorySampler(pid_provider, interval_s=0.2) as sampler:
                        collector = SubprocAsyncRolloutCollector(
                            env_cfg=env_cfg,
                            perception_cfg=perception_cfg,
                            calibration_cfg=calibration_cfg,
                            actor_critic_cfg=actor_critic_cfg,
                            rl_cfg=rl_cfg,
                            num_workers=num_workers,
                            observation_spec=getattr(actor_critic, "observation_spec", None),
                        )
                        collector_holder["collector"] = collector
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
                        trainer.iteration = 0
                        trainer.collect_rollout(trainer.batch_episodes)

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
                        trainer.agent.update(batch)
                        trainer.update_calibrator()
                        trainer.buffer.clear()
                        debug_states = collector.get_worker_debug_states()
                finally:
                    if collector is not None:
                        collector.close()
                    del trainer
                    del agent
                    del optimizer
                    del actor_critic
                    torch.cuda.empty_cache()

                self.assertEqual(len(debug_states), num_workers)
                self.assertTrue(all(state["device"] == "cuda:0" for state in debug_states))
                self.assertGreater(sampler.total_peak_mb, 0)

                profile_results.append(
                    {
                        "num_workers": num_workers,
                        "global_baseline_used_mb": int(sampler.global_baseline_used_mb),
                        "global_peak_used_mb": int(sampler.global_peak_used_mb),
                        "global_peak_delta_mb": int(sampler.total_peak_mb),
                        "compute_pid_peak_mb": {str(pid): int(value) for pid, value in sampler.per_pid_peak_mb.items()},
                    }
                )

            report = {
                "gpu_name": torch.cuda.get_device_name(0),
                "worker_counts": list(worker_counts),
                "results": profile_results,
            }
            report_path = output_root / "worker_gpu_memory_profile.json"
            report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

            self.assertTrue(report_path.exists())
            self.assertEqual([item["num_workers"] for item in profile_results], list(worker_counts))
            for item in profile_results:
                self.assertTrue(np.isfinite(float(item["global_peak_delta_mb"])))
                self.assertGreater(float(item["global_peak_delta_mb"]), 0.0)


if __name__ == "__main__":
    unittest.main()
