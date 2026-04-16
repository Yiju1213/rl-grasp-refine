from __future__ import annotations

import json
import os
import platform
import statistics
import subprocess
import sys
import time
import unittest
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_PATH = REPO_ROOT / "configs/experiment/exp_debug_stb5x_latefus_128_epi.yaml"


def _env_enabled() -> bool:
    return os.environ.get("RL_GRASP_RUN_COMPLEXITY_BENCH", "").strip() == "1"


def _env_int(name: str, default: int) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None or not raw_value.strip():
        return int(default)
    return max(int(raw_value), 1)


def _safe_run_stdout(command: list[str]) -> str | None:
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _cpu_model_name() -> str:
    stdout = _safe_run_stdout(["lscpu"])
    if stdout:
        for line in stdout.splitlines():
            if line.startswith("Model name:"):
                return line.split(":", maxsplit=1)[1].strip()
    return platform.processor() or platform.machine()


def _hardware_summary(device: torch.device) -> dict[str, Any]:
    gpu_summary: dict[str, Any] | None = None
    if device.type == "cuda" and torch.cuda.is_available():
        index = torch.cuda.current_device() if device.index is None else int(device.index)
        properties = torch.cuda.get_device_properties(index)
        gpu_summary = {
            "name": torch.cuda.get_device_name(index),
            "total_memory_mb": float(properties.total_memory / (1024.0 * 1024.0)),
            "driver_cuda_version": torch.version.cuda,
        }
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu": _cpu_model_name(),
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device": str(device),
        "gpu": gpu_summary,
    }


def _resolve_runtime_path(path_value: str | os.PathLike[str], source_root: str | os.PathLike[str] | None = None) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path.expanduser().resolve()
    if source_root is not None:
        candidate = (Path(source_root).expanduser().resolve() / path).resolve()
        if candidate.exists():
            return candidate
    return (REPO_ROOT / path).resolve()


def _assert_sgagsn_runtime_resources(runtime_cfg: dict) -> None:
    if not torch.cuda.is_available():
        raise unittest.SkipTest("CUDA is required for real SGA-GSN complexity benchmark.")
    source_root = _resolve_runtime_path(runtime_cfg["source_root"])
    required_paths = {
        "source_root": source_root,
        "config_path": _resolve_runtime_path(runtime_cfg["config_path"], source_root=source_root),
        "shape_checkpoint": _resolve_runtime_path(runtime_cfg["shape_checkpoint"], source_root=source_root),
        "grasp_checkpoint": _resolve_runtime_path(runtime_cfg["grasp_checkpoint"], source_root=source_root),
    }
    missing = {name: str(path) for name, path in required_paths.items() if not Path(path).exists()}
    if missing:
        raise unittest.SkipTest(f"Missing SGA-GSN resources: {json.dumps(missing, sort_keys=True)}")


def _load_full_bundle() -> tuple[dict, dict]:
    from src.runtime.experiment_config import apply_experiment_overrides
    from src.utils.config import load_config

    experiment_cfg = load_config(EXPERIMENT_PATH)
    bundle = {
        key: load_config(REPO_ROOT / relative_path)
        for key, relative_path in experiment_cfg.get("configs", {}).items()
    }
    return apply_experiment_overrides(deepcopy(experiment_cfg), bundle)


def _sync_cuda(device: torch.device | None) -> None:
    if device is not None and device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _memory_allocated_mb(device: torch.device) -> float:
    return float(torch.cuda.memory_allocated(device) / (1024.0 * 1024.0))


def _measure(
    fn: Callable[[], Any],
    *,
    repeats: int,
    warmup: int,
    cuda_device: torch.device | None = None,
) -> dict[str, float | None]:
    for _ in range(warmup):
        fn()
    _sync_cuda(cuda_device)

    allocated_before_mb: float | None = None
    if cuda_device is not None and cuda_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(cuda_device)
        allocated_before_mb = _memory_allocated_mb(cuda_device)

    durations_ms: list[float] = []
    for _ in range(repeats):
        _sync_cuda(cuda_device)
        start = time.perf_counter()
        fn()
        _sync_cuda(cuda_device)
        durations_ms.append(float((time.perf_counter() - start) * 1000.0))

    peak_allocated_mb: float | None = None
    peak_delta_mb: float | None = None
    if cuda_device is not None and cuda_device.type == "cuda":
        peak_allocated_mb = float(torch.cuda.max_memory_allocated(cuda_device) / (1024.0 * 1024.0))
        peak_delta_mb = float(max(peak_allocated_mb - float(allocated_before_mb or 0.0), 0.0))

    return {
        "mean_ms": float(statistics.fmean(durations_ms)),
        "std_ms": float(statistics.pstdev(durations_ms)) if len(durations_ms) > 1 else 0.0,
        "cuda_allocated_before_mb": allocated_before_mb,
        "cuda_peak_allocated_mb": peak_allocated_mb,
        "cuda_peak_delta_mb": peak_delta_mb,
    }


def _finite_nonnegative(value: Any) -> bool:
    if value is None:
        return True
    value = float(value)
    return bool(np.isfinite(value) and value >= 0.0)


def _module_row(
    *,
    module: str,
    group: str,
    input_desc: str,
    forward: dict[str, float | None] | None,
    update: dict[str, float | None] | None,
    learnable: bool,
    notes: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = {
        "module": module,
        "group": group,
        "input": input_desc,
        "forward_ms_mean": None if forward is None else forward["mean_ms"],
        "forward_ms_std": None if forward is None else forward["std_ms"],
        "update_ms_mean": None if update is None else update["mean_ms"],
        "update_ms_std": None if update is None else update["std_ms"],
        "cuda_allocated_before_mb": None if forward is None else forward["cuda_allocated_before_mb"],
        "cuda_peak_allocated_mb": None if forward is None else forward["cuda_peak_allocated_mb"],
        "cuda_peak_delta_mb": None if forward is None else forward["cuda_peak_delta_mb"],
        "learnable": bool(learnable),
        "notes": notes,
    }
    if update is not None and row["cuda_peak_delta_mb"] is None:
        row["cuda_allocated_before_mb"] = update["cuda_allocated_before_mb"]
        row["cuda_peak_allocated_mb"] = update["cuda_peak_allocated_mb"]
        row["cuda_peak_delta_mb"] = update["cuda_peak_delta_mb"]
    if extra:
        row.update(extra)
    return row


def _build_prepared_vtg_inputs(runtime_cfg: dict, *, seed: int = 7) -> PreparedVTGInputs:
    from src.perception.sga_gsn_types import PreparedVTGInputs

    rng = np.random.default_rng(seed)
    sc_points = int(runtime_cfg["sc_input_points"])
    tac_points = int(runtime_cfg["tac_points_per_side"]) * 2
    return PreparedVTGInputs(
        sc_input=rng.normal(size=(sc_points, 3)).astype(np.float32),
        gs_input=rng.normal(size=(tac_points, 4)).astype(np.float32),
        zero_mean=np.zeros(3, dtype=np.float32),
        debug_visual_world_points=np.zeros((0, 3), dtype=np.float32),
        debug_tactile_left_world_points=np.zeros((0, 3), dtype=np.float32),
        debug_tactile_right_world_points=np.zeros((0, 3), dtype=np.float32),
        debug_tactile_left_contact_world_points=np.zeros((0, 3), dtype=np.float32),
        debug_tactile_right_contact_world_points=np.zeros((0, 3), dtype=np.float32),
        debug_tactile_left_gel_mask=np.zeros((0,), dtype=bool),
        debug_tactile_right_gel_mask=np.zeros((0,), dtype=bool),
    )


def _build_raw_contact_observation(*, seed: int = 7) -> RawSensorObservation:
    from src.structures.observation import RawSensorObservation

    rng = np.random.default_rng(seed)
    contact_map = rng.uniform(low=0.0, high=0.6, size=(2, 320, 240)).astype(np.float32)
    return RawSensorObservation(
        visual_data={},
        tactile_data={"contact_map": contact_map},
        grasp_metadata={},
    )


def _build_synthetic_observations(batch_size: int, latent_dim: int, *, seed: int = 7) -> list[Observation]:
    from src.structures.action import GraspPose
    from src.structures.observation import Observation

    rng = np.random.default_rng(seed)
    observations: list[Observation] = []
    for _ in range(batch_size):
        observations.append(
            Observation(
                latent_feature=rng.normal(size=(latent_dim,)).astype(np.float32),
                contact_semantic=rng.uniform(low=0.0, high=1.0, size=(2,)).astype(np.float32),
                grasp_pose=GraspPose(
                    position=rng.normal(scale=0.01, size=(3,)).astype(np.float32),
                    rotation=rng.normal(scale=0.1, size=(3,)).astype(np.float32),
                ),
                raw_stability_logit=float(rng.normal()),
            )
        )
    return observations


@unittest.skipUnless(_env_enabled(), "Set RL_GRASP_RUN_COMPLEXITY_BENCH=1 to run real SGA-GSN complexity benchmark.")
class TestCoreModuleCostRealSGAGSN(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.experiment_cfg, cls.bundle = _load_full_bundle()
        cls.perception_cfg = cls.bundle["perception"]
        cls.calibration_cfg = cls.bundle["calibration"]
        cls.rl_cfg = cls.bundle["rl"]
        cls.actor_critic_cfg = cls.bundle["actor_critic"]
        cls.runtime_cfg = cls.perception_cfg["sga_gsn"]["runtime"]
        _assert_sgagsn_runtime_resources(cls.runtime_cfg)

    def test_core_module_cost_report(self):
        from src.calibration.online_logit_calibrator import OnlineLogitCalibrator
        from src.perception.contact_semantics import ContactSemanticsExtractor
        from src.perception.sga_gsn_runtime import get_shared_sga_gsn_runtime, infer_sga_gsn_body_feature_dim
        from src.rl.ppo_agent import PPOAgent
        from src.runtime.builders import build_actor_critic

        warmup = _env_int("RL_GRASP_COMPLEXITY_WARMUP", 5)
        repeats = _env_int("RL_GRASP_COMPLEXITY_REPEATS", 30)
        batch_size = int(self.rl_cfg.get("batch_episodes", 128))
        device = torch.device(str(self.rl_cfg.get("device", "cuda:0")))
        if device.type != "cuda":
            raise unittest.SkipTest("Core complexity benchmark expects a CUDA PPO device.")

        modules: list[dict[str, Any]] = []

        torch.cuda.empty_cache()
        _sync_cuda(device)
        perception_alloc_before_mb = _memory_allocated_mb(device)
        runtime = get_shared_sga_gsn_runtime(self.runtime_cfg)
        _sync_cuda(device)
        perception_resident_allocated_mb = max(_memory_allocated_mb(device) - perception_alloc_before_mb, 0.0)
        latent_dim = int(infer_sga_gsn_body_feature_dim(self.runtime_cfg))
        self.assertEqual(latent_dim, 576)

        prepared_inputs = _build_prepared_vtg_inputs(self.runtime_cfg)
        perception_result = runtime.run_prepared(prepared_inputs)
        self.assertEqual(tuple(perception_result.body_feature.shape), (latent_dim,))
        self.assertTrue(np.all(np.isfinite(perception_result.body_feature)))
        self.assertTrue(np.isfinite(perception_result.raw_logit))

        perception_forward = _measure(
            lambda: runtime.run_prepared(prepared_inputs),
            repeats=repeats,
            warmup=warmup,
            cuda_device=device,
        )
        modules.append(
            _module_row(
                module="Perception backbone network",
                group="inference",
                input_desc=(
                    f"PreparedVTGInputs(sc_input=({int(self.runtime_cfg['sc_input_points'])},3), "
                    f"gs_input=({int(self.runtime_cfg['tac_points_per_side']) * 2},4))"
                ),
                forward=perception_forward,
                update=None,
                learnable=False,
                notes="Frozen SGA-GSN runtime; excludes adapter, rendering, and environment interaction.",
                extra={"cuda_runtime_resident_allocated_mb": perception_resident_allocated_mb},
            )
        )

        contact_extractor = ContactSemanticsExtractor(self.perception_cfg.get("contact_semantics", {}))
        raw_contact_obs = _build_raw_contact_observation()
        contact_value = contact_extractor.extract(raw_contact_obs)
        self.assertEqual(tuple(contact_value.shape), (2,))
        self.assertTrue(np.all(np.isfinite(contact_value)))
        contact_forward = _measure(
            lambda: contact_extractor.extract(raw_contact_obs),
            repeats=repeats,
            warmup=warmup,
            cuda_device=None,
        )
        modules.append(
            _module_row(
                module="Contact semantics",
                group="inference",
                input_desc="RawSensorObservation(tactile contact_map=(2,320,240))",
                forward=contact_forward,
                update=None,
                learnable=False,
                notes="Non-parametric CPU extraction of t_cover and t_edge.",
            )
        )

        rng = np.random.default_rng(17)
        logits = rng.normal(size=(batch_size,)).astype(np.float64)
        labels = rng.integers(0, 2, size=(batch_size,)).astype(np.float64)
        calibrator = OnlineLogitCalibrator(self.calibration_cfg)
        calibrated_probs = calibrator.predict(logits)
        self.assertEqual(tuple(np.asarray(calibrated_probs).shape), (batch_size,))
        self.assertTrue(np.all(np.isfinite(calibrated_probs)))
        calibrator.update(logits, labels)
        calibrator_state = calibrator.get_state()
        self.assertEqual(tuple(np.asarray(calibrator_state["posterior_cov"]).shape), (2, 2))
        self.assertTrue(np.all(np.isfinite(np.asarray(calibrator_state["posterior_cov"]))))

        calibration_forward = _measure(
            lambda: calibrator.predict(logits),
            repeats=repeats,
            warmup=warmup,
            cuda_device=None,
        )
        calibration_update = _measure(
            lambda: calibrator.update(logits, labels),
            repeats=repeats,
            warmup=warmup,
            cuda_device=None,
        )
        modules.append(
            _module_row(
                module="Online calibration",
                group="training_update",
                input_desc=f"logits/labels batch=({batch_size},)",
                forward=calibration_forward,
                update=calibration_update,
                learnable=True,
                notes="Two-parameter online logistic calibration with Laplace covariance.",
            )
        )

        actor_critic = build_actor_critic(self.perception_cfg, self.actor_critic_cfg)
        actor_critic.to(device)
        actor_critic.train()
        optimizer = torch.optim.Adam(actor_critic.parameters(), lr=float(self.rl_cfg.get("learning_rate", 3e-4)))
        agent = PPOAgent(actor_critic=actor_critic, optimizer=optimizer, cfg=self.rl_cfg)
        observation_spec = getattr(actor_critic, "observation_spec", None)
        self.assertIsNotNone(observation_spec)
        self.assertEqual(int(observation_spec.latent_dim), latent_dim)
        self.assertEqual(tuple(observation_spec.components), ("latent_feature", "contact_semantic"))
        self.assertEqual(int(observation_spec.obs_dim), latent_dim + 2)

        observations = _build_synthetic_observations(batch_size, latent_dim)
        obs_tensor = torch.randn(batch_size, int(observation_spec.obs_dim), dtype=torch.float32, device=device)
        with torch.no_grad():
            action, log_prob, value, entropy = actor_critic.act(obs_tensor, deterministic=True)
        self.assertEqual(tuple(action.shape), (batch_size, 6))
        self.assertEqual(tuple(log_prob.shape), (batch_size,))
        self.assertEqual(tuple(value.shape), (batch_size,))
        self.assertEqual(tuple(entropy.shape), (batch_size,))

        action_np = rng.uniform(low=-0.75, high=0.75, size=(batch_size, 6)).astype(np.float32)
        values_np = rng.normal(size=(batch_size,)).astype(np.float32)
        batch = {
            "obs": observations,
            "actions": action_np,
            "log_probs": rng.normal(scale=0.1, size=(batch_size,)).astype(np.float32),
            "returns": rng.normal(size=(batch_size,)).astype(np.float32),
            "advantages": rng.normal(size=(batch_size,)).astype(np.float32),
            "values": values_np,
        }
        ppo_stats = agent.update(batch)
        self.assertTrue(ppo_stats)
        for value_item in ppo_stats.values():
            self.assertTrue(np.isfinite(float(value_item)))

        actor_forward = _measure(
            lambda: actor_critic.act(obs_tensor, deterministic=True),
            repeats=repeats,
            warmup=warmup,
            cuda_device=device,
        )
        ppo_update = _measure(
            lambda: agent.update(batch),
            repeats=repeats,
            warmup=warmup,
            cuda_device=device,
        )
        modules.append(
            _module_row(
                module="Actor-critic update (PPO)",
                group="training_update",
                input_desc=f"Full late-fusion obs batch=({batch_size},{int(observation_spec.obs_dim)})",
                forward=actor_forward,
                update=ppo_update,
                learnable=True,
                notes="Full policy/value MLP update on latent-feature inputs; excludes rollout collection.",
            )
        )

        report = {
            "hardware": _hardware_summary(device),
            "config": {
                "experiment": str(EXPERIMENT_PATH),
                "batch_size": batch_size,
                "latent_dim": latent_dim,
                "contact_dim": 2,
                "ppo_update_epochs": int(self.rl_cfg.get("update_epochs", 4)),
                "ppo_minibatch_size": int(self.rl_cfg.get("minibatch_size", 16)),
                "warmup": warmup,
                "repeats": repeats,
            },
            "modules": modules,
        }

        for module in modules:
            for key in (
                "forward_ms_mean",
                "forward_ms_std",
                "update_ms_mean",
                "update_ms_std",
                "cuda_allocated_before_mb",
                "cuda_peak_allocated_mb",
                "cuda_peak_delta_mb",
            ):
                self.assertTrue(_finite_nonnegative(module[key]), msg=f"{module['module']} {key}={module[key]}")
            if module["module"] == "Perception backbone network":
                self.assertTrue(_finite_nonnegative(module["cuda_runtime_resident_allocated_mb"]))

        report_json = json.dumps(report, indent=2, sort_keys=True)
        print("\n[core-module-cost-report]")
        print(report_json)

        report_path_raw = os.environ.get("RL_GRASP_COMPLEXITY_REPORT_PATH", "").strip()
        if report_path_raw:
            report_path = Path(report_path_raw).expanduser().resolve()
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(report_json + "\n", encoding="utf-8")
            self.assertTrue(report_path.exists())


if __name__ == "__main__":
    unittest.main()
