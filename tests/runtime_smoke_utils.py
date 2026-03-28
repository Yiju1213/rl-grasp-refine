from __future__ import annotations

import json
import os
import subprocess
import threading
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import yaml

from src.runtime.builders import build_env as runtime_build_env
from src.runtime.experiment_config import apply_experiment_overrides
from src.utils.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_EXPERIMENT_PATH = REPO_ROOT / "configs/experiment/exp_debug.yaml"
TIMING_KEYS = (
    "env_reset_total_s",
    "policy_forward_s",
    "env_step_total_s",
    "obs_build_total_s",
    "feature_encode_s",
    "contact_semantic_s",
    "stability_predict_s",
    "calibrator_predict_total_s",
    "reward_compute_s",
)


def _perf_counter() -> float:
    import time

    return time.perf_counter()


def assert_real_sgagsn_resources() -> None:
    configure_headless_render_env()
    experiment_cfg = load_config(BASE_EXPERIMENT_PATH)
    env_cfg = load_config(REPO_ROOT / experiment_cfg["configs"]["env"])
    perception_cfg = load_config(REPO_ROOT / experiment_cfg["configs"]["perception"])
    runtime_cfg = perception_cfg.get("sga_gsn", {}).get("runtime", {})

    if not torch.cuda.is_available():
        raise AssertionError("CUDA is required for real SGA-GSN smoke tests, but no CUDA device is available.")

    dataset_root = Path(env_cfg.get("dataset", {}).get("dataset_root", "")).expanduser().resolve()
    if not dataset_root.exists():
        raise AssertionError(f"Dataset root does not exist: {dataset_root}")

    required_paths = {
        "source_root": runtime_cfg.get("source_root"),
        "config_path": runtime_cfg.get("config_path"),
        "shape_checkpoint": runtime_cfg.get("shape_checkpoint"),
        "grasp_checkpoint": runtime_cfg.get("grasp_checkpoint"),
    }
    missing = [name for name, path in required_paths.items() if not path or not Path(path).expanduser().resolve().exists()]
    if missing:
        missing_payload = {name: required_paths[name] for name in missing}
        raise AssertionError(f"Missing required SGA-GSN resources: {json.dumps(missing_payload, ensure_ascii=True)}")


def configure_headless_render_env() -> dict[str, str]:
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    return {
        "PYOPENGL_PLATFORM": os.environ["PYOPENGL_PLATFORM"],
    }


def query_compute_gpu_memory_by_pid(pids: set[int] | None = None) -> dict[int, int]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    pid_filter = None if pids is None else {int(pid) for pid in pids}
    usage_by_pid: dict[int, int] = {}
    for raw_line in completed.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        pid_str, mem_str = [part.strip() for part in line.split(",", maxsplit=1)]
        pid = int(pid_str)
        if pid_filter is not None and pid not in pid_filter:
            continue
        usage_by_pid[pid] = int(mem_str)
    return usage_by_pid


def query_total_gpu_memory_used_mb() -> int:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    first_line = next((line.strip() for line in completed.stdout.splitlines() if line.strip()), "")
    return int(first_line) if first_line else 0


class GpuMemorySampler:
    def __init__(self, pid_provider, interval_s: float = 0.2):
        self._pid_provider = pid_provider
        self._interval_s = float(interval_s)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.per_pid_peak_mb: dict[int, int] = {}
        self.total_peak_mb = 0
        self.global_baseline_used_mb = 0
        self.global_peak_used_mb = 0

    def _sample_once(self) -> None:
        global_used_mb = query_total_gpu_memory_used_mb()
        if self.global_peak_used_mb == 0 and self.global_baseline_used_mb == 0:
            self.global_baseline_used_mb = global_used_mb
        if global_used_mb > self.global_peak_used_mb:
            self.global_peak_used_mb = global_used_mb
        global_delta_mb = max(global_used_mb - self.global_baseline_used_mb, 0)
        if global_delta_mb > self.total_peak_mb:
            self.total_peak_mb = global_delta_mb

        pids = {int(pid) for pid in self._pid_provider() if pid is not None}
        if not pids:
            return
        snapshot = query_compute_gpu_memory_by_pid(pids)
        for pid, used_mb in snapshot.items():
            used_mb = int(used_mb)
            previous_peak = self.per_pid_peak_mb.get(pid, 0)
            if used_mb > previous_peak:
                self.per_pid_peak_mb[pid] = used_mb

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self._sample_once()
            self._stop_event.wait(self._interval_s)

    def __enter__(self):
        self._sample_once()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(self._interval_s * 4.0, 1.0))
        self._sample_once()


def load_real_training_bundle() -> tuple[dict, dict]:
    experiment_cfg = load_config(BASE_EXPERIMENT_PATH)
    bundle = {
        key: load_config(REPO_ROOT / relative_path)
        for key, relative_path in experiment_cfg.get("configs", {}).items()
    }
    return apply_experiment_overrides(deepcopy(experiment_cfg), bundle)


def build_real_training_bundle(
    *,
    output_root: Path,
    num_envs: int,
    batch_episodes: int,
    num_iterations: int,
    device: str,
    worker_policy_device: str,
) -> tuple[dict, dict]:
    experiment_cfg, bundle = load_real_training_bundle()
    bundle["perception"]["adapter_type"] = "sga_gsn"
    bundle["rl"]["num_envs"] = int(num_envs)
    bundle["rl"]["batch_episodes"] = int(batch_episodes)
    bundle["rl"]["device"] = str(device)
    bundle["rl"]["worker_policy_device"] = str(worker_policy_device)
    experiment_cfg["num_iterations"] = int(num_iterations)
    experiment_cfg.setdefault("logging", {})
    experiment_cfg["logging"]["log_dir"] = str((output_root / "logs").resolve())
    experiment_cfg["logging"]["checkpoint_dir"] = str((output_root / "logs" / "checkpoints").resolve())
    experiment_cfg["logging"].setdefault("tensorboard", {})
    experiment_cfg["logging"]["tensorboard"]["dir"] = str((output_root / "logs" / "tensorboard").resolve())
    experiment_cfg["logging"].setdefault("sample_metrics", {})
    experiment_cfg["logging"]["sample_metrics"]["path"] = str((output_root / "logs" / "episode_metrics.jsonl").resolve())
    return experiment_cfg, bundle


def write_temp_experiment_bundle(
    output_root: Path,
    *,
    num_envs: int,
    batch_episodes: int,
    num_iterations: int,
    device: str,
    worker_policy_device: str,
) -> Path:
    experiment_cfg, bundle = build_real_training_bundle(
        output_root=output_root,
        num_envs=num_envs,
        batch_episodes=batch_episodes,
        num_iterations=num_iterations,
        device=device,
        worker_policy_device=worker_policy_device,
    )
    output_root.mkdir(parents=True, exist_ok=True)

    config_paths: dict[str, str] = {}
    for key, config in bundle.items():
        path = output_root / f"{key}.yaml"
        with path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)
        config_paths[key] = str(path.resolve())

    experiment_cfg = deepcopy(experiment_cfg)
    experiment_cfg["configs"] = config_paths
    experiment_path = output_root / "experiment.yaml"
    with experiment_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(experiment_cfg, handle, sort_keys=False)
    return experiment_path.resolve()


def summarize_timing_records(records: dict[str, list[float]]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for key in TIMING_KEYS:
        values = [float(value) for value in records.get(key, [])]
        count = len(values)
        total = float(sum(values))
        mean = float(total / count) if count else 0.0
        max_value = float(max(values)) if values else 0.0
        summary[key] = {
            "count": float(count),
            "total_s": total,
            "mean_s": mean,
            "max_s": max_value,
        }
    return summary


class InstrumentedRealEnvProxy:
    def __init__(self, env):
        self._env = env
        self._timing_records: dict[str, list[float]] = defaultdict(list)
        self._install_timing_wrappers()

    def _install_timing_wrappers(self) -> None:
        observation_builder = self._env.observation_builder
        feature_extractor = observation_builder.feature_extractor
        contact_semantics_extractor = observation_builder.contact_semantics_extractor
        stability_predictor = observation_builder.stability_predictor
        reward_manager = self._env.reward_manager
        calibrator = self._env.calibrator

        original_build = observation_builder.build
        original_encode = feature_extractor.encode
        original_contact_extract = contact_semantics_extractor.extract
        original_predict_logit = stability_predictor.predict_logit
        original_reward_compute = reward_manager.compute
        original_calibrator_predict = calibrator.predict

        def timed_build(raw_obs, grasp_pose):
            start = _perf_counter()
            try:
                return original_build(raw_obs, grasp_pose)
            finally:
                self.record_timing("obs_build_total_s", _perf_counter() - start)

        def timed_encode(raw_obs):
            start = _perf_counter()
            try:
                return original_encode(raw_obs)
            finally:
                self.record_timing("feature_encode_s", _perf_counter() - start)

        def timed_contact_extract(raw_obs):
            start = _perf_counter()
            try:
                return original_contact_extract(raw_obs)
            finally:
                self.record_timing("contact_semantic_s", _perf_counter() - start)

        def timed_predict_logit(latent_feature):
            start = _perf_counter()
            try:
                return original_predict_logit(latent_feature)
            finally:
                self.record_timing("stability_predict_s", _perf_counter() - start)

        def timed_reward_compute(*args, **kwargs):
            start = _perf_counter()
            try:
                return original_reward_compute(*args, **kwargs)
            finally:
                self.record_timing("reward_compute_s", _perf_counter() - start)

        def timed_calibrator_predict(*args, **kwargs):
            start = _perf_counter()
            try:
                return original_calibrator_predict(*args, **kwargs)
            finally:
                self.record_timing("calibrator_predict_total_s", _perf_counter() - start)

        observation_builder.build = timed_build
        feature_extractor.encode = timed_encode
        contact_semantics_extractor.extract = timed_contact_extract
        stability_predictor.predict_logit = timed_predict_logit
        reward_manager.compute = timed_reward_compute
        calibrator.predict = timed_calibrator_predict

    def record_timing(self, name: str, duration_s: float) -> None:
        self._timing_records[str(name)].append(float(duration_s))

    def reset(self):
        start = _perf_counter()
        try:
            return self._env.reset()
        finally:
            self.record_timing("env_reset_total_s", _perf_counter() - start)

    def step(self, action):
        start = _perf_counter()
        try:
            return self._env.step(action)
        finally:
            self.record_timing("env_step_total_s", _perf_counter() - start)

    def sync_calibrator(self, state: dict) -> None:
        return self._env.sync_calibrator(state)

    def close(self) -> None:
        return self._env.close()

    def get_debug_snapshot(self) -> dict[str, Any]:
        base_snapshot = {}
        get_debug_snapshot = getattr(self._env, "get_debug_snapshot", None)
        if callable(get_debug_snapshot):
            base_snapshot = get_debug_snapshot()
        return {
            "env": base_snapshot,
            "timings": summarize_timing_records(self._timing_records),
        }

    def __getattr__(self, item):
        return getattr(self._env, item)


def build_instrumented_real_env_for_worker(
    env_cfg: dict,
    perception_cfg: dict,
    calibration_cfg: dict,
    worker_id: int | None = None,
    num_workers: int | None = None,
    worker_seed: int | None = None,
    worker_generation: int | None = None,
):
    env, _ = runtime_build_env(
        env_cfg=env_cfg,
        perception_cfg=perception_cfg,
        calibration_cfg=calibration_cfg,
        worker_id=worker_id,
        num_workers=num_workers,
        worker_seed=worker_seed,
        worker_generation=worker_generation,
    )
    return InstrumentedRealEnvProxy(env)
