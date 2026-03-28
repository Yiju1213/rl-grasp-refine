from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

import torch


_BYTES_PER_MB = 1024.0 * 1024.0
_NVIDIA_SMI_USABLE: bool | None = None
_PROC_STATUS_KEYS = {
    "VmRSS": "system/process_rss_mb",
    "VmHWM": "system/process_rss_peak_mb",
    "VmSize": "system/process_vms_mb",
    "VmSwap": "system/process_swap_mb",
    "RssAnon": "system/process_rss_anon_mb",
    "RssFile": "system/process_rss_file_mb",
    "RssShmem": "system/process_rss_shmem_mb",
}
_CGROUP_FILE_KEYS = {
    "/sys/fs/cgroup/memory.current": "system/cgroup_memory_current_mb",
    "/sys/fs/cgroup/memory.swap.current": "system/cgroup_memory_swap_current_mb",
}
_CGROUP_STAT_KEYS = {
    "anon": "system/cgroup_memory_anon_mb",
    "file": "system/cgroup_memory_file_mb",
    "kernel": "system/cgroup_memory_kernel_mb",
    "slab": "system/cgroup_memory_slab_mb",
    "shmem": "system/cgroup_memory_shmem_mb",
}


def _bytes_to_mb(value: int) -> float:
    return float(value) / _BYTES_PER_MB


def _read_int_file(path: str | Path) -> int | None:
    try:
        raw = Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not raw or raw == "max":
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _run_nvidia_smi(args: list[str]) -> str | None:
    global _NVIDIA_SMI_USABLE
    if _NVIDIA_SMI_USABLE is False:
        return None
    try:
        completed = subprocess.run(
            ["nvidia-smi", *args],
            capture_output=True,
            text=True,
            check=True,
            timeout=2.0,
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        _NVIDIA_SMI_USABLE = False
        return None
    _NVIDIA_SMI_USABLE = True
    return completed.stdout


def read_process_memory_stats(pid: int | None = None) -> dict[str, float]:
    target_pid = int(os.getpid() if pid is None else pid)
    status_path = Path(f"/proc/{target_pid}/status")
    try:
        lines = status_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}

    stats: dict[str, float] = {}
    for line in lines:
        key, _, value = line.partition(":")
        metric_name = _PROC_STATUS_KEYS.get(key.strip())
        if metric_name is None:
            continue
        tokens = value.strip().split()
        if not tokens:
            continue
        try:
            value_kib = int(tokens[0])
        except ValueError:
            continue
        stats[metric_name] = float(value_kib) / 1024.0
    return stats


def read_cgroup_memory_stats() -> dict[str, float]:
    stats: dict[str, float] = {}
    for path, metric_name in _CGROUP_FILE_KEYS.items():
        value = _read_int_file(path)
        if value is not None:
            stats[metric_name] = _bytes_to_mb(value)

    stat_path = Path("/sys/fs/cgroup/memory.stat")
    try:
        lines = stat_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return stats

    for line in lines:
        key, _, value = line.partition(" ")
        metric_name = _CGROUP_STAT_KEYS.get(key.strip())
        if metric_name is None:
            continue
        try:
            stats[metric_name] = _bytes_to_mb(int(value.strip()))
        except ValueError:
            continue
    return stats


def query_total_gpu_memory_used_mb() -> float | None:
    stdout = _run_nvidia_smi(["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
    if stdout is None:
        return None
    values: list[int] = []
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            values.append(int(line))
        except ValueError:
            continue
    if not values:
        return None
    return float(sum(values))


def query_compute_gpu_memory_by_pid(pids: set[int] | None = None) -> dict[int, float]:
    stdout = _run_nvidia_smi(
        [
            "--query-compute-apps=pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    if stdout is None:
        return {}

    pid_filter = None if pids is None else {int(pid) for pid in pids}
    usage_by_pid: dict[int, float] = {}
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        pid_str, sep, mem_str = line.partition(",")
        if not sep:
            continue
        try:
            pid = int(pid_str.strip())
            used_mb = float(mem_str.strip())
        except ValueError:
            continue
        if pid_filter is not None and pid not in pid_filter:
            continue
        usage_by_pid[pid] = used_mb
    return usage_by_pid


def read_torch_cuda_memory_stats(device: str | torch.device | None) -> dict[str, float]:
    if device is None:
        return {}
    device_obj = torch.device(device)
    if device_obj.type != "cuda" or not torch.cuda.is_available():
        return {}

    try:
        device_index = torch.cuda.current_device() if device_obj.index is None else int(device_obj.index)
        return {
            "system/gpu_torch_allocated_mb": _bytes_to_mb(torch.cuda.memory_allocated(device_index)),
            "system/gpu_torch_reserved_mb": _bytes_to_mb(torch.cuda.memory_reserved(device_index)),
            "system/gpu_torch_max_allocated_mb": _bytes_to_mb(torch.cuda.max_memory_allocated(device_index)),
            "system/gpu_torch_max_reserved_mb": _bytes_to_mb(torch.cuda.max_memory_reserved(device_index)),
        }
    except Exception:
        return {}


def summarize_worker_process_states(worker_process_states: list[dict[str, Any]] | None) -> dict[str, float]:
    states = list(worker_process_states or [])
    if not states:
        return {}

    alive = sum(1 for item in states if bool(item.get("is_alive")))
    exited = sum(1 for item in states if item.get("exitcode") is not None)
    nonzero_exitcodes = sum(1 for item in states if item.get("exitcode") not in (None, 0))
    return {
        "system/collector_workers_total": float(len(states)),
        "system/collector_workers_alive": float(alive),
        "system/collector_workers_exited": float(exited),
        "system/collector_workers_dead": float(len(states) - alive),
        "system/collector_workers_nonzero_exitcodes": float(nonzero_exitcodes),
    }


def collect_system_metrics(
    *,
    main_device: str | torch.device | None = None,
    worker_process_states: list[dict[str, Any]] | None = None,
) -> dict[str, float]:
    stats: dict[str, float] = {}
    stats.update(read_process_memory_stats())
    stats.update(read_cgroup_memory_stats())
    stats.update(read_torch_cuda_memory_stats(main_device))
    stats.update(summarize_worker_process_states(worker_process_states))

    should_query_gpu = main_device is not None and torch.device(main_device).type == "cuda"
    if not should_query_gpu:
        return stats

    gpu_total_used_mb = query_total_gpu_memory_used_mb()
    if gpu_total_used_mb is not None:
        stats["system/gpu_total_used_mb"] = float(gpu_total_used_mb)

    tracked_pids = {int(os.getpid())}
    for state in list(worker_process_states or []):
        pid = state.get("pid")
        if pid is not None:
            tracked_pids.add(int(pid))

    gpu_by_pid = query_compute_gpu_memory_by_pid(tracked_pids)
    if gpu_by_pid:
        stats["system/gpu_process_reporting_count"] = float(len(gpu_by_pid))
        main_pid = int(os.getpid())
        if main_pid in gpu_by_pid:
            stats["system/gpu_main_process_used_mb"] = float(gpu_by_pid[main_pid])

        worker_values = [
            float(gpu_by_pid[int(state["pid"])])
            for state in list(worker_process_states or [])
            if state.get("pid") is not None and int(state["pid"]) in gpu_by_pid
        ]
        if worker_values:
            stats["system/gpu_worker_process_reporting_count"] = float(len(worker_values))
            stats["system/gpu_worker_process_used_mb_sum"] = float(sum(worker_values))
            stats["system/gpu_worker_process_used_mb_max"] = float(max(worker_values))
    return stats
