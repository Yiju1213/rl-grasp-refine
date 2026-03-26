from __future__ import annotations

import numpy as np


def _extract_tactile_signal(raw_obs):
    tactile = raw_obs.tactile_data
    if isinstance(tactile, dict):
        return tactile.get("contact_map", tactile.get("depth", tactile.get("rgb")))
    return tactile


def _signal_to_scalar_map(signal) -> np.ndarray:
    signal_array = np.asarray(signal, dtype=np.float32)
    if signal_array.ndim >= 4 and signal_array.shape[-1] == 3:
        signal_array = signal_array.mean(axis=-1)
    return signal_array


class ContactSemanticsExtractor:
    """Extract a compact contact semantic vector."""

    def __init__(self, cfg: dict):
        self.tactile_threshold = float(cfg.get("tactile_threshold", 0.2))
        self.edge_scale = float(cfg.get("edge_scale", 0.05))

    def extract(self, raw_obs) -> np.ndarray:
        coverage_ratio = self.compute_coverage_ratio(raw_obs)
        edge_proximity = self.compute_edge_proximity(raw_obs)
        return np.asarray([coverage_ratio, edge_proximity], dtype=np.float32)

    def compute_coverage_ratio(self, raw_obs) -> float:
        tactile_signal = _extract_tactile_signal(raw_obs)
        if tactile_signal is None:
            return 0.0
        # TODO: Replace this placeholder with the actual tactile coverage metric
        # computed from calibrated tac RGB/depth observations.
        contact_array = _signal_to_scalar_map(tactile_signal).reshape(-1)
        if contact_array.size == 0:
            return 0.0
        active = np.abs(contact_array) > self.tactile_threshold
        return float(np.mean(active))

    def compute_edge_proximity(self, raw_obs) -> float:
        tactile_signal = _extract_tactile_signal(raw_obs)
        if tactile_signal is None:
            return 0.0
        # TODO: Replace this placeholder with a tactile edge metric derived from
        # tac RGB/depth geometry instead of visual segmentation.
        scalar_map = _signal_to_scalar_map(tactile_signal)
        if scalar_map.ndim == 2:
            scalar_map = scalar_map[None, ...]
        if scalar_map.size == 0:
            return 0.0

        sensor_maps = scalar_map.reshape(scalar_map.shape[0], -1)
        if sensor_maps.shape[0] == 1:
            active = np.abs(sensor_maps[0]) > self.tactile_threshold
            return float(np.mean(active))

        active_ratio = [float(np.mean(np.abs(sensor_map) > self.tactile_threshold)) for sensor_map in sensor_maps[:2]]
        return float(np.clip(abs(active_ratio[0] - active_ratio[1]), 0.0, 1.0))
