from __future__ import annotations

import numpy as np


def _extract_contact_signal(raw_obs):
    tactile = raw_obs.tactile_data
    if isinstance(tactile, dict):
        return tactile.get("contact_map", tactile.get("depth"))
    return tactile


def _signal_to_scalar_map(signal) -> np.ndarray:
    signal_array = np.asarray(signal, dtype=np.float32)
    if signal_array.ndim >= 4 and signal_array.shape[-1] == 3:
        signal_array = signal_array.mean(axis=-1)
    return signal_array


def _to_sensor_maps(signal) -> np.ndarray:
    scalar_map = _signal_to_scalar_map(signal)
    if scalar_map.ndim == 0:
        return scalar_map.reshape(1, 1, 1)
    if scalar_map.ndim == 1:
        return scalar_map.reshape(1, 1, -1)
    if scalar_map.ndim == 2:
        return scalar_map.reshape(1, *scalar_map.shape)
    return scalar_map.reshape(-1, scalar_map.shape[-2], scalar_map.shape[-1])


def _normalized_boundary_distance(height: int, width: int) -> np.ndarray:
    if height <= 0 or width <= 0:
        return np.zeros((0, 0), dtype=np.float32)

    y_coords = np.arange(height, dtype=np.float32)[:, None]
    x_coords = np.arange(width, dtype=np.float32)[None, :]
    if height == 1 and width == 1:
        boundary_distance = np.zeros((1, 1), dtype=np.float32)
    elif height == 1:
        boundary_distance = np.minimum(x_coords, float(width - 1) - x_coords)
    elif width == 1:
        boundary_distance = np.minimum(y_coords, float(height - 1) - y_coords)
    else:
        boundary_distance = np.minimum(
            np.minimum(y_coords, float(height - 1) - y_coords),
            np.minimum(x_coords, float(width - 1) - x_coords),
        )

    max_distance = float(np.max(boundary_distance))
    if max_distance <= 0.0:
        return np.zeros((height, width), dtype=np.float32)
    return (boundary_distance / max_distance).astype(np.float32)


class ContactSemanticsExtractor:
    """Extract paper-aligned contact semantics from tactile maps."""

    def __init__(self, cfg: dict):
        self.tactile_threshold = float(cfg.get("tactile_threshold", 0.2))

    def extract(self, raw_obs) -> np.ndarray:
        tactile_signal = _extract_contact_signal(raw_obs)
        if tactile_signal is None:
            return np.zeros(2, dtype=np.float32)

        sensor_maps = _to_sensor_maps(tactile_signal)
        valid_points = 0
        active_points = 0
        edge_distance_values: list[np.ndarray] = []

        for sensor_map in sensor_maps:
            valid_mask = np.isfinite(sensor_map)
            if not np.any(valid_mask):
                continue
            active_mask = valid_mask & (sensor_map > self.tactile_threshold)
            valid_points += int(np.sum(valid_mask))
            active_points += int(np.sum(active_mask))
            if np.any(active_mask):
                normalized_distance = _normalized_boundary_distance(sensor_map.shape[0], sensor_map.shape[1])
                edge_distance_values.append(normalized_distance[active_mask].astype(np.float32).reshape(-1))

        if valid_points == 0:
            return np.zeros(2, dtype=np.float32)

        t_cover = float(active_points / valid_points)
        if active_points == 0:
            t_edge = 0.0
        else:
            t_edge = float(np.mean(np.concatenate(edge_distance_values, axis=0)))
        return np.asarray([t_cover, t_edge], dtype=np.float32)
