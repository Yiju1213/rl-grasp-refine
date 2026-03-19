from __future__ import annotations

import numpy as np


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
        tactile = raw_obs.tactile_data
        if isinstance(tactile, dict):
            contact_map = tactile.get("contact_map", tactile.get("values", tactile.get("embedding")))
        else:
            contact_map = tactile
        if contact_map is None:
            return 0.0
        contact_array = np.asarray(contact_map, dtype=np.float32).reshape(-1)
        if contact_array.size == 0:
            return 0.0
        active = np.abs(contact_array) > self.tactile_threshold
        return float(np.mean(active))

    def compute_edge_proximity(self, raw_obs) -> float:
        metadata = raw_obs.grasp_metadata
        distance_to_edge = metadata.get("distance_to_edge")
        if distance_to_edge is None and isinstance(raw_obs.visual_data, dict):
            distance_to_edge = raw_obs.visual_data.get("distance_to_edge")
        if distance_to_edge is None:
            return 0.0
        distance = max(float(distance_to_edge), 0.0)
        proximity = 1.0 - min(distance / max(self.edge_scale, 1e-6), 1.0)
        return float(np.clip(proximity, 0.0, 1.0))
