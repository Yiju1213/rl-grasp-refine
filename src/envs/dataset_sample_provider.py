from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

from src.utils.geometry import quaternion_to_rotvec


class DatasetSampleProvider:
    """Iterate tactile-extended entries with object-block shuffle and worker sharding."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.dataset_root = Path(cfg.get("dataset_root", "/Datasets/GraspNet-1billion/tactile-extended")).resolve()
        if not self.dataset_root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.dataset_root}")

        self.seed = int(cfg.get("seed", 0))
        self.cache_size = int(cfg.get("metadata_cache_size", 4))
        self.object_block_size = max(int(cfg.get("object_block_size", 4)), 1)
        self.worker_id = int(cfg.get("worker_id", 0))
        self.num_workers = max(int(cfg.get("num_workers", 1)), 1)
        self.worker_generation = max(int(cfg.get("worker_generation", 0)), 0)
        if self.worker_id < 0 or self.worker_id >= self.num_workers:
            raise ValueError(
                f"worker_id must be in [0, num_workers). Got worker_id={self.worker_id}, "
                f"num_workers={self.num_workers}."
            )
        self.runtime_defaults = cfg.get("runtime_defaults", {})
        shuffle_seed = self._derive_shuffle_seed(self.seed, self.worker_generation)
        self.rng = __import__("numpy").random.default_rng(shuffle_seed)
        self._metadata_cache: OrderedDict[int, dict[str, Any]] = OrderedDict()
        # TODO: Provider startup is still dominated by eager metadata indexing. Leave
        # this as-is for now and revisit only if build/startup time becomes a bottleneck.
        self._object_entries = self._build_object_entries()
        self._epoch_sample_pairs: list[tuple[int, int]] = []

    @staticmethod
    def _derive_shuffle_seed(base_seed: int, worker_generation: int) -> int:
        modulus = 2**32
        return int((int(base_seed) + int(worker_generation) * 9_000_019) % modulus)

    def _build_object_entries(self) -> dict[int, list[int]]:
        object_entries: dict[int, list[int]] = {}
        object_dirs = sorted(path for path in self.dataset_root.iterdir() if path.is_dir() and path.name.isdigit())
        if not object_dirs:
            raise RuntimeError(f"No object directories found under {self.dataset_root}")

        for object_dir in object_dirs:
            object_id = int(object_dir.name)
            metadata_path = object_dir / "_metadata.json"
            if not metadata_path.exists():
                continue
            with metadata_path.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            global_ids = [int(global_id) for global_id in metadata.keys()]
            if global_ids:
                object_entries[object_id] = global_ids
        if not object_entries:
            raise RuntimeError(f"No dataset entries found under {self.dataset_root}")
        return object_entries

    def _load_object_metadata(self, object_id: int) -> dict[str, Any]:
        cached = self._metadata_cache.get(object_id)
        if cached is not None:
            self._metadata_cache.move_to_end(object_id)
            return cached

        metadata_path = self.dataset_root / f"{object_id:03d}" / "_metadata.json"
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)

        self._metadata_cache[object_id] = metadata
        while len(self._metadata_cache) > self.cache_size:
            self._metadata_cache.popitem(last=False)
        return metadata

    def _build_epoch_sample_pairs(self) -> list[tuple[int, int]]:
        object_ids = list(self._object_entries.keys())
        self.rng.shuffle(object_ids)

        blocks: list[list[tuple[int, int]]] = []
        for object_id in object_ids:
            shuffled_global_ids = list(self._object_entries[object_id])
            self.rng.shuffle(shuffled_global_ids)
            for start in range(0, len(shuffled_global_ids), self.object_block_size):
                block_global_ids = shuffled_global_ids[start : start + self.object_block_size]
                blocks.append([(int(object_id), int(global_id)) for global_id in block_global_ids])

        self.rng.shuffle(blocks)
        worker_blocks = blocks[self.worker_id :: self.num_workers]
        epoch_sample_pairs = [sample_pair for block in worker_blocks for sample_pair in block]
        epoch_sample_pairs.reverse()
        return epoch_sample_pairs

    def _prepare_next_epoch(self) -> None:
        if self._epoch_sample_pairs:
            return
        self._epoch_sample_pairs = self._build_epoch_sample_pairs()
        if not self._epoch_sample_pairs:
            raise RuntimeError(
                f"No dataset samples assigned to worker {self.worker_id} out of {self.num_workers} workers."
            )

    def sample(self) -> dict[str, Any]:
        self._prepare_next_epoch()
        object_id, global_id = self._epoch_sample_pairs.pop()
        metadata = self._load_object_metadata(object_id)
        entry = metadata[str(global_id)]
        return self._entry_to_sample_cfg(object_id=object_id, global_id=global_id, entry=entry)

    def _entry_to_sample_cfg(self, object_id: int, global_id: int, entry: dict[str, Any]) -> dict[str, Any]:
        object_dir = self.dataset_root / f"{object_id:03d}"
        before_paths = {
            "tac_rgb_l": str(object_dir / "tac_rgb" / f"{global_id}_l.png"),
            "tac_rgb_r": str(object_dir / "tac_rgb" / f"{global_id}_r.png"),
            "tac_dep_l": str(object_dir / "tac_dep" / f"{global_id}_l.png"),
            "tac_dep_r": str(object_dir / "tac_dep" / f"{global_id}_r.png"),
            "vis_rgb": str(object_dir / "vis_rgb" / f"{global_id}.png"),
            "vis_dep": str(object_dir / "vis_dep" / f"{global_id}.png"),
            "vis_seg": str(object_dir / "vis_seg" / f"{global_id}.png"),
        }

        pre_grasp_hand_pos, pre_grasp_hand_quat = entry["pre-grasp"]["hand-pose"]
        pre_grasp_obj_pos, pre_grasp_obj_quat = entry["pre-grasp"]["obj-pose"]
        left_gel_pose = entry.get("grasping", {}).get("left-gel-pose")
        right_gel_pose = entry.get("grasping", {}).get("right-gel-pose")
        grasping_obj_pose = entry.get("grasping", {}).get("obj-pose")

        return {
            "source": {
                "object_id": object_id,
                "global_id": global_id,
                # Retained only for offline auditing / consistency checks against the
                # legacy dataset labels. It is not consumed anywhere in training.
                "legacy_drop_success": bool(entry.get("isPositive", False)),
                "graspnet_score": float(entry.get("graspnet-score", 0.0)),
                "before_paths": before_paths,
                "segmentation_ids": {"object": 1, "hand": 3},
            },
            "pre_grasp": {
                "hand_pose_world": {
                    "position": list(pre_grasp_hand_pos),
                    "quaternion": list(pre_grasp_hand_quat),
                },
                "object_pose_world": {
                    "position": list(pre_grasp_obj_pos),
                    "quaternion": list(pre_grasp_obj_quat),
                },
            },
            "grasping": {
                "left_gel_pose_world": {
                    "position": list(left_gel_pose[0]) if left_gel_pose else None,
                    "quaternion": list(left_gel_pose[1]) if left_gel_pose else None,
                },
                "right_gel_pose_world": {
                    "position": list(right_gel_pose[0]) if right_gel_pose else None,
                    "quaternion": list(right_gel_pose[1]) if right_gel_pose else None,
                },
                "object_pose_world": {
                    "position": list(grasping_obj_pose[0]) if grasping_obj_pose else list(pre_grasp_obj_pos),
                    "quaternion": list(grasping_obj_pose[1]) if grasping_obj_pose else list(pre_grasp_obj_quat),
                },
            },
            "camera": {
                "view_matrix": entry["viewMat"],
                "visual_proj_matrix": entry["visCamProjMat"],
                "tactile_proj_matrix": entry["tacCamProjMat"],
            },
            "runtime": dict(self.runtime_defaults),
            "initial_grasp_pose": {
                "position": list(pre_grasp_hand_pos),
                "rotation": quaternion_to_rotvec(pre_grasp_hand_quat).tolist(),
            },
            "object_pose": {
                "position": list(pre_grasp_obj_pos),
                "rotation": quaternion_to_rotvec(pre_grasp_obj_quat).tolist(),
            },
        }
