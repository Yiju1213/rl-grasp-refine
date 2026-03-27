from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.envs.dataset_sample_provider import DatasetSampleProvider
from src.utils.geometry import quaternion_to_rotvec


def _write_png(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), array)


def _write_entry(root: Path, object_id: int, global_id: int) -> None:
    object_root = root / f"{object_id:03d}"
    metadata_path = object_root / "_metadata.json"
    metadata = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    metadata[str(global_id)] = {
        "pre-grasp": {
            "hand-pose": [[0.1, 0.2, 0.3], [0.0, 0.0, 0.0, 1.0]],
            "obj-pose": [[0.4, 0.5, 0.6], [0.0, 0.0, 0.0, 1.0]],
        },
        "grasping": {
            "left-gel-pose": [[0.11, 0.21, 0.31], [0.0, 0.0, 0.0, 1.0]],
            "right-gel-pose": [[0.12, 0.22, 0.32], [0.0, 0.0, 0.0, 1.0]],
            "obj-pose": [[0.41, 0.51, 0.61], [0.0, 0.0, 0.0, 1.0]],
        },
        "viewMat": np.eye(4, dtype=np.float32).T.reshape(-1).tolist(),
        "visCamProjMat": np.eye(4, dtype=np.float32).T.reshape(-1).tolist(),
        "tacCamProjMat": np.eye(4, dtype=np.float32).reshape(-1).tolist(),
        "isPositive": True,
        "graspnet-score": 0.9,
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    dep_vis = np.full((8, 8), 500, dtype=np.uint16)
    dep_tac = np.full((8, 8), 12, dtype=np.uint8)
    seg = np.ones((8, 8), dtype=np.uint8)

    _write_png(object_root / "tac_rgb" / f"{global_id}_l.png", rgb)
    _write_png(object_root / "tac_rgb" / f"{global_id}_r.png", rgb)
    _write_png(object_root / "tac_dep" / f"{global_id}_l.png", dep_tac)
    _write_png(object_root / "tac_dep" / f"{global_id}_r.png", dep_tac)
    _write_png(object_root / "vis_rgb" / f"{global_id}.png", rgb)
    _write_png(object_root / "vis_dep" / f"{global_id}.png", dep_vis)
    _write_png(object_root / "vis_seg" / f"{global_id}.png", seg)


class TestDatasetSampleProvider(unittest.TestCase):
    def test_provider_reads_entry_and_builds_sample_cfg(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_root = Path(tmpdir) / "tactile-extended"
            root = dataset_root / "000"
            _write_entry(dataset_root, object_id=0, global_id=42)

            provider = DatasetSampleProvider(
                {
                    "dataset_root": str(dataset_root),
                    "seed": 3,
                    "metadata_cache_size": 1,
                    "runtime_defaults": {"time_step": 0.005},
                }
            )
            sample = provider.sample()

            self.assertEqual(sample["source"]["object_id"], 0)
            self.assertEqual(sample["source"]["global_id"], 42)
            self.assertEqual(sample["source"]["before_paths"]["vis_rgb"], str(root / "vis_rgb" / "42.png"))
            self.assertIn("hand_pose_world", sample["pre_grasp"])
            self.assertIn("left_gel_pose_world", sample["grasping"])
            np.testing.assert_allclose(
                sample["initial_grasp_pose"]["rotation"],
                quaternion_to_rotvec([0.0, 0.0, 0.0, 1.0]),
            )

    def test_provider_shuffles_by_object_blocks_per_epoch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_root = Path(tmpdir) / "tactile-extended"
            expected_pairs = {
                (0, 10),
                (0, 11),
                (0, 12),
                (0, 13),
                (1, 20),
                (1, 21),
                (1, 22),
                (1, 23),
            }
            for object_id, global_id in expected_pairs:
                _write_entry(dataset_root, object_id=object_id, global_id=global_id)

            provider_a = DatasetSampleProvider(
                {
                    "dataset_root": str(dataset_root),
                    "seed": 7,
                    "object_block_size": 2,
                    "metadata_cache_size": 1,
                    "runtime_defaults": {"time_step": 0.005},
                }
            )
            provider_b = DatasetSampleProvider(
                {
                    "dataset_root": str(dataset_root),
                    "seed": 7,
                    "object_block_size": 2,
                    "metadata_cache_size": 1,
                    "runtime_defaults": {"time_step": 0.005},
                }
            )

            epoch_a = [
                (sample["source"]["object_id"], sample["source"]["global_id"])
                for sample in (provider_a.sample() for _ in range(len(expected_pairs)))
            ]
            epoch_b = [
                (sample["source"]["object_id"], sample["source"]["global_id"])
                for sample in (provider_b.sample() for _ in range(len(expected_pairs)))
            ]

            self.assertEqual(epoch_a, epoch_b)
            self.assertEqual(set(epoch_a), expected_pairs)
            self.assertEqual(len(epoch_a), len(set(epoch_a)))
            self.assertLessEqual(len(provider_a._metadata_cache), 1)
            for block_start in range(0, len(epoch_a), 2):
                self.assertEqual(epoch_a[block_start][0], epoch_a[block_start + 1][0])

    def test_provider_shards_epoch_blocks_across_three_workers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_root = Path(tmpdir) / "tactile-extended"
            expected_pairs = {
                (0, 10),
                (0, 11),
                (0, 12),
                (0, 13),
                (1, 20),
                (1, 21),
                (1, 22),
                (1, 23),
                (2, 30),
                (2, 31),
                (2, 32),
                (2, 33),
            }
            for object_id, global_id in expected_pairs:
                _write_entry(dataset_root, object_id=object_id, global_id=global_id)

            providers = [
                DatasetSampleProvider(
                    {
                        "dataset_root": str(dataset_root),
                        "seed": 11,
                        "object_block_size": 2,
                        "worker_id": worker_id,
                        "num_workers": 3,
                        "metadata_cache_size": 1,
                        "runtime_defaults": {"time_step": 0.005},
                    }
                )
                for worker_id in range(3)
            ]
            provider_worker_0_again = DatasetSampleProvider(
                {
                    "dataset_root": str(dataset_root),
                    "seed": 11,
                    "object_block_size": 2,
                    "worker_id": 0,
                    "num_workers": 3,
                    "metadata_cache_size": 1,
                    "runtime_defaults": {"time_step": 0.005},
                }
            )

            samples_per_worker = len(expected_pairs) // 3
            worker_epochs = [
                [
                    (sample["source"]["object_id"], sample["source"]["global_id"])
                    for sample in (provider.sample() for _ in range(samples_per_worker))
                ]
                for provider in providers
            ]
            epoch_worker_0_again = [
                (sample["source"]["object_id"], sample["source"]["global_id"])
                for sample in (provider_worker_0_again.sample() for _ in range(samples_per_worker))
            ]

            self.assertEqual(worker_epochs[0], epoch_worker_0_again)
            combined = set()
            for worker_epoch in worker_epochs:
                self.assertEqual(len(worker_epoch), samples_per_worker)
                self.assertTrue(combined.isdisjoint(set(worker_epoch)))
                combined.update(worker_epoch)
            self.assertEqual(combined, expected_pairs)

    def test_provider_uses_configured_seed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_root = Path(tmpdir) / "tactile-extended"
            expected_pairs = {
                (0, 10),
                (0, 11),
                (0, 12),
                (0, 13),
                (1, 20),
                (1, 21),
                (1, 22),
                (1, 23),
            }
            for object_id, global_id in expected_pairs:
                _write_entry(dataset_root, object_id=object_id, global_id=global_id)

            epoch_orders = []
            for seed in (1, 2, 3):
                provider = DatasetSampleProvider(
                    {
                        "dataset_root": str(dataset_root),
                        "seed": seed,
                        "object_block_size": 2,
                        "metadata_cache_size": 1,
                        "runtime_defaults": {"time_step": 0.005},
                    }
                )
                epoch_orders.append(
                    [
                        (sample["source"]["object_id"], sample["source"]["global_id"])
                        for sample in (provider.sample() for _ in range(len(expected_pairs)))
                    ]
                )

            self.assertEqual(len({tuple(order) for order in epoch_orders}), len(epoch_orders))


if __name__ == "__main__":
    unittest.main()
