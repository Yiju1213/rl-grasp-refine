from __future__ import annotations

import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np

from src.utils.geometry import depth_to_camera_points, tactile_depth_to_gel_points_and_mask


@contextmanager
def _adapointr_utils_context():
    source_root = Path("/AdaPoinTr")
    source_root_str = str(source_root)
    inserted = False
    if source_root_str not in sys.path:
        sys.path.insert(0, source_root_str)
        inserted = True
    try:
        yield
    finally:
        if inserted:
            try:
                sys.path.remove(source_root_str)
            except ValueError:
                pass


@unittest.skipUnless(Path("/AdaPoinTr").exists(), "AdaPoinTr source tree is required for geometry alignment tests.")
class TestGeometryVTG3DAlignment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            with _adapointr_utils_context():
                from utils import vtg3d_utils
        except Exception as exc:
            raise unittest.SkipTest(f"Failed to import AdaPoinTr vtg3d_utils: {exc}") from exc

        cls.vtg3d_utils = vtg3d_utils

    def test_visual_camera_points_match_vtg3d_utils(self):
        proj = np.eye(4, dtype=np.float32)
        depth_m = np.asarray(
            [
                [0.10, 0.12, 0.00, 0.00],
                [0.20, 0.00, 0.18, 0.00],
                [0.00, 0.22, 0.25, 0.30],
            ],
            dtype=np.float32,
        )
        seg = np.asarray(
            [
                [1, 1, 0, 0],
                [1, 0, 1, 0],
                [0, 1, 1, 1],
            ],
            dtype=np.int16,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            depth_path = Path(temp_dir) / "depth.png"
            seg_path = Path(temp_dir) / "seg.png"
            cv2.imwrite(str(depth_path), np.round(depth_m * 1000.0).astype(np.uint16))
            cv2.imwrite(str(seg_path), (seg + 1).astype(np.uint16))

            point_cloud = self.vtg3d_utils.getVisualPointCloud(
                dep_path=str(depth_path),
                proj_mat=proj,
                rgb_path=None,
                seg_path=str(seg_path),
                seg_id=1,
                align_to_view_coordinate=True,
            )
            expected = np.asarray(point_cloud.points, dtype=np.float32)

        actual = depth_to_camera_points(depth_m=depth_m, proj_matrix=proj, mask=seg == 1)
        self.assertEqual(expected.shape, actual.shape)
        self.assertTrue(np.allclose(expected, actual, atol=1e-6))

    def test_tactile_gel_points_match_vtg3d_utils(self):
        proj = np.eye(4, dtype=np.float32)
        depth_m = np.zeros((320, 240), dtype=np.float32)
        depth_m[80:160, 60:120] = 0.0015
        depth_m[180:240, 120:180] = 0.0008

        with tempfile.TemporaryDirectory() as temp_dir:
            depth_path = Path(temp_dir) / "tactile.png"
            cv2.imwrite(str(depth_path), np.round(depth_m * 10000.0).astype(np.uint16))
            point_cloud, gel_mask = self.vtg3d_utils.getTactilePointCloud(
                dep_path=str(depth_path),
                proj_mat=proj,
                distance_cam2gel=0.02315,
                align_to_gel_coordinate=True,
                step=8,
            )
            expected_points = np.asarray(point_cloud.points, dtype=np.float32)
            expected_mask = np.asarray(gel_mask, dtype=bool)

        actual_points, actual_mask = tactile_depth_to_gel_points_and_mask(
            depth_m=depth_m,
            proj_matrix=proj,
            camera_distance_to_gel_m=0.02315,
            step=8,
        )
        self.assertEqual(expected_points.shape, actual_points.shape)
        self.assertEqual(expected_mask.shape, actual_mask.shape)
        self.assertTrue(np.array_equal(expected_mask, actual_mask))
        self.assertTrue(np.allclose(expected_points, actual_points, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
