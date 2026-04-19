from __future__ import annotations

from abc import ABC, abstractmethod
from binascii import crc32

import numpy as np
import torch
from PIL import Image

from src.perception.cnnmca_types import PreparedCNNMCAInputs
from src.perception.sga_gsn_types import PreparedVTGInputs
from src.structures.observation import RawSensorObservation
from src.utils.geometry import (
    DEFAULT_TACTILE_CAMERA_TO_GEL_M,
    apply_zero_means,
    depth_to_world_points,
    downsample_by_dist_ratio,
    downsample_points_with_indices,
    gel_points_to_world,
    get_zero_mean,
    tactile_depth_to_gel_points_and_mask,
)


def _to_float_tensor(value, add_batch_dim: bool = True) -> torch.Tensor:
    if value is None:
        tensor = torch.zeros(1, dtype=torch.float32)
    elif isinstance(value, torch.Tensor):
        tensor = value.float()
    else:
        tensor = torch.as_tensor(np.asarray(value), dtype=torch.float32)

    if add_batch_dim and tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    elif add_batch_dim and tensor.dim() >= 2 and tensor.shape[0] != 1:
        tensor = tensor.unsqueeze(0)
    return tensor


def _extract_point_cloud(raw_obs: RawSensorObservation):
    visual = raw_obs.visual_data
    if isinstance(visual, dict):
        point_cloud = visual.get("point_cloud")
        if point_cloud is not None:
            return point_cloud
    # TODO: Replace this dummy tensor with world-frame point cloud reconstruction
    # from visual depth/segmentation plus view/projection matrices and pose metadata.
    return np.zeros((1, 3), dtype=np.float32)


def _extract_tactile(raw_obs: RawSensorObservation):
    tactile = raw_obs.tactile_data
    if isinstance(tactile, dict):
        return tactile.get("contact_map", tactile.get("embedding", tactile.get("values")))
    return tactile


class PerceptionInputAdapter(ABC):
    """Adapter interface between raw observations and model inputs."""

    @abstractmethod
    def adapt_feature_input(self, raw_obs: RawSensorObservation) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def prepare_inputs(self, raw_obs: RawSensorObservation) -> PreparedVTGInputs:
        raise NotImplementedError("This adapter does not support VTG3D input preparation.")


class SGAGSNAdapter(PerceptionInputAdapter):
    """Bridge raw env observations to VTG3D-style SGA-GSN inputs."""

    def __init__(self, cfg: dict | None = None):
        cfg = cfg or {}
        runtime_cfg = cfg.get("sga_gsn", {}).get("runtime", {})
        self.vis_points = int(runtime_cfg.get("vis_points", 2048))
        self.tac_points_per_side = int(runtime_cfg.get("tac_points_per_side", 1200))
        self.sc_input_points = int(runtime_cfg.get("sc_input_points", 2048))
        self.tactile_step = int(runtime_cfg.get("tactile_step", 8))
        self.camera_distance_to_gel_m = float(
            runtime_cfg.get("camera_distance_to_gel_m", DEFAULT_TACTILE_CAMERA_TO_GEL_M)
        )
        self.tactile_noise_eps = float(runtime_cfg.get("tactile_noise_eps", 1e-4))
        self.seed = int(runtime_cfg.get("seed", 0))

    def adapt_feature_input(self, raw_obs: RawSensorObservation) -> dict[str, torch.Tensor]:
        return {
            "point_cloud": _to_float_tensor(_extract_point_cloud(raw_obs), add_batch_dim=True),
            "tactile": _to_float_tensor(_extract_tactile(raw_obs), add_batch_dim=True),
        }

    def prepare_inputs(self, raw_obs: RawSensorObservation) -> PreparedVTGInputs:
        rng = np.random.default_rng(self._observation_seed(raw_obs))

        visual_world_points = self._extract_visual_world_points(raw_obs)
        visual_world_points, _ = downsample_points_with_indices(
            visual_world_points,
            method="random",
            num_points=self.vis_points,
            rng=rng,
        )

        tactile_left_world_points, tactile_left_gel_mask = self._extract_tactile_world_points(raw_obs, side="left")
        tactile_right_world_points, tactile_right_gel_mask = self._extract_tactile_world_points(raw_obs, side="right")
        tactile_left_world_points, tactile_left_gel_mask = self._ensure_tactile_point_count(
            tactile_left_world_points,
            tactile_left_gel_mask,
            rng=rng,
        )
        tactile_right_world_points, tactile_right_gel_mask = self._ensure_tactile_point_count(
            tactile_right_world_points,
            tactile_right_gel_mask,
            rng=rng,
        )

        tactile_left_contact_world_points = tactile_left_world_points[~tactile_left_gel_mask].copy()
        tactile_right_contact_world_points = tactile_right_world_points[~tactile_right_gel_mask].copy()
        zero_mean = get_zero_mean(
            [
                visual_world_points,
                tactile_left_contact_world_points,
                tactile_right_contact_world_points,
            ]
        )
        (
            tactile_left_world_points_zero_mean,
            tactile_right_world_points_zero_mean,
            visual_world_points_zero_mean,
            tactile_left_contact_world_points_zero_mean,
            tactile_right_contact_world_points_zero_mean,
        ) = apply_zero_means(
            [
                tactile_left_world_points,
                tactile_right_world_points,
                visual_world_points,
                tactile_left_contact_world_points,
                tactile_right_contact_world_points,
            ],
            zero_mean,
        )

        sc_input = self._build_sc_input(
            visual_world_points_zero_mean,
            tactile_left_contact_world_points_zero_mean,
            tactile_right_contact_world_points_zero_mean,
            rng=rng,
        )
        gs_input = self._build_gs_input(
            tactile_left_world_points_zero_mean,
            tactile_right_world_points_zero_mean,
            tactile_left_gel_mask,
            tactile_right_gel_mask,
        )

        return PreparedVTGInputs(
            sc_input=sc_input,
            gs_input=gs_input,
            zero_mean=zero_mean,
            debug_visual_world_points=visual_world_points,
            debug_tactile_left_world_points=tactile_left_world_points,
            debug_tactile_right_world_points=tactile_right_world_points,
            debug_tactile_left_contact_world_points=tactile_left_contact_world_points,
            debug_tactile_right_contact_world_points=tactile_right_contact_world_points,
            debug_tactile_left_gel_mask=tactile_left_gel_mask,
            debug_tactile_right_gel_mask=tactile_right_gel_mask,
        )

    def _observation_seed(self, raw_obs: RawSensorObservation) -> int:
        metadata = raw_obs.grasp_metadata
        global_id = int(metadata.get("source_global_id", 0))
        stage = str(metadata.get("observation_stage", ""))
        return int((self.seed + global_id * 1_000_003 + crc32(stage.encode("utf-8"))) % (2**32))

    def _extract_visual_world_points(self, raw_obs: RawSensorObservation) -> np.ndarray:
        visual = raw_obs.visual_data if isinstance(raw_obs.visual_data, dict) else {}
        if "point_cloud" in visual:
            return np.asarray(visual["point_cloud"], dtype=np.float32).reshape(-1, 3)

        depth = visual.get("depth")
        proj_matrix = visual.get("proj_matrix")
        view_matrix = visual.get("view_matrix")
        if depth is None or proj_matrix is None or view_matrix is None:
            return np.zeros((0, 3), dtype=np.float32)

        seg = visual.get("seg")
        object_id = raw_obs.grasp_metadata.get("segmentation_ids", {}).get("object")
        mask = None
        if seg is not None and object_id is not None:
            mask = np.asarray(seg) == int(object_id)
        return depth_to_world_points(
            depth_m=np.asarray(depth, dtype=np.float32),
            proj_matrix=np.asarray(proj_matrix, dtype=np.float32),
            view_matrix=np.asarray(view_matrix, dtype=np.float32),
            mask=mask,
            max_points=None,
        )

    def _extract_tactile_world_points(self, raw_obs: RawSensorObservation, side: str) -> tuple[np.ndarray, np.ndarray]:
        tactile = raw_obs.tactile_data if isinstance(raw_obs.tactile_data, dict) else {}
        depth_all = tactile.get("depth")
        proj_matrix = tactile.get("proj_matrix")
        gel_pose = raw_obs.grasp_metadata.get("gel_pose_world", {}).get(side)
        if depth_all is None or proj_matrix is None or gel_pose is None:
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=bool)

        side_index = 0 if side == "left" else 1
        depth = np.asarray(depth_all, dtype=np.float32)[side_index]
        camera_distance = float(tactile.get("camera_distance_to_gel_m", self.camera_distance_to_gel_m))
        gel_points, gel_mask = tactile_depth_to_gel_points_and_mask(
            depth_m=depth,
            proj_matrix=np.asarray(proj_matrix, dtype=np.float32),
            camera_distance_to_gel_m=camera_distance,
            noise_eps=self.tactile_noise_eps,
            step=self.tactile_step,
        )
        if gel_points.shape[0] == 0:
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=bool)

        world_points = gel_points_to_world(
            gel_points=gel_points,
            gel_position=gel_pose["position"],
            gel_quaternion=gel_pose["quaternion"],
        )
        world_points, indices = downsample_points_with_indices(
            world_points,
            method="uniform",
            num_points=self.tac_points_per_side,
        )
        if indices.size == 0:
            return world_points, np.zeros((0,), dtype=bool)
        return world_points.astype(np.float32), gel_mask[indices].astype(bool)

    def _ensure_tactile_point_count(
        self,
        points: np.ndarray,
        gel_mask: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        gel_mask = np.asarray(gel_mask, dtype=bool).reshape(-1)
        if points.shape[0] == self.tac_points_per_side:
            return points, gel_mask
        if points.shape[0] == 0:
            return (
                np.zeros((self.tac_points_per_side, 3), dtype=np.float32),
                np.ones((self.tac_points_per_side,), dtype=bool),
            )

        if points.shape[0] > self.tac_points_per_side:
            points, indices = downsample_points_with_indices(
                points,
                method="uniform",
                num_points=self.tac_points_per_side,
            )
            return points, gel_mask[indices].astype(bool)

        extra_needed = self.tac_points_per_side - points.shape[0]
        extra_indices = rng.choice(points.shape[0], size=extra_needed, replace=True)
        padded_points = np.concatenate([points, points[extra_indices]], axis=0).astype(np.float32)
        padded_gel_mask = np.concatenate([gel_mask, gel_mask[extra_indices]], axis=0).astype(bool)
        return padded_points, padded_gel_mask

    def _build_sc_input(
        self,
        visual_world_points_zero_mean: np.ndarray,
        tactile_left_contact_world_points_zero_mean: np.ndarray,
        tactile_right_contact_world_points_zero_mean: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        try:
            vis_sc, tactile_left_sc, tactile_right_sc = downsample_by_dist_ratio(
                vis_points=visual_world_points_zero_mean,
                tactile_points_left=tactile_left_contact_world_points_zero_mean,
                tactile_points_right=tactile_right_contact_world_points_zero_mean,
                num_points=self.sc_input_points,
                rng=rng,
            )
            sc_input = np.concatenate([vis_sc, tactile_left_sc, tactile_right_sc], axis=0).astype(np.float32)
        except ValueError:
            sc_input = np.zeros((0, 3), dtype=np.float32)
        return self._ensure_xyz_point_count(sc_input, self.sc_input_points, rng)

    def _build_gs_input(
        self,
        tactile_left_world_points_zero_mean: np.ndarray,
        tactile_right_world_points_zero_mean: np.ndarray,
        tactile_left_gel_mask: np.ndarray,
        tactile_right_gel_mask: np.ndarray,
    ) -> np.ndarray:
        tactile_left_with_gel = np.concatenate(
            [tactile_left_world_points_zero_mean, tactile_left_gel_mask.astype(np.float32)[:, None]],
            axis=1,
        )
        tactile_right_with_gel = np.concatenate(
            [tactile_right_world_points_zero_mean, tactile_right_gel_mask.astype(np.float32)[:, None]],
            axis=1,
        )
        return np.concatenate([tactile_left_with_gel, tactile_right_with_gel], axis=0).astype(np.float32)

    @staticmethod
    def _ensure_xyz_point_count(
        points: np.ndarray,
        target_points: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        if points.shape[0] == target_points:
            return points
        if points.shape[0] == 0:
            return np.zeros((target_points, 3), dtype=np.float32)
        if points.shape[0] > target_points:
            indices = rng.choice(points.shape[0], size=target_points, replace=False)
            return points[indices].astype(np.float32)
        extra_indices = rng.choice(points.shape[0], size=target_points - points.shape[0], replace=True)
        return np.concatenate([points, points[extra_indices]], axis=0).astype(np.float32)


class DGCNNAdapter(SGAGSNAdapter):
    """Stub-compatible adapter matching the DGCNN placeholder."""

    pass


class CNNMCAAdapter(PerceptionInputAdapter):
    """Bridge raw env RGB observations to CNNMCA's 2D VTG inputs."""

    _IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
    _IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]

    def __init__(self, cfg: dict | None = None):
        cfg = cfg or {}
        runtime_cfg = cfg.get("cnnmca", {}).get("runtime", {})
        self.image_size = int(runtime_cfg.get("image_size", 224))
        self.concat_two_tactile = bool(runtime_cfg.get("concat_two_tactile", True))
        if self.image_size <= 0:
            raise ValueError(f"CNNMCA image_size must be positive. Got {self.image_size}.")

    def adapt_feature_input(self, raw_obs: RawSensorObservation) -> dict[str, torch.Tensor]:
        prepared = self.prepare_inputs(raw_obs)
        return {
            "visual_img": torch.from_numpy(prepared.visual_img).unsqueeze(0),
            "tactile_img": torch.from_numpy(prepared.tactile_img).unsqueeze(0),
        }

    def prepare_inputs(self, raw_obs: RawSensorObservation) -> PreparedCNNMCAInputs:
        visual_rgb = self._require_rgb(raw_obs.visual_data, key="rgb", source="visual_data")
        tactile_rgb = self._require_rgb(raw_obs.tactile_data, key="rgb", source="tactile_data")
        tactile_rgb = np.asarray(tactile_rgb)
        if tactile_rgb.ndim != 4 or tactile_rgb.shape[0] < 2 or tactile_rgb.shape[-1] != 3:
            raise ValueError(
                "CNNMCA tactile_data['rgb'] must have shape (2, H, W, 3) or compatible. "
                f"Got {tactile_rgb.shape}."
            )
        if not self.concat_two_tactile:
            raise NotImplementedError("CNNMCAAdapter currently supports concat_two_tactile=true only.")

        tactile_concat = np.concatenate([tactile_rgb[0], tactile_rgb[1]], axis=1)
        return PreparedCNNMCAInputs(
            visual_img=self._to_chw_normalized_rgb(visual_rgb),
            tactile_img=self._to_chw_normalized_rgb(tactile_concat),
        )

    @staticmethod
    def _require_rgb(container, *, key: str, source: str):
        if not isinstance(container, dict) or key not in container:
            raise ValueError(f"CNNMCAAdapter requires {source}[{key!r}].")
        return container[key]

    def _to_chw_normalized_rgb(self, rgb) -> np.ndarray:
        array = np.asarray(rgb)
        if array.ndim != 3 or array.shape[-1] != 3:
            raise ValueError(f"CNNMCA RGB input must have shape (H, W, 3). Got {array.shape}.")
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)

        image = Image.fromarray(array, mode="RGB")
        image = image.resize((self.image_size, self.image_size), resample=Image.BILINEAR)
        image_array = np.asarray(image, dtype=np.float32) / 255.0
        chw = np.transpose(image_array, (2, 0, 1))
        return ((chw - self._IMAGENET_MEAN) / self._IMAGENET_STD).astype(np.float32)
