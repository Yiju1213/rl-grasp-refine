from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch

from src.structures.action import GraspPose
from src.structures.observation import RawSensorObservation


def _to_float_tensor(value, add_batch_dim: bool = True) -> torch.Tensor:
    if value is None:
        tensor = torch.zeros(1, dtype=torch.float32)
    elif isinstance(value, torch.Tensor):
        tensor = value.float()
    else:
        tensor = torch.as_tensor(np.asarray(value), dtype=torch.float32)

    if add_batch_dim and tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    elif add_batch_dim and tensor.dim() == 2 and tensor.shape[0] != 1:
        tensor = tensor.unsqueeze(0)
    return tensor


def _extract_point_cloud(raw_obs: RawSensorObservation):
    visual = raw_obs.visual_data
    if isinstance(visual, dict):
        return visual.get("point_cloud")
    return visual


def _extract_tactile(raw_obs: RawSensorObservation):
    tactile = raw_obs.tactile_data
    if isinstance(tactile, dict):
        return tactile.get("contact_map", tactile.get("embedding", tactile.get("values")))
    return tactile


def _grasp_pose_to_array(grasp_pose) -> np.ndarray:
    if isinstance(grasp_pose, GraspPose):
        return grasp_pose.as_array()
    return np.asarray(grasp_pose, dtype=np.float32).reshape(6)


def _sensor_summary(raw_obs: RawSensorObservation) -> torch.Tensor:
    point_cloud = np.asarray(_extract_point_cloud(raw_obs) if _extract_point_cloud(raw_obs) is not None else [0.0])
    tactile = np.asarray(_extract_tactile(raw_obs) if _extract_tactile(raw_obs) is not None else [0.0])
    summary = np.asarray(
        [
            float(np.mean(point_cloud)),
            float(np.std(point_cloud)),
            float(np.mean(tactile)),
            float(np.std(tactile)),
        ],
        dtype=np.float32,
    )
    return torch.from_numpy(summary).unsqueeze(0)


class PerceptionInputAdapter(ABC):
    """Adapter interface between raw observations and model inputs."""

    @abstractmethod
    def adapt_feature_input(self, raw_obs: RawSensorObservation) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def adapt_predictor_input(
        self,
        raw_obs: RawSensorObservation,
        latent_feature,
    ) -> dict[str, torch.Tensor]:
        raise NotImplementedError


class SGAGSNAdapter(PerceptionInputAdapter):
    """Default adapter for the minimal summary encoder."""

    def adapt_feature_input(self, raw_obs: RawSensorObservation) -> dict[str, torch.Tensor]:
        return {
            "point_cloud": _to_float_tensor(_extract_point_cloud(raw_obs), add_batch_dim=True),
            "tactile": _to_float_tensor(_extract_tactile(raw_obs), add_batch_dim=True),
        }

    def adapt_predictor_input(
        self,
        raw_obs: RawSensorObservation,
        latent_feature,
    ) -> dict[str, torch.Tensor]:
        if isinstance(latent_feature, torch.Tensor):
            latent_tensor = latent_feature.float()
        else:
            latent_tensor = torch.as_tensor(np.asarray(latent_feature), dtype=torch.float32)
        if latent_tensor.dim() == 1:
            latent_tensor = latent_tensor.unsqueeze(0)

        grasp_pose = raw_obs.grasp_metadata.get("grasp_pose")
        contact_semantic = raw_obs.grasp_metadata.get("contact_semantic", np.zeros(2, dtype=np.float32))
        return {
            "latent_feature": latent_tensor,
            "grasp_pose": torch.from_numpy(_grasp_pose_to_array(grasp_pose)).unsqueeze(0),
            "contact_feature": torch.as_tensor(contact_semantic, dtype=torch.float32).reshape(1, -1),
            "sensor_summary": _sensor_summary(raw_obs),
        }


class DGCNNAdapter(SGAGSNAdapter):
    """Stub-compatible adapter matching the DGCNN placeholder."""

    pass
