from __future__ import annotations

import torch
import torch.nn as nn

from src.models.predictors.base_predictor import BasePredictor


class StabilityHead(BasePredictor):
    """Small MLP that predicts a raw stability logit."""

    def __init__(self, cfg: dict):
        super().__init__()
        latent_dim = int(cfg.get("latent_dim", 32))
        hidden_dim = int(cfg.get("hidden_dim", 64))
        input_dim = latent_dim + 6 + 2 + 4
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, **kwargs) -> torch.Tensor:
        latent_feature = kwargs["latent_feature"]
        batch_size = latent_feature.shape[0]

        grasp_pose = kwargs.get("grasp_pose")
        if grasp_pose is None:
            grasp_pose = torch.zeros(batch_size, 6, device=latent_feature.device, dtype=latent_feature.dtype)

        contact_feature = kwargs.get("contact_feature")
        if contact_feature is None:
            contact_feature = torch.zeros(
                batch_size,
                2,
                device=latent_feature.device,
                dtype=latent_feature.dtype,
            )

        sensor_summary = kwargs.get("sensor_summary")
        if sensor_summary is None:
            sensor_summary = torch.zeros(
                batch_size,
                4,
                device=latent_feature.device,
                dtype=latent_feature.dtype,
            )

        features = torch.cat([latent_feature, grasp_pose, contact_feature, sensor_summary], dim=1)
        return self.net(features).squeeze(-1)
