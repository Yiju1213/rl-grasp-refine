from __future__ import annotations

import numpy as np

from src.perception.sga_gsn_types import PerceptionResult


class FeatureExtractor:
    """Wrap a backbone model and adapter behind a stable interface."""

    def __init__(self, backbone_model, adapter, freeze: bool = True, runtime=None):
        self.backbone_model = backbone_model
        self.adapter = adapter
        self.freeze = freeze
        self.runtime = runtime

    def encode(self, raw_obs) -> PerceptionResult:
        if self.runtime is not None:
            result = self.runtime.infer(raw_obs, self.adapter)
            return PerceptionResult(
                latent_feature=result.body_feature.copy(),
                raw_stability_logit=result.raw_logit,
                runtime_payload=result,
            )

        model_inputs = self.adapter.adapt_feature_input(raw_obs)
        self.backbone_model.eval()
        if self.freeze:
            import torch

            with torch.no_grad():
                latent = self.backbone_model(**model_inputs)
        else:
            latent = self.backbone_model(**model_inputs)
        return PerceptionResult(
            latent_feature=np.asarray(latent.squeeze(0).detach().cpu().numpy(), dtype=np.float32),
        )
