from __future__ import annotations

import numpy as np


class FeatureExtractor:
    """Wrap a backbone model and adapter behind a stable interface."""

    def __init__(self, backbone_model, adapter, freeze: bool = True, runtime=None):
        self.backbone_model = backbone_model
        self.adapter = adapter
        self.freeze = freeze
        self.runtime = runtime

    def extract(self, raw_obs):
        if self.runtime is not None:
            result = self.runtime.infer(raw_obs, self.adapter)
            return result.body_feature.copy()

        model_inputs = self.adapter.adapt_feature_input(raw_obs)
        self.backbone_model.eval()
        if self.freeze:
            import torch

            with torch.no_grad():
                latent = self.backbone_model(**model_inputs)
        else:
            latent = self.backbone_model(**model_inputs)
        return np.asarray(latent.squeeze(0).detach().cpu().numpy(), dtype=np.float32)
