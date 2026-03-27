from __future__ import annotations

import numpy as np
import torch


class StabilityPredictor:
    """Wrap a feature-only stability prediction head."""

    def __init__(self, predictor_model, freeze: bool = True):
        self.predictor_model = predictor_model
        self.freeze = freeze

    def predict_logit(self, latent_feature) -> float:
        if self.predictor_model is None:
            raise RuntimeError("StabilityPredictor has no predictor_model; raw logit must come from PerceptionResult.")

        latent_tensor = torch.as_tensor(np.asarray(latent_feature), dtype=torch.float32)
        if latent_tensor.dim() == 1:
            latent_tensor = latent_tensor.unsqueeze(0)

        self.predictor_model.eval()
        if self.freeze:
            with torch.no_grad():
                logit = self.predictor_model(latent_feature=latent_tensor)
        else:
            logit = self.predictor_model(latent_feature=latent_tensor)
        return float(logit.squeeze(0).detach().cpu().item())
