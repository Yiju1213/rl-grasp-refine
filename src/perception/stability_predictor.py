from __future__ import annotations


class StabilityPredictor:
    """Wrap a stability prediction head and input adapter."""

    def __init__(self, predictor_model, adapter, freeze: bool = True):
        self.predictor_model = predictor_model
        self.adapter = adapter
        self.freeze = freeze

    def predict_logit(self, raw_obs, latent_feature) -> float:
        model_inputs = self.adapter.adapt_predictor_input(raw_obs, latent_feature)
        self.predictor_model.eval()
        if self.freeze:
            import torch

            with torch.no_grad():
                logit = self.predictor_model(**model_inputs)
        else:
            logit = self.predictor_model(**model_inputs)
        return float(logit.squeeze(0).detach().cpu().item())
