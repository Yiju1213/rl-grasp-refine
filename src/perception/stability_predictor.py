from __future__ import annotations


class StabilityPredictor:
    """Wrap a stability prediction head and input adapter."""

    def __init__(self, predictor_model, adapter, freeze: bool = True, runtime=None):
        self.predictor_model = predictor_model
        self.adapter = adapter
        self.freeze = freeze
        self.runtime = runtime

    def predict_logit(self, raw_obs, latent_feature) -> float:
        if self.runtime is not None:
            result = self.runtime.infer(raw_obs, self.adapter)
            return float(result.raw_logit)

        model_inputs = self.adapter.adapt_predictor_input(raw_obs, latent_feature)
        self.predictor_model.eval()
        if self.freeze:
            import torch

            with torch.no_grad():
                logit = self.predictor_model(**model_inputs)
        else:
            logit = self.predictor_model(**model_inputs)
        return float(logit.squeeze(0).detach().cpu().item())
