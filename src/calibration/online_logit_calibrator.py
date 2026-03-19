from __future__ import annotations

import numpy as np

from src.calibration.base_calibrator import BaseCalibrator


class OnlineLogitCalibrator(BaseCalibrator):
    """Two-parameter online logistic calibrator."""

    def __init__(self, cfg: dict):
        self.init_a = float(cfg.get("init_a", 1.0))
        self.init_b = float(cfg.get("init_b", 0.0))
        self.learning_rate = float(cfg.get("learning_rate", 0.05))
        self.l2_reg = float(cfg.get("l2_reg", 0.001))
        self.prior_var = float(cfg.get("prior_var", 1.0))
        self.uncertainty_base = float(cfg.get("uncertainty_base", 0.05))
        self.reset()

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    def predict(self, logits):
        logits_array = np.asarray(logits, dtype=np.float64)
        z = self.a * logits_array + self.b
        calibrated_prob = self._sigmoid(z)

        feature_matrix = np.stack([logits_array.reshape(-1), np.ones(logits_array.size)], axis=1)
        variances = np.einsum("bi,ij,bj->b", feature_matrix, self.posterior_cov, feature_matrix)
        uncertainty = np.sqrt(np.maximum(variances, 0.0)) * calibrated_prob.reshape(-1) * (
            1.0 - calibrated_prob.reshape(-1)
        )
        uncertainty = uncertainty + self.uncertainty_base

        if np.isscalar(logits) or logits_array.ndim == 0:
            return float(np.asarray(calibrated_prob).item()), float(np.asarray(uncertainty).item())
        return calibrated_prob.astype(np.float32), uncertainty.astype(np.float32)

    def update(self, logits, labels) -> None:
        logits_array = np.asarray(logits, dtype=np.float64).reshape(-1)
        labels_array = np.asarray(labels, dtype=np.float64).reshape(-1)
        if logits_array.size == 0:
            return
        if logits_array.shape != labels_array.shape:
            raise ValueError("logits and labels must have the same shape.")

        design = np.stack([logits_array, np.ones_like(logits_array)], axis=1)
        theta = np.asarray([self.a, self.b], dtype=np.float64)
        logits_calibrated = design @ theta
        probs = self._sigmoid(logits_calibrated)

        grad = (design.T @ (probs - labels_array)) / logits_array.size
        grad += self.l2_reg * theta

        weights = probs * (1.0 - probs)
        hessian = (design.T * weights) @ design / logits_array.size
        hessian += (self.l2_reg + 1.0 / max(self.prior_var, 1e-6)) * np.eye(2)

        step = np.linalg.solve(hessian, grad)
        theta = theta - self.learning_rate * step
        self.a = float(theta[0])
        self.b = float(theta[1])
        self.posterior_cov = np.linalg.inv(hessian)

    def get_state(self) -> dict:
        return {
            "a": self.a,
            "b": self.b,
            "posterior_cov": self.posterior_cov.copy(),
        }

    def reset(self) -> None:
        self.a = self.init_a
        self.b = self.init_b
        self.posterior_cov = np.eye(2, dtype=np.float64) * self.prior_var
