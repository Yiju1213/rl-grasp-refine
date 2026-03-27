from __future__ import annotations

import numpy as np

from src.calibration.base_calibrator import BaseCalibrator


class OnlineLogitCalibrator(BaseCalibrator):
    """Two-parameter online logistic calibrator with Laplace covariance."""

    def __init__(self, cfg: dict):
        self.init_a = float(cfg.get("init_a", 1.0))
        self.init_b = float(cfg.get("init_b", 0.0))
        self.lambda_reg = float(cfg.get("lambda", 1.0))
        if self.lambda_reg <= 0.0:
            raise ValueError("OnlineLogitCalibrator requires a positive 'lambda' regularization term.")
        self.reset()

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    def predict(self, logits):
        logits_array = np.asarray(logits, dtype=np.float64)
        z = self.a * logits_array + self.b
        calibrated_prob = self._sigmoid(z)

        if np.isscalar(logits) or logits_array.ndim == 0:
            return float(np.asarray(calibrated_prob).item())
        return calibrated_prob.astype(np.float32)

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
        grad += self.lambda_reg * theta

        weights = probs * (1.0 - probs)
        hessian = (design.T * weights) @ design / logits_array.size
        hessian += self.lambda_reg * np.eye(2)

        step = np.linalg.solve(hessian, grad)
        theta = theta - step
        self.a = float(theta[0])
        self.b = float(theta[1])
        self.posterior_cov = np.linalg.inv(hessian)

    def posterior_trace(self) -> float:
        return float(np.trace(self.posterior_cov))

    def get_state(self) -> dict:
        return {
            "a": self.a,
            "b": self.b,
            "posterior_cov": self.posterior_cov.copy(),
        }

    def load_state(self, state: dict) -> None:
        self.a = float(state["a"])
        self.b = float(state["b"])
        self.posterior_cov = np.asarray(state["posterior_cov"], dtype=np.float64).copy().reshape(2, 2)

    def reset(self) -> None:
        self.a = self.init_a
        self.b = self.init_b
        self.posterior_cov = np.eye(2, dtype=np.float64) / self.lambda_reg
