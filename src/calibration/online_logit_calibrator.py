from __future__ import annotations

import numpy as np

from src.calibration.base_calibrator import BaseCalibrator


class OnlineLogitCalibrator(BaseCalibrator):
    """Two-parameter online logistic calibrator with Laplace covariance."""

    def __init__(self, cfg: dict):
        self.init_a = float(cfg.get("init_a", 1.0))
        self.init_b = float(cfg.get("init_b", 0.0))
        self.lambda_reg = float(cfg.get("lambda", 1.0))
        self.online_update_enabled = bool(cfg.get("online_update_enabled", True))
        self.signal_mode = str(cfg.get("signal_mode", "calibrated_probability") or "calibrated_probability").strip()
        self.uncertainty_discount_enabled = bool(cfg.get("uncertainty_discount_enabled", True))
        if self.lambda_reg <= 0.0:
            raise ValueError("OnlineLogitCalibrator requires a positive 'lambda' regularization term.")
        if self.signal_mode not in {"calibrated_probability", "identity_probability"}:
            raise ValueError(
                "OnlineLogitCalibrator 'signal_mode' must be one of "
                "{'calibrated_probability', 'identity_probability'}."
            )
        self.reset()

    def _diagnostics_for_no_update(self) -> dict[str, float]:
        posterior_trace = self.posterior_trace()
        return {
            "a_before": float(self.a),
            "b_before": float(self.b),
            "a_after": float(self.a),
            "b_after": float(self.b),
            "da": 0.0,
            "db": 0.0,
            "grad_norm": 0.0,
            "hessian_cond": 0.0,
            "scale_negative_flag": float(self.a < 0.0),
            "posterior_trace": float(posterior_trace),
        }

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    def predict(self, logits):
        logits_array = np.asarray(logits, dtype=np.float64)
        if self.signal_mode == "identity_probability":
            identity_prob = self._sigmoid(logits_array)
            if np.isscalar(logits) or logits_array.ndim == 0:
                return float(np.asarray(identity_prob).item())
            return identity_prob.astype(np.float32)
        z = self.a * logits_array + self.b
        calibrated_prob = self._sigmoid(z)

        if np.isscalar(logits) or logits_array.ndim == 0:
            return float(np.asarray(calibrated_prob).item())
        return calibrated_prob.astype(np.float32)

    def update(self, logits, labels) -> None:
        if not self.online_update_enabled or self.signal_mode != "calibrated_probability":
            self.last_update_diagnostics = self._diagnostics_for_no_update()
            return
        logits_array = np.asarray(logits, dtype=np.float64).reshape(-1)
        labels_array = np.asarray(labels, dtype=np.float64).reshape(-1)
        if logits_array.size == 0:
            self.last_update_diagnostics = self._diagnostics_for_no_update()
            return
        if logits_array.shape != labels_array.shape:
            raise ValueError("logits and labels must have the same shape.")

        design = np.stack([logits_array, np.ones_like(logits_array)], axis=1)
        theta = np.asarray([self.a, self.b], dtype=np.float64)
        theta_before = theta.copy()
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
        self.last_update_diagnostics = {
            "a_before": float(theta_before[0]),
            "b_before": float(theta_before[1]),
            "a_after": float(theta[0]),
            "b_after": float(theta[1]),
            "da": float(theta[0] - theta_before[0]),
            "db": float(theta[1] - theta_before[1]),
            "grad_norm": float(np.linalg.norm(grad)),
            "hessian_cond": float(np.linalg.cond(hessian)),
            "scale_negative_flag": float(theta[0] < 0.0),
            "posterior_trace": float(np.trace(self.posterior_cov)),
        }

    def posterior_trace(self) -> float:
        if not self.uncertainty_discount_enabled:
            return 0.0
        return float(np.trace(self.posterior_cov))

    def get_update_diagnostics(self) -> dict[str, float]:
        return dict(self.last_update_diagnostics)

    def get_state(self) -> dict:
        return {
            "a": self.a,
            "b": self.b,
            "posterior_cov": self.posterior_cov.copy(),
        }

    def load_state(self, state: dict) -> None:
        self.a = float(state["a"])
        self.b = float(state["b"])
        if not self.uncertainty_discount_enabled:
            self.posterior_cov = np.zeros((2, 2), dtype=np.float64)
            self.last_update_diagnostics = self._diagnostics_for_no_update()
            return
        self.posterior_cov = np.asarray(state["posterior_cov"], dtype=np.float64).copy().reshape(2, 2)
        self.last_update_diagnostics = self._diagnostics_for_no_update()

    def reset(self) -> None:
        self.a = self.init_a
        self.b = self.init_b
        if self.uncertainty_discount_enabled:
            self.posterior_cov = np.eye(2, dtype=np.float64) / self.lambda_reg
        else:
            self.posterior_cov = np.zeros((2, 2), dtype=np.float64)
        self.last_update_diagnostics = self._diagnostics_for_no_update()
