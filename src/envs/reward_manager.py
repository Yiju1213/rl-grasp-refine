from __future__ import annotations

import numpy as np

from src.structures.reward import RewardBreakdown


class RewardManager:
    """Compute the paper-aligned single-step reward."""

    def __init__(self, cfg: dict):
        self.stability_kappa = float(cfg.get("stability_kappa", 1.0))
        self.contact_lambda_cover = float(cfg.get("contact_lambda_cover", 0.1))
        self.contact_lambda_edge = float(cfg.get("contact_lambda_edge", 0.1))
        self.contact_threshold_cover = float(cfg.get("contact_threshold_cover", 0.2))
        self.contact_threshold_edge = float(cfg.get("contact_threshold_edge", 0.2))
        self.drop_success_reward = float(cfg.get("drop_success_reward", 1.0))
        self.drop_failure_reward = float(cfg.get("drop_failure_reward", -1.0))

    def compute(
        self,
        drop_success: int,
        calibrated_before: float,
        calibrated_after: float,
        posterior_trace: float,
        contact_after,
    ) -> RewardBreakdown:
        reward_drop = self.compute_drop_reward(drop_success)
        reward_stability = self.compute_stability_reward(
            calibrated_before=calibrated_before,
            calibrated_after=calibrated_after,
            posterior_trace=posterior_trace,
        )
        reward_contact = self.compute_contact_reward(contact_after=contact_after)
        total = reward_drop + reward_stability + reward_contact
        return RewardBreakdown(
            total=float(total),
            drop=float(reward_drop),
            stability=float(reward_stability),
            contact=float(reward_contact),
        )

    def compute_drop_reward(self, drop_success: int) -> float:
        return self.drop_success_reward if int(drop_success) == 1 else self.drop_failure_reward

    def compute_stability_reward(
        self,
        calibrated_before: float,
        calibrated_after: float,
        posterior_trace: float,
    ) -> float:
        delta_p = float(calibrated_after - calibrated_before)
        denom = 1.0 + self.stability_kappa * max(float(posterior_trace), 0.0)
        return float(delta_p / denom)

    def compute_contact_reward(self, contact_after) -> float:
        after = np.asarray(contact_after, dtype=np.float32).reshape(-1)
        if after.size < 2:
            raise ValueError("Contact semantics must provide at least two values: coverage and edge proximity.")
        t_cover = float(after[0])
        t_edge = float(after[1])
        coverage_penalty = self.contact_lambda_cover * max(0.0, self.contact_threshold_cover - t_cover)
        edge_penalty = self.contact_lambda_edge * max(0.0, self.contact_threshold_edge - t_edge)
        return float(-(coverage_penalty + edge_penalty))
