from __future__ import annotations

import numpy as np

from src.structures.reward import RewardBreakdown


class RewardManager:
    """Compute the decomposed v1 reward."""

    def __init__(self, cfg: dict):
        weights = cfg.get("weights", {})
        self.w_drop = float(weights.get("drop", 1.0))
        self.w_stability = float(weights.get("stability", 0.5))
        self.w_contact = float(weights.get("contact", 0.1))
        self.stability_alpha = float(cfg.get("stability_alpha", 0.1))
        self.stability_clip = tuple(cfg.get("stability_clip", [-1.0, 1.0]))
        self.contact_beta = float(cfg.get("contact_beta", 0.5))
        self.contact_clip = tuple(cfg.get("contact_clip", [-0.25, 0.25]))
        self.drop_success_reward = float(cfg.get("drop_success_reward", 1.0))
        self.drop_failure_reward = float(cfg.get("drop_failure_reward", -1.0))

    def compute(
        self,
        drop_success: int,
        calibrated_before: float,
        calibrated_after: float,
        uncertainty_before: float,
        uncertainty_after: float,
        contact_before,
        contact_after,
    ) -> RewardBreakdown:
        reward_drop = self.compute_drop_reward(drop_success)
        reward_stability = self.compute_stability_reward(
            calibrated_before=calibrated_before,
            calibrated_after=calibrated_after,
            uncertainty_before=uncertainty_before,
            uncertainty_after=uncertainty_after,
        )
        reward_contact = self.compute_contact_reward(contact_before=contact_before, contact_after=contact_after)
        total = (
            self.w_drop * reward_drop
            + self.w_stability * reward_stability
            + self.w_contact * reward_contact
        )
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
        uncertainty_before: float,
        uncertainty_after: float,
    ) -> float:
        gain = float(calibrated_after - calibrated_before)
        uncertainty_penalty = self.stability_alpha * max(float(uncertainty_after - uncertainty_before), 0.0)
        reward = gain - uncertainty_penalty
        return float(np.clip(reward, self.stability_clip[0], self.stability_clip[1]))

    def compute_contact_reward(self, contact_before, contact_after) -> float:
        before = np.asarray(contact_before, dtype=np.float32).reshape(-1)
        after = np.asarray(contact_after, dtype=np.float32).reshape(-1)
        if before.size < 2 or after.size < 2:
            raise ValueError("Contact semantics must provide at least two values: coverage and edge proximity.")
        coverage_gain = float(after[0] - before[0])
        edge_penalty = self.contact_beta * float(after[1] - before[1])
        reward = coverage_gain - edge_penalty
        return float(np.clip(reward, self.contact_clip[0], self.contact_clip[1]))
