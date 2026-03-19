from __future__ import annotations

from src.structures.observation import Observation, RawSensorObservation


class ObservationBuilder:
    """Build structured observations from raw scene outputs."""

    def __init__(
        self,
        feature_extractor,
        contact_semantics_extractor,
        stability_predictor,
    ):
        self.feature_extractor = feature_extractor
        self.contact_semantics_extractor = contact_semantics_extractor
        self.stability_predictor = stability_predictor

    def build(self, raw_obs: RawSensorObservation, grasp_pose) -> Observation:
        raw_obs.grasp_metadata["grasp_pose"] = grasp_pose
        latent_feature = self.feature_extractor.extract(raw_obs)
        contact_semantic = self.contact_semantics_extractor.extract(raw_obs)
        raw_obs.grasp_metadata["contact_semantic"] = contact_semantic
        raw_stability_logit = self.stability_predictor.predict_logit(raw_obs, latent_feature)
        return Observation(
            latent_feature=latent_feature,
            contact_semantic=contact_semantic,
            grasp_pose=grasp_pose,
            raw_stability_logit=raw_stability_logit,
        )
