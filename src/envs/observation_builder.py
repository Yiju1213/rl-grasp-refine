from __future__ import annotations

from src.structures.observation import Observation, RawSensorObservation

# TODO 换个跟perception部分更贴近的名称，同时可以尝试移过去
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
        perception_result = self.feature_extractor.encode(raw_obs)
        contact_semantic = self.contact_semantics_extractor.extract(raw_obs)
        raw_stability_logit = perception_result.raw_stability_logit
        if raw_stability_logit is None:
            raw_stability_logit = self.stability_predictor.predict_logit(perception_result.latent_feature)
        return Observation(
            latent_feature=perception_result.latent_feature,
            contact_semantic=contact_semantic,
            grasp_pose=grasp_pose,
            raw_stability_logit=raw_stability_logit,
        )
