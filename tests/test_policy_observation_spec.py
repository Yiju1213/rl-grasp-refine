from __future__ import annotations

import unittest

import numpy as np

from src.rl.observation_spec import infer_obs_dim_from_spec, resolve_policy_observation_spec
from src.structures.action import GraspPose
from src.structures.observation import Observation
from src.utils.tensor_utils import observation_to_tensor
from tests.fakes import make_actor_critic_cfg, make_perception_cfg


def _make_observation(latent_dim: int = 32) -> Observation:
    return Observation(
        latent_feature=np.arange(latent_dim, dtype=np.float32),
        contact_semantic=np.asarray([0.25, -0.5], dtype=np.float32),
        grasp_pose=GraspPose(
            position=np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
            rotation=np.asarray([4.0, 5.0, 6.0], dtype=np.float32),
        ),
        raw_stability_logit=7.0,
    )


class TestPolicyObservationSpec(unittest.TestCase):
    def test_current_matches_legacy_layout(self):
        perception_cfg = make_perception_cfg()
        actor_critic_cfg = make_actor_critic_cfg()
        spec = resolve_policy_observation_spec(perception_cfg, actor_critic_cfg)

        self.assertEqual(spec.components[0], "latent_feature")
        self.assertEqual(spec.components, ("latent_feature", "contact_semantic", "grasp_position", "grasp_rotation", "raw_stability_logit"))
        self.assertEqual(infer_obs_dim_from_spec(spec), 41)

        obs = _make_observation()
        obs_tensor = observation_to_tensor(obs, spec=spec)
        self.assertEqual(tuple(obs_tensor.shape), (1, 41))
        np.testing.assert_allclose(obs_tensor.squeeze(0).numpy()[:32], obs.latent_feature)

    def test_paper_preset_keeps_only_latent_and_contact(self):
        perception_cfg = make_perception_cfg()
        actor_critic_cfg = make_actor_critic_cfg()
        actor_critic_cfg["policy_observation"] = {"preset": "paper"}
        spec = resolve_policy_observation_spec(perception_cfg, actor_critic_cfg)

        self.assertEqual(spec.components[0], "latent_feature")
        self.assertEqual(spec.components, ("latent_feature", "contact_semantic"))
        self.assertEqual(infer_obs_dim_from_spec(spec), 34)

        obs_tensor = observation_to_tensor(_make_observation(), spec=spec)
        self.assertEqual(tuple(obs_tensor.shape), (1, 34))

    def test_component_override_uses_canonical_order(self):
        perception_cfg = make_perception_cfg()
        actor_critic_cfg = make_actor_critic_cfg()
        actor_critic_cfg["policy_observation"] = {
            "preset": "current",
            "components": ["raw_stability_logit", "contact_semantic", "latent_feature"],
        }
        spec = resolve_policy_observation_spec(perception_cfg, actor_critic_cfg)

        self.assertEqual(spec.components[0], "latent_feature")
        self.assertEqual(spec.components, ("latent_feature", "contact_semantic", "raw_stability_logit"))
        self.assertEqual(infer_obs_dim_from_spec(spec), 35)

    def test_no_pose_and_no_logit_dims(self):
        perception_cfg = make_perception_cfg()

        no_pose_cfg = make_actor_critic_cfg()
        no_pose_cfg["policy_observation"] = {"preset": "no_pose"}
        no_pose_spec = resolve_policy_observation_spec(perception_cfg, no_pose_cfg)
        self.assertEqual(no_pose_spec.components[0], "latent_feature")
        self.assertEqual(infer_obs_dim_from_spec(no_pose_spec), 35)

        no_logit_cfg = make_actor_critic_cfg()
        no_logit_cfg["policy_observation"] = {"preset": "no_logit"}
        no_logit_spec = resolve_policy_observation_spec(perception_cfg, no_logit_cfg)
        self.assertEqual(no_logit_spec.components[0], "latent_feature")
        self.assertEqual(infer_obs_dim_from_spec(no_logit_spec), 40)


if __name__ == "__main__":
    unittest.main()
