from __future__ import annotations

import unittest

import numpy as np
import torch

from src.runtime.builders import build_actor_critic
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

    def test_paper_camgeom_preset_adds_camera_geometry_context(self):
        perception_cfg = make_perception_cfg()
        actor_critic_cfg = make_actor_critic_cfg()
        actor_critic_cfg["policy_observation"] = {"preset": "paper_camgeom"}
        spec = resolve_policy_observation_spec(perception_cfg, actor_critic_cfg)

        self.assertEqual(
            spec.components,
            ("latent_feature", "contact_semantic", "action_axes_in_camera", "hand_pose_in_camera"),
        )
        self.assertEqual(infer_obs_dim_from_spec(spec), 55)

        obs = _make_observation()
        obs.action_axes_in_camera = np.arange(9, dtype=np.float32)
        obs.hand_pose_in_camera = np.arange(12, dtype=np.float32) + 10.0
        obs_tensor = observation_to_tensor(obs, spec=spec).squeeze(0).numpy()

        self.assertEqual(obs_tensor.shape, (55,))
        np.testing.assert_allclose(obs_tensor[:32], obs.latent_feature)
        np.testing.assert_allclose(obs_tensor[32:34], obs.contact_semantic)
        np.testing.assert_allclose(obs_tensor[34:43], obs.action_axes_in_camera)
        np.testing.assert_allclose(obs_tensor[43:55], obs.hand_pose_in_camera)

    def test_paper_fingergeom_preset_adds_compact_finger_geometry(self):
        perception_cfg = make_perception_cfg()
        actor_critic_cfg = make_actor_critic_cfg()
        actor_critic_cfg["policy_observation"] = {"preset": "paper_fingergeom"}
        spec = resolve_policy_observation_spec(perception_cfg, actor_critic_cfg)

        self.assertEqual(
            spec.components,
            ("latent_feature", "contact_semantic", "finger_geometry_in_camera"),
        )
        self.assertEqual(infer_obs_dim_from_spec(spec), 43)

        obs = _make_observation()
        obs.finger_geometry_in_camera = np.arange(9, dtype=np.float32) + 20.0
        obs_tensor = observation_to_tensor(obs, spec=spec).squeeze(0).numpy()

        self.assertEqual(obs_tensor.shape, (43,))
        np.testing.assert_allclose(obs_tensor[:32], obs.latent_feature)
        np.testing.assert_allclose(obs_tensor[32:34], obs.contact_semantic)
        np.testing.assert_allclose(obs_tensor[34:43], obs.finger_geometry_in_camera)

    def test_paper_allgeom_preset_adds_camera_and_finger_geometry(self):
        perception_cfg = make_perception_cfg()
        actor_critic_cfg = make_actor_critic_cfg()
        actor_critic_cfg["policy_observation"] = {"preset": "paper_allgeom"}
        spec = resolve_policy_observation_spec(perception_cfg, actor_critic_cfg)

        self.assertEqual(
            spec.components,
            (
                "latent_feature",
                "contact_semantic",
                "action_axes_in_camera",
                "hand_pose_in_camera",
                "finger_geometry_in_camera",
            ),
        )
        self.assertEqual(infer_obs_dim_from_spec(spec), 64)

        obs = _make_observation()
        obs.action_axes_in_camera = np.arange(9, dtype=np.float32)
        obs.hand_pose_in_camera = np.arange(12, dtype=np.float32) + 10.0
        obs.finger_geometry_in_camera = np.arange(9, dtype=np.float32) + 30.0
        obs_tensor = observation_to_tensor(obs, spec=spec).squeeze(0).numpy()

        self.assertEqual(obs_tensor.shape, (64,))
        np.testing.assert_allclose(obs_tensor[:32], obs.latent_feature)
        np.testing.assert_allclose(obs_tensor[32:34], obs.contact_semantic)
        np.testing.assert_allclose(obs_tensor[34:43], obs.action_axes_in_camera)
        np.testing.assert_allclose(obs_tensor[43:55], obs.hand_pose_in_camera)
        np.testing.assert_allclose(obs_tensor[55:64], obs.finger_geometry_in_camera)

    def test_late_fusion_actor_critic_accepts_camgeom_aux_features(self):
        perception_cfg = make_perception_cfg()
        actor_critic_cfg = make_actor_critic_cfg()
        actor_critic_cfg["architecture"] = {"type": "latent_first_late_fusion"}
        actor_critic_cfg["policy_observation"] = {"preset": "paper_camgeom"}

        actor_critic = build_actor_critic(perception_cfg, actor_critic_cfg)
        obs_dim = infer_obs_dim_from_spec(actor_critic.observation_spec)

        self.assertEqual(obs_dim, 55)
        self.assertEqual(actor_critic.policy_net.aux_dim, 23)
        self.assertEqual(actor_critic.value_net.aux_dim, 23)
        action, log_prob, value, entropy = actor_critic.act(torch.zeros(2, obs_dim, dtype=torch.float32))
        self.assertEqual(tuple(action.shape), (2, 6))
        self.assertEqual(tuple(log_prob.shape), (2,))
        self.assertEqual(tuple(value.shape), (2,))
        self.assertEqual(tuple(entropy.shape), (2,))

    def test_late_fusion_actor_critic_accepts_fingergeom_aux_features(self):
        perception_cfg = make_perception_cfg()
        actor_critic_cfg = make_actor_critic_cfg()
        actor_critic_cfg["architecture"] = {"type": "latent_first_late_fusion"}
        actor_critic_cfg["policy_observation"] = {"preset": "paper_fingergeom"}

        actor_critic = build_actor_critic(perception_cfg, actor_critic_cfg)
        obs_dim = infer_obs_dim_from_spec(actor_critic.observation_spec)

        self.assertEqual(obs_dim, 43)
        self.assertEqual(actor_critic.policy_net.aux_dim, 11)
        self.assertEqual(actor_critic.value_net.aux_dim, 11)
        action, log_prob, value, entropy = actor_critic.act(torch.zeros(2, obs_dim, dtype=torch.float32))
        self.assertEqual(tuple(action.shape), (2, 6))
        self.assertEqual(tuple(log_prob.shape), (2,))
        self.assertEqual(tuple(value.shape), (2,))
        self.assertEqual(tuple(entropy.shape), (2,))

    def test_late_fusion_actor_critic_accepts_allgeom_aux_features(self):
        perception_cfg = make_perception_cfg()
        actor_critic_cfg = make_actor_critic_cfg()
        actor_critic_cfg["architecture"] = {"type": "latent_first_late_fusion"}
        actor_critic_cfg["policy_observation"] = {"preset": "paper_allgeom"}

        actor_critic = build_actor_critic(perception_cfg, actor_critic_cfg)
        obs_dim = infer_obs_dim_from_spec(actor_critic.observation_spec)

        self.assertEqual(obs_dim, 64)
        self.assertEqual(actor_critic.policy_net.aux_dim, 32)
        self.assertEqual(actor_critic.value_net.aux_dim, 32)
        action, log_prob, value, entropy = actor_critic.act(torch.zeros(2, obs_dim, dtype=torch.float32))
        self.assertEqual(tuple(action.shape), (2, 6))
        self.assertEqual(tuple(log_prob.shape), (2,))
        self.assertEqual(tuple(value.shape), (2,))
        self.assertEqual(tuple(entropy.shape), (2,))


if __name__ == "__main__":
    unittest.main()
