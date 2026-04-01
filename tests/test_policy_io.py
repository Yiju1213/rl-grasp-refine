from __future__ import annotations

import unittest

import torch

from src.models.rl.actor_critic import ActorCritic
from src.models.rl.policy_network import LatentFirstLateFusionPolicyNetwork, PolicyNetwork
from src.models.rl.value_network import LatentFirstLateFusionValueNetwork, ValueNetwork
from tests.fakes import make_actor_critic_cfg


class TestPolicyIO(unittest.TestCase):
    def test_policy_and_value_shapes(self):
        cfg = make_actor_critic_cfg()
        policy = PolicyNetwork(obs_dim=41, action_dim=6, cfg=cfg)
        value = ValueNetwork(obs_dim=41, cfg=cfg)
        actor_critic = ActorCritic(policy_net=policy, value_net=value)
        obs_tensor = torch.zeros(3, 41)

        action_mean, action_log_std = policy(obs_tensor)
        self.assertEqual(action_mean.shape, (3, 6))
        self.assertEqual(action_log_std.shape, (3, 6))

        value_tensor = value(obs_tensor)
        self.assertEqual(value_tensor.shape, (3, 1))

        action, log_prob, predicted_value, entropy = actor_critic.act(obs_tensor)
        self.assertEqual(action.shape, (3, 6))
        self.assertEqual(log_prob.shape, (3,))
        self.assertEqual(predicted_value.shape, (3,))
        self.assertEqual(entropy.shape, (3,))

    def test_late_fusion_policy_and_value_shapes(self):
        cfg = make_actor_critic_cfg()
        cfg["architecture"] = {"type": "latent_first_late_fusion"}
        policy = LatentFirstLateFusionPolicyNetwork(latent_dim=32, aux_dim=9, action_dim=6, cfg=cfg)
        value = LatentFirstLateFusionValueNetwork(latent_dim=32, aux_dim=9, cfg=cfg)
        actor_critic = ActorCritic(policy_net=policy, value_net=value)
        obs_tensor = torch.zeros(3, 41)

        action_mean, action_log_std = policy(obs_tensor)
        self.assertEqual(action_mean.shape, (3, 6))
        self.assertEqual(action_log_std.shape, (3, 6))

        value_tensor = value(obs_tensor)
        self.assertEqual(value_tensor.shape, (3, 1))

        action, log_prob, predicted_value, entropy = actor_critic.act(obs_tensor)
        self.assertEqual(action.shape, (3, 6))
        self.assertEqual(log_prob.shape, (3,))
        self.assertEqual(predicted_value.shape, (3,))
        self.assertEqual(entropy.shape, (3,))

    def test_late_fusion_supports_paper_preset_aux_only_contact(self):
        cfg = make_actor_critic_cfg()
        cfg["architecture"] = {"type": "latent_first_late_fusion"}
        cfg["policy_observation"] = {"preset": "paper"}
        policy = LatentFirstLateFusionPolicyNetwork(latent_dim=32, aux_dim=2, action_dim=6, cfg=cfg)
        value = LatentFirstLateFusionValueNetwork(latent_dim=32, aux_dim=2, cfg=cfg)

        obs_tensor = torch.zeros(2, 34)
        action_mean, action_log_std = policy(obs_tensor)
        value_tensor = value(obs_tensor)

        self.assertEqual(action_mean.shape, (2, 6))
        self.assertEqual(action_log_std.shape, (2, 6))
        self.assertEqual(value_tensor.shape, (2, 1))


if __name__ == "__main__":
    unittest.main()
