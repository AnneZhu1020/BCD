from typing import Optional, Tuple, Union

import numpy as np
import torch as th
from torch import nn
from torch.nn import functional as F

from minirl.common.actor_critic import Actor, QCritic
from minirl.common.policy import ParamsMixin
from minirl.utils import polyak_update


class DDPGContinuousPolicy(ParamsMixin, nn.Module):
    def __init__(
        self,
        extractor_fn: str,
        extractor_kwargs: dict,
        action_dim: int,
        actor_hiddens: Tuple[int, ...] = (),
        critic_hiddens: Tuple[int, ...] = (),
        preprocess_obs_fn: str = "obs:none",
        preprocess_obs_kwargs: Optional[dict] = None,
        target_update_ratio: float = 1.0,
        device: Union[str, th.device] = "cpu",
    ) -> None:
        super().__init__(device)

        # Build actor
        common_kwargs = {
            "extractor_fn": extractor_fn,
            "extractor_kwargs": extractor_kwargs,
            "preprocess_obs_fn": preprocess_obs_fn,
            "preprocess_obs_kwargs": preprocess_obs_kwargs,
        }
        actor_kwargs = common_kwargs.copy()
        actor_kwargs.update(
            {"squash_output": True, "hiddens": actor_hiddens, "n_outputs": action_dim}
        )
        self.actor = Actor(**actor_kwargs)
        self.actor_target = Actor(**actor_kwargs)

        # Build critic
        critic_kwargs = common_kwargs.copy()
        critic_kwargs.update({"hiddens": critic_hiddens, "action_dim": action_dim})
        self.critic = QCritic(**critic_kwargs)
        self.critic_target = QCritic(**critic_kwargs)

        # Copy weights to target networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.action_dim = action_dim
        self.target_update_ratio = target_update_ratio

    @th.no_grad()
    def step(self, obs: np.ndarray) -> np.ndarray:
        obs = th.as_tensor(obs).to(self.device).float()
        actions = self.actor(obs)
        return actions.cpu().numpy()

    def step_random(self) -> np.ndarray:
        actions = np.random.uniform(-1.0, 1.0, size=(self.action_dim, 1))
        return actions

    def critic_loss(self, obs, actions, next_obs, rewards, firsts, gamma, weights):
        # Convert from numpy array to torch tensor
        obs = th.as_tensor(obs).to(self.device).float()
        actions = th.as_tensor(actions).to(self.device).float()
        next_obs = th.as_tensor(next_obs).to(self.device).float()
        rewards = th.as_tensor(rewards).to(self.device).unsqueeze(-1).float()
        firsts = th.as_tensor(firsts).to(self.device).unsqueeze(-1).float()

        # Compute target Q-values
        with th.no_grad():
            next_q_values = self.critic_target(next_obs, self.actor_target(next_obs))
            target_q_values = rewards + (1.0 - firsts) * gamma * next_q_values

        # Compute current Q-values
        current_q_values = self.critic(obs, actions)

        # Compute critic loss
        critic_loss = F.mse_loss(current_q_values, target_q_values)

        extra_out = {}
        return critic_loss, extra_out

    def actor_loss(self, obs):
        # Convert from numpy array to torch tensor
        obs = th.as_tensor(obs).to(self.device).float()
        # Compute actor loss
        actor_loss = -self.critic(obs, self.actor(obs)).mean()
        extra_out = {}
        return actor_loss, extra_out

    def update_target_net(self):
        polyak_update(
            params=self.actor.parameters(),
            target_params=self.actor_target.parameters(),
            tau=self.target_update_ratio,
        )
        polyak_update(
            params=self.critic.parameters(),
            target_params=self.critic_target.parameters(),
            tau=self.target_update_ratio,
        )
