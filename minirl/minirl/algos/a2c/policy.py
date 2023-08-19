from typing import Optional, Tuple, Union

import numpy as np
import torch as th
from torch import nn
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal

from minirl.common.actor_critic import ActorVCritic
from minirl.common.policy import ParamsMixin
from minirl.utils import get_callable


class A2CBasePolicy(ParamsMixin, nn.Module):
    def __init__(
        self,
        extractor_fn: str,
        extractor_kwargs: dict,
        n_outputs: int,
        actor_hiddens: Tuple[int, ...] = (),
        critic_hiddens: Tuple[int, ...] = (),
        activation_fn: str = "torch.nn.ReLU",
        preprocess_obs_fn: str = "obs:none",
        preprocess_obs_kwargs: Optional[dict] = None,
        init_weight_fn: Optional[str] = None,
        device: Union[str, th.device] = "cpu",
    ):
        super().__init__(device)
        self.actor_vcritic = ActorVCritic(
            n_outputs=n_outputs,
            extractor_fn=extractor_fn,
            extractor_kwargs=extractor_kwargs,
            actor_hiddens=actor_hiddens,
            critic_hiddens=critic_hiddens,
            activation=get_callable(activation_fn),
            preprocess_obs_fn=preprocess_obs_fn,
            preprocess_obs_kwargs=preprocess_obs_kwargs,
        )
        if init_weight_fn is not None:
            get_callable(init_weight_fn)(self)

    @th.no_grad()
    def step(self, obs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        obs = th.as_tensor(obs).to(self.device).float()
        pi, value = self.actor_vcritic(obs)
        dist = self.distribution_cls(pi)
        action = dist.sample()
        logpacs = dist.log_prob(action)
        if len(logpacs.shape) > 1:
            logpacs = logpacs.sum(dim=1)
        return (
            action.cpu().numpy(),
            value.squeeze(-1).cpu().numpy(),
            logpacs.cpu().numpy(),
        )

    @th.no_grad()
    def value(self, obs):
        obs = th.as_tensor(obs).to(self.device).float()
        value = self.actor_vcritic.forward_critic(obs)
        return value.squeeze(-1).cpu().numpy()

    def loss(self, obs, advs, firsts, actions, old_values, normalize_adv: bool = False):
        # Convert from numpy array to torch tensor
        obs = th.as_tensor(obs).to(self.device).float()
        advs = th.as_tensor(advs).to(self.device).float()
        firsts = th.as_tensor(firsts).to(self.device)
        actions = th.as_tensor(actions).to(self.device)
        old_values = th.as_tensor(old_values).to(self.device).float()

        # Calculate returns
        returns = advs + old_values
        # Advantage normalization
        if normalize_adv:
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        # Forward
        pi, values = self.actor_vcritic(obs)
        values = values.squeeze(-1)

        # Compute policy loss
        dist = self.distribution_cls(pi)
        logpacs = dist.log_prob(actions)
        if len(logpacs.shape) > 1:
            logpacs = logpacs.sum(dim=1)
        pg_loss = th.mean(-advs * logpacs)

        # Compute value loss
        vf_loss = 0.5 * th.mean(th.square(values - returns))

        # Compute entropy
        entropy = dist.entropy()
        if len(entropy.shape) > 1:
            entropy = entropy.sum(dim=1)
        entropy = th.mean(entropy)

        # Calculate additional quantities
        extra_out = {}

        return pg_loss, vf_loss, entropy, extra_out


class A2CDiscretePolicy(A2CBasePolicy):
    def __init__(
        self,
        extractor_fn: str,
        extractor_kwargs: dict,
        n_actions: int,
        actor_hiddens: Tuple[int, ...] = (),
        critic_hiddens: Tuple[int, ...] = (),
        activation_fn: str = "torch.nn.ReLU",
        preprocess_obs_fn: str = "obs:none",
        preprocess_obs_kwargs: Optional[dict] = None,
        init_weight_fn: Optional[str] = None,
        device: Union[str, th.device] = "cpu",
    ):
        super().__init__(
            extractor_fn=extractor_fn,
            extractor_kwargs=extractor_kwargs,
            n_outputs=n_actions,
            actor_hiddens=actor_hiddens,
            critic_hiddens=critic_hiddens,
            activation_fn=activation_fn,
            preprocess_obs_fn=preprocess_obs_fn,
            preprocess_obs_kwargs=preprocess_obs_kwargs,
            init_weight_fn=init_weight_fn,
            device=device,
        )
        self.n_actions = n_actions
        self.distribution_cls = lambda pi: Categorical(logits=pi)


class A2CContinuousPolicy(A2CBasePolicy):
    def __init__(
        self,
        extractor_fn: str,
        extractor_kwargs: dict,
        action_dim: int,
        log_std_init: float = 0.0,
        actor_hiddens: Tuple[int, ...] = (),
        critic_hiddens: Tuple[int, ...] = (),
        activation_fn: str = "torch.nn.ReLU",
        preprocess_obs_fn: str = "obs:none",
        preprocess_obs_kwargs: Optional[dict] = None,
        init_weight_fn: Optional[str] = None,
        device: Union[str, th.device] = "cpu",
    ):
        super().__init__(
            extractor_fn=extractor_fn,
            extractor_kwargs=extractor_kwargs,
            n_outputs=action_dim,
            actor_hiddens=actor_hiddens,
            critic_hiddens=critic_hiddens,
            activation_fn=activation_fn,
            preprocess_obs_fn=preprocess_obs_fn,
            preprocess_obs_kwargs=preprocess_obs_kwargs,
            init_weight_fn=init_weight_fn,
            device=device,
        )
        self.action_dim = action_dim
        self.log_std_init = log_std_init
        self.log_std = nn.Parameter(th.ones(action_dim, device=device) * log_std_init)
        self.distribution_cls = lambda pi: Normal(
            loc=pi, scale=th.ones_like(pi) * self.log_std.exp()
        )
