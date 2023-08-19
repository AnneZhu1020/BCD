from typing import Optional, Tuple, Union

import numpy as np
import torch as th
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal

from minirl.common.actor_critic import ActorVCritic
from minirl.common.policy import ParamsMixin
from minirl.utils import get_callable
from minirl.torch_utils import clamp


class PPOBasePolicy(ParamsMixin, nn.Module):
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
        bisim: bool = False,
        coef_bisim: float = 1.0,
        lap: bool = False,
        coef_lap: float = 1.0,
        contrastive_batch_wise: bool = False,
        coef_contrastive: float = 1.0,
        bisim_delta: bool = False,
        coef_bisim_delta: float = 1.0,
        uncert: bool = False,
        permute: bool = False,
        similarity: str = "smooth_l1",
        uniform: bool = False,
        coef_uniform: float = 1.0,
        fb: bool = False,
        sr: bool = False,
        coef_sr: float = 1.0,
        hinge: bool = False,
        coef_hinge: float = 1.0,
        lat_sim: str = 'cosine',
        tran_sim: str = 'cosine',
        sr_sim: str = 'cosine',
        load_file: str = None,
        vq_vae: bool = False,
        coef_vq: float = 1.0,
        backward: bool = False,
        coef_back: float = 1.0,
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
        self.rnn_states = None
        self.is_recurrent = self.actor_vcritic.extractor.is_recurrent
        if init_weight_fn is not None:
            get_callable(init_weight_fn)(self)
        self.bisim = bisim
        self.coef_bisim = coef_bisim
        self.lap = lap
        self.coef_lap = coef_lap
        self.contrastive_batch_wise = contrastive_batch_wise
        self.coef_contrastive = coef_contrastive
        self.bisim_delta = bisim_delta
        self.coef_bisim_delta = coef_bisim_delta
        self.uncert = uncert
        self.permute = permute
        self.similarity = similarity
        self.uniform = uniform
        self.coef_uniform = coef_uniform
        self.fb = fb
        self.sr = sr
        self.coef_sr = coef_sr
        self.hinge = hinge
        self.coef_hinge = coef_hinge
        self.lat_sim = lat_sim
        self.tran_sim = tran_sim
        self.sr_sim = sr_sim
        self.load_file = load_file
        self.vq_vae = vq_vae
        self.coef_vq = coef_vq
        self.backward = backward
        self.coef_back = coef_back

    @th.no_grad()
    def step(self, obs, first) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        obs = th.as_tensor(obs).to(self.device).float()
        first = th.as_tensor(first).to(self.device)
        pi, value, self.rnn_states = self.actor_vcritic(obs, first, self.rnn_states)
        dist = self.distribution_cls(pi)
        action = dist.sample()
        logpacs = dist.log_prob(action)
        if isinstance(dist, Normal):
            logpacs = logpacs.sum(dim=-1)
        return (
            action.cpu().numpy(),
            value.squeeze(-1).cpu().numpy(),
            logpacs.cpu().numpy(),
        )

    @th.no_grad()
    def value(self, obs, first):
        obs = th.as_tensor(obs).to(self.device).float()
        first = th.as_tensor(first).to(self.device)
        value, _ = self.actor_vcritic.forward_critic(obs, first)
        return value.squeeze(-1).cpu().numpy()

    def loss(
        self,
        obs,
        advs,
        firsts,
        next_firsts,
        actions,
        old_values,
        old_logpacs,
        rnn_states,
        clip_range: float,
        vf_clip_range: float,
        normalize_adv: bool = False,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, dict]:
        # Convert from numpy array to torch tensor
        obs = th.as_tensor(obs).to(self.device).float()
        advs = th.as_tensor(advs).to(self.device).float()
        firsts = th.as_tensor(firsts).to(self.device)
        next_firsts = th.as_tensor(next_firsts).to(self.device)
        actions = th.as_tensor(actions).to(self.device)
        old_values = th.as_tensor(old_values).to(self.device).float()
        old_logpacs = th.as_tensor(old_logpacs).to(self.device).float()
        if rnn_states is not None:
            rnn_states = th.as_tensor(rnn_states).to(self.device).float().contiguous()
            if rnn_states.shape[-1] > 1:
                rnn_states = rnn_states.unbind(-1)
            else:
                rnn_states = rnn_states.squeeze(-1)

        # Calculate returns
        returns = advs + old_values
        # Advantage normalization
        if normalize_adv:
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        # Forward
        pi, values, _ = self.actor_vcritic(obs, firsts, rnn_states)
        values = values.squeeze(-1)

        # Compute policy loss
        dist = self.distribution_cls(pi)
        logpacs = dist.log_prob(actions)
        if isinstance(dist, Normal):
            logpacs = logpacs.sum(dim=-1)
        ratio = th.exp(logpacs - old_logpacs)
        pg_losses1 = -advs * ratio
        pg_losses2 = -advs * clamp(ratio, min=1.0 - clip_range, max=1.0 + clip_range)
        pg_loss = th.mean(th.max(pg_losses1, pg_losses2))

        # Compute value loss
        vf_losses = th.square(values - returns)
        values_clipped = clamp(
            values, min=old_values - vf_clip_range, max=old_values + vf_clip_range
        )
        vf_losses = th.max(vf_losses, th.square(values_clipped - returns))
        vf_loss = 0.5 * th.mean(vf_losses)

        # Compute entropy
        entropy = dist.entropy()
        if isinstance(dist, Normal):
            entropy = entropy.sum(dim=-1)
        entropy = th.mean(entropy)

        # Calculate additional quantities
        extra_out = {}
        with th.no_grad():
            extra_out["approx_kl"] = 0.5 * th.mean(th.square(logpacs - old_logpacs))
            extra_out["clip_frac"] = th.mean(((ratio - 1.0).abs() > clip_range).float())

        return pg_loss, vf_loss, entropy, extra_out


class PPODiscretePolicy(PPOBasePolicy):
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
        bisim: bool = False,
        coef_bisim: float = 1.0,
        lap: bool = False,
        coef_lap: float = 1.0,
        contrastive_batch_wise: bool = False,
        coef_contrastive: float = 1.0,
        bisim_delta: bool = False,
        coef_bisim_delta: float = 1.0,
        uncert: bool = False,
        permute: bool = False,
        similarity: str = "smooth_l1",
        uniform: bool = False,
        coef_uniform: float = 1.0,
        fb: bool = False,
        sr: bool = False,
        coef_sr: float = 1.0,
        hinge: bool = False,
        coef_hinge: float = 1.0,
        lat_sim: str = 'cosine',
        tran_sim: str = 'cosine',
        sr_sim: str = 'cosine',
        load_file: str = None,
        vq_vae: bool = False,
        coef_vq: float = 1.0,
        backward: bool = False,
        coef_back: float = 1.0,
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


class PPOContinuousPolicy(PPOBasePolicy):
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
