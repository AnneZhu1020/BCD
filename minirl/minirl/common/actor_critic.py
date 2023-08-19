from typing import Optional, Sequence, Tuple, Type

import torch as th
import torch.nn as nn

from minirl.network import MLP
from minirl.common.policy import Extractor


class ActorHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_outputs: int,
        hiddens: Tuple[int, ...] = (),
        activation: Type[nn.Module] = nn.ReLU,
        squash_output: bool = False,
    ) -> None:
        super().__init__()
        self.pi = MLP(
            input_dim=input_dim,
            hiddens=(*hiddens, n_outputs),
            activation=activation,
            final_activation=nn.Tanh if squash_output else nn.Identity,
        )

    def forward(self, features: th.Tensor) -> th.Tensor:
        T, B, _ = features.shape
        return self.pi(features.view(T * B, -1)).view(T, B, -1)


class QCriticHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hiddens: Tuple[int, ...] = (),
        activation: Type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        self.qf = MLP(
            input_dim=input_dim + action_dim,
            hiddens=(*hiddens, 1),
            activation=activation,
        )

    def forward(self, features: th.Tensor, actions: th.Tensor) -> th.Tensor:
        T, B, _ = features.shape
        qf_input = th.cat([features, actions], dim=2).view(T * B, -1)
        return self.qf(qf_input).view(T, B, -1)


class VCriticHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hiddens: Tuple[int, ...] = (),
        activation: Type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        self.vf = MLP(input_dim=input_dim, hiddens=(*hiddens, 1), activation=activation)

    def forward(self, features: th.Tensor) -> th.Tensor:
        T, B, _ = features.shape
        return self.vf(features.view(T * B, -1)).view(T, B, -1)


class Actor(Extractor):
    def __init__(
        self,
        n_outputs: int,
        extractor_fn: str,
        extractor_kwargs: dict,
        hiddens: Tuple[int, ...] = (),
        activation: Type[nn.Module] = nn.ReLU,
        squash_output: bool = False,
        preprocess_obs_fn: str = "obs:none",
        preprocess_obs_kwargs: Optional[dict] = None,
    ) -> None:
        super().__init__(
            extractor_fn=extractor_fn,
            extractor_kwargs=extractor_kwargs,
            preprocess_obs_fn=preprocess_obs_fn,
            preprocess_obs_kwargs=preprocess_obs_kwargs,
        )
        self.actor_head = ActorHead(
            input_dim=self.extractor.output_dim,
            n_outputs=n_outputs,
            hiddens=hiddens,
            activation=activation,
            squash_output=squash_output,
        )

    def forward(self, obs: th.Tensor, first: th.Tensor, states=None):
        features, states = self.extract_features(obs, first, states)
        pi = self.actor_head(features)
        return pi, states


class QCritic(Extractor):
    def __init__(
        self,
        action_dim: int,
        extractor_fn: str = None,
        extractor_kwargs: dict = None,
        hiddens: Tuple[int, ...] = (),
        activation: Type[nn.Module] = nn.ReLU,
        preprocess_obs_fn: str = "obs:none",
        preprocess_obs_kwargs: Optional[dict] = None,
    ) -> None:
        super().__init__(
            extractor_fn=extractor_fn,
            extractor_kwargs=extractor_kwargs,
            preprocess_obs_fn=preprocess_obs_fn,
            preprocess_obs_kwargs=preprocess_obs_kwargs,
        )
        self.qcritic_head = QCriticHead(
            input_dim=self.extractor.output_dim,
            action_dim=action_dim,
            hiddens=hiddens,
            activation=activation,
        )

    def forward(
        self, obs: th.Tensor, actions: th.Tensor, first: th.Tensor, states=None
    ):
        features, states = self.extract_features(obs, first, states)
        qf = self.qcritic_head(features, actions)
        return qf, states


class VCritic(Extractor):
    def __init__(
        self,
        extractor_fn: str = None,
        extractor_kwargs: dict = None,
        hiddens: Tuple[int, ...] = (),
        activation: Type[nn.Module] = nn.ReLU,
        preprocess_obs_fn: str = "obs:none",
        preprocess_obs_kwargs: Optional[dict] = None,
    ) -> None:
        super().__init__(
            extractor_fn=extractor_fn,
            extractor_kwargs=extractor_kwargs,
            preprocess_obs_fn=preprocess_obs_fn,
            preprocess_obs_kwargs=preprocess_obs_kwargs,
        )
        self.vcritic_head = VCriticHead(
            input_dim=self.extractor.output_dim, hiddens=hiddens, activation=activation
        )

    def forward(self, obs: th.Tensor, first: th.Tensor, states=None):
        features, states = self.extract_features(obs, first, states)
        value = self.vcritic_head(features)
        return value, states


class ActorVCritic(Extractor):
    def __init__(
        self,
        n_outputs: int,
        extractor_fn: str = None,
        extractor_kwargs: dict = None,
        actor_hiddens: Tuple[int, ...] = (),
        critic_hiddens: Tuple[int, ...] = (),
        activation: Type[nn.Module] = nn.ReLU,
        preprocess_obs_fn: str = "obs:none",
        preprocess_obs_kwargs: Optional[dict] = None,
    ) -> None:
        super().__init__(
            extractor_fn=extractor_fn,
            extractor_kwargs=extractor_kwargs,
            preprocess_obs_fn=preprocess_obs_fn,
            preprocess_obs_kwargs=preprocess_obs_kwargs,
        )
        self.actor_head = ActorHead(
            input_dim=self.extractor.output_dim,
            n_outputs=n_outputs,
            hiddens=actor_hiddens,
            activation=activation,
        )
        self.vcritic_head = VCriticHead(
            input_dim=self.extractor.output_dim,
            hiddens=critic_hiddens,
            activation=activation,
        )

    def forward(self, obs: th.Tensor, first: th.Tensor, states=None):
        features, states = self.extract_features(obs, first, states)
        pi = self.actor_head(features)
        value = self.vcritic_head(features)
        return pi, value, states

    def forward_actor(self, obs: th.Tensor, first: th.Tensor, states=None):
        features, states = self.extract_features(obs, first, states)
        pi = self.actor_head(features)
        return pi, states

    def forward_critic(self, obs: th.Tensor, first: th.Tensor, states=None):
        features, states = self.extract_features(obs, first, states)
        value = self.vcritic_head(features)
        return value, states
