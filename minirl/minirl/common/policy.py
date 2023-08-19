from collections import OrderedDict
from typing import Optional, Sequence, Tuple, Union

import torch as th
from torch import nn

from minirl.network import get_network_class
from minirl.preprocess import get_preprocess_fn


class Extractor(nn.Module):
    def __init__(
        self,
        extractor_fn: str,
        extractor_kwargs: dict,
        preprocess_obs_fn: str = "obs:none",
        preprocess_obs_kwargs: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.extractor = get_network_class(extractor_fn)(**extractor_kwargs)
        self.is_recurrent = any(
            isinstance(m, nn.RNNBase) for m in self.extractor.modules()
        )
        self.preprocess_obs_fn = preprocess_obs_fn
        self.preprocess_obs_kwargs = preprocess_obs_kwargs or {}

    def preprocess_obs(self, obs: th.Tensor) -> th.Tensor:
        preprocess_fn = get_preprocess_fn(self.preprocess_obs_fn)
        return preprocess_fn(obs, **self.preprocess_obs_kwargs)

    def extract_features(self, obs: th.Tensor, first: th.Tensor, states=None):
        # obs: T x B x shape
        # first: T x B
        # states: (N x B x C)
        obs = self.preprocess_obs(obs)
        if self.is_recurrent:
            features_lst = []
            for ob, mask in zip(obs.unbind(), first.unbind()):
                if states is not None:
                    if isinstance(states, tuple):
                        states = tuple(mask[..., None] * s for s in states)
                    else:
                        states = mask[..., None] * states
                features, states = self.extractor(ob[None, ...], states)
                features_lst.append(features)
            features = th.cat(features_lst, dim=0)
        else:
            T, B, *shape = obs.shape
            features = self.extractor(obs.view(T * B, *shape)).view(T, B, -1)
        return features, states


class ParamsMixin(nn.Module):
    def __init__(self, device: Union[str, th.device] = "cpu") -> None:
        super().__init__()
        self.device = device

    def get_params(self) -> OrderedDict:
        params = OrderedDict(
            {name: weight.detach().cpu() for name, weight in self.state_dict().items()}
        )
        return params

    def set_params(self, params: OrderedDict) -> None:
        self.load_state_dict(params)
        self.to(self.device)
