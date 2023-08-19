from typing import Callable

import torch as th

mapping = {}


def register(name: str):
    def _thunk(preprocess_fn: Callable[..., th.Tensor]):
        mapping[name] = preprocess_fn
        return preprocess_fn

    return _thunk


def get_preprocess_fn(name: str):
    if name in mapping:
        return mapping[name]
    else:
        raise ValueError("Unknown preprocess function: {}".format(name))


@register("obs:none")
def dummy_obs_preprocessor(obs: th.Tensor):
    return obs
