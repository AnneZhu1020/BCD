from copy import deepcopy
from collections import deque, defaultdict
from functools import partial
from typing import Any, Optional, Sequence

import gym3
import numpy as np

from minirl.utils import RunningMeanStd


class Wrapper(gym3.Wrapper):
    def callmethod(self, method: str, *args: Sequence[Any], **kwargs: Sequence[Any]):
        try:
            return getattr(self, method)(*args, **kwargs)
        except AttributeError:
            return self.env.callmethod(method, *args, **kwargs)


class ObsTransposeWrapper(Wrapper):
    def __init__(self, env, axes):
        self.axes = np.asarray(axes)
        ob_space = deepcopy(env.ob_space)
        new_shape = tuple((ob_space.shape[i] for i in axes))
        ob_space.shape = new_shape
        super().__init__(env, ob_space)

    def observe(self):
        rew, obs, first = self.env.observe()
        obs = np.transpose(obs, np.insert(self.axes + 1, 0, 0))
        return rew, obs, first
    
    def render(self):
        rgb_img = self.env.render(
            mode='rgb_array',
            highlight=False,
            tile_size=self.tile_size
        )
        return rgb_img


class EpisodeStatsWrapper(Wrapper):
    def __init__(self, env, buffer_size=100):
        super().__init__(env)
        self.stats_buffers = defaultdict(partial(deque, maxlen=buffer_size))
        self.ep_returns = np.zeros(self.num, dtype=np.float)
        self.ep_lengths = np.zeros(self.num, dtype=np.int)
        self.ep_count = 0

    def act(self, ac):
        _, _, first = self.observe()
        self.ep_returns[first] = 0
        self.ep_lengths[first] = 0
        self.env.act(ac)
        rew, _, first = self.observe()
        self.ep_returns += rew
        self.ep_lengths += 1
        self.ep_count += first.sum()
        self.stats_buffers["r"].extend(self.ep_returns[first])
        self.stats_buffers["l"].extend(self.ep_lengths[first])

    def get_info(self):
        infos = self.env.get_info()
        _, _, first = self.observe()
        for i, info in enumerate(infos):
            if first[i] and self.ep_lengths[i] > 0:
                ep_info = {
                    "r": self.ep_returns[i],
                    "l": self.ep_lengths[i],
                }
                info["episode"] = ep_info
        return infos

    def get_ep_stat_mean(self, key):
        return np.mean(self.stats_buffers[key]) if self.stats_buffers[key] else 0


class CollectEpisodeStatsWrapper(Wrapper):
    def __init__(self, env, ep_stat_keys, buffer_size=100):
        super().__init__(env)
        self.stats_buffers = defaultdict(partial(deque, maxlen=buffer_size))
        self.ep_stat_keys = ep_stat_keys
        self.buffer_size = buffer_size

    def act(self, ac: Any):
        super().act(ac)
        infos = self.get_info()
        for info in infos:
            for key in info.get("episode", {}):
                if key in self.ep_stat_keys:
                    self.stats_buffers[key].append(info["episode"][key])

    def get_ep_stat_mean(self, key):
        return np.mean(self.stats_buffers[key]) if self.stats_buffers[key] else 0


class NormalizeWrapper(Wrapper):
    def __init__(
        self, env, normalize_obs=True, normalize_rew=True, gamma=0.99, epsilon=1e-8,
    ):
        super().__init__(env)
        self.obs_rms = (
            RunningMeanStd(shape=self.ob_space.shape) if normalize_obs else None
        )
        self.ret_rms = RunningMeanStd(shape=()) if normalize_rew else None
        self.ret = np.zeros(self.num)
        self.gamma = gamma
        self.epsilon = epsilon
        # Update with initial obs
        rew, obs, self.first = self.env.observe()
        if self.obs_rms:
            self.obs_rms.update(obs)

    def observe(self):
        rew, obs, first = self.env.observe()
        if self.obs_rms:
            obs = self.normalize_obs(obs)
        if self.ret_rms:
            rew = self.normalize_rew(rew)
        return rew, obs, first

    def act(self, ac):
        self.ret[self.first] = 0.0
        self.env.act(ac)
        rew, obs, self.first = self.env.observe()
        if self.obs_rms:
            self.obs_rms.update(obs)
        if self.ret_rms:
            self.ret = self.ret * self.gamma + rew
            self.ret_rms.update(self.ret)

    def normalize_obs(self, obs):
        obs = (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon)
        return obs

    def normalize_rew(self, rew):
        rew = rew / np.sqrt(self.ret_rms.var + self.epsilon)
        return rew


class ClipWrapper(Wrapper):
    def __init__(
        self, env, clip_obs: Optional[float] = None, clip_rew: Optional[float] = None
    ):
        super().__init__(env)
        assert clip_obs is not None or clip_rew is not None
        self.clip_obs = clip_obs
        self.clip_rew = clip_rew

    def observe(self):
        rew, obs, first = self.env.observe()
        if self.clip_obs is not None:
            obs = np.clip(obs, -self.clip_obs, self.clip_obs)
        if self.clip_rew is not None:
            rew = np.clip(rew, -self.clip_rew, self.clip_rew)
        return rew, obs, first


class FrameStackWrapper(Wrapper):
    """
    Frame stacking wrapper
    """

    def __init__(self, env, n_stack, stack_axis=-1):
        self.n_stack = n_stack
        self.stack_axis = stack_axis
        ob_space = deepcopy(env.ob_space)
        shape = np.array(ob_space.shape)
        shape[stack_axis] = shape[stack_axis] * n_stack
        ob_space.shape = tuple(shape)
        super().__init__(env, ob_space)
        # Build obs buffer and add initial obs
        self.stacked_obs = np.zeros((env.num, *shape), ob_space.eltype.dtype_name)
        _, obs, first = self.env.observe()
        self.update_stacked_obs(obs=obs, first=first)

    def observe(self):
        rew, _, first = self.env.observe()
        return rew, self.stacked_obs, first

    def act(self, ac: Any):
        self.env.act(ac)
        _, obs, first = self.env.observe()
        self.update_stacked_obs(obs=obs, first=first)

    def update_stacked_obs(self, obs, first):
        # Roll the buffer
        self.stacked_obs = np.roll(
            self.stacked_obs, shift=-obs.shape[self.stack_axis], axis=self.stack_axis
        )
        # Create slices for indexing stacked obs
        slc = [slice(None) for _ in range(self.stacked_obs.ndim)]
        slc[self.stack_axis] = slice(-obs.shape[self.stack_axis], None)
        # Index stacked obs and rewrite it
        self.stacked_obs[first] = 0
        self.stacked_obs[tuple(slc)] = obs
