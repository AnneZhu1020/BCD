import time
from typing import Any, Dict, List, Tuple

import numpy as np
import gym

from minirl.envs.atari.dm_wrappers import wrap_deepmind


class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """
        Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        :param env: (Gym Environment) the environment to wrap
        :param noop_max: (int) the maximum value of no-ops to run
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == "NOOP"

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1)
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, action):
        return self.env.step(action)


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """
        Return only every `skip`-th frame (frameskipping)
        :param env: (Gym Environment) the environment
        :param skip: (int) number of `skip`-th frame
        """
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros(
            (2,) + env.observation_space.shape, dtype=env.observation_space.dtype
        )
        self._skip = skip

    def step(self, action):
        """
        Step the environment with the given action
        Repeat action, sum reward, and max over last observations.
        :param action: ([int] or [float]) the action
        :return: ([int] or [float], [float], [bool], dict) observation, reward, done, information
        """
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class Monitor(gym.Wrapper):
    """
    A monitor wrapper for Gym environments to record the episode infos (e.g. reward, length)
    """

    def __init__(self, env: gym.Env, info_keywords=()):
        super().__init__(env)
        self.t_start = time.time()
        self.info_keywords = info_keywords
        self.rewards = None
        self.needs_reset = True
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_times = []
        self.total_steps = 0

    def reset(self, **kwargs) -> np.ndarray:
        """
        Calls the Gym environment reset.
        """
        self.rewards = []
        self.needs_reset = False
        return self.env.reset(**kwargs)

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, Dict[Any, Any]]:
        """
        Step the environment with the given action
        """
        if self.needs_reset:
            raise RuntimeError("Tried to step environment that needs reset")
        observation, reward, done, info = self.env.step(action)
        self.rewards.append(reward)
        if done:
            self.needs_reset = True
            ep_rew = sum(self.rewards)
            eplen = len(self.rewards)
            ep_info = {
                "r": round(ep_rew, 6),
                "l": eplen,
                "t": round(time.time() - self.t_start, 6),
            }
            for key in self.info_keywords:
                ep_info[key] = info[key]
            self.episode_rewards.append(ep_rew)
            self.episode_lengths.append(eplen)
            self.episode_times.append(time.time() - self.t_start)
            info["episode"] = ep_info
        self.total_steps += 1
        return observation, reward, done, info

    def get_total_steps(self) -> int:
        """
        Returns the total number of timesteps
        :return: (int)
        """
        return self.total_steps

    def get_episode_rewards(self) -> List[float]:
        """
        Returns the rewards of all the episodes
        :return: ([float])
        """
        return self.episode_rewards

    def get_episode_lengths(self) -> List[int]:
        """
        Returns the number of timesteps of all the episodes
        :return: ([int])
        """
        return self.episode_lengths

    def get_episode_times(self) -> List[float]:
        """
        Returns the runtime in seconds of all the episodes
        :return: ([float])
        """
        return self.episode_times


def make_atari(env_id):
    """
    Create a wrapped atari Environment
    :param env_id: (str) the environment ID
    :return: (Gym Environment) the wrapped atari environment
    """
    env = gym.make(env_id)
    assert "NoFrameskip" in env.spec.id
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    return env


def make_atari_deepmind(
    env_id: str,
    episode_life: bool = True,
    clip_rewards: bool = True,
    frame_stack: bool = False,
    scale: bool = False,
):
    """
    Create a wrapped atari Environment
    :param env_id: (str) the environment ID
    :return: (Gym Environment) the wrapped atari environment
    """
    env = make_atari(env_id)
    env = Monitor(env)
    env = wrap_deepmind(
        env,
        episode_life=episode_life,
        clip_rewards=clip_rewards,
        frame_stack=frame_stack,
        scale=scale,
    )
    return env
