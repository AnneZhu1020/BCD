import gym
import numpy as np


class ScaleActionWrapper(gym.ActionWrapper):
    """
    Rescale the action from/to [low, high] to/from [-1, 1]
    """

    def action(self, action):
        low, high = self.action_space.low, self.action_space.high
        return (high - low) / 2 * action + (high + low) / 2

    def reverse_action(self, action):
        low, high = self.action_space.low, self.action_space.high
        return 2.0 / (high - low) * (action - (high + low) / 2.0)


class ClipActionWrapper(gym.ActionWrapper):
    """
    Clip the action to [low, high]
    """

    def action(self, action):
        return np.clip(action, self.action_space.low, self.action_space.high)
