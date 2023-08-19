import gym
import numpy as np

from minirl.envs.gym3_wrapper import EpisodeStatsWrapper


class ModifiedEpisodeStatsWrapper(EpisodeStatsWrapper):
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
        self.stats_buffers["finish"].extend(np.logical_and(first, rew > 0)[first])
        # num_explored_room = np.asarray(self.callmethod("get_num_explored_room"))
        # self.stats_buffers["avg_rooms"].extend(num_explored_room[first])


class RecordRoomWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.n_explored_rooms = 1
        self.need_reset = True

    def step(self, action):
        if self.need_reset:
            self.n_explored_rooms = 1
            self.need_reset = False
        obs, rew, done, info = super().step(action)
        if self.n_explored_rooms < len(self.env.unwrapped.rooms):
            room = self.env.unwrapped.rooms[self.n_explored_rooms]
            top_x, top_y = room.top
            x_len, y_len = room.size
            bottom_x, bottom_y = top_x + x_len - 1, top_y + y_len - 1
            agent_x, agent_y = self.env.unwrapped.agent_pos
            if top_x < agent_x < bottom_x and top_y < agent_y < bottom_y:
                self.n_explored_rooms += 1
        return obs, rew, done, info

    def reset(self, **kwargs):
        self.need_reset = True
        return super().reset(**kwargs)

    def get_num_explored_room(self):
        return self.n_explored_rooms


class CheapReseedWrapper(gym.Wrapper):
    def __init__(self, env, max_seed=10000000, seed_idx=0):
        self.max_seed = max_seed
        self.seed_idx = seed_idx
        super().__init__(env)

    def reset(self, **kwargs):
        self.env.seed(self.seed_idx)
        self.seed_idx = (self.seed_idx + 1) % self.max_seed
        return super().reset(**kwargs)
