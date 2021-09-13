import gym
from gym import spaces
import numpy as np
import torch
import torch.nn as nn
from collections import deque


class AtariWrapper(gym.Wrapper):

    def __init__(self, env=None, skip=4):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = deque(maxlen=2)
        self._skip = skip
        self.pooling = nn.MaxPool2d((2, 2))
        h, w, c = self.observation_space.shape
        self.observation_space.shape = (h//2, w//2, c)

    def _pipeline(self, data: np.ndarray) -> torch.Tensor:
        data = torch.tensor(data)
        h, w, c = data.shape[-3:]
        data = data.view(-1, c, h, w).to(torch.float32).cuda()
        data = self.pooling.forward(data)
        return data

    def step(self, action):
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break

        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        max_frame = self._pipeline(max_frame)

        return max_frame, total_reward, done, info

    def reset(self):
        """Clear past frame buffer and init. to first obs. from inner env."""
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        obs = self._pipeline(obs)
        return obs


class MountainCarWrapper(gym.Wrapper):
    def __init__(self, env, discrete_num=100):
        super().__init__(env)
        self.env = env
        self.discrete_num = discrete_num
        self.pos_range = self.env.max_position - self.env.min_position
        self.vel_range = self.env.max_speed * 2

    def _discretize(self, state):
        sp = int((state[0] - self.env.min_position) / self.pos_range *
                 (self.discrete_num - 1))
        sv = int((state[1] + self.env.max_speed) / self.vel_range *
                 (self.discrete_num - 1))
        return (sp, sv)

    def step(self, action):
        s_, r, done, info = super().step(action)
        s_ = self._discretize(s_)
        return s_, r, done, info

    def reset(self):
        s = super().reset()
        return self._discretize(s)


class DQNMountainCarWrapper(gym.Wrapper):
    
    def __init__(self, env, discrete_num=100):
        super().__init__(env)
        self.env = env
    
    def step(self, action):
        s_, r, done, info = super().step(action)
        return torch.tensor(s_).float(), r, done, info
    
    def reset(self):
        return torch.tensor(super().reset()).float()