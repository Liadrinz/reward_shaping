from gym import spaces
from gym.envs.classic_control.mountain_car import MountainCarEnv
import numpy as np
import matplotlib.pyplot as plt
import gym
from gym import spaces

class MazeEnv(gym.Env):

    def __init__(self, size):
        self.size = size
        self.beginning = np.array([0, 0])
        self.terminal = np.array([size-1, size-1])

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(0, size-1, (2,), dtype=np.int)

        self.state = None

    def step(self, action):
        if   action == 0: delta = np.array([-1, 0])
        elif action == 1: delta = np.array([1, 0])
        elif action == 2: delta = np.array([0, -1])
        elif action == 3: delta = np.array([0, 1])
        else:             raise ValueError("Invalid action")
        self.state = self.state + delta  # 不可写成self.state += delta
        done = (self.state == self.terminal).all()
        if done:
            reward = 2 * self.size - 3
        else:
            reward = -1
        return self.state, reward, done, {}
    
    def reset(self):
        self.state = self.beginning
        return self.state
