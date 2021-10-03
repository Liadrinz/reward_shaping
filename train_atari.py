import numpy as np
from exp_pool import ExpPool
import gym
from policy import EpsilonGreedyDQNPolicy
from agents import DQNAgent
from trainer import Trainer
from ray.rllib.env.wrappers.atari_wrappers import wrap_deepmind
from wrappers import TensorWrapper

env = TensorWrapper(wrap_deepmind(gym.make("BreakoutNoFrameskip-v4"), dim=42))
policy = EpsilonGreedyDQNPolicy(env, 4, env.action_space.n, epsilon=0.01)
h, w, c = env.observation_space.shape
exp_pool = ExpPool(10000, (c, h, w))
agents = DQNAgent("breakout", env, policy, exp_pool, 200, 100)
Trainer("baseline", agents, 10000).start()
