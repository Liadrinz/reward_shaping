import numpy as np
from exp_pool import ExpPool
import gym
from wrappers import AtariWrapper
from policy import EpsilonGreedyDQNPolicy
from agents import DQNAgent
from trainer import Trainer

env = AtariWrapper(gym.make("Pong-v0"))
policy = EpsilonGreedyDQNPolicy(env, 3, env.action_space.n, epsilon=0.01)
exp_pool = ExpPool(2000, env.observation_space.shape)
agent = DQNAgent("pong", env, policy, exp_pool, 32, 100)
Trainer("baseline", agent, 10000).start()
