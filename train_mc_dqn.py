import gym
from shaper import MountainCarAssistantShaper
from policy import EpsilonGreedyDQNPolicy
from wrappers import DQNMountainCarWrapper
from agents import DQNAgent
from trainer import Trainer, AssistantsTrainer
from networks import FCDQN
from exp_pool import ExpPool

env1 = DQNMountainCarWrapper(gym.make("MountainCar-v0"))
policy1 = EpsilonGreedyDQNPolicy(env1, 2, 3, epsilon=0.01, DQN_type=FCDQN)
exp_pool = ExpPool(2000, env1.observation_space.shape)
agent1 = DQNAgent("mc", env1, policy1, exp_pool, 32, 100)

Trainer("baseline", agent1, 100000).start()
