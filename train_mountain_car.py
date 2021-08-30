import gym
from shaper import MountainCarAssistantShaper
from policy import EpsilonGreedyQPolicy
from wrappers import MountainCarWrapper
from agents import QAgent
from trainer import Trainer, AssistantsTrainer

env1 = MountainCarWrapper(gym.make("MountainCar-v0"))
policy1 = EpsilonGreedyQPolicy(env1, (100, 100), 3, epsilon=0.01)
agent1 = QAgent("mc", env1, policy1)

env2 = MountainCarWrapper(gym.make("MountainCar-v0"))
policy2 = EpsilonGreedyQPolicy(env2, (100, 100), 3, epsilon=0.01)
agent2 = QAgent("mc", env2, policy2)

env3 = MountainCarWrapper(gym.make("MountainCar-v0"))
policy3 = EpsilonGreedyQPolicy(env3, (100, 100), 3, epsilon=0.01)
agent3 = QAgent("mc", env3, policy3)

env4 = MountainCarWrapper(gym.make("MountainCar-v0"))
policy4 = EpsilonGreedyQPolicy(env4, (100, 100), 3, epsilon=0.01)
agent4 = QAgent("mc", env4, policy4, MountainCarAssistantShaper(env4, [agent2, agent3]))

Trainer("baseline", agent1, 100000).start()
AssistantsTrainer("co1", agent4, [agent2, agent3], 100000).start()
