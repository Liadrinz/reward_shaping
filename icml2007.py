import gym
from shaper import MountainCarAssistantShaper
from policy import EpsilonGreedyQPolicy
from wrappers import MountainCarWrapper
from agents import QAgent
from trainer import Trainer, AssistantsTrainer

env_abs = MountainCarWrapper(gym.make("MountainCar-v0"), 25)
policy_abs = EpsilonGreedyQPolicy(env_abs, (25, 25), 3, epsilon=0.01)
agent_abs = QAgent("mc-abs", env_abs, policy_abs)

Trainer("abs", agent_abs, 50000).run()

env1 = MountainCarWrapper(gym.make("MountainCar-v0"))
policy1 = EpsilonGreedyQPolicy(env1, (100, 100), 3, epsilon=0.01)
agent1 = QAgent("mc", env1, policy1)

env2 = MountainCarWrapper(gym.make("MountainCar-v0"))
policy2 = EpsilonGreedyQPolicy(env2, (100, 100), 3, epsilon=0.01)
agent2 = QAgent("mc", env2, policy2, MountainCarAssistantShaper(env2, [agent_abs]))

Trainer("baseline", agent1, 100000).start()
AssistantsTrainer("co1", agent2, [agent_abs], 100000, True).start()
