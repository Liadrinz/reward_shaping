
from gym.envs.classic_control.mountain_car import MountainCarEnv
from shaper import MountainCarAssistantShaper
from policy import EpsilonGreedyQPolicy
from env_wrapper import MountainCarWrapper
from agents import QAgent
from trainer import Trainer, AssistantsTrainer

env1 = MountainCarEnv()
env_wrapper1 = MountainCarWrapper(env1)
policy1 = EpsilonGreedyQPolicy(env_wrapper1, (100, 100), 3, epsilon=0.01)
agent1 = QAgent("mc", env1, env_wrapper1, policy1)

env2 = MountainCarEnv()
env_wrapper2 = MountainCarWrapper(env2)
policy2 = EpsilonGreedyQPolicy(env_wrapper2, (100, 100), 3, epsilon=0.01)
agent2 = QAgent("mc", env2, env_wrapper2, policy2)

env3 = MountainCarEnv()
env_wrapper3 = MountainCarWrapper(env3)
policy3 = EpsilonGreedyQPolicy(env_wrapper3, (100, 100), 3, epsilon=0.01)
agent3 = QAgent("mc", env3, env_wrapper3, policy3)

env4 = MountainCarEnv()
env_wrapper4 = MountainCarWrapper(env4)
policy4 = EpsilonGreedyQPolicy(env_wrapper4, (100, 100), 3, epsilon=0.01)
agent4 = QAgent("mc", env4, env_wrapper4, policy4, MountainCarAssistantShaper(env4, [agent2, agent3]))

Trainer("baseline", agent1, 100000).start()
AssistantsTrainer("co1", agent4, [agent2, agent3], 100000).start()
