from gym.core import Env
import numpy as np


class Shaper(object):

    def __init__(self, env: Env):
        self.env = env

    def shaping_reward(self, state_, state, i_episode):
        return 0


class MazeManhattanShaper(Shaper):

    def shaping_reward(self, state_, state):
        return np.sum(state_ - state)


class MountainCarEnergyShaper(Shaper):

    def shaping_reward(self, state_, state, i_episode):
        def compute_energy(s):
            kinetic = s[1] * s[1] / 0.0098
            gravity = (self.env._height(s[0]) - 0.1) / 0.9 - 0.5
            return (kinetic + gravity) / 2
        return compute_energy(state_) - compute_energy(state)


class MountainCarAssistantShaper(Shaper):

    def __init__(self, env: Env, assist_agents, gamma=0.9):
        super().__init__(env)
        self.energy_shaper = MountainCarEnergyShaper(env)
        self.assist_agents = assist_agents
        self.gamma = gamma
        self.global_n = 1
        self.global_loss = 0
        self.global_rs = 0
        self.global_correct = 0
        self.global_eps = 0
    
    def shaping_reward(self, state_, state, i_episode):
        rs = 0
        for assist_agent in self.assist_agents:
            compute_v = lambda avs: np.mean(assist_agent.policy.epsilon * avs) + (1 - assist_agent.policy.epsilon) * np.max(avs)
            ratio = self.env.discrete_num // assist_agent.env.discrete_num
            action_values_ = assist_agent.policy.Q[state_[0] // ratio, state_[1] // ratio]
            action_values = assist_agent.policy.Q[state[0] // ratio, state[1] // ratio]
            v_ = compute_v(action_values_)
            v = compute_v(action_values)
            loss = assist_agent.get_loss()
            delta = (loss - self.global_loss) / self.global_n
            self.global_loss += delta
            self.global_n += 1
            rs += (v_ - v)
            self.global_correct += loss / 10000
            self.global_rs += (v_ - v)
        if i_episode % 100 == 0 and i_episode > self.global_eps:
            print(self.global_rs, self.global_correct)
            self.global_eps = i_episode
        return rs
