from envs import MazeEnv
from gym.core import Env


class EnvWrapper(object):

    def __init__(self, env: Env):
        self.env = env

    def state_to_index(self, state):
        return state

    def get_valid_and_invalid_actions(self, state):
        return list(range(self.env.action_space.n)), []


class MazeWrapper(EnvWrapper):

    def __init__(self, env: MazeEnv):
        super().__init__(env)

    def state_to_index(self, state):
        return (state[0], state[1])

    def get_valid_and_invalid_actions(self, state):
        invalid_actions = []
        valid_actions = list(range(self.env.action_space.n))
        if state[0] == 0:
            valid_actions.remove(0)
            invalid_actions.append(0)
        elif state[0] == self.env.size - 1:
            valid_actions.remove(1)
            invalid_actions.append(1)
        if state[1] == 0:
            valid_actions.remove(2)
            invalid_actions.append(2)
        elif state[1] == self.env.size - 1:
            valid_actions.remove(3)
            invalid_actions.append(3)
        return valid_actions, invalid_actions


class MountainCarWrapper(EnvWrapper):
    
    def __init__(self, env: Env, discrete_num=100):
        super().__init__(env)
        self.discrete_num = discrete_num
        self.pos_range = self.env.max_position - self.env.min_position
        self.vel_range = self.env.max_speed * 2

    def state_to_index(self, state):
        sp = int((state[0] - self.env.min_position) / self.pos_range * (self.discrete_num - 1))
        sv = int((state[1] + self.env.max_speed) / self.vel_range * (self.discrete_num - 1))
        return (sp, sv)
