from env_wrapper import EnvWrapper
import numpy as np


class Policy(object):

    def __init__(self, env_wrapper: EnvWrapper):
        self.env_wrapper = env_wrapper

    def compute_action(self, state):
        raise NotImplementedError
    
    def compute_state_value(self, state):
        raise NotImplementedError

    def save(self, path):
        raise NotImplementedError
    
    def restore(self, path):
        raise NotImplementedError


class QPolicy(Policy):

    def __init__(self, env_wrapper: EnvWrapper, state_shape, n_actions):
        super().__init__(env_wrapper)
        self.n_states = state_shape
        self.n_actions = n_actions
        self.Q = np.zeros((*state_shape, n_actions))
    
    def save(self, path):
        np.save(path, self.Q)
    
    def restore(self, path):
        self.Q = np.load(path)


class EpsilonGreedyQPolicy(QPolicy):

    def __init__(self, env_wrapper: EnvWrapper, state_shape, n_actions, epsilon=0.01):
        super().__init__(env_wrapper, state_shape, n_actions)
        self.epsilon = epsilon

    def compute_action(self, state):
        valid_actions, invalid_actions = self.env_wrapper.get_valid_and_invalid_actions(state)
        values = self.Q.__getitem__(self.env_wrapper.state_to_index(state)).copy()
        values[invalid_actions] = -np.inf
        if np.random.random() > self.epsilon:
            max_actions = np.where(values == np.max(values))[0]
            res = np.random.choice(max_actions)
            if res in invalid_actions:
                print(values, max_actions)
            return res
        return np.random.choice(valid_actions)

    def compute_state_value(self, state):
        action_values = self.Q[self.env_wrapper.state_to_index(state)]
        return np.mean(self.epsilon * action_values) + (1 - self.epsilon) * np.max(action_values)
