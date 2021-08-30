from copy import deepcopy
from networks import DQN
import numpy as np
import torch


class Policy(object):

    def __init__(self, env):
        self.env = env

    def compute_action(self, state):
        raise NotImplementedError
    
    def compute_state_value(self, state):
        raise NotImplementedError

    def save(self, path):
        raise NotImplementedError
    
    def restore(self, path):
        raise NotImplementedError


class QPolicy(Policy):

    def __init__(self, env, state_shape, n_actions):
        super().__init__(env)
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.Q = np.zeros((*state_shape, n_actions))
    
    def save(self, path):
        np.save(path, self.Q)
    
    def restore(self, path):
        self.Q = np.load(path)


class EpsilonGreedyQPolicy(QPolicy):

    def __init__(self, env, state_shape, n_actions, epsilon=0.01):
        super().__init__(env, state_shape, n_actions)
        self.epsilon = epsilon

    def compute_action(self, state):
        values = self.Q.__getitem__(state).copy()
        if np.random.random() > self.epsilon:
            max_actions = np.where(values == np.max(values))[0]
            res = np.random.choice(max_actions)
            return res
        return np.random.randint(0, self.n_actions)

    def compute_state_value(self, state):
        action_values = self.Q[state]
        return np.mean(self.epsilon * action_values) + (1 - self.epsilon) * np.max(action_values)


class DQNPolicy(Policy):

    def __init__(self, env, in_channels, n_actions):
        super().__init__(env)
        self.in_channels = in_channels
        self.n_actions = n_actions
        self.eval_net = DQN(in_channels, n_actions)
        self.target_net = deepcopy(self.eval_net)

    def save(self, path):
        torch.save(self.eval_net.state_dict(), path)
    
    def restore(self, path):
        self.eval_net.load_state_dict(torch.load(path))
        self.target_net = deepcopy(self.eval_net)


class EpsilonGreedyDQNPolicy(DQNPolicy):

    def __init__(self, env, in_channels, n_actions, epsilon=0.01):
        super().__init__(env, in_channels, n_actions)
        self.epsilon = epsilon
        
    
    def compute_action(self, state):
        values = self.eval_net.forward(state)
        if type(values) == torch.Tensor:
            values = values.detach().numpy()
        if np.random.random() > self.epsilon:
            max_actions = np.where(values == np.max(values))[0]
            res = np.random.choice(max_actions)
            return res
        return np.random.randint(0, self.n_actions)
