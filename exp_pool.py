import numpy as np
import torch

class ExpPool(object):

    def __init__(self, capacity, state_shape):
        self.capacity = capacity
        self.state_shape = state_shape
        self.n_state = np.prod(self.state_shape)
        self.memory = np.zeros((capacity, self.n_state * 2 + 2))
        self.memory_counter = 0
    
    def store_trainsition(self, s, a, r, s_):
        s = s.cpu()
        s_ = s_.cpu()
        s = np.reshape(s, (-1,))
        s_ = np.reshape(s_, (-1,))
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % self.capacity
        self.memory[index,:] = transition
        self.memory_counter += 1
    
    def sample_transition(self, batch_size):
        sample_index = np.random.choice(self.capacity, batch_size)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :self.n_state]).view(-1, *self.state_shape).cuda()
        b_a = torch.LongTensor(b_memory[:, self.n_state:self.n_state + 1].astype(int)).cuda()
        b_r = torch.FloatTensor(b_memory[:, self.n_state + 1:self.n_state + 2]).cuda()
        b_s_ = torch.FloatTensor(b_memory[:, -self.n_state:]).view(-1, *self.state_shape).cuda()
        return b_s, b_a, b_r, b_s_


class PrioritizedExpPool(ExpPool):
    pass
