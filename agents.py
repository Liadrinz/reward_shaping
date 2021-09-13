from copy import deepcopy
import torch
from torch import nn
from exp_pool import ExpPool
from shaper import Shaper
from policy import DQNPolicy, Policy, QPolicy
from typing import List
import numpy as np
from gym import Env


class Agent(object):

    def __init__(self,
                 name: str,
                 env: Env,\
                 policy: Policy,
                 shaper: Shaper = None,
                 gamma=0.9,
                 lr=1,
                 render=False):
        self.name = name
        self.env = env
        self.gamma = gamma
        self.lr = lr
        self.render = render
        self.policy = policy
        self.shaper = shaper
    
    def learn(self, state, action, reward, state_):
        raise NotImplementedError
    
    def run_episode(self, i_episode):
        raise NotImplementedError


class QAgent(Agent):

    def __init__(self,
                 name: str,
                 env: Env,
                 policy: QPolicy,
                 shaper: Shaper = None,
                 gamma=0.9,
                 lr=1,
                 render=False):
        super().__init__(name, env, policy, shaper, gamma, lr, render)
        self._loss = 0
    
    def get_loss(self):
        return self._loss

    def _shaping_reward(self, state_, state, i_episode):
        if self.shaper is not None:
            return self.shaper.shaping_reward(state_, state, i_episode)
        return 0
    
    def learn(self, state, action, reward, state_):
        compute_v = lambda avs: np.mean(self.policy.epsilon * avs) + (1 - self.policy.epsilon) * np.max(avs)
        sidx = state
        sidx_ = state_
        values = self.policy.Q.__getitem__((*sidx, action))
        values_ = self.policy.Q.__getitem__(sidx_)
        update = self.lr * (reward + self.gamma * np.max(values_) - values)
        self.policy.Q.__setitem__(
            (*sidx, action),
            values + update
        )
        update_v = self.lr * (reward + self.gamma * compute_v(values_) - compute_v(values))
        return update_v
    
    def run_episode(self, i_episode):
        state = self.env.reset()
        sum_reward = 0
        n_steps = 0
        while True:
            if self.render == True or (i_episode >= self.render and self.render != False):
                self.env.render()
            action = self.policy.compute_action(state)
            state_, real_reward, done, _ = self.env.step(action)
            sum_reward += real_reward
            reward = real_reward + self._shaping_reward(state_, state, i_episode)
            self._loss = self.learn(state, action, reward, state_)
            state = state_
            n_steps += 1
            if done:
                return n_steps, sum_reward


class DQNAgent(QAgent):
    
    def __init__(self,
                 name: str,
                 env: Env,
                 policy: DQNPolicy,
                 exp_pool: ExpPool,
                 batch_size: int,
                 target_repl_iter: int,
                 shaper: Shaper = None,
                 gamma=0.9,
                 lr=0.01,
                 render=False):
        super().__init__(name, env, policy, shaper, gamma, lr, render)
        self.exp_pool = exp_pool
        self.batch_size = batch_size
        self.target_repl_iter = target_repl_iter

        self.learn_step_counter = 0
        self.optimizer = torch.optim.Adam(self.policy.eval_net.parameters(), lr=self.lr)
        self.loss_func = nn.MSELoss()
    
    def learn(self, state, action, reward, state_):
        self.exp_pool.store_trainsition(state, action, reward, state_)

        if self.learn_step_counter % self.target_repl_iter == 0:
            self.policy.target_net.load_state_dict(self.policy.eval_net.state_dict())
        self.learn_step_counter += 1
        
        
        if self.exp_pool.memory_counter > self.exp_pool.capacity:

            b_s, b_a, b_r, b_s_ = self.exp_pool.sample_transition(self.batch_size)

            q_eval = self.policy.eval_net(b_s).gather(1, b_a)
            q_next = self.policy.target_net(b_s).detach()
            q_target = b_r + self.gamma * q_next.max(1)[0].view(self.batch_size, 1)
            
            loss = self.loss_func(q_eval, q_target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
