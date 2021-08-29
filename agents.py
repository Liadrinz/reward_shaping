from shaper import Shaper
from policy import Policy, QPolicy
from env_wrapper import EnvWrapper
import os
from typing import List
import tqdm
import numpy as np
import threading
from gym import Env
from tensorboardX import SummaryWriter
from datetime import datetime


class Agent(object):

    def __init__(self,
                 name: str,
                 env: Env,
                 env_wrapper: EnvWrapper,
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
        self.env_wrapper = env_wrapper
        self.policy = policy
        self.shaper = shaper
    
    def learn(self, state, action, reward, state_):
        raise NotImplementedError
    
    def run_episode(self, i_episode):
        raise NotImplementedError


class QAgent(Agent):

    def _shaping_reward(self, state_, state, i_episode):
        if self.shaper is not None:
            return self.shaper.shaping_reward(state_, state, i_episode)
        return 0
    
    def learn(self, state, action, reward, state_):
        sidx = self.env_wrapper.state_to_index(state)
        sidx_ = self.env_wrapper.state_to_index(state_)
        value = self.policy.Q.__getitem__((*sidx, action))
        values_ = self.policy.Q.__getitem__(sidx_)
        self.policy.Q.__setitem__(
            (*sidx, action),
            value + self.lr * (reward + self.gamma * np.max(values_) - value)
        )
    
    def run_episode(self, i_episode):
        state = self.env.reset()
        reward_list = []
        n_steps = 0
        while True:
            if self.render == True or (i_episode >= self.render and self.render != False):
                self.env.render()
            action = self.policy.compute_action(state)
            state_, real_reward, done, _ = self.env.step(action)
            reward_list.append(real_reward)
            reward = real_reward + self._shaping_reward(state_, state, i_episode)
            self.learn(state, action, reward, state_)
            state = state_
            n_steps += 1
            if done or n_steps == 200:
                reward_list = np.array(reward_list)
                return {
                    "sum_reward": np.sum(reward_list),
                    "mean_reward": np.mean(reward_list),
                    "max_reward": np.max(reward_list),
                    "min_reward": np.min(reward_list)
                }


class QLearningAgent(threading.Thread):

    def __init__(self, env: Env,
                       name="maze",
                       recover=None,
                       episodes=1000,
                       epsilon=0.01,
                       gamma=1,
                       lr=1,
                       action="epsilon_greedy",
                       shaping="original",
                       training=True,
                       render=False):
        super().__init__()
        self.name = name
        self.episodes = episodes
        self.env = env
        self.epsilon = epsilon
        self.gamma = gamma
        self.lr = lr
        self.action = eval(f"self.action_{action}")
        self.shaping = eval(f"self.shaping_{shaping}")
        self.training = training
        self.render = render
        if recover is None:
            self.Q = self.get_init_q_table()
        else:
            self.Q = np.load(recover)
        self.writer = SummaryWriter(f"log/{self.name}/{datetime.now()}/".replace(":", "-").replace(" ", "-"))

    def get_init_q_table(self):
        raise NotImplementedError
    
    def state_to_index(self, state):
        raise NotImplementedError

    def get_valid_and_invalid_actions(self, state):
        raise NotImplementedError

    def action_epsilon_greedy(self, state, random=True) -> int:
        valid_actions, invalid_actions = self.get_valid_and_invalid_actions(state)
        values = self.Q.__getitem__(self.state_to_index(state)).copy()
        values[invalid_actions] = -np.inf
        if np.random.random() > self.epsilon or not random:
            max_actions = np.where(values == np.max(values))[0]
            res = np.random.choice(max_actions)
            if res in invalid_actions:
                print(values, max_actions)
            return res
        return np.random.choice(valid_actions)

    def action_boltzmann(self, state, random=True) -> int:
        _, invalid_actions = self.get_valid_and_invalid_actions(state)
        values = self.Q.__getitem__(self.state_to_index(state)).copy()
        values[invalid_actions] = -np.inf
        dist = np.exp(values) / np.sum(np.exp(values))
        action = np.random.choice(np.arange(self.env.action_space.n), p=dist)
        return action
    
    def shaping_original(self, state_, state):
        return 0
    
    def learn(self, state, action, reward, state_):
        sidx = self.state_to_index(state)
        sidx_ = self.state_to_index(state_)
        value = self.Q.__getitem__((*sidx, action))
        values_ = self.Q.__getitem__(sidx_)
        self.Q.__setitem__(
            (*sidx, action),
            value + self.lr * (reward + self.gamma * np.max(values_) - value)
        )
    
    def run_episode(self, i_episode):
        if hasattr(self, "assistants") and self.assistants is not None:
            for assist in self.assistants:
                assist.run_episode(i_episode)
        state = self.env.reset()
        reward_list = []
        n_steps = 0
        while True:
            if self.render == True or (i_episode >= self.render and self.render != False):
                self.env.render()
            action = self.action(state, random=self.training)
            state_, real_reward, done, _ = self.env.step(action)
            reward_list.append(real_reward)
            if self.training:
                reward = real_reward + self.shaping(state_, state) / (0.001 * i_episode + 0.1)
                # reward = self.shaping(state_, state)
                self.learn(state, action, reward, state_)
            state = state_
            n_steps += 1
            if done or n_steps == 200:
                return {
                    "sum_reward": np.sum(reward_list),
                    "mean_reward": np.mean(reward_list),
                    "max_reward": np.max(reward_list),
                    "min_reward": np.min(reward_list)
                }
            
    def run(self):
        os.makedirs("rewards", exist_ok=True)
        pbar = tqdm.tqdm(total=self.episodes)
        for i_episode in range(self.episodes):
            result = self.run_episode(i_episode)
            for key, value in result.items():
                self.writer.add_scalar(f"reward/{key}", value, i_episode)
            pbar.update()

    def save(self):
        os.makedirs("q_tables", exist_ok=True)
        np.save(f"q_tables/Q.{self.name}.npy", np.array(self.Q))


class MazeAgent(QLearningAgent):

    def shaping_manhattan(self, state_, state):
        return np.sum(state_ - state)
    
    def get_init_q_table(self):
        return np.zeros((self.env.size, self.env.size, self.env.action_space.n))

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


class MountainCarAgent(QLearningAgent):

    def __init__(self, env: Env,
                       name="mountain-car",
                       recover=None,
                       episodes=1000,
                       epsilon=0.01,
                       gamma=0.9,
                       lr=1,
                       discrete_num=100,
                       action="epsilon_greedy",
                       shaping="original",
                       assistants: List[QLearningAgent]=None,
                       training=True,
                       render=False):
        self.discrete_num = discrete_num
        super().__init__(env,
                         name=name,
                         recover=recover,
                         episodes=episodes,
                         epsilon=epsilon,
                         gamma=gamma,
                         lr=lr,
                         action=action,
                         shaping=shaping,
                         training=training,
                         render=render)
        self.pos_range = self.env.max_position - self.env.min_position
        self.vel_range = self.env.max_speed * 2
        self.assistants = assistants

    def shaping_distance(self, state_, state):
        return (state_[0] - state[0]) * self.discrete_num
    
    def shaping_energy(self, state_, state):
        def compute_energy(s):
            kinetic = s[1] * s[1] / 0.0098
            gravity = (self.env._height(s[0]) - 0.1) / 0.9 - 0.5
            return kinetic + gravity
        # compute_energy = lambda s: 1/2 * s[1] * s[1] + self.env.gravity * self.env._height(s[0])
        return compute_energy(state_) - compute_energy(state)
    
    def shaping_co(self, state_, state):
        assert self.assistants is not None
        compute_v = lambda avs: np.mean(self.epsilon * avs) + (1 - self.epsilon) * np.max(avs)
        rs = 0
        for assist in self.assistants:
            action_values_ = assist.Q[assist.state_to_index(state_)]
            action_values = assist.Q[assist.state_to_index(state)]
            v_ = compute_v(action_values_)
            v = compute_v(action_values)
            rs += (v_ - v) / len(self.assistants)
        return rs + self.shaping_energy(state_, state)

    def get_init_q_table(self):
        return np.zeros((self.discrete_num, self.discrete_num, self.env.action_space.n))
    
    def state_to_index(self, state):
        sp = int((state[0] - self.env.min_position) / self.pos_range * (self.discrete_num - 1))
        sv = int((state[1] + self.env.max_speed) / self.vel_range * (self.discrete_num - 1))
        return (sp, sv)

    def get_valid_and_invalid_actions(self, state):
        return list(range(self.env.action_space.n)), []
