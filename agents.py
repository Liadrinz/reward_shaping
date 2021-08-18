import os
from typing import List
import tqdm
import numpy as np
import threading
from gym import Env
from envs import MazeEnv
from gym.envs.classic_control import MountainCarEnv


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
        sum_real_reward = 0
        n_steps = 0
        while True:
            if self.render == True or (i_episode >= self.render and self.render != False):
                self.env.render()
            action = self.action(state, random=self.training)
            state_, real_reward, done, _ = self.env.step(action)
            sum_real_reward += real_reward
            if self.training:
                reward = real_reward / 2 + self.shaping(state_, state)
                # reward = self.shaping(state_, state)
                self.learn(state, action, reward, state_)
            state = state_
            n_steps += 1
            if done or n_steps == 200:
                return sum_real_reward

    def run(self):
        real_rewards = []
        os.makedirs("rewards", exist_ok=True)
        pbar = tqdm.tqdm(total=self.episodes)
        for i_episode in range(self.episodes):
            sum_reward = self.run_episode(i_episode)
            real_rewards.append(sum_reward)
            pbar.update()
            if i_episode > 0 and i_episode % 100 == 0:
                np.save(f"rewards/rewards.{self.name}{'' if self.training else '-test'}.npy", np.array(real_rewards))
                self.save()

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
            return (kinetic + gravity) / 2
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
