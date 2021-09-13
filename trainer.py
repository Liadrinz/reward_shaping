from typing import List
from agents import Agent
from datetime import datetime
import threading
from numpy.lib.type_check import real
import tqdm
import os
import numpy as np
from tensorboardX import SummaryWriter


class Trainer(threading.Thread):

    def __init__(self, name: str, agent: Agent, episodes: int):
        super().__init__()
        self.name = name
        self.agent = agent
        self.episodes = episodes
        self.writer = SummaryWriter(f"log/{self.name}/{datetime.now()}/".replace(":", "-").replace(" ", "-"))
        self.time_steps = 0
    
    def run(self):
        pbar = tqdm.tqdm(total=self.episodes)
        for i_episode in range(self.episodes):
            n_steps, sum_reward = self.agent.run_episode(i_episode)
            self.time_steps += n_steps
            self.writer.add_scalar(f"reward", sum_reward, self.time_steps)
            pbar.update()
    

class AssistantsTrainer(Trainer):

    def __init__(self, name: str, main_agent: Agent, assist_agents: List[Agent], episodes: str, freeze_assistants=False):
        super().__init__(name, main_agent, episodes)
        self.assist_agents = assist_agents
        self.freeze_assistants = freeze_assistants

    def run(self):
        pbar = tqdm.tqdm(total=self.episodes)
        for i_episode in range(self.episodes):
            if not self.freeze_assistants:
                for assist_agent in self.assist_agents:
                    assist_agent.run_episode(i_episode)
            result = self.agent.run_episode(i_episode)
            for key, value in result.items():
                self.writer.add_scalar(f"reward/{key}", value, i_episode)
            pbar.update()
