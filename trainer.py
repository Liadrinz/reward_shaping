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
    
    def run(self):
        pbar = tqdm.tqdm(total=self.episodes)
        for i_episode in range(self.episodes):
            result = self.agent.run_episode(i_episode)
            for key, value in result.items():
                self.writer.add_scalar(f"reward/{key}", value, i_episode)
            pbar.update()
    

class AssistantsTrainer(Trainer):

    def __init__(self, name: str, main_agent: Agent, assist_agents: List[Agent], episodes: str):
        super().__init__(name, main_agent, episodes)
        self.assist_agents = assist_agents

    def run(self):
        pbar = tqdm.tqdm(total=self.episodes)
        for i_episode in range(self.episodes):
            for assist_agent in self.assist_agents:
                assist_agent.run_episode(i_episode)
            result = self.agent.run_episode(i_episode)
            for key, value in result.items():
                self.writer.add_scalar(f"reward/{key}", value, i_episode)
            pbar.update()
