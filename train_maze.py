import numpy as np

from envs import MazeEnv
from agents import MazeAgent

if __name__ == "__main__":
    agents = [
        MazeAgent(MazeEnv(100), name="baseline"),
        MazeAgent(MazeEnv(100), name="shaping", shaping="manhattan")
    ]
    [agent.start() for agent in agents]
    [agent.join() for agent in agents]
    [agent.save() for agent in agents]
