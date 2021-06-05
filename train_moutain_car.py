from gym.envs.classic_control import MountainCarEnv
from agents import MountainCarAgent

if __name__ == "__main__":
    agents = [
        MountainCarAgent(MountainCarEnv(), name="mc-baseline", discrete_num=50, episodes=3000),
        MountainCarAgent(MountainCarEnv(), name="mc-shaping-distance", discrete_num=50, shaping="distance", episodes=3000),
        MountainCarAgent(MountainCarEnv(), name="mc-shaping-energy", discrete_num=50, shaping="energy", episodes=3000),
    ]
    [agent.start() for agent in agents]
    [agent.join() for agent in agents]
    [agent.save() for agent in agents]
