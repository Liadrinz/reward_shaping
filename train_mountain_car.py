from gym.envs.classic_control import MountainCarEnv
from agents import MountainCarAgent

if __name__ == "__main__":
    agents = [
        MountainCarAgent(MountainCarEnv(), name="mc-energy", discrete_num=100, shaping="energy", episodes=100000),
        MountainCarAgent(MountainCarEnv(), name="mc-shaping-co", discrete_num=100, shaping="co", episodes=100000, assistants=[
            MountainCarAgent(MountainCarEnv(), name="assist", discrete_num=100, shaping="energy", episodes=1),
        ]),
        MountainCarAgent(MountainCarEnv(), name="mc-shaping-co2", discrete_num=100, shaping="co", episodes=100000, assistants=[
            MountainCarAgent(MountainCarEnv(), name="assist", discrete_num=100, shaping="energy", episodes=1),
            MountainCarAgent(MountainCarEnv(), name="assist", discrete_num=100, shaping="energy", episodes=1),
        ]),
        MountainCarAgent(MountainCarEnv(), name="mc-shaping-co4", discrete_num=100, shaping="co", episodes=100000, assistants=[
            MountainCarAgent(MountainCarEnv(), name="assist", discrete_num=100, shaping="energy", episodes=1) for i in range(4)
        ]),
        # MountainCarAgent(MountainCarEnv(), name="mc-shaping-co8", discrete_num=100, shaping="co", episodes=100000, assistants=[
        #     MountainCarAgent(MountainCarEnv(), name="assist", discrete_num=100, episodes=1) for _ in range(8)
        # ]),
        # MountainCarAgent(MountainCarEnv(), name="mc-shaping-co16", discrete_num=100, shaping="co", episodes=100000, assistants=[
        #     MountainCarAgent(MountainCarEnv(), name="mc-energy", discrete_num=100, shaping="energy", episodes=1) for _ in range(16)
        # ]),
    ]
    [agent.start() for agent in agents]
    [agent.join() for agent in agents]
    [agent.save() for agent in agents]
