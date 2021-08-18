import numpy as np
from envs import MazeEnv
from agents import MazeAgent

def transform(env, agent):
    x, y = env.terminal[0] + 1, env.terminal[1] + 1

    top_left = agent.Q.copy()[-x:,-y:]
    top_right = np.rot90(agent.Q.copy(), k=-1)[-x:,:-y][:, :, [2, 3, 1, 0]]
    bottom_left = np.rot90(agent.Q.copy())[:-x,-y:][:, :, [3, 2, 0, 1]]
    bottom_right = np.rot90(agent.Q.copy(), k=2)[:-x,:-y][:, :, [1, 0, 3, 2]]
    top = np.concatenate([top_left, top_right], axis=1)
    bottom = np.concatenate([bottom_left, bottom_right], axis=1)
    Q_1 = np.concatenate([top, bottom], axis=0)

    top_left = agent.Q.copy()[:x,:y]
    top_right = np.rot90(agent.Q.copy(), k=-1)[:x,y:][:, :, [2, 3, 1, 0]]
    bottom_left = np.rot90(agent.Q.copy())[x:,:y][:, :, [3, 2, 0, 1]]
    bottom_right = np.rot90(agent.Q.copy(), k=2)[x:,y:][:, :, [1, 0, 3, 2]]
    top = np.concatenate([top_left, top_right], axis=1)
    bottom = np.concatenate([bottom_left, bottom_right], axis=1)
    Q_2 = np.concatenate([top, bottom], axis=0)

    # agent.Q = (Q_1 + Q_2) / 2
    return Q_1

if __name__ == "__main__":
    env = MazeEnv(100)
    agent = MazeAgent(env,
                    name="trans.shaping",
                    recover="q_tables/Q.shaping.npy",
                    shaping="manhattan",
                    training=False)
    env.terminal = np.array([0, 99])
    Q_ = transform(env, agent)
    for i in range(1000):
        s = env.observation_space.sample()
        a = env.action_space.sample()
        s_ = env.step(a)
    agent.run()
