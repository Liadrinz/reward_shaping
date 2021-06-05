import sys
import numpy as np
import matplotlib.pyplot as plt
from smooth_signal import smooth

for fname in sys.argv[1::2]:
    rewards = np.load(fname)
    plt.plot(np.arange(rewards.shape[0]), rewards)
plt.legend([name for name in sys.argv[2::2]])
plt.show()

for fname in sys.argv[1::2]:
    rewards = np.load(fname)
    rewards = smooth(rewards, window_len=111)
    plt.plot(np.arange(rewards.shape[0]), rewards)
plt.legend([name for name in sys.argv[2::2]])
plt.show()
