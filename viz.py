import sys
import numpy as np
import matplotlib.pyplot as plt
from smooth_signal import smooth

win_len = int(sys.argv[-1])
sys.argv = sys.argv[:-1]
for fname in sys.argv[1::2]:
    rewards = np.load(fname)
    rewards = smooth(rewards, window_len=win_len)
    plt.plot(np.arange(rewards.shape[0]), rewards)
plt.legend([name for name in sys.argv[2::2]])
plt.show()
