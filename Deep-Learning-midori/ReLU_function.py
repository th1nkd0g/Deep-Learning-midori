"""This is test program of Deep Learning Training"""

import numpy as np
import matplotlib.pylab as plt


def relu(x):
    return np.maximum(0, x)


x = np.arange(-5.0, 5.0, 0.1)
y = relu(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)  # y軸の範囲を指定
plt.show()
