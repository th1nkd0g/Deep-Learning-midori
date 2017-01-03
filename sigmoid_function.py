"""This is test program of Deep Learning Training"""

import numpy as np
import matplotlib.pylab as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))  # exp(-x)はeの-x乗を意味します。eはネイピア数の2.7182……


x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)  # y軸の範囲を指定
plt.show()
