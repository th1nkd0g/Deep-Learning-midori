"""This is test program of Deep Learning Training"""

import numpy as np


def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)


# 「２」を正解とする
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
# 例１：「２」の確率が最も高い場合(0.6)
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
mean_squared_error(np.array(y), np.array(t))
# 例２：「７」の確率が最も高い場合(0.6)
y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
mean_squared_error(np.array(y), np.array(t))
