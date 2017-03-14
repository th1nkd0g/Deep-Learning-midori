"""This is test program of Deep Learning Training"""

# 変数が二つある偏微分
# f(x[0], x[1]) = x[0]**2 + x[1]**2
# 上記式をPythonで実装すると以下になる

import numpy as np
import matplotlib.pylab as plt


def function_2(x):
    return x[0]**2 + x[1]**2  # または return np.sum(x**2)


def numerical_diff(f, x):
    h = 1e-4  # 0.0001
    return (f(x+h) - f(x-h)) / (2*h)


# x[0]=3, x[1]=4の時の x[0] に対する偏微分x[0]
def function_tmp1(x0):
    return x0*x0 + 4.0**2.0


numerical_diff(function_tmp1, 3.0)

# x[0]=3, x[1]=4の時の x[1] に対する偏微分x[1]
def function_tmp2(x1):
    return 3.0**2.0 + x1*x1


numerical_diff(function_tmp2, 4.0)
