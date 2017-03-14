"""This is test program of Deep Learning Training"""

import numpy as np


def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)  # オーバーフロー対策した指数関数
    sum_exp_a = np.sum(exp_a)  # 指数関数の和
    y = exp_a / sum_exp_a
    return y
