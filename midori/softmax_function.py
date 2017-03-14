"""This is test program of Deep Learning Training"""

import numpy as np


def softmax(a):  # ソフトマックス関数、但しオーバーフローする欠点あり
    exp_a = np.exp(a)  # 指数関数
    sum_exp_a = np.sum(exp_a)  # 指数関数の和
    y = exp_a / sum_exp_a
    return y
