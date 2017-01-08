"""This is test program of Deep Learning Training"""


import numpy as np


def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)  # オーバーフロー対策した指数関数
    sum_exp_a = np.sum(exp_a)  # 指数関数の和
    y = exp_a / sum_exp_a
    return y


def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None  # 損失
        self.y = None  # softmaxの出力
        self.t = None  # 教師データ(one-hot vector)

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size

        return dx
