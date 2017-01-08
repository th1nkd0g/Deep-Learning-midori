"""This is test program of Deep Learning Training"""
# 誤差逆伝播法の勾配確認


import sys
import os
import numpy as np
sd = os.path.dirname('/Users/mi2/dev/deep-learning-from-scratch-master/dataset')
sys.path.append(sd)
md = os.path.dirname('/Users/mi2/dev/deep-learning-from-scratch-master/common')
sys.path.append(md)
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet
from common.layers import Relu, Affine, SoftmaxWithLoss
from common.gradient import numerical_gradient
from collections import OrderedDict

# データの読み込み
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

x_batch = x_train[:3]
t_batch = t_train[:3]

grad_numerical = network.numerical_gradient(x_batch, t_batch)
grad_backprop = network.gradient(x_batch, t_batch)

# 各重みの絶対誤差の平均を求める
for key in grad_numerical.keys():
    diff = np.average(np.abs(grad_backprop[key] - grad_numerical[key]))
    print(key + ":" + str(diff))

