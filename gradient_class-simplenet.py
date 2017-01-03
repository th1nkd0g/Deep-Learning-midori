"""This is test program of Deep Learning Training"""

import numpy as np
import pickle
import sys
import os
sd = os.path.dirname('/Users/mi2/dev/deep-learning-from-scratch-master/dataset')
sys.path.append(sd)
md = os.path.dirname('/Users/mi2/dev/deep-learning-from-scratch-master/common')
sys.path.append(md)
from dataset.mnist import load_mnist
from PIL import Image
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient


class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)
        return loss


net = simpleNet()
print(net.W)

x = np.array([0.6, 0.9])
p = net.predict(x)
print(p)

np.argmax(p)

t = np.array([0, 0, 1])
net.loss(x, t)

def f(W):
    return net.loss(x, t)


dW = numerical_gradient(f, net.W)
print(dW)

f = lambda w: net.loss(x, t)
dW = numerical_gradient(f, net.W)
