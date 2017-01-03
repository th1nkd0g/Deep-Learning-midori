"""This is test program of Deep Learning Training"""

import numpy as np
import pickle
import sys
import os
sd = os.path.dirname('/Users/mi2/dev/deep-learning-from-scratch-master/dataset')
sys.path.append(sd)
md = os.path.dirname('/Users/mi2/dev/deep-learning-from-scratch-master/ch03')
sys.path.append(md)
from dataset.mnist import load_mnist
from PIL import Image


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)  # オーバーフロー対策した指数関数
    sum_exp_a = np.sum(exp_a)  # 指数関数の和
    y = exp_a / sum_exp_a
    return y


def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test


def init_network():
    with open("/Users/mi2/dev/deep-learning-from-scratch-master/ch03/sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
        return network


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)
    return y


x, t = get_data()
network = init_network()
batch_size = 100  # バッチの数
accuracy_cnt = 0
for i in range(0, len(x), batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)  # 最も確率の高い要素のインデックスを取得
    accuracy_cnt += np.sum(p == t[i:i+batch_size])

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))

x = np.array([[0.1, 0.8, 0.1], [0.3, 0.1, 0.6], [0.2, 0.5, 0.3], [0.8, 0.1, 0.1]])
y = np.argmax(x, axis=1)
print(y)

y = np.array([1, 2, 1, 0])
t = np.array([1, 2, 0, 0])
print(y == t)

np.sum(y == t)
