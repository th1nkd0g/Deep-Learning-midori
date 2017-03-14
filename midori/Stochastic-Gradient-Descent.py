"""This is test program of Deep Learning Training"""

# 学習に関するテクニック
# 確率的降下法


class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]
