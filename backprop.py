#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Cube26 product code
#
# (C) Copyright 2015 Cube26 Software Pvt Ltd
# All right reserved.
#
# This file is confidential and NOT open source.  Do not distribute.
#

"""

"""

import numpy as np
from theano import shared, function
import theano.tensor as T
from utils import get_xor_blobs
import matplotlib.pyplot as plt


def _get_weights(shape):
    std = 1.0 / np.sqrt(shape[1] + 1)
    return np.random.normal(scale=std, size=shape)


class Backpropagation(object):

    def __init__(self, layers, alpha=0.3, gradcheck=1e-4):
        self.alpha = alpha
        self.layers = layers
        self.weights = []
        self.biases = []
        self._x = T.dmatrix('x')
        self._y = T.dmatrix('y')
        for i, n in enumerate(layers):
            if i != len(layers) - 1:
                w = shared(_get_weights((layers[i + 1], n)),
                           name="w{}".format(i))
                b = shared(_get_weights((layers[i + 1], 1)),
                           name="b{}".format(i))
                self.weights.append(w)
                self.biases.append(b)

    def predict(self, X):
        self.layer_activations = [self._x.T]
        for i, weight in enumerate(self.weights):
            ai = self.layer_activations[i]
            activation = T.dot(weight, ai) + \
                self.biases[i].repeat(ai.shape[1], axis=1)
            self.layer_activations.append(1.0 / (1 + T.exp(-activation)))
        self._predict = function([self._x], [self.layer_activations[-1]])
        return self._predict(X)[0].T

    def fit(self, X, y, n_iter=1000, showloss=False, meanGrad=True):
        self.losses = []
        loss = T.sum((self.layer_activations[-1] - self._y.T) ** 2)
        updates = []
        for i in range(len(self.layers) - 1):
            w = self.weights[i]
            b = self.biases[i]
            grad_w, grad_b = T.grad(loss, [w, b])
            w_update = self.alpha * grad_w
            b_update = self.alpha * grad_b
            if meanGrad:
                w_update = w_update / self._x.shape[0]
                b_update = b_update / self._x.shape[0]
            updates.append((w, w - w_update))
            updates.append((b, b - b_update))
        self._fit = function([self._x, self._y], [loss], updates=updates)
        for i in xrange(n_iter):
            self.losses.append(self._fit(X, y))
            if showloss:
                print self.losses[-1]

if __name__ == '__main__':
    X, y = get_xor_blobs()
    bp1 = Backpropagation(layers=[2, 3, 2])
    bp1.fit(X, y, n_iter=10000, showloss=True)
    bp2 = Backpropagation(layers=[2, 3, 2])
    bp2.fit(X, y, n_iter=10000, showloss=True, meanGrad=False)
    plt.subplot(211), plt.plot(bp1.losses), plt.title("Mean gradient update")
    plt.xlim(0, 2000)
    plt.subplot(212), plt.plot(bp2.losses), plt.title("Total gradient update")
    plt.xlim(0, 2000)
    plt.show()
