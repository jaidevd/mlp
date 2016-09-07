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

from theano import shared, function
import theano.tensor as T
from utils import make_circles, get_weights, draw_decision_boundary
from sklearn.preprocessing import OneHotEncoder


class Backpropagation(object):

    def __init__(self, layers, alpha=0.3, C=0.0, gradcheck=1e-4):
        self.alpha = alpha
        self._lambda = C
        self.layers = layers
        self.weights = []
        self.biases = []
        self._x = T.dmatrix('x')
        self._y = T.dmatrix('y')
        for i, n in enumerate(layers):
            if i != len(layers) - 1:
                w = shared(get_weights((layers[i + 1], n)),
                           name="w{}".format(i))
                b = shared(get_weights((layers[i + 1], 1)),
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
        self.predict(X)
        loss = T.sum((self.layer_activations[-1] - self._y.T) ** 2)
        # Adding the regularization term:
        loss += self._lambda / 2.0 * T.sum([(w ** 2).sum() for w in self.weights])
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
    X, y = make_circles(factor=0.1, noise=0.22)
    y = OneHotEncoder().fit_transform(y.reshape(-1, 1)).toarray()
    bp = Backpropagation(layers=[2, 10, 2], C=0.1)
    bp.fit(X, y, n_iter=100000, showloss=True)
    draw_decision_boundary(bp, X, y)
    X, y = make_circles(factor=0.2)
    y = OneHotEncoder().fit_transform(y.reshape(-1, 1)).toarray()
    draw_decision_boundary(bp, X, y, marker='x')
