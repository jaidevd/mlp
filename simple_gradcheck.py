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
from theano import function, shared
import theano.tensor as T
from utils import get_xor_blobs, get_weights, draw_decision_boundary
import matplotlib.pyplot as plt
# from backprop import Backpropagation


EPSILON = 1e-4

X, Y = get_xor_blobs()

# bp = Backpropagation(layers=[2, 3, 2])
# bp.fit(X, y, n_iter=10000)
# draw_decision_boundary(bp, X, y)


w1 = shared(get_weights((3, 2)), name="w1")
w2 = shared(get_weights((2, 3)), name="w2")
b1 = shared(get_weights((3, 1)), name="b1")
b2 = shared(get_weights((2, 1)), name="b2")

x = T.dmatrix('x')
y = T.dmatrix('y')

a1 = x.T
z2 = T.dot(w1, a1) + b1.repeat(a1.shape[1], axis=1)
a2 = 1.0 / (1 + T.exp(-z2))
z3 = T.dot(w2, a2) + b2.repeat(a2.shape[1], axis=1)
a3 = 1.0 / (1 + T.exp(-z3))

predict = function([x], [a3.T])

loss = T.sum((a3 - y.T) ** 2) / 2

cost = function([x, y], [loss])

grad_w2, grad_b2 = T.grad(loss, [w2, b2])
grad_w1, grad_b1 = T.grad(loss, [w1, b1])


gradients = function([x, y], [grad_w1, grad_w2, grad_b1, grad_b2])


def g_theta(X, Y):
    w1.set_value(w1.get_value() + EPSILON)
    w2.set_value(w2.get_value() + EPSILON)
    b1.set_value(b1.get_value() + EPSILON)
    b2.set_value(b2.get_value() + EPSILON)
    cost_old = cost(X, Y)[0]

    w1.set_value(w1.get_value() - 2 * EPSILON)
    w2.set_value(w2.get_value() - 2 * EPSILON)
    b1.set_value(b1.get_value() - 2 * EPSILON)
    b2.set_value(b2.get_value() - 2 * EPSILON)
    cost_new = cost(X, Y)[0]
    return (cost_old - cost_new) / (2 * EPSILON)

updates = [
        (w1, w1 - 0.3 * grad_w1 / x.shape[0]),
        (b1, b1 - 0.3 * grad_b1 / x.shape[0]),
        (w2, w2 - 0.3 * grad_w2 / x.shape[0]),
        (b2, b2 - 0.3 * grad_b2 / x.shape[0])]

train = function([x, y], [loss], updates=updates)


class SimpleBack(object):
    def predict(self, X):
        return predict(X)[0]

p = []
costs = []
for i in xrange(1000000):
    print train(X, Y)
    gt = g_theta(X, Y)
    grads = [g.ravel().sum() for g in gradients(X, Y)]
    p.append(np.abs(gt - np.sum(grads)))
    costs.append(cost(X, Y)[0])
    print p[-1]
draw_decision_boundary(SimpleBack(), X, Y)
