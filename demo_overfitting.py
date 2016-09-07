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

from backprop import Backpropagation
from sklearn.preprocessing import OneHotEncoder
from utils import make_circles, draw_decision_boundary
import matplotlib.pyplot as plt

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

# Unregularized noisy
X, y = make_circles(factor=0.1, noise=0.22)
y = OneHotEncoder().fit_transform(y.reshape(-1, 1)).toarray()
bp = Backpropagation(layers=[2, 10, 2], C=0.0)
bp.fit(X, y, n_iter=100000)
draw_decision_boundary(bp, X, y, show=False, ax=ax[0, 0])
ax[0, 0].set_title("Unreg noisy")

# Unregulated ideal
X, y = make_circles(factor=0.2)
y = OneHotEncoder().fit_transform(y.reshape(-1, 1)).toarray()
draw_decision_boundary(bp, X, y, show=False, ax=ax[0, 1])
ax[0, 1].set_title("Unreg ideal")

# regularized noisy
X, y = make_circles(factor=0.1, noise=0.22)
y = OneHotEncoder().fit_transform(y.reshape(-1, 1)).toarray()
bp = Backpropagation(layers=[2, 10, 2], C=0.1)
bp.fit(X, y, n_iter=100000)
draw_decision_boundary(bp, X, y, show=False, ax=ax[1, 0])
ax[1, 0].set_title("reg noisy")

# regulated ideal
X, y = make_circles(factor=0.2)
y = OneHotEncoder().fit_transform(y.reshape(-1, 1)).toarray()
draw_decision_boundary(bp, X, y, show=False, ax=ax[1, 1])
ax[1, 1].set_title("reg ideal")

plt.show()
