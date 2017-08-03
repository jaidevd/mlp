# coding: utf-8
import numpy as np
x = np.random.rand(10,10)
H = np.random.rand(10,)
wxh = np.random.rand(10,10)
x = np.random.rand(10,)
np.dot(wxh, x)
get_ipython().magic('ls ')
get_ipython().magic('less proc.txt')
get_ipython().magic('clear ')
data = open("proc.txt", "r").read()
chars = set(data)
char2ix = {c:i for i, c in enumerate(chars)}
ix2char = {i:c for c, i in char2ix.items()}
len(chars)
wxh = np.random.rand(68, 100)
whh = np.random.rand(100,100)
why = np.random.rand(100, 68)
len(data)
X = np.zeros(len(data), len(chars))
X = np.zeros((len(data), len(chars)), dtype=int)
for i, char in enumerate(data):
    X[i, char2ix[char]] = 1
    
X.sum(1)
_.max()
X.sum(1).min()
X.sum(0)
np.argmax(_)
ix2char[65]
