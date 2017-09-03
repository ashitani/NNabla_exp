#!/usr/bin/env python
from __future__ import print_function

import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S

import numpy as np

def get_batch(n):
    x = np.random.random(n)
    y = np.exp(x)
    return x,y

batch_size = 100

x = nn.Variable((batch_size,1))
h1 = F.elu(PF.affine(x, 16,name="affine1"))
h2 = F.elu(PF.affine(h1, 32,name="affine2"))
y = F.elu(PF.affine(h2, 1,name="affine3"))

t = nn.Variable((batch_size,1))
loss = F.mean(F.squared_error(y, t))

solver = S.Adam()
solver.set_parameters(nn.get_parameters())


losses=[]
for i in range(10000):
    dat=get_batch(batch_size)
    x.d=dat[0].reshape((batch_size,1))
    t.d=dat[1].reshape((batch_size,1))
    loss.forward()
    solver.zero_grad()
    loss.backward()
    solver.update()
    losses.append(loss.d.copy())
    if i % 1000 == 0:  # Print for each 1000 iterations
        print(i, loss.d)

# save as H5 for python prediction

nn.save_parameters("exp_net.h5")

# save as NNP for C++ prediction

import nnabla.utils.save

runtime_contents = {
        'networks': [
            {'name': 'runtime',
             'batch_size': 1,
             'outputs': {'y': y},
             'names': {'x': x}}],
        'executors': [
            {'name': 'runtime',
             'network': 'runtime',
             'data': ['x'],
             'output': ['y']}]}
nnabla.utils.save.save('exp_net.nnp', runtime_contents)