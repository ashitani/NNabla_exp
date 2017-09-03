#!/usr/bin/env python

import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF

import numpy as np
import matplotlib.pyplot as plt

batch_size = 1

x = nn.Variable((batch_size,1))
h1 = F.elu(PF.affine(x, 16,name="affine1"))
h2 = F.elu(PF.affine(h1, 32,name="affine2"))
y = F.elu(PF.affine(h2, 1,name="affine3"))

nn.load_parameters("exp_net.h5")

xi=np.linspace(0,1,100)
plt.plot(xi,np.exp(xi))

ys=[]
for i in range(100):
    x.d = xi[i]
    y.forward()
    ys.append(y.d.copy())
_=plt.plot(xi, np.array(ys).reshape((100,)),"r")

plt.show()