#%%
import numpy as np
import matplotlib.pyplot as plt
from numpy import random as rnd
import seaborn as sns
import networkx as nx

# %%
# utils
def sigmoid(s, alpha):
    # alpha: sigmoid gain
    # s: weighted sum of neuron
    return 1 / (1+np.exp(-alpha*s))


def act_stoc(s, alpha):
    dice = abs(rnd.rand())
    res = float(sigmoid(s, alpha) > dice)
    return res

def calc_energy(w, v, n, t, x0, C):
    energy = C
    for i in range(n):
        energy += t[i]*v[i]
        for j in range(n):
            energy += -0.5*w[i, j]*v[i]*v[j]
    return energy

#%%
def nqueen_energy(x):
    nparam = len(x)
    
    # S{i=0...2}(  (S{j=1...3}( x_{i*3+j} ) - 1)**2 )  ) +
    # S{i=1...3}(  (S{j=0...2}( x_{i+j*3} ) - 1)**2 )  )
    # --->> Generalize 
    # S{i=0...N-1}(  (S{j=1...N  }( x_{i*3+j} ) - 1)**2 )  ) +
    # S{i=1...N  }(  (S{j=0...N-1}( x_{i+j*3} ) - 1)**2 )  )
    energy = 0

    for i in range(int(np.sqrt(nparam))):
        term = 0
        for j in range(1, int(np.sqrt(nparam)) + 1):
            term += x[i*3 + j - 1]
        energy += (term -1) ** 2

    for i in range(1, int(np.sqrt(nparam)) + 1):
        term = 0
        for j in range(int(np.sqrt(nparam))):
            term += x[i + j*3 - 1]
        energy += (term -1) ** 2

    return energy

#%%
def ecalc_weights(fenergy, nparam, IF_SYMM = True):
    # calculates the weights of a BM from the energy function
    # f-energy sould be in the format of f([x0, x1, x2... xn]), the calculation of parameters at each position are completed
    # inside of energy function, we do not care about them here.
    # the example of n-queens problem is given above.

    wxkl = np.zeros((nparam, nparam))   # conn weight xk->xl, assume that xkl = xlk
                # for  Chinese-postman problem, the graph in assumed to be complete
    thek = np.zeros(nparam) # theta-k

    x = np.zeros(nparam) # 1 by nparam

    C = fenergy(x) # params are all-zero

    for k in range(nparam):
        xk1 = x.copy(); xk1[k] = 1
        thek[k] = fenergy(xk1) - fenergy(x)
    
    for k in range(nparam):
        if IF_SYMM:
            lbl = k+1
        else:
            lbl = 0
        for l in range(lbl, nparam): # assume symmetry
            xkl = x.copy(); xkl[k] = 1; xkl[l] = 1
            wxkl[k][l] = -fenergy(xkl) + thek[k] + thek[l] + C
            if IF_SYMM:
                wxkl[l][k] = wxkl[k][l]
    return (wxkl, thek, C) # notice the sequence

# %%
