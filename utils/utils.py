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
    # for example in a 3x3 chesspad, we place queens (castles)
    # S{i=0...2}(  (S{j=1...3}( x_{i*3+j} ) - 1)**2 )  ) +
    # S{i=1...3}(  (S{j=0...2}( x_{i+j*3} ) - 1)**2 )  )
    # --->> Generalize 
    # S{i=0...N-1}(  (S{j=1...N  }( x_{i*3+j} ) - 1)**2 )  ) +
    # S{i=1...N  }(  (S{j=0...N-1}( x_{i+j*3} ) - 1)**2 )  )
    energy = 0
    pad_width = int(np.sqrt(len(x)))
    for i in range(pad_width):
        term = 0
        for j in range(1, pad_width + 1):
            term += x[i*pad_width + j - 1]
        energy += (term -1) ** 2

    for i in range(1, pad_width + 1):
        term = 0
        for j in range(pad_width):
            term += x[i + j*pad_width - 1]
        energy += (term -1) ** 2

    return energy


def lineq_energy(x, A, b):
    # Equations are expressed in Ax = b
    assert(A.shape[0] == A.shape[1] and A.shape[0] == len(b))
    energy = 0
    for k in range(len(b)):
        term = 0
        for j in range(len(b)):
            term += x[j] * A[k][j]
        term -= b[k]
        energy += term**2
    return energy

def cnpost_energy(x, we, beta=1E3):
    # we: weight of edges
    E_c = nqueen_energy(x)
    E_l = 0
    p = int(np.sqrt(len(x)))
    x = np.reshape(x, (p,p))
    print(x)
    for i in range(p-1):
        for j in range(p):
            for k in range(p):
                E_l += we[j][k]*x[i][j]*x[i+1][k]
    energy = E_c + E_l
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
            if l == k:
                wxkl[k][l]=0
            else:
                xkl = x.copy(); xkl[k] = 1; xkl[l] = 1
                wxkl[k][l] = -fenergy(xkl) + thek[k] + thek[l] + C
            if IF_SYMM:
                wxkl[l][k] = wxkl[k][l]
    return (wxkl, thek, C) # notice the sequence

def diag0(x):
    for k in range(x.shape[0]):
        x[k,k]=0
    return x
# %%

# test energy funcs
print(nqueen_energy([1,0,0,0,0,1,0,1,0]), "\n")
print(ecalc_weights(nqueen_energy, 9, False), "\n")

A = np.array([[1,-1,1,1], [2,0,-1,1], [0,1,-1,-1], [-1,1,1,-1]])
b = np.array([2,2,-2,-1])
print( lineq_energy([1,1,0,0], A, b), "\n")

# using lambda function to fix certain values
print(ecalc_weights(lambda x: lineq_energy(x, A, b), 4, False), "\n")

# %%
