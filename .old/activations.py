import numpy as np
from numpy import random as rnd
import matplotlib.pyplot as plt

def sigmoid(s, aplha):
    # aplha: sigmoid gain
    # s: weighted sum of neuron
    return 1 / (1+np.exp(-aplha*s))

def act_stoc(s,aplha):
    dice = rnd.randn()
    res = s.copy()
    for k in range(s.shape[0]):
        res[k] = float(sigmoid(s[k], aplha) > dice)

    return res

def test():
    x=np.linspace(-10,10,1000)
    for alpha in np.linspace(0, 10, 30):
        plt.plot(x, sigmoid(x, alpha))
    plt.show()

test()