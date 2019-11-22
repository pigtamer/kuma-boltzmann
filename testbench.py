# %%
import sys
sys.path.append("./utils")

from utils import *
from boltz import BoltzMachine

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx


# %%


def test(taskid = 1, iternum=10000):
    # w = np.array([[0, -1, -1], [-1, 0, -1], [-1, -1, 0]])
    # theta = -np.array([-0.5, -0.5, -0.5])

    # Task 1: winner takes all
    if taskid == 1:
        w = np.array([[0, 1, 1, 1, 0, 0, 1, 0, 0],
                    [1, 0, 1, 0, 1, 0, 0, 1, 0],
                    [1, 1, 0, 0, 0, 1, 0, 0, 1],
                    [1, 0, 0, 0, 1, 1, 1, 0, 0],
                    [0, 1, 0, 1, 0, 1, 0, 1, 0],
                    [0, 0, 1, 1, 1, 0, 0, 0, 1],
                    [1, 0, 0, 1, 0, 0, 0, 1, 1],
                    [0, 1, 0, 0, 1, 0, 1, 0, 1],
                    [0, 0, 1, 0, 0, 1, 1, 1, 0]]) * (-2)
        theta = -2*np.ones(9)
        x0 = 1
        v = np.zeros(9)
        n = 9
        C = 6
    elif taskid == 2:
        pass
    elif taskid == 3:

        # Task 3: prob linear eq
        w = np.array([[0, 4, 0,-8],
                    [4, 0, 2, 10],
                    [0, 2, 0, 4],
                    [-8,10, 4, 0]])
        theta = [-8, 13, 2, -6]
        x0 = 1
        v = np.zeros(4)
        n = 4
        C = 13         
    elif taskid == 4:
        pass
    elif taskid == 5:
        pass
    elif taskid == 6:
        pass

    
    bm = BoltzMachine(n, v, w, theta, x0); alpha=5
    bm.show()
    
    iter_num = iternum
    en_this = np.zeros(iter_num, dtype=float)
    # en_all = []
    for k in range(iter_num):
        bm.randinit()
        for j in range(3):
            bm.update_all(FLAG_STOCH=True, alpha=alpha)
        en_this[k] = calc_energy(w, bm.get_value(), n, theta, x0, C)
        if iter_num <= 100:
            bm.show()
    sns.set()
    plt.hist(en_this)
    # plt.hist(np.array(en_all))

    bm.view_graph()
    # plt.show()

# %%
test(iternum=10000)


# %%
