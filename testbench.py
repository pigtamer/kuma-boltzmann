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


def test(taskid = 1, iternum=10000, alpha=1):

    # Task 1: winner takes all
    if taskid == 1:
        x0 = 1; n = 9
        v = np.zeros(n)

        w, theta, C = ecalc_weights(nqueen_energy, n)
    elif taskid == 2:
        x0 = 1; n = 4
        v = np.zeros(n)
        A = np.array([[1,-1,2,1], [2,1,-2,1], [-1,2,1,2], [0,1,-1,-1]])
        b = np.array([3,0,0,-1])

        w, theta, C = ecalc_weights(lambda x: lineq_energy(x, A, b), n, False)
    elif taskid == 3:  
        x0 = 1; n = 4
        v = np.zeros(n)
        A = np.array([[1,-1,1,1], [2,0,-1,1], [0,1,-1,-1], [-1,1,1,-1]])
        b = np.array([2,2,-2,-1])

        w, theta, C = ecalc_weights(lambda x: lineq_energy(x, A, b), n, False)
    elif taskid == 4:
        pass
    elif taskid == 5:
        pass
    elif taskid == 6:
        pass

    
    bm = BoltzMachine(n, v, w, theta, x0)
    bm.show()
    
    iter_num = iternum
    en_this = np.zeros(iter_num, dtype=float)
    # en_all = []
    for k in range(iter_num):
        bm.randinit()
        for j in range(10):
            bm.update_all(FLAG_STOCH=True, alpha=alpha)
        en_this[k] = calc_energy(w, bm.get_value(), n, theta, x0, C)
        if iter_num < 100:
            bm.show()
    sns.set()
    plt.hist(en_this)
    # plt.hist(np.array(en_all))

    bm.view_graph()
    # plt.show()

# %%
test(iternum=100, taskid=1, alpha=10)


# %%
