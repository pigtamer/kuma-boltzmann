# %%
import sys
sys.path.append("./utils")

from utils import *
from boltz import BoltzMachine

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import networkx as nx

sns.set()
# %%

def test(taskid=1, iternum=10000, alpha=1):
    if taskid == 1:
        x0 = 1
        n = 9
        w, theta, C = ecalc_weights(nqueen_energy, n)
    elif taskid == 2:
        x0 = 1
        n = 4
        A = np.array([[1, -1, 2, 1], [2, 1, -2, 1],
                      [-1, 2, 1, 2], [0, 1, -1, -1]])
        b = np.array([3, 0, 0, -1])
        w, theta, C = ecalc_weights(lambda x: lineq_energy(x, A, b), n, False)
    elif taskid == 3:
        x0 = 1
        n = 4
        v = np.zeros(n)
        A = np.array([[1, -1, 1, 1], [2, 0, -1, 1],
                      [0, 1, -1, -1], [-1, 1, 1, -1]])
        b = np.array([2, 2, -2, -1])
        w, theta, C = ecalc_weights(lambda x: lineq_energy(x, A, b), n, False)
    elif taskid == 4:
        x0 = 1
        n = 16
        m_dis = [[100, 1, 2, 3], [1, 100, 4, 5],
                 [2, 4, 100, 6], [3, 5, 6, 100]]
        w, theta, C = ecalc_weights(
            lambda x: cnpost_energy(x, m_dis, beta=10), 16, True)
    elif taskid == 5:
        n_nodes = 6
        bm = BoltzMachine(n_nodes)

        adja_mask = np.ones((n_nodes, n_nodes))
        for k in range(n_nodes):
            adja_mask[1, k] = 0
            adja_mask[0, k] = 0
            adja_mask[k, -1] = 0
            adja_mask[k, k] = 0
        
        adja_mask[2, [3,4]] = 0
        adja_mask[3, [2,4]] = 0
        adja_mask[4, [3,2]] = 0
        adja_mask[n_nodes-1, 0] = 0
        adja_mask[n_nodes-1, 1] = 0

        # A:
        N = iternum

        wavg = np.zeros((n_nodes, n_nodes))
        tavg = np.zeros(n_nodes)
        bm.randinit()
        bm.randinit_weight()
        bm.weights *= adja_mask

        bm.fix_val([0, 1], [0, 0])  # fix x1, x2 to (0,0)
        for t in range(N):
            wavg += np.matmul(np.matrix(bm.value_nodes).T,
                              np.matrix(bm.value_nodes))

            tavg += bm.x0*bm.value_nodes
            bm.update_all(FLAG_STOCH=True, alpha=alpha)
        wavg *= adja_mask
        wavg_a = wavg / N
        tavg_a = tavg / N
        # B:

        print(bm.weights)
        print(bm.theta)

        wavg = np.zeros((n_nodes, n_nodes))
        tavg = np.zeros(n_nodes)
        eps = 0.1

        bm.randinit()
        for t in range(N):
            # bm.randinit_weight()
            # bm.weights*=adja_mask

            dice = rnd.rand()
            y = 0 if dice < 0.8 else 1
            bm.fix_val([0, 1, -1], [0, 0, y])
            wavg += np.matmul(np.matrix(bm.value_nodes).T,
                              np.matrix(bm.value_nodes))
            wavg = wavg * adja_mask

            tavg += bm.x0*bm.value_nodes
            tavg[[0,1]] = 0
            bm.update_all(FLAG_STOCH=True, alpha=alpha)
        wavg_b = wavg / N
        tavg_b = tavg / N

        dw = (wavg_b - wavg_a) * eps
        dt = (tavg_b - tavg_a) * eps

        bm.weights += dw
        bm.theta += dt
        print(bm.weights)
        print(bm.theta)
        bm.unlock()
        N = 10000
        en_this = np.zeros(N, dtype=float)
        bm.fix_val([0, 1], [0, 0])
        bm.randinit()

        for k in range(N):
            # for j in range(10):
            # print(bm.weights)
            bm.update_all(FLAG_STOCH=True, alpha=alpha)
            en_this[k] = bm.value_nodes[-1]
        plt.hist(en_this)
        plt.show()
        return

    # unified process for tasks 1~4
    v = np.zeros(n)
    bm = BoltzMachine(n, v, w, theta, x0)

    iter_num = iternum
    en_this = np.zeros(iter_num, dtype=float)
    enwave_t2 = []
    stts = [None] * iter_num

    bm.randinit()
    if taskid != 2:
        for k in range(iter_num):
            bm.update_all(FLAG_STOCH=True, alpha=alpha)
            en_this[k] = calc_energy(w, bm.value_nodes, n, theta, x0, C)
            stts[k] = np.array2string(bm.value_nodes)

            vseq = pd.Series(stts).value_counts()
            eseq = pd.Series(en_this).value_counts()
        plt.figure()
        vseq.plot('bar')
        plt.figure()
        eseq.plot('bar')
    else:
        for idx in range(n):
            bm.update_single(idx, alpha=alpha)
            enwave_t2.append(calc_energy(
                w, bm.value_nodes, n, theta, x0, C))
        plt.plot(enwave_t2)

    bm.view_graph()
    plt.show()
    return


# %%
test(iternum=10000, taskid=5, alpha=0.01)


# %%
