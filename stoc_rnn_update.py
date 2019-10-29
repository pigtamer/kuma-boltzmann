# %%
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

# %%


class BoltzMachine():
    def __init__(self, n_nodes, value_nodes, weights, theta, x0, init_method='rand'):
        assert(n_nodes == weights.shape[0] &
               weights.shape[0] == weights.shape[1])
        self.weights = weights
        self.n_nodes = n_nodes
        self.value_nodes = value_nodes
        self.theta = theta
        self.x0 = x0
        if init_method == 'rand':
            self.randinit()

    def update_single(self, idx, FLAG_STOCH=False):
        # updates the value of a single node according to the partial summary and activation function
        s = -self.theta[idx]*self.x0 + \
            np.sum(self.weights[idx, :]*self.value_nodes)
        if FLAG_STOCH:
            new_value_i_j = act_stoc(s, 0.3)  # i set an arbitrary alpha.
        else:
            new_value_i_j = s > 0
        return new_value_i_j

    def update_all(self, FLAG_STOCH=False):
        # update ALL the values of BM
        for i in range(self.n_nodes):
            self.value_nodes[i] = self.update_single(
                i, FLAG_STOCH)  # one node at a time

    def randinit(self):
        # this func is supposed to initialize the values of our BM
        self.value_nodes = rnd.randn(self.n_nodes)

    def show(self):
        print(self.value_nodes, "\n")

    def get_value(self):
        return self.value_nodes

    def set_single(self):
        # set the value of a node manually
        pass

    def view_graph(self, layout=nx.DiGraph()):
        G = nx.convert_matrix.from_numpy_array(self.weights, create_using=layout)
        layout = nx.circular_layout(G)
        plt.figure(figsize=(8,8))
        nx.draw(G, layout)
        labels = nx.get_edge_attributes(G, "weight")
        nx.draw_networkx_edge_labels(G, pos=layout, edge_labels=labels)
        plt.show()
# %%


def calc_energy(w, v, n, t, x0, C):
    energy = C
    for i in range(n):
        energy += t[i]*v[i]
        for j in range(n):
            energy += -0.5*w[i, j]*v[i]*v[j]
    return energy
# %%


def test():
    # w = np.array([[0, -1, -1], [-1, 0, -1], [-1, -1, 0]])
    # theta = -np.array([-0.5, -0.5, -0.5])
    w = np.array([[0, 1, 1, 1, 0, 0, 1, 0, 0],
                  [1, 0, 1, 0, 1, 0, 0, 1, 0],
                  [1, 1, 0, 0, 0, 1, 0, 0, 1],
                  [1, 0, 0, 0, 1, 1, 1, 0, 0],
                  [0, 1, 0, 1, 0, 1, 0, 1, 0],
                  [0, 0, 1, 1, 1, 0, 0, 0, 1],
                  [1, 0, 0, 1, 0, 0, 0, 1, 1],
                  [0, 1, 0, 0, 1, 0, 1, 0, 1],
                  [0, 0, 1, 0, 0, 1, 1, 1, 0]]) * (-2)

    # w = np.array([[0, 1, 1, 1, 1, 1, 1, 1, 1],
    #               [1, 0, 1, 1, 1, 1, 1, 1, 1],
    #               [1, 1, 0, 1, 1, 1, 1, 1, 1],
    #               [1, 1, 1, 0, 1, 1, 1, 1, 1],
    #               [1, 1, 1, 1, 0, 1, 1, 1, 1],
    #               [1, 1, 1, 1, 1, 0, 1, 1, 1],
    #               [1, 1, 1, 1, 1, 1, 0, 1, 1],
    #               [1, 1, 1, 1, 1, 1, 1, 0, 1],
    #               [1, 1, 1, 1, 1, 1, 1, 1, 0]]) * (-2)
          
    theta = -2*np.ones(9)
    x0 = 1
    v = np.zeros(9)
    n = 9
    C = 6
    bm = BoltzMachine(n, v, w, theta, x0)
    bm.show()
    bm.view_graph()
    iter_num = 5000
    en_this = np.zeros(iter_num, dtype=float)
    stt_this = np.zeros(iter_num, dtype=int)
    for k in range(iter_num):
        bm.randinit()
        for j in range(3):
            bm.update_all(FLAG_STOCH=False)
        stt_this[k] = sum(bm.get_value()*2**np.arange(n))
        en_this[k] = calc_energy(w, bm.get_value(), n, theta, x0, 6)
        if iter_num <= 100:
            bm.show()
    sns.set()
    plt.hist(en_this)

    # en_theo = np.linspace(0, 8, 100)
    # plt.figure(2)
    # plt.plot(en_theo, en_theo*np.exp(-0.8*en_theo))
    # plt.show()
    plt.figure(3)
    sns.distplot(stt_this)
    plt.show()
    print(np.unique(stt_this))


# %%
test()
