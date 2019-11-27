#%%
from utils import *

class BoltzMachine():
    def __init__(self, n_nodes, value_nodes=None, weights=None, theta=None, x0=1, init_method='rand'):
        self.n_nodes = n_nodes
        self.x0 = x0
        self.fix_list = np.zeros(n_nodes)
        if weights==None or value_nodes==None:
            if init_method == 'rand':
                self.randinit_weight()
                self.randinit()
        else:
            assert(n_nodes == weights.shape[0] &
               weights.shape[0] == weights.shape[1])
            self.weights = weights
            self.theta = theta
            self.value_nodes = value_nodes
            


    def update_single(self, idx, FLAG_STOCH=False, alpha=1):
        # updates the value of a single node according to the partial summary and activation function
        s = -self.theta[idx]*self.x0 + \
            np.sum(self.weights[idx, :]*self.value_nodes)
        if FLAG_STOCH:
            new_value_i_j = act_stoc(s, alpha)  # i set an arbitrary alpha.
        else:
            new_value_i_j = s > 0
        return new_value_i_j

    def update_all(self, FLAG_STOCH=False, alpha=1):
        # update ALL the values of BM
        for i in range(self.n_nodes):
            if self.fix_list[i] == 0:
                self.value_nodes[i] = self.update_single(
                    i, FLAG_STOCH, alpha)  # one node at a time
            else:
                continue

    def randinit(self):
        # this func is supposed to initialize the values of our BM
        self.value_nodes = rnd.rand(self.n_nodes)

    def randinit_weight(self):
        self.weights = rnd.rand(self.n_nodes, self.n_nodes)
        self.theta = rnd.rand(self.n_nodes)
        # print("Randinit weights. All presets of connection weights are hereby cleared! \n")

    def show(self):
        print(self.value_nodes, "\n")

    def get_value(self):
        return self.value_nodes

    def set_single(self):
        # set the value of a node manually
        raise Exception(DeprecationWarning)

    def fix_val(self, idxs, vals):
        # fix the values of certain nodes
        # idxs: list or nparray, indices of fixed points
        # vals: list of node values to be fixed on
        self.fix_list[idxs] = 1
        self.value_nodes[idxs] = vals

    def unlock(self):
        self.fix_list = np.zeros(self.n_nodes)

    def view_graph(self, layout=nx.DiGraph()):
        G = nx.convert_matrix.from_numpy_array(self.weights, create_using=layout)
        layout = nx.circular_layout(G)
        plt.figure(figsize=(4,4))
        nx.draw(G, layout)
        labels = nx.get_edge_attributes(G, "weight")
        nx.draw_networkx_edge_labels(G, pos=layout, edge_labels=labels)
        plt.show()

# %%
