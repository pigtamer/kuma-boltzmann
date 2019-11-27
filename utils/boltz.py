#%%
from utils import *

class BoltzMachine():
    def __init__(self, n_nodes, value_nodes, weights, theta, x0, init_method='rand'):
        assert(n_nodes == weights.shape[0] &
               weights.shape[0] == weights.shape[1])
        self.weights = weights
        self.n_nodes = n_nodes
        self.value_nodes = value_nodes
        self.theta = theta
        self.x0 = x0
        self.fix_list = np.zeros(n_nodes)
        if init_method == 'rand':
            self.randinit()

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
        self.value_nodes = 1/abs(rnd.randn(self.n_nodes))

    def show(self):
        print(self.value_nodes, "\n")

    def get_value(self):
        return self.value_nodes

    def set_single(self):
        # set the value of a node manually
        pass

    def fix_val(self, idxs, vals):
        # fix the values of certain nodes
        pass

    def view_graph(self, layout=nx.DiGraph()):
        G = nx.convert_matrix.from_numpy_array(self.weights, create_using=layout)
        layout = nx.circular_layout(G)
        plt.figure(figsize=(4,4))
        nx.draw(G, layout)
        labels = nx.get_edge_attributes(G, "weight")
        nx.draw_networkx_edge_labels(G, pos=layout, edge_labels=labels)
        plt.show()