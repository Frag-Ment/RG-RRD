import torch
import torch.nn as nn
import torch.nn.functional as F
from model.mlp import MLP


class GraphCNN(nn.Module):
    def __init__(self,
                 num_layers,
                 num_mlp_layers,
                 input_dim,
                 hidden_dim,
                 # final_dropout,
                 learn_eps,
                 neighbor_pooling_type,
                 device):
        '''
        num_layers: number of layers in the neural networks (INCLUDING the input layer)
        num_mlp_layers: number of layers in mlps (EXCLUDING the input layer)
        input_dim: dimensionality of input features
        hidden_dim: dimensionality of hidden units at ALL layers
        output_dim: number of classes for prediction
        final_dropout: dropout ratio on the final linear layer
        learn_eps: If True, learn epsilon to distinguish center nodes from neighboring nodes. If False, aggregate neighbors and center nodes altogether.
        neighbor_pooling_type: how to aggregate neighbors (mean, average, or max)
        device: which device to use
        '''
        super(GraphCNN, self).__init__()

        # self.final_dropout = final_dropout
        self.device = device
        self.num_layers = num_layers
        self.neighbor_pooling_type = neighbor_pooling_type
        self.learn_eps = learn_eps

        # common out the eps if you do not need to use it, otherwise the it will cause
        # error "not in the computational graph"
        # if self.learn_eps:
        #     self.eps = nn.Parameter(torch.zeros(self.num_layers - 1))

        # List of MLPs
        self.mlps = torch.nn.ModuleList()

        # List of batchnorms applied to the output of MLP (input of the final prediction linear layer)
        # self.batch_norms = torch.nn.ModuleList()

        for layer in range(self.num_layers-1):
            if layer == 0:
                self.mlps.append(MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim))
            else:
                self.mlps.append(MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim))

            # self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

    def next_layer(self, h, layer, padded_neighbor_list = None, Adj_block = None):
        ''' 6月12日改回去了 '''
        # h = self.mlps[layer](h)
        pooled = torch.mm(Adj_block, h)
        # representation of neighboring and center nodes
        pooled_rep = self.mlps[layer](pooled)

        # h = self.batch_norms[layer](pooled_rep)
        h = pooled_rep


        # non-linearity
        h = F.leaky_relu(h)
        return h


    def forward(self,
                x,
                adj,
                graph_pool,
                ):

        padded_nei = None,
        x_concat = x
        graph_pool = graph_pool
        Adj_block = adj

        # list of hidden representation at each layer (including input)
        h = x_concat

        for layer in range(self.num_layers-1):
            h = self.next_layer(h, layer, Adj_block=Adj_block)


        h_nodes = h.clone()
        # print(graph_pool.shape, h.shape)
        pooled_h = torch.sparse.mm(graph_pool, h)
        # pooled_h = graph_pool.spmm(h)

        # print('pooled h', pooled_h)
        # print('h nodes', h_nodes)
        return pooled_h, h_nodes

