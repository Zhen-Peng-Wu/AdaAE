from copy import deepcopy
import numpy as np
import torch
from torch.nn import Sequential, Linear, ReLU
import torch.nn.functional as F
from torch_scatter import scatter_add
import torch_geometric.utils as tg_utils
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GINConv
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.nn.inits import glorot, zeros

def attribute_masking(data, aug_ratio=0.2):
    data = deepcopy(data)

    node_num, feat_dim = data.x.size()
    mask_num = int(node_num * aug_ratio)

    idx_mask = np.random.choice(node_num, mask_num, replace=False)
    if data.edge_attr is None:
        token = torch.tensor(np.random.normal(loc=0.5, scale=0.5, size=(mask_num, feat_dim)),
                             dtype=torch.float32).to(data.x.device)
    else:
        token = data.x.float().mean(dim=0).long()
    data.x[idx_mask] = token
    return data


def edge_perturbation(data, aug_ratio=0.2):
    data = deepcopy(data)

    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    permute_num = int(edge_num * aug_ratio)

    idx_add = np.random.choice(node_num, (2, permute_num))
    idx_drop = np.random.choice(edge_num, edge_num - permute_num, replace=False)

    edge_index = data.edge_index[:, idx_drop]
    data.edge_index = edge_index

    if data.edge_attr is not None:
        edge_attr = data.edge_attr[idx_drop]
        data.edge_attr = edge_attr
    return data


def identical(data, aug_ratio=0.2):
    data = deepcopy(data)
    return data


def node_dropping(data, aug_ratio=0.2):
    data = deepcopy(data)

    x = data.x
    edge_index = data.edge_index

    drop_num = int(data.num_nodes * aug_ratio)
    keep_num = data.num_nodes - drop_num

    keep_idx = torch.randperm(data.num_nodes)[:keep_num]
    if data.edge_attr is None:
        edge_index, edge_attr = tg_utils.subgraph(keep_idx, edge_index)
    else:
        edge_index, edge_attr = tg_utils.subgraph(keep_idx, edge_index, edge_attr=data.edge_attr, relabel_nodes=True, num_nodes=data.num_nodes)

    drop_idx = torch.ones(x.shape[0], dtype=bool)
    drop_idx[keep_idx] = False
    x[drop_idx] = 0

    data.x = x
    data.edge_index = edge_index
    data.edge_attr = edge_attr

    return data


def subgraph(data, aug_ratio=0.2):
    data = deepcopy(data)

    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    sub_num = int(node_num * (1 - aug_ratio))

    x = data.x
    edge_index = data.edge_index

    idx_sub = torch.randint(0, data.num_nodes, (1,)).to(edge_index.device)
    last_idx = idx_sub

    keep_idx = None
    diff = None

    # print("sub_num:", sub_num)
    for k in range(1, sub_num):
        keep_idx, _, _, _ = tg_utils.k_hop_subgraph(last_idx, 1, edge_index)
        # print("subgraph: {}, keep_idx size: {}".format(k, keep_idx.shape[0]) )
        if keep_idx.shape[0] == last_idx.shape[0] or keep_idx.shape[0] >= sub_num or k == sub_num - 1:
            combined = torch.cat((last_idx, keep_idx)).to(edge_index.device)
            uniques, counts = combined.unique(return_counts=True)
            diff = uniques[counts == 1]
            break

        last_idx = keep_idx

    diff_keep_num = min(sub_num - last_idx.shape[0], diff.shape[0])
    diff_keep_idx = torch.randperm(diff.shape[0])[:diff_keep_num].to(edge_index.device)
    final_keep_idx = torch.cat((last_idx, diff_keep_idx))

    drop_idx = torch.ones(x.shape[0], dtype=bool)
    drop_idx[final_keep_idx] = False
    x[drop_idx] = 0

    if data.edge_attr is None:
        edge_index, edge_attr = tg_utils.subgraph(final_keep_idx, edge_index)
    else:
        edge_index, edge_attr = tg_utils.subgraph(final_keep_idx, edge_index, edge_attr=data.edge_attr, relabel_nodes=True, num_nodes=data.num_nodes)

    data.x = x
    data.edge_index = edge_index
    data.edge_attr = edge_attr
    return data


def aug_map(aug_type):
    if aug_type == 'attribute_masking':
        return attribute_masking
    elif aug_type == 'edge_perturbation':
        return edge_perturbation
    elif aug_type == 'identical':
        return identical
    elif aug_type == 'node_dropping':
        return node_dropping
    elif aug_type == 'subgraph':
        return subgraph
    else:
        raise Exception("wrong augmentation function")


def act_map(activation_type):
    if activation_type == "elu":
        act = torch.nn.functional.elu
    elif activation_type == "leaky_relu":
        act = torch.nn.functional.leaky_relu
    elif activation_type == "relu":
        act = torch.nn.functional.relu
    elif activation_type == "relu6":
        act = torch.nn.functional.relu6
    elif activation_type == "sigmoid":
        act = torch.nn.functional.sigmoid
    elif activation_type == "softplus":
        act = torch.nn.functional.softplus
    elif activation_type == "tanh":
        act = torch.nn.functional.tanh
    elif activation_type == "linear":
        act = lambda x: x
    else:
        raise Exception("Wrong activate function")
    return act

def pooling_map(pooling_type):
    if pooling_type == 'global_max':
        pooling = global_max_pool
    elif pooling_type == 'global_mean':
        pooling = global_mean_pool
    elif pooling_type == 'global_add':
        pooling = global_add_pool
    else:
        raise Exception("Wrong pooling function")
    return pooling

def conv_map(attention_type, aggregator_type, input_dim, hidden_dimension_num, bias, withEdge):
    ### withEdge is False(or edge_attr is None), represent the unsupervised learning and semisupervised learning tasks
    ### withEdge is True(or edge_attr is exist), represent the transfer learning task
    if attention_type == 'gcn':
        if withEdge == False:
            conv_layer = GCNConv(input_dim,
                                 hidden_dimension_num,
                                 bias=bias,
                                 aggr=aggregator_type,
                                 normalize=True)
        else:
            conv_layer = GCNConvWithEdge(input_dim,
                                         hidden_dimension_num,
                                         aggr=aggregator_type)
    elif attention_type == 'gat':
        if withEdge == False:
            conv_layer = GATConv(input_dim,
                                 hidden_dimension_num,
                                 bias=bias,
                                 aggr=aggregator_type,
                                 heads=4,
                                 concat=False)
        else:
            conv_layer = GATConvWithEdge(input_dim,
                                         hidden_dimension_num,
                                         aggr=aggregator_type,
                                         heads=4)
    elif attention_type == 'graphsage':
        if withEdge == False:
            conv_layer = SAGEConv(input_dim,
                                  hidden_dimension_num,
                                  bias=bias,
                                  aggr=aggregator_type,
                                  normalize=True)
        else:
            conv_layer = SAGEConvWithEdge(input_dim,
                                          hidden_dimension_num,
                                          aggr=aggregator_type)
    elif attention_type == 'gin':
        if withEdge == False:
            conv_layer = GINConv(Sequential(Linear(input_dim, 2*hidden_dimension_num),
                                            ReLU(),
                                            Linear(2*hidden_dimension_num, hidden_dimension_num)),
                                 # train_eps=True,
                                 aggr=aggregator_type)
        else:
            conv_layer = GINConvWithEdge(input_dim,
                                         hidden_dimension_num,
                                         aggr=aggregator_type)
    else:
        raise Exception("Wrong conv function")
    return conv_layer





num_bond_type = 6  # including aromatic and self-loop edge, and extra masked tokens
num_bond_direction = 3


class GCNConvWithEdge(MessagePassing):

    def __init__(self, input_dim, hidden_dimension_num, aggr="add"):
        super(GCNConvWithEdge, self).__init__()

        self.input_dim = input_dim
        self.hidden_dimension_num = hidden_dimension_num
        self.aggr = aggr

        self.linear = torch.nn.Linear(input_dim, hidden_dimension_num)
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, hidden_dimension_num)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, hidden_dimension_num)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

    def norm(self, edge_index, num_nodes, dtype):
        ### assuming that self-loops have been already added in edge_index
        edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                                 device=edge_index.device)
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])

        norm = self.norm(edge_index, x.size(0), x.dtype)

        x = self.linear(x)

        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings, norm=norm)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * (x_j + edge_attr)


class GATConvWithEdge(MessagePassing):
    def __init__(self, input_dim, hidden_dimension_num, aggr="add", heads=2, negative_slope=0.2):
        super(GATConvWithEdge, self).__init__()

        self.input_dim = input_dim
        self.hidden_dimension_num = hidden_dimension_num
        self.aggr = aggr
        self.heads = heads
        self.negative_slope = negative_slope

        self.weight_linear = torch.nn.Linear(input_dim, heads * hidden_dimension_num)
        self.att = torch.nn.Parameter(torch.Tensor(1, heads, 2 * hidden_dimension_num))
        self.bias = torch.nn.Parameter(torch.Tensor(hidden_dimension_num))

        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, heads * hidden_dimension_num)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, heads * hidden_dimension_num)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])

        x = self.weight_linear(x)

        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, edge_index, x_i, x_j, edge_attr):

        x_j += edge_attr

        e_ = torch.cat([x_i, x_j], dim=-1).view(-1, self.heads, 2*self.hidden_dimension_num)
        alpha = (e_ * self.att).sum(dim=-1).sum(dim=-1).view(-1, 1)

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index[1])

        return x_j * alpha

    def update(self, aggr_out):
        aggr_out = aggr_out.mean(dim=1).view(-1,1)
        aggr_out = aggr_out + self.bias

        return aggr_out


class SAGEConvWithEdge(MessagePassing):
    def __init__(self, input_dim, hidden_dimension_num, aggr="mean"):
        super(SAGEConvWithEdge, self).__init__()

        self.input_dim = input_dim
        self.hidden_dimension_num = hidden_dimension_num
        self.aggr = aggr

        self.linear = torch.nn.Linear(input_dim, hidden_dimension_num)
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, hidden_dimension_num)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, hidden_dimension_num)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])

        x = self.linear(x)

        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return F.normalize(aggr_out, p=2, dim=-1)


class GINConvWithEdge(MessagePassing):
    """
    Extension of GIN aggregation to incorporate edge information by concatenation.
    Args:
        hidden_dimension_num (int): dimensionality of embeddings for nodes and edges.
        embed_input (bool): whether to embed input or not.

    See https://arxiv.org/abs/1810.00826
    """

    def __init__(self, input_dim, hidden_dimension_num, aggr="add"):
        super(GINConvWithEdge, self).__init__()

        # multi-layer perceptron
        self.input_dim = input_dim
        self.hidden_dimension_num = hidden_dimension_num
        self.aggr = aggr

        self.mlp = torch.nn.Sequential(torch.nn.Linear(input_dim, 2 * hidden_dimension_num), torch.nn.ReLU(),
                                       torch.nn.Linear(2 * hidden_dimension_num, hidden_dimension_num))

        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, input_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, input_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])

        # return self.propagate(self.aggr, edge_index, x=x, edge_attr=edge_embeddings)
        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)

