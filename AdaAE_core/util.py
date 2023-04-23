import torch
from numpy import *

def data_information(data):

    the_graph_num = data.len_dataset

    graph_size_list = []
    for sample in data.dataset[:the_graph_num]:
        graph_size_list.append(sample.x.shape[0])

    graph_max_size = max(graph_size_list)
    graph_min_size = min(graph_size_list)
    graph_mean_size = mean(graph_size_list)

    edge_size_list = []
    for sample in data.dataset[:the_graph_num]:
        edge_size_list.append(sample.edge_index[0].shape[0])

    graph_edge_max_size = max(edge_size_list)
    graph_edge_min_size = min(edge_size_list)
    graph_edge_mean_size = mean(edge_size_list)

    node_feature_num = data.num_features
    label_num = data.num_labels
    data_name = data.data_name

    data_information_dict = {"data_name": data_name,
                             "the graph num": the_graph_num,
                             "graph_max_size": graph_max_size,
                             "graph_min_size": graph_min_size,
                             "graph_mean_size": graph_mean_size,
                             "graph_edge_max_size": graph_edge_max_size,
                             "graph_edge_min_size": graph_edge_min_size,
                             "graph_edge_mean_size": graph_edge_mean_size,
                             "node_feature_num": node_feature_num}

    return data_information_dict

def cl_loss_function(x1, x2, T = 0.5):

    batch_size, _ = x1.size()
    x1_abs = x1.norm(dim=1)
    x2_abs = x2.norm(dim=1)

    sim_matrix_a = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
    sim_matrix_a = torch.exp(sim_matrix_a / T)
    pos_sim_a = sim_matrix_a[range(batch_size), range(batch_size)]
    loss_a = pos_sim_a / (sim_matrix_a.sum(dim=1) - pos_sim_a)
    loss_a = - torch.log(loss_a).mean()

    sim_matrix_b = torch.einsum('ik,jk->ij', x2, x1) / torch.einsum('i,j->ij', x2_abs, x1_abs)
    sim_matrix_b = torch.exp(sim_matrix_b / T)
    pos_sim_b = sim_matrix_b[range(batch_size), range(batch_size)]
    loss_b = pos_sim_b / (sim_matrix_b.sum(dim=1) - pos_sim_b)
    loss_b = - torch.log(loss_b).mean()

    loss = (loss_a + loss_b) / 2
    return loss
