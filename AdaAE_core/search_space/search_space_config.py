class SearchSpace(object):
    """
    Loading the search space dict
    """
    def __init__(self, gnn_layers=2):
        self.gnn_layers = int(gnn_layers)
        self.stack_gnn_architecture = ['augmentation', 'augmentation'] +\
                                      ['attention-aggregation-hidden_dimension', 'activation'] * self.gnn_layers +\
                                      ['pooling']
        self.space_dict = {
            'augmentation': ['attribute_masking', 'edge_perturbation', 'identical', 'node_dropping', 'subgraph'],
            'attention-aggregation-hidden_dimension': ['gcn-max-32', 'gcn-mean-32', 'gcn-add-32',
                                                       'gat-max-32', 'gat-mean-32', 'gat-add-32',
                                                       'graphsage-max-32', 'graphsage-mean-32', 'graphsage-add-32',
                                                       'gin-max-32', 'gin-mean-32', 'gin-add-32',
                                                       'gcn-max-64', 'gcn-mean-64', 'gcn-add-64',
                                                       'gat-max-64', 'gat-mean-64', 'gat-add-64',
                                                       'graphsage-max-64', 'graphsage-mean-64', 'graphsage-add-64',
                                                       'gin-max-64', 'gin-mean-64', 'gin-add-64',
                                                       'gcn-max-128', 'gcn-mean-128', 'gcn-add-128',
                                                       'gat-max-128', 'gat-mean-128', 'gat-add-128',
                                                       'graphsage-max-128', 'graphsage-mean-128', 'graphsage-add-128',
                                                       'gin-max-128', 'gin-mean-128', 'gin-add-128',
                                                       'gcn-max-256', 'gcn-mean-256', 'gcn-add-256',
                                                       'gat-max-256', 'gat-mean-256', 'gat-add-256',
                                                       'graphsage-max-256', 'graphsage-mean-256', 'graphsage-add-256',
                                                       'gin-max-256', 'gin-mean-256', 'gin-add-256'],
            'activation': ['elu', 'leaky_relu', 'linear', 'relu', 'relu6', 'sigmoid', 'softplus', 'tanh'],
            'pooling': ['global_max', 'global_mean', 'global_add']
        }
