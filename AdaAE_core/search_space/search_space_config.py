class SearchSpace(object):
    """
    Loading the search space dict
    """
    def __init__(self, gnn_layers=2):
        self.gnn_layers = int(gnn_layers)
        self.stack_gnn_architecture = ['augmentation', 'augmentation'] +\
                                      ['attention-aggregation-hidden_dimension', 'activation'] * self.gnn_layers +\
                                      ['pooling']

        self.attention = ['gcn', 'gat', 'graphsage', 'gin']
        self.aggregation = ['max', 'mean', 'add']
        self.hidden_dimension = ['32', '64', '128', '256']
        self.space_dict = {
            'augmentation': ['attribute_masking', 'edge_perturbation', 'identical', 'node_dropping', 'subgraph'],
            'attention-aggregation-hidden_dimension': ['-'.join([i,j,k]) for k in self.hidden_dimension for j in self.aggregation for i in self.attention],
            'activation': ['elu', 'leaky_relu', 'linear', 'relu', 'relu6', 'sigmoid', 'softplus', 'tanh'],
            'pooling': ['global_max', 'global_mean', 'global_add']
        }
