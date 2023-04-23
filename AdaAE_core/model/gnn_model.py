import torch
from torch.nn import BatchNorm1d
from AdaAE_core.search_space.utils import aug_map, conv_map, act_map, pooling_map

class GnnModel(torch.nn.Module):
    """
     build gnn model based on gumbel softmax sample
        with architecture alpha parameter sample distribution.

    Args:
        architecture: list
            the stack gnn architecture describe
            for example: ['subgraph', 'identical', 'gcn-sum', 'tanh', 'gcn-sum', 'tanh', 'global_sum']
        original_feature_num: int
            the original input dimension for the stack gnn model

    Returns:
        output: tensor
            the output of the stack gnn model.
    """

    def __init__(self,
                 sample_architecture,
                 args,
                 original_feature_num,
                 withEdge=False):

        super(GnnModel, self).__init__()

        self.sample_architecture = sample_architecture
        self.args = args
        self.original_feature_num = original_feature_num
        self.withEdge = withEdge
        

        if self.withEdge == True:
            self.original_feature_num = 300
            self.num_atom_type = 120  # including the extra mask tokens
            self.num_chirality_tag = 3
            self.x_embedding1 = torch.nn.Embedding(self.num_atom_type, self.original_feature_num)
            self.x_embedding2 = torch.nn.Embedding(self.num_chirality_tag, self.original_feature_num)
            torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)
            torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        
        self.aug_method1 = aug_map(self.sample_architecture[0])
        self.aug_method2 = aug_map(self.sample_architecture[1])
    
        self.conv_layers = torch.nn.ModuleList()
        self.activation_operators = []
        self.normalization_layers = torch.nn.ModuleList()

        for layer in range(self.args.gnn_layers):
            if layer == 0:
                input_dim = self.original_feature_num
            else:
                input_dim = hidden_dimension

            convolution_type = self.sample_architecture[2 + layer * 2 + 0]
            attention_type, aggregator_type, hidden_dimension = convolution_type.split('-')
            hidden_dimension = int(hidden_dimension)
            conv = conv_map(attention_type, aggregator_type, input_dim, hidden_dimension, self.args.bias, self.withEdge)
            self.conv_layers.append(conv)

            activation_type = self.sample_architecture[2 + layer * 2 + 1]
            act = act_map(activation_type)
            self.activation_operators.append(act)

            graphnorm = BatchNorm1d(hidden_dimension)
            self.normalization_layers.append(graphnorm)

        pooling_type = self.sample_architecture[-1]
        self.global_pooling = pooling_map(pooling_type)


    ### the unsupervised learning and semisupervised learning tasks set the edge_attr as None
    ### the transfer learning task has the edge_attr
    def forward(self, data):
        aug1_data = self.aug_method1(data)
        aug2_data = self.aug_method2(data)

        x1 = aug1_data.x
        if self.withEdge == True:
            x1 = self.x_embedding1(x1[:, 0].long()) + self.x_embedding2(x1[:, 1].long())
        for layer in range(self.args.gnn_layers):
            # convolution for node embedding matrix
            if self.withEdge == False:
                x1 = self.conv_layers[layer](x1, aug1_data.edge_index)
            else:
                x1 = self.conv_layers[layer](x1, aug1_data.edge_index, aug1_data.edge_attr)
            # norm
            x1 = self.normalization_layers[layer](x1)
            # activation for node embedding matrix
            x1 = self.activation_operators[layer](x1)
            # dropout for node embedding matrix
            # x1 = F.dropout(x1, p=self.args.dropout, training=self.training)
        # last layer readout
        x1 = self.global_pooling(x1, aug1_data.batch)


        x2 = aug2_data.x
        if self.withEdge == True:
            x2 = self.x_embedding1(x2[:, 0].long()) + self.x_embedding2(x2[:, 1].long())
        for layer in range(self.args.gnn_layers):
            # convolution for node embedding matrix
            if self.withEdge == False:
                x2 = self.conv_layers[layer](x2, aug2_data.edge_index)
            else:
                x2 = self.conv_layers[layer](x2, aug2_data.edge_index, aug2_data.edge_attr)
            # norm
            x2 = self.normalization_layers[layer](x2)
            # activation for node embedding matrix
            x2 = self.activation_operators[layer](x2)
            # dropout for node embedding matrix
            # x2 = F.dropout(x2, p=self.args.dropout, training=self.training)
        # last layer readout
        x2 = self.global_pooling(x2, aug2_data.batch)

        return x1, x2

    def forward_gumbel(self, data, gumbel_softmax_sample_ret_list, sample_candidate_index_list):
        aug1_data = self.aug_method1(data)
        aug1_data.x = aug1_data.x * gumbel_softmax_sample_ret_list[0][0][sample_candidate_index_list[0]]

        aug2_data = self.aug_method2(data)
        aug2_data.x = aug2_data.x * gumbel_softmax_sample_ret_list[1][0][sample_candidate_index_list[1]]


        x1 = aug1_data.x
        if self.withEdge == True:
            x1 = self.x_embedding1(x1[:, 0].long()) + self.x_embedding2(x1[:, 1].long())
        for layer in range(self.args.gnn_layers):
            # convolution for node embedding matrix
            if self.withEdge == False:
                x1 = self.conv_layers[layer](x1, aug1_data.edge_index) * gumbel_softmax_sample_ret_list[2+layer*2+0][0][sample_candidate_index_list[2+layer*2+0]]
            else:
                x1 = self.conv_layers[layer](x1, aug1_data.edge_index, aug1_data.edge_attr) * gumbel_softmax_sample_ret_list[2+layer*2+0][0][sample_candidate_index_list[2+layer*2+0]]

            # norm
            x1 = self.normalization_layers[layer](x1)
            # activation for node embedding matrix
            x1 = self.activation_operators[layer](x1) * gumbel_softmax_sample_ret_list[2+layer*2+1][0][sample_candidate_index_list[2+layer*2+1]]
            # dropout for node embedding matrix
            # x1 = F.dropout(x1, p=self.args.dropout, training=self.training)
        # last layer readout
        x1 = self.global_pooling(x1, aug1_data.batch) * gumbel_softmax_sample_ret_list[-1][0][sample_candidate_index_list[-1]]


        x2 = aug2_data.x
        if self.withEdge == True:
            x2 = self.x_embedding1(x2[:, 0].long()) + self.x_embedding2(x2[:, 1].long())
        for layer in range(self.args.gnn_layers):
            # convolution for node embedding matrix
            if self.withEdge == False:
                x2 = self.conv_layers[layer](x2, aug2_data.edge_index) * gumbel_softmax_sample_ret_list[2+layer*2+0][0][sample_candidate_index_list[2+layer*2+0]]
            else:
                x2 = self.conv_layers[layer](x2, aug2_data.edge_index, aug2_data.edge_attr) * gumbel_softmax_sample_ret_list[2+layer*2+0][0][sample_candidate_index_list[2+layer*2+0]]

            # norm
            x2 = self.normalization_layers[layer](x2)
            # activation for node embedding matrix
            x2 = self.activation_operators[layer](x2) * gumbel_softmax_sample_ret_list[2+layer*2+1][0][sample_candidate_index_list[2+layer*2+1]]
            # dropout for node embedding matrix
            # x2 = F.dropout(x2, p=self.args.dropout, training=self.training)
        # last layer readout
        x2 = self.global_pooling(x2, aug2_data.batch) * gumbel_softmax_sample_ret_list[-1][0][sample_candidate_index_list[-1]]

        return x1, x2

    def forward_encoder(self, data):
        x = data.x
        if self.withEdge == True:
            x = self.x_embedding1(x[:, 0].long()) + self.x_embedding2(x[:, 1].long())

        for layer in range(self.args.gnn_layers):
            # convolution for node embedding matrix
            if self.withEdge == False:
                x = self.conv_layers[layer](x, data.edge_index)
            else:
                x = self.conv_layers[layer](x, data.edge_index, data.edge_attr)
            # norm
            x = self.normalization_layers[layer](x)
            # activation for node embedding matrix
            x = self.activation_operators[layer](x)
            # dropout for node embedding matrix
            # x = F.dropout(x, p=self.args.dropout, training=self.training)
        # last layer readout
        x = self.global_pooling(x, data.batch)

        return x
