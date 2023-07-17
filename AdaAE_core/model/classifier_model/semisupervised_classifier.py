import torch.nn.functional as F
import torch.nn as nn
from torch.nn import BatchNorm1d, Linear

class SemisupervisedClassifier(nn.Module):
    """
    The custom downstream task class,
    using the mlp to realize the inductive graph
    classification based on node embedding from
    stack gcn model

    Args:
        gnn_embedding_dim: int
            the input node embedding dimension
        output_dim: int
            the number of classes of graph dataset
        node_embedding_matrix: tensor
            the output node embedding matrix of stack gcn model

    Returns:
        x: tensor
            the output tensor of predicting
    """

    def __init__(self,
                 gnn_embedding_dim,
                 output_dim,
                 mlp_layers=2):
        super(SemisupervisedClassifier, self).__init__()
        self.bns_fc = nn.ModuleList()
        self.lins = nn.ModuleList()
        for i in range(mlp_layers):
            self.bns_fc.append(BatchNorm1d(gnn_embedding_dim))
            self.lins.append(Linear(gnn_embedding_dim, gnn_embedding_dim))
        self.bn_hidden = BatchNorm1d(gnn_embedding_dim)
        self.lin_class = Linear(gnn_embedding_dim, output_dim)

    def forward(self, x):
        for i, lin in enumerate(self.lins):
            x = self.bns_fc[i](x)
            x = lin(x)
            x = F.relu(x)

        x = self.bn_hidden(x)
        x = self.lin_class(x)
        return F.log_softmax(x, dim=-1)
