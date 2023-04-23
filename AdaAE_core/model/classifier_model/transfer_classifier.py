import torch.nn as nn
from torch.nn import Linear

class TransferClassifier(nn.Module):

    def __init__(self,
                 gnn_embedding_dim,
                 num_tasks):
        super(TransferClassifier, self).__init__()

        self.graph_pred_linear = Linear(gnn_embedding_dim, num_tasks)

    def forward(self, x):
        return self.graph_pred_linear(x)
