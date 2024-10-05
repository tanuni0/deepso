import torch
from torch import nn
import torch.nn.functional as F

import torch_geometric.nn as gnn
from torch_geometric.data import Data


class PSOGNN(nn.Module):
    def __init__(self, node_input_dim, hidden_dim=32):
        super(PSOGNN, self).__init__()
        #GNN
        self.conv1 = gnn.GCNConv(node_input_dim, hidden_dim)
        self.conv2 = gnn.GCNConv(hidden_dim, hidden_dim)
        
        self.fc = nn.Linear(hidden_dim, 3)  # 3 outputs for w, c1, c2


    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        # x = gnn.global_mean_pool(x, data.batch)  # Pooling cho đầu ra

        
        return torch.sigmoid(self.fc(x))  # Sử dụng sigmoid để giới hạn các giá trị trong khoảng [0, 1]