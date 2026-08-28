import torch
from torch_geometric.nn import global_mean_pool, GATv2Conv
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

hidden_channels = 64
out_channels = 14
heads = 4
epochs = 50

class GAT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, heads, date_dim=2):
        super(GAT, self).__init__()
        self.conv1 = GATv2Conv(-1, hidden_channels, heads=heads, concat=True, add_self_loops=True)
        self.conv2 = GATv2Conv(-1, hidden_channels, heads=heads, concat=True, add_self_loops=True)
        self.conv3 = GATv2Conv(-1, hidden_channels, heads=1, concat=False, add_self_loops=True)
        self.lin = torch.nn.Linear(hidden_channels + date_dim + 1, out_channels)

        self.dropout = torch.nn.Dropout(p=0.2)
    
    def forward(self, x, edge_index, batch, date_feat):
        
        if self.training:
            x.requires_grad_(True)
        
        def custom_forward(feat):
            x1 = F.elu(self.conv1(feat, edge_index))
            x1 = self.dropout(x1)
            x2 = F.elu(self.conv2(x1, edge_index))
            x2 = self.dropout(x2) + x1
            x3 = F.elu(self.conv3(x2, edge_index))
            return x3
        
        if self.training:
            x = checkpoint(custom_forward, x, use_reentrant=False)
        else:
            x = custom_forward(x)

        x = global_mean_pool(x, batch)
        num_nodes = torch.bincount(batch).float().log1p().unsqueeze(1).to(x.device)
        x = torch.cat([x, num_nodes, date_feat], dim=1)
        return self.lin(x)