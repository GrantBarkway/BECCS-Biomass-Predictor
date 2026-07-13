import torch
from torch_geometric.nn import GATv2Conv, global_mean_pool
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

hidden_channels = 64
heads = 4
out_channels = 14
epochs = 200

class GAT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, heads, date_dim=2):
        super(GAT, self).__init__()
        self.conv1 = GATv2Conv(-1, hidden_channels, heads=heads, concat=True, add_self_loops=False)
        self.conv2 = GATv2Conv(-1, hidden_channels, heads=heads, concat=True, add_self_loops=False)
        self.conv3 = GATv2Conv(-1, hidden_channels, heads=1, concat=False, add_self_loops=False)
        self.lin = torch.nn.Linear(hidden_channels + date_dim, out_channels)
        
        self.dropout = torch.nn.Dropout(p=0.2)
    
    def forward(self, x, edge_index, batch, date_feat):
        
        if self.training:
            x.requires_grad_(True)
        
        def custom_forward(feat):
            x = F.elu(self.conv1(feat, edge_index))
            x = self.dropout(x)
            x = F.elu(self.conv2(x, edge_index))
            x = self.dropout(x)
            x = F.elu(self.conv3(x, edge_index))
            return x
        
        if self.training:
            x = checkpoint(custom_forward, x, use_reentrant=False)
        else:
            x = custom_forward(x)
        
        x = global_mean_pool(x, batch)
        x = torch.cat([x, date_feat], dim=1)  # merge in date info before final layer
        return self.lin(x)