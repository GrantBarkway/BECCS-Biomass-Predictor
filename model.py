import torch
from torch_geometric.nn import GINConv, global_add_pool
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

hidden_channels = 128
out_channels = 14
epochs = 50

def make_gin_mlp(in_dim, hidden_dim):
    return torch.nn.Sequential(
        torch.nn.Linear(in_dim, hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, hidden_dim),
    )

class GIN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, in_channels=10, date_dim=2):
        super(GIN, self).__init__()
        self.conv1 = GINConv(make_gin_mlp(in_channels, hidden_channels), train_eps=True)
        self.conv2 = GINConv(make_gin_mlp(hidden_channels, hidden_channels // 2), train_eps=True)
        self.conv3 = GINConv(make_gin_mlp(hidden_channels // 2, hidden_channels // 4), train_eps=True)
        self.conv4 = GINConv(make_gin_mlp(hidden_channels // 4, hidden_channels // 8), train_eps=True)
        self.lin = torch.nn.Linear(hidden_channels // 8 + date_dim, out_channels)
        
        self.dropout = torch.nn.Dropout(p=0.2)
    
    def forward(self, x, edge_index, batch, date_feat):

        def custom_forward(feat):
            h = F.elu(self.conv1(feat, edge_index))
            h = self.dropout(h)
            h = F.elu(self.conv2(h, edge_index))
            h = self.dropout(h)
            h = F.elu(self.conv3(h, edge_index))
            h = self.dropout(h)
            h = F.elu(self.conv4(h, edge_index))
            return h
        
        if self.training:
            x = x.clone().requires_grad_(True)
            x = checkpoint(custom_forward, x, use_reentrant=False)
        else:
            x = custom_forward(x)

        x = global_add_pool(x, batch)
        x = torch.cat([x, date_feat], dim=1)
        return self.lin(x)