import torch
from torch_geometric.nn import GINConv, global_mean_pool
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

hidden_channels = 256
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
        self.conv2 = GINConv(make_gin_mlp(hidden_channels, hidden_channels), train_eps=True)
        self.lin = torch.nn.Linear(hidden_channels + 1 + date_dim, out_channels)

        self.dropout = torch.nn.Dropout(p=0.2)

    def forward(self, x, edge_index, batch, date_feat):

        def custom_forward(feat):
            h = F.elu(self.conv1(feat, edge_index))
            h = self.dropout(h)
            h = h + F.elu(self.conv2(h, edge_index))
            return h

        if self.training:
            x = x.clone().requires_grad_(True)
            x = checkpoint(custom_forward, x, use_reentrant=False)
        else:
            x = custom_forward(x)

        x = global_mean_pool(x, batch)
        num_nodes = torch.bincount(batch).float().log1p().unsqueeze(1).to(x.device)
        x = torch.cat([x, num_nodes, date_feat], dim=1)
        return self.lin(x)