import torch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, global_mean_pool
import torch.nn.functional as F
from processing import images_to_graph, encode_date
from data.data import data

hidden_channels_1 = 64
heads = 4
out_channels = 14

class GAT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, heads, date_dim=2):
        super(GAT, self).__init__()
        self.conv1 = GATv2Conv(-1, hidden_channels, heads=heads, concat=True, add_self_loops=False)
        self.conv2 = GATv2Conv(-1, hidden_channels, heads=1, concat=False, add_self_loops=False)
        self.lin = torch.nn.Linear(hidden_channels + date_dim, out_channels)
        
        self.dropout = torch.nn.Dropout(p=0.2)
    
    def forward(self, x, edge_index, batch, date_feat):
        x = F.elu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.elu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        x = torch.cat([x, date_feat], dim=1)  # merge in date info before final layer
        return self.lin(x)

def train(graph_list, graph_information, epochs, batch_size=32):
    """
    data_list: list of torch_geometric.data.Data objects (one per image/graph)
    targets:   list or tensor of target values, one per graph
    """
    model = GAT(hidden_channels_1, out_channels, heads)

    optimiser = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser,
        mode='min',
        factor=0.5,
        patience=50,
        min_lr=1e-5
    )
    criterion = torch.nn.MSELoss()
    
    # attach targets to each Data object so they travel with the batch correctly
    targets = torch.as_tensor([g['targets'] for g in graph_information], dtype=torch.float)
    dates = torch.as_tensor([g['day'] for g in graph_information], dtype=torch.float)
    for d, y, date in zip(graph_list, targets, dates):
        d.y = y.view(1, -1) if y.dim() == 0 else y.unsqueeze(0)
        d.date_feat = encode_date(date).unsqueeze(0)
    
    loader = DataLoader(graph_list, batch_size=batch_size, shuffle=True)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in loader:
            optimiser.zero_grad()
            output = model(batch.x, batch.edge_index, batch.batch, batch.date_feat)
            loss = criterion(output, batch.y)
            loss.backward()
            optimiser.step()
            total_loss += loss.item() * batch.num_graphs
        
        avg_loss = total_loss / len(graph_list)
        scheduler.step(avg_loss)
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {avg_loss:.4f}')
    
    return model

graph_data = images_to_graph(data.keys(), 1000)
graph_information = list(data.values())

train(graph_data, graph_information, 500)