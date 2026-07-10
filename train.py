import torch
from torch_geometric.loader import DataLoader
from processing import images_to_graph, encode_date
from data.data import data
from model import GAT, hidden_channels, out_channels, heads, epochs

def train(graph_list, graph_information, epochs, batch_size=32):
    """
    data_list: list of torch_geometric.data.Data objects (one per image/graph)
    targets:   list or tensor of target values, one per graph
    """
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    model = GAT(hidden_channels, out_channels, heads).to(device)

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
    
    loader = DataLoader(
        graph_list,
        batch_size=batch_size, # Fine now, check GPU memory when dataset gets bigger
        shuffle=True,
        pin_memory=True # Faster data transfer from CPU to GPU
    )
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in loader:
            batch.to(device, non_blocking=True)
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
        
        if epoch % 50 == 0 or epoch == epochs - 1:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimiser_state_dict': optimiser.state_dict(),
                'loss': avg_loss,
            }, f'data/checkpoints/checkpoint_epoch{epoch}.pt')
    
    return model

graph_data = images_to_graph(data.keys(), 1000, 10)
graph_information = list(data.values())

train(graph_data, graph_information, epochs)