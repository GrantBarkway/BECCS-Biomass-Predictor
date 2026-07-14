import torch
from torch_geometric.loader import DataLoader
from processing import images_to_graph, encode_date
from data.data import data
from torch.amp import autocast
from model import GIN, hidden_channels, out_channels, epochs
import os

# Stops fragmentation of memory, freeing up allocated but unused memory
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

def train(graph_list, graph_information, epochs, batch_size=32):
    """
    data_list: list of torch_geometric.data.Data objects (one per image/graph)
    targets:   list or tensor of target values, one per graph
    """
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    
    model = GIN(hidden_channels, out_channels).to(device)

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
    
    # Transform targets for more efficient training use torch.expm1 to reverse when evaluating/running
    targets_transformed = torch.log1p(targets)
    
    dates = torch.as_tensor([g['day'] for g in graph_information], dtype=torch.float)
    for d, y, date in zip(graph_list, targets_transformed, dates):
        d.y = y.view(1, -1) if y.dim() == 0 else y.unsqueeze(0)
        d.date_feat = encode_date(date).view(1,-1)
    
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
            batch = batch.to(device, non_blocking=True)
            assert not torch.isnan(batch.x).any(), "NaN in node features"
            assert not torch.isnan(batch.y).any(), "NaN in targets"
            assert not torch.isnan(batch.date_feat).any(), "NaN in date_feat"
            optimiser.zero_grad()
            with autocast(device_type='cuda', dtype=torch.bfloat16):
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

if __name__ == "__main__":

    print("torch version : ", torch.__version__)
    print("cuda available? : ", torch.cuda.is_available())
    print("cuda version: ", torch.version.cuda)
    print("cuda device count: ", torch.cuda.device_count())
    device_ids = list(range(torch.cuda.device_count()))
    print("cuda device id: ", *device_ids, sep=", ")

    graph_data = images_to_graph(data.keys(), 50, 16)
    print("Graph data length: ", len(graph_data))
    graph_information = list(data.values())
    print("Graph information length: ", len(graph_information))

    train(graph_data, graph_information, epochs, 1)
    
    print(f"gpu used {torch.cuda.max_memory_allocated(device=None)} memory")