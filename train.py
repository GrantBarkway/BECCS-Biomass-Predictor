import torch
from torch_geometric.loader import DataLoader
from processing import images_to_graph, encode_date
from data.data import data
from torch.amp import autocast
from model import GAT, hidden_channels, out_channels, heads, epochs
import os

# Stops fragmentation of memory, freeing up allocated but unused memory
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def weighted_mse(output, target, zero_weight=1.0, nonzero_weight=15.0):
    weights = torch.where(target > 0, nonzero_weight, zero_weight)
    return (weights * (output - target) ** 2).mean()

def train(graph_list, graph_information, epochs, feature_mean, feature_std, accumulation_steps=16):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    
    model = GAT(hidden_channels, out_channels, heads).to(device)

    optimiser = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode='min', factor=0.5, patience=10, min_lr=1e-5
    )
    criterion = weighted_mse
    
    targets = torch.as_tensor([g['targets'] for g in graph_information], dtype=torch.float)
    targets_transformed = torch.log1p(targets)
    dates = torch.as_tensor([g['day'] for g in graph_information], dtype=torch.float)
    for d, y, date in zip(graph_list, targets_transformed, dates):
        d.y = y.view(1, -1) if y.dim() == 0 else y.unsqueeze(0)
        d.date_feat = encode_date(date).view(1, -1)
    
    loader = DataLoader(
        graph_list,
        batch_size=4,
        shuffle=True,
        pin_memory=True
    )
    
    model.train()
    best_loss = float('inf')
    for epoch in range(epochs):
        total_loss = 0.0
        optimiser.zero_grad()

        for step, batch in enumerate(loader):
            batch = batch.to(device, non_blocking=True)
            assert not torch.isnan(batch.x).any(), "NaN in node features"
            assert not torch.isnan(batch.y).any(), "NaN in targets"
            assert not torch.isnan(batch.date_feat).any(), "NaN in date_feat"

            with autocast(device_type='cuda', dtype=torch.bfloat16):
                output = model(batch.x, batch.edge_index, batch.batch, batch.date_feat)
                loss = criterion(output, batch.y) / accumulation_steps

            loss.backward()
            total_loss += loss.item() * accumulation_steps * batch.num_graphs

            if (step + 1) % accumulation_steps == 0 or (step + 1) == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                optimiser.step()
                optimiser.zero_grad()

        avg_loss = total_loss / len(graph_list)
        scheduler.step(avg_loss)
        print(f'Epoch {epoch}, Loss: {avg_loss:.4f}')

        epoch_info = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimiser_state_dict': optimiser.state_dict(),
            'loss': avg_loss,
            'feature_mean': feature_mean,
            'feature_std': feature_std,
        }
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(epoch_info, 'data/checkpoints/checkpoint_best.pt')
        if epoch % 10 == 0 or epoch == epochs - 1:
            torch.save(epoch_info, f'data/checkpoints/checkpoint_epoch{epoch}.pt')

    return model

if __name__ == "__main__":
    
    print("torch version : ", torch.__version__)
    print("cuda available? : ", torch.cuda.is_available())
    print("cuda version: ", torch.version.cuda)
    print("cuda device count: ", torch.cuda.device_count())
    device_ids = list(range(torch.cuda.device_count()))
    print("cuda device id: ", *device_ids, sep=", ")

    graph_data, feature_mean, feature_std = images_to_graph(data.keys(), 300, 10)
    print("Graph data length: ", len(graph_data))
    graph_information = list(data.values())
    print("Graph information length: ", len(graph_information))
    
    train(graph_data, graph_information, epochs, feature_mean, feature_std, 4)
    
    print(f"gpu used {torch.cuda.max_memory_allocated(device=None)} memory")