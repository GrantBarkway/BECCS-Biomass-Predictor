from model import GAT, hidden_channels, out_channels, heads

def run():
    model = GAT(hidden_channels, out_channels, heads)
    