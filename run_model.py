from model import GIN, hidden_channels, out_channels

def run():
    model = GIN(hidden_channels, out_channels)
    