import rasterio
import torch
import numpy as np
from torch_geometric.data import Data
from skimage.segmentation import slic
from skimage.graph import rag_mean_color
import os
import math
from multiprocessing import Pool
from functools import partial

def image_to_graph(filepath, target_pixels_per_segment):
    """
    Converts a multi-band GeoTIFF to a PyG Data object. Each node = one
    superpixel region. Edges connect adjacent regions. Node features = 
    mean value per band (all 10 Sentinel-2 bands) within the region.
    """
    
    filename = os.path.basename(filepath).split(".")[0]
    image_data_file = "data/processed/" + filename + ".pt"
    if os.path.isfile(image_data_file):
        return torch.load(image_data_file, weights_only=False)
    
    # Read all bands directly: shape (bands, height, width)
    with rasterio.open(filepath) as src:
        image_tensor = torch.from_numpy(src.read().astype(np.float32))
    
    channels, height, width = image_tensor.shape
    n_segments = (height * width) / target_pixels_per_segment
    
    # Segmentation needs 3 bands to segment off
    rgb_for_slic = image_tensor[[2, 1, 0]].permute(1, 2, 0).numpy()
    rgb_for_slic = (rgb_for_slic / rgb_for_slic.max() * 255).astype(np.uint8)

    segments = slic(rgb_for_slic, n_segments=n_segments, start_label=0)
    rag = rag_mean_color(rgb_for_slic, segments)
    
    # Full HWC array across all bands for computing per-segment means
    all_bands = image_tensor.permute(1, 2, 0).numpy()  # H, W, 10
    
    x = np.array([
        all_bands[segments == label].mean(axis=0)
        for label in rag.nodes
    ])
    x = torch.tensor(x, dtype=torch.float)
    
    label_to_idx = {label: i for i, label in enumerate(rag.nodes)}
    edges = [[label_to_idx[u], label_to_idx[v]] for u, v in rag.edges]
    edges += [[j, i] for i, j in edges]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    data = Data(x=x, edge_index=edge_index, num_nodes=len(rag.nodes))
    data.validate(raise_on_error=True)
    data.contiguous()
    torch.save(data, image_data_file)

    return data

def images_to_graph(filepaths, target_pixels_per_segment, n_workers):
    # Partial function for worker process to use
    worker_fn = partial(image_to_graph, target_pixels_per_segment=target_pixels_per_segment)
    
    with Pool(n_workers) as pool:
        result = pool.map(worker_fn, filepaths)
    
    return result

def encode_date(day_of_year, period=365.25):
    """
    Encodes date in 2 geometric value. This
    is so December 31st and January 1st are 
    only 1 day apart, not 364.
    """
    angle = 2 * math.pi * day_of_year / period
    return torch.tensor([math.sin(angle), math.cos(angle)], dtype=torch.float)