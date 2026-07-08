from PIL import Image
import torchvision.transforms.v2 as transforms
import torch
import numpy as np
from torch_geometric.data import Data
from skimage.segmentation import slic
from skimage.graph import rag_mean_color
import os

def image_to_graph(filepath, target_pixels_per_segment):
    """
    Converts an image to a CHW image tensor, and then converts that CHW 
    image tensor into a PyG Data object. Each node = one superpixel region. 
    Edges connect adjacent regions. Node features = mean RGB color of the region.
    """
    
    # Determines if the image has already been processed
    filename = os.path.basename(filepath).split(".")[0]
    image_data_file = "data/processed/" + filename + ".pt"
    if os.path.isfile(image_data_file):
        return torch.load(image_data_file, weights_only=False)
    
    # Converts image to a CHW tensor
    img = Image.open(filepath)
    transform = transforms.ToImage()
    image_tensor = transform(img)
    
    # Determines number of superpixels
    channels, height, width = image_tensor.shape
    n_segments = (height * width)/target_pixels_per_segment
    
    rgb = image_tensor[:3].permute(1, 2, 0).numpy().astype(np.uint8)  # CHW -> HWC uint8
 
    segments = slic(rgb, n_segments=n_segments, start_label=0)
    rag = rag_mean_color(rgb, segments)
 
    x = np.array([rgb[segments == label].mean(axis=0) / 255.0 for label in rag.nodes])
    x = torch.tensor(x, dtype=torch.float)
 
    label_to_idx = {label: i for i, label in enumerate(rag.nodes)}
    edges = [[label_to_idx[u], label_to_idx[v]] for u, v in rag.edges]
    edges += [[j, i] for i, j in edges]  # make undirected
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
 
    # Avoids reprocessing by saving data object
    data = Data(x=x, edge_index=edge_index, num_nodes=len(rag.nodes))
    data.validate(raise_on_error=True)
    data.contiguous()
    torch.save(data, image_data_file)
    
    return data

def images_to_graph(filepaths, target_pixels_per_segment):
    return [image_to_graph(filepath, target_pixels_per_segment) for filepath in filepaths]