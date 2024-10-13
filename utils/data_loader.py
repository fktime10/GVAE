import numpy as np
import torch
import pandas as pd
from torch_geometric.data import Data

def load_citeseer_data(content_path="data/citeseer/citeseer.content", cites_path="data/citeseer/citeseer.cites"):
    """
    Load Citeseer data from content and citation files.
    
    :param content_path: Path to the citeseer.content file (node features and labels)
    :param cites_path: Path to the citeseer.cites file (citation edges)
    
    :return: A PyTorch Geometric Data object
    """
    
    # Load citeseer.content file: node features and labels
    # Treat the first column (node IDs) as strings to avoid dtype warnings
    content = pd.read_csv(content_path, sep='\t', header=None, dtype={0: str})
    
    # The first column is the node ID, the next columns are features, and the last column is the label
    node_ids = content[0].values  # Node IDs (strings)
    features = content.iloc[:, 1:-1].values  # Features (all columns except first and last)
    labels = pd.factorize(content.iloc[:, -1])[0]  # Convert labels to numeric (factorize)
    
    # Create a mapping from node ID to index for use in the adjacency matrix
    node_id_map = {node_id: i for i, node_id in enumerate(node_ids)}
    
    # Load citeseer.cites file: citation edges (directed graph)
    edges = pd.read_csv(cites_path, sep='\t', header=None, dtype={0: str, 1: str})
    
    # Create edge list: map node IDs to indices using node_id_map
    edge_index = []
    for row in edges.itertuples(index=False):
        if row[0] in node_id_map and row[1] in node_id_map:
            edge_index.append([node_id_map[row[0]], node_id_map[row[1]]])
    
    # Convert edge list to PyTorch tensor
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    
    # Convert features and labels to PyTorch tensors
    x = torch.tensor(features, dtype=torch.float)
    y = torch.tensor(labels, dtype=torch.long)
    
    # Create a PyTorch Geometric data object
    data = Data(x=x, edge_index=edge_index, y=y)
    
    return data

if __name__ == "__main__":
    # Example usage
    data = load_citeseer_data("data/citeseer/citeseer.content", "data/citeseer/citeseer.cites")
    print(data)
