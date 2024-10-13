import torch
import torch.optim as optim
import torch.nn.functional as F
import yaml
import logging
import os
import time
import random
from tqdm import tqdm
from model.gala import GALA
from utils.data_loader import load_citeseer_data
from torch_geometric.utils import to_dense_adj
from sklearn.metrics import roc_auc_score, average_precision_score

# Define directories for logs and checkpoints
log_dir = "logs"
checkpoint_dir = "checkpoints"
os.makedirs(log_dir, exist_ok=True)          # Create the logs directory if it doesn't exist
os.makedirs(checkpoint_dir, exist_ok=True)   # Create the checkpoints directory if it doesn't exist

# Generate a unique timestamp for the log file
timestamp = time.strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_dir, f"training_{timestamp}.log")

# Set up logging configuration (both console and file)
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(log_file),        # Log to file with timestamped filename
                        logging.StreamHandler()              # Log to console
                    ])

# Load configuration from YAML
with open("config/config.yaml", "r") as file:
    config = yaml.safe_load(file)

def train():
    # Load model parameters from config
    encoder_layers = config['model']['encoder_layers']
    hidden_channels = config['model']['hidden_channels']
 
    # Initialize GALA model with dynamic config parameters
    model = GALA(in_channels=config['dataset']['num_features'], 
                 hidden_channels=hidden_channels, 
                 encoder_layers=encoder_layers)

    # Initialize optimizer
    train_lr = float(config['training']['train_lr'])
    optimizer = optim.Adam(model.parameters(), 
                           lr=train_lr, 
                           weight_decay=config['training']['weight_decay'])

    # Load dataset
    data = load_citeseer_data(config['dataset']['content_path'], 
                              config['dataset']['cites_path'])
    x, edge_index, labels = data.x, data.edge_index, data.y

    # Initialize tqdm progress bar
    epochs = config['training']['epochs']
    progress_bar = tqdm(range(epochs), desc="Training Epochs", unit="epoch")

    # Variables to track best validation accuracy
    best_auc = 0

    for epoch in progress_bar:
        model.train()
        optimizer.zero_grad()

        # Forward pass
        x_hat, z = model(x, edge_index)

        # Loss calculation
        reconstruction_loss = F.mse_loss(x_hat, x)

        # Clustering loss
        adj = to_dense_adj(edge_index).squeeze()
        affinity_matrix = torch.mm(z, z.t())
        clustering_loss = F.mse_loss(affinity_matrix, adj)

        # Total loss
        total_loss = reconstruction_loss + config['training']['clustering_loss_weight'] * clustering_loss
        total_loss.backward()
        optimizer.step()

        # Update tqdm with loss information
        progress_bar.set_postfix({'Loss': total_loss.item()})

        # Log epoch and loss
        if epoch % config['training']['val_interval'] == 0:
            logging.info(f"Epoch {epoch}: Loss = {total_loss.item()}")

            # Validate the model and save if validation improves
            auc, ap = validate(model, data)
            if auc > best_auc:
                best_auc = auc
                # Save the model checkpoint if AUC improves
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best_model.pt'))
                logging.info(f"Saved new best model with AUC: {best_auc:.4f}")


def validate(model, data):
    model.eval()
    x, edge_index, labels = data.x, data.edge_index, data.y

    with torch.no_grad():
        _, z = model(x, edge_index)

    # Evaluate both true and false edges
    auc, ap = evaluate_with_true_and_false_edges(z, edge_index)

    # Log validation results
    logging.info(f"Validation (True & False Edges) - AUC: {auc:.4f}, AP: {ap:.4f}")
    
    return auc, ap

def evaluate_with_true_and_false_edges(latent_rep, edge_index, num_false_edges=1000):
    # Reconstruct the adjacency matrix from latent embeddings
    reconstructed_adj = torch.mm(latent_rep, latent_rep.t())

    # Get true edges from the edge_index
    true_edges = edge_index.T.tolist()

    # Generate false edges (random pairs of nodes that don't exist in the graph)
    false_edges = generate_false_edges(edge_index, latent_rep.size(0), num_false_edges)

    # Calculate scores for both true and false edges
    true_edge_scores = [reconstructed_adj[edge[0], edge[1]].item() for edge in true_edges]
    false_edge_scores = [reconstructed_adj[edge[0], edge[1]].item() for edge in false_edges]

    # Combine the scores and labels
    edge_scores = true_edge_scores + false_edge_scores
    edge_labels = [1] * len(true_edges) + [0] * len(false_edges)

    # Calculate ROC AUC and Average Precision (AP)
    auc = roc_auc_score(edge_labels, edge_scores)
    ap = average_precision_score(edge_labels, edge_scores)

    return auc, ap

def generate_false_edges(edge_index, num_nodes, num_false_edges=1000):
    # Generate a set of all possible edges in the graph
    all_possible_edges = set((i, j) for i in range(num_nodes) for j in range(i+1, num_nodes))

    # Remove the true edges (existing edges) from all possible edges
    true_edges = set(tuple(edge) for edge in edge_index.T.tolist())
    false_edges = list(all_possible_edges - true_edges)

    # Randomly sample the false edges
    return random.sample(false_edges, num_false_edges)

if __name__ == "__main__":
    train()
