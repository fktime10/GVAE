import torch
import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support
from model.gala import GALA
from utils.data_loader import load_citeseer_data
import yaml

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log_file = "inference.log"
if os.path.exists(log_file):
    os.remove(log_file)
logging.getLogger().addHandler(logging.FileHandler(log_file))

# Load configuration from YAML (for consistency)
with open("config/config.yaml", "r") as file:
    config = yaml.safe_load(file)

def infer():
    # Load the checkpoint
    checkpoint_path = "checkpoints/best_model.pt"
    
    if not os.path.exists(checkpoint_path):
        logging.error(f"Checkpoint file '{checkpoint_path}' not found!")
        return

    # Load the checkpoint to get the model architecture
    try:
        checkpoint = torch.load(checkpoint_path)
        logging.info("Checkpoint loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading checkpoint: {str(e)}")
        return

    # Extract model configuration from YAML or checkpoint
    encoder_layer_sizes = config['model']['encoder_layers']
    hidden_channels = config['model']['hidden_channels']
    in_channels = config['dataset']['num_features']

    # Load the trained model with dynamic architecture
    model = GALA(in_channels=in_channels, hidden_channels=hidden_channels, encoder_layers=encoder_layer_sizes)
    
    try:
        model.load_state_dict(checkpoint)
        model.eval()
        logging.info("Model loaded successfully with dynamic architecture.")
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        return

    # Load the dataset
    try:
        data = load_citeseer_data(config['dataset']['content_path'], config['dataset']['cites_path'])
        x, edge_index, labels = data.x, data.edge_index, data.y
        logging.info("Dataset loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading dataset: {str(e)}")
        return

    # Inference: Get the latent representation
    with torch.no_grad():
        _, latent_rep = model(x, edge_index)
        logging.info("Inference completed successfully.")

    # t-SNE visualization
    try:
        visualize_tsne(latent_rep, labels)
    except Exception as e:
        logging.error(f"Error during t-SNE visualization: {str(e)}")

    # Generate false edges and plot graphs
    true_edges = edge_index.T.tolist()  # True edges from the edge index
    false_edges = generate_false_edges(edge_index, latent_rep.size(0), num_false_edges=1000)
    
    # Test thresholds between 0.05 and 0.08
    threshold_range = np.arange(0.05, 0.09, 0.01)
    best_f1 = -1
    best_threshold = 0

    for threshold in threshold_range:
        auc, ap, precision, recall, f1 = evaluate_link_prediction(latent_rep, true_edges, false_edges, threshold)
        logging.info(f"Threshold: {threshold:.2f} - AUC: {auc:.4f}, AP: {ap:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    logging.info(f"Best Threshold: {best_threshold:.2f}, Best F1-Score: {best_f1:.4f}")
    
    # Visualize original and predicted graphs using the best threshold
    try:
        visualize_graph_comparison(true_edges, latent_rep, best_threshold)
    except Exception as e:
        logging.error(f"Error during graph visualization: {str(e)}")

def generate_false_edges(edge_index, num_nodes, num_false_edges=1000):
    all_possible_edges = set((i, j) for i in range(num_nodes) for j in range(i+1, num_nodes))
    true_edges = set(tuple(edge) for edge in edge_index.T.tolist())
    false_edges = list(all_possible_edges - true_edges)
    return random.sample(false_edges, num_false_edges)

def evaluate_link_prediction(latent_rep, true_edges, false_edges, threshold=0.05):
    reconstructed_adj = torch.mm(latent_rep, latent_rep.t())
    visualize_adjacency_matrix(reconstructed_adj)

    true_edge_scores = [reconstructed_adj[edge[0], edge[1]].item() for edge in true_edges]
    false_edge_scores = [reconstructed_adj[edge[0], edge[1]].item() for edge in false_edges]

    edge_scores = true_edge_scores + false_edge_scores
    edge_labels = [1] * len(true_edges) + [0] * len(false_edges)

    # Apply threshold to calculate predicted edges
    predicted_labels = [1 if score > threshold else 0 for score in edge_scores]

    auc = roc_auc_score(edge_labels, edge_scores)
    ap = average_precision_score(edge_labels, edge_scores)
    precision, recall, f1, _ = precision_recall_fscore_support(edge_labels, predicted_labels, average='binary')

    # Plot the score distribution
    plot_score_distribution(true_edge_scores, false_edge_scores)

    return auc, ap, precision, recall, f1

def plot_score_distribution(true_edge_scores, false_edge_scores):
    plt.figure(figsize=(10, 6))
    plt.hist(true_edge_scores, bins=50, alpha=0.5, label="True Edges")
    plt.hist(false_edge_scores, bins=50, alpha=0.5, label="False Edges")
    plt.title("Edge Score Distribution")
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

def visualize_adjacency_matrix(reconstructed_adj, output_path="plots/adj_matrix.png"):
    plt.figure(figsize=(8, 6))
    plt.imshow(reconstructed_adj.cpu().detach().numpy(), cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title("Reconstructed Adjacency Matrix")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.show()

def visualize_tsne(latent_rep, labels, output_path="plots/tsne_plot.png"):
    latent_np = latent_rep.cpu().numpy()
    tsne = TSNE(n_components=2, random_state=42)
    tsne_result = tsne.fit_transform(latent_np)
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=labels.cpu().numpy(), cmap='viridis', alpha=0.7)
    plt.colorbar(scatter)
    plt.title("t-SNE Visualization of Latent Space")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.show()

def visualize_graph_comparison(true_edges, latent_rep, threshold, output_path="plots/graph_comparison.png"):
    G_true = nx.Graph()
    G_true.add_edges_from(true_edges)

    reconstructed_adj = torch.mm(latent_rep, latent_rep.t()).cpu().detach().numpy()

    # Use dynamic threshold for predicted edges
    predicted_edges = [(i, j) for i in range(reconstructed_adj.shape[0]) for j in range(i+1, reconstructed_adj.shape[1]) if reconstructed_adj[i, j] > threshold]

    G_predicted = nx.Graph()
    G_predicted.add_edges_from(predicted_edges)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    nx.draw(G_true, node_size=30, node_color='b', edge_color='gray', with_labels=False)
    plt.title("Original Graph (True Edges)")

    plt.subplot(1, 2, 2)
    nx.draw(G_predicted, node_size=30, node_color='r', edge_color='gray', with_labels=False)
    plt.title(f"Predicted Graph (Threshold: {threshold:.2f})")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.show()

if __name__ == "__main__":
    infer()
