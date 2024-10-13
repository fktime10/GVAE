import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

def plot_training_metrics(log_file, output_path='plots/'):
    """
    Plots loss, accuracy, NMI, and ARI from the training log.
    
    :param log_file: str, path to the log file containing training metrics
    :param output_path: str, folder where the plots will be saved
    """
    # Load training logs (assumes the log file contains lines like "Epoch: {epoch}, Loss: {loss}, ACC: {acc}, NMI: {nmi}, ARI: {ari}")
    epochs, loss_vals, acc_vals, nmi_vals, ari_vals = [], [], [], [], []
    
    with open(log_file, 'r') as f:
        for line in f:
            if 'Epoch' in line:
                parts = line.split(',')
                epoch = int(parts[0].split(':')[1].strip())
                loss = float(parts[1].split(':')[1].strip())
                acc = float(parts[2].split(':')[1].strip())
                nmi = float(parts[3].split(':')[1].strip())
                ari = float(parts[4].split(':')[1].strip())
                
                epochs.append(epoch)
                loss_vals.append(loss)
                acc_vals.append(acc)
                nmi_vals.append(nmi)
                ari_vals.append(ari)
    
    # Plot loss curve
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, loss_vals, label="Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.savefig(output_path + 'loss_curve.png')
    plt.close()

    # Plot Accuracy, NMI, ARI curves
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, acc_vals, label="Accuracy")
    plt.plot(epochs, nmi_vals, label="NMI")
    plt.plot(epochs, ari_vals, label="ARI")
    plt.xlabel('Epoch')
    plt.ylabel('Metrics')
    plt.title('Clustering Metrics During Training')
    plt.legend()
    plt.savefig(output_path + 'clustering_metrics.png')
    plt.close()

def plot_tsne(latent_embeddings, labels, output_path='plots/tsne.png'):
    """
    Plots t-SNE visualization of the latent embeddings.
    
    :param latent_embeddings: np.array, latent space representations (shape: [num_nodes, latent_dim])
    :param labels: np.array, ground-truth labels for the nodes
    :param output_path: str, path to save the t-SNE plot
    """
    tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
    tsne_results = tsne.fit_transform(latent_embeddings)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap='Spectral', s=10)
    plt.colorbar(scatter)
    plt.title('t-SNE Visualization of Latent Space')
    plt.savefig(output_path)
    plt.close()

def plot_inference_results(predicted_edges, actual_edges, output_path='plots/inference.png'):
    """
    Plots a visualization for the inference results (e.g., link prediction).
    
    :param predicted_edges: list of tuples, predicted edges (node pairs)
    :param actual_edges: list of tuples, actual edges (node pairs)
    :param output_path: str, path to save the plot
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Convert edges to arrays for easier plotting
    predicted_x = [edge[0] for edge in predicted_edges]
    predicted_y = [edge[1] for edge in predicted_edges]
    
    actual_x = [edge[0] for edge in actual_edges]
    actual_y = [edge[1] for edge in actual_edges]

    # Plot predicted edges in one color, actual edges in another
    ax.scatter(predicted_x, predicted_y, c='blue', label="Predicted Edges", s=10, alpha=0.6)
    ax.scatter(actual_x, actual_y, c='red', label="Actual Edges", s=10, alpha=0.6)

    ax.set_title('Link Prediction Results')
    ax.set_xlabel('Node Index (Source)')
    ax.set_ylabel('Node Index (Target)')
    ax.legend()
    plt.savefig(output_path)
    plt.close()
