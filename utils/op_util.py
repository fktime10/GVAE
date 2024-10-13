import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

def Optimizer(model, learning_rates):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rates[0], weight_decay=1e-5)  # Add weight decay

    finetune_optimizer = torch.optim.Adam(model.parameters(), lr=learning_rates[1])
    
    train_loss = 0
    ACC, NMI, ARI = [], [], []

    def train_step(input_data, edge_index, weight_decay=1e-5):
        model.train()
        optimizer.zero_grad()
        reconstructed_x, z = model(input_data, edge_index)
        loss = F.mse_loss(reconstructed_x, input_data) + weight_decay * sum(torch.norm(param) for param in model.parameters())
        loss.backward()
        optimizer.step()
        return loss.item()

    def finetune(input_data, edge_index, labels, k=20):
        model.train()
        finetune_optimizer.zero_grad()
        reconstructed_x, z = model(input_data, edge_index)
        loss = F.mse_loss(reconstructed_x, input_data)

        # Perform Spectral Clustering on latent space z
        latent = z.detach().cpu().numpy()
        spectral = SpectralClustering(n_clusters=k, affinity='precomputed')
        spectral.fit(latent)

        # Calculate clustering metrics
        acc = (spectral.labels_ == labels).mean()
        nmi = adjusted_mutual_info_score(labels, spectral.labels_)
        ari = adjusted_rand_score(labels, spectral.labels_)

        ACC.append(acc)
        NMI.append(nmi)
        ARI.append(ari)

        loss.backward()
        finetune_optimizer.step()
        return loss.item()

    def validate(input_data, edge_index, labels, k=20):
        model.eval()
        with torch.no_grad():
            _, z = model(input_data, edge_index)
            latent = z.cpu().numpy()

            # Debug: Print summary of latent embeddings
            print("Latent embeddings - Min:", np.min(latent), "Max:", np.max(latent), "Mean:", np.mean(latent))

            # Check for zero vectors or invalid embeddings and clip them
            norms = np.linalg.norm(latent, axis=1)
            if norms.min() < 1e-6:
                print("Warning: Some latent embeddings have very small norms. Clipping them.")
                latent[norms < 1e-6] = latent[norms < 1e-6] + np.random.normal(0, 1e-6, size=latent[norms < 1e-6].shape)

            # Step 1: Compute affinity matrix (e.g., cosine similarity)
            affinity_matrix = cosine_similarity(latent)

            # Debug: Print summary of affinity matrix
            print("Affinity Matrix - Min:", np.min(affinity_matrix), "Max:", np.max(affinity_matrix))

            # Step 2: Check for NaNs or Infs in the affinity matrix
            if np.isnan(affinity_matrix).any() or np.isinf(affinity_matrix).any():
                print("Error: Affinity matrix contains NaN or Inf values.")
                return 0, 0, 0  # Handle this gracefully by returning default scores

            # Step 3: Perform Spectral Clustering on the valid affinity matrix
            spectral = SpectralClustering(n_clusters=k, affinity='precomputed')
            predictions = spectral.fit_predict(affinity_matrix)

            # Step 4: Calculate clustering metrics
            acc = (predictions == labels).mean()
            nmi = adjusted_mutual_info_score(labels, predictions)
            ari = adjusted_rand_score(labels, predictions)
            
            return acc, nmi, ari

    return train_step, train_loss, finetune, validate, ACC, NMI, ARI
