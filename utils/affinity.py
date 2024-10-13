import numpy as np
from sklearn.neighbors import kneighbors_graph
import scipy.sparse as sp

def build_affinity_matrix(features, k=20):
    """
    Builds the k-nearest neighbor affinity matrix for the graph.
    
    :param features: np.array, Node features (shape: [num_nodes, num_features])
    :param k: int, Number of nearest neighbors for constructing the affinity matrix
    
    :return: Tuple of (DADsm, DADsp), which are the matrices for Laplacian smoothing and sharpening
    """
    
    # Step 1: Build k-nearest neighbor graph (sparse matrix)
    A = kneighbors_graph(features, n_neighbors=k, mode='connectivity', include_self=False)
    
    # Step 2: Convert to dense matrix and symmetrize the adjacency matrix
    A = A + A.T
    A = A.todense()

    # Add self-loops
    eye = np.eye(A.shape[0])
    Asm = A + eye
    
    # Step 3: Laplacian Smoothing (DADsm)
    Dsm = np.diag(1 / np.sqrt(Asm.sum(axis=-1).A1))  # Diagonal matrix with degrees
    DADsm = Dsm @ Asm @ Dsm  # Symmetric normalized Laplacian

    # Step 4: Laplacian Sharpening (DADsp)
    Asp = 2 * eye - A
    Dsp = np.diag(1 / np.sqrt((2 * eye + A).sum(axis=-1).A1))
    DADsp = Dsp @ Asp @ Dsp  # Laplacian sharpening
    
    return DADsm, DADsp

def save_affinity_matrix(path, DADsm, DADsp):
    """
    Saves the computed affinity matrices as a .npz file.
    
    :param path: str, File path to save the matrices
    :param DADsm: np.array, Laplacian smoothing matrix
    :param DADsp: np.array, Laplacian sharpening matrix
    """
    np.savez(path, DADsm=DADsm, DADsp=DADsp)

def load_affinity_matrix(path):
    """
    Loads the affinity matrices from a .npz file.
    
    :param path: str, File path to load the matrices
    :return: Tuple of (DADsm, DADsp)
    """
    data = np.load(path)
    return data['DADsm'], data['DADsp']
