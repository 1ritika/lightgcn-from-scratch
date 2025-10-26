import torch
import pandas as pd
import scipy.sparse as sp
import numpy as np
import os
from tqdm import tqdm

def build_normalized_adjacency_matrix(data_dir):
    """
    Builds the normalized adjacency matrix A_tilde as described in the LightGCN paper.
    """
    print("--- Starting graph building ---")
    
    # 1. Load data info and training data
    info_path = os.path.join(data_dir, 'data_info.pt')
    data_info = torch.load(info_path, weights_only=False)
    num_users = data_info['num_users']
    num_items = data_info['num_items']
    num_nodes = num_users + num_items
    
    train_path = os.path.join(data_dir, 'train_df.csv')
    train_df = pd.read_csv(train_path)
    
    print(f"Loaded {num_users} users, {num_items} items. Total nodes: {num_nodes}")
    print(f"Building graph from {len(train_df)} training interactions.")

    # 2. Build the adjacency matrix A in COO format (list of edges)
    # We use scipy.sparse.coo_matrix to build the graph efficiently
    
    # Get user and item indices from the training data
    user_indices = train_df['user_index'].values
    item_indices = train_df['item_index'].values
    
    # Item indices need to be offset by the number of users
    # e.g., if num_users = 1000, item 0 becomes node 1000
    item_indices_shifted = item_indices + num_users
    
    # Create the edge list for the User-Item interaction part (R)
    # This represents R
    ratings = np.ones_like(user_indices, dtype=np.float32)
    R = sp.coo_matrix((ratings, (user_indices, item_indices_shifted)), shape=(num_nodes, num_nodes))
    
    # The full adjacency matrix A is R + R.T
    # This is because the graph is bipartite, and we want A = [[0, R], [R.T, 0]]
    A = R + R.T
    print("Built A (R + R.T)")

    # 3. Add self-connections (Identity matrix I)
    # A_hat = A + I
    A_hat = A + sp.eye(num_nodes)
    print("Built A_hat (A + I)")

    # 4. Compute the degree matrix D
    # D is a diagonal matrix where D_ii is the degree of node i
    degrees = np.array(A_hat.sum(axis=1)).flatten()
    
    # 5. Compute D^(-1/2)
    # We add a small epsilon (1e-6) to avoid division by zero
    D_inv_sqrt = sp.diags(1.0 / (np.sqrt(degrees) + 1e-6), format='csr')
    print("Calculated D_inv_sqrt")

    # 6. Compute the normalized adjacency matrix: A_tilde = D^(-1/2) * A_hat * D^(-1/2)
    # This is the core formula from the paper
    A_tilde = D_inv_sqrt.dot(A_hat).dot(D_inv_sqrt)
    print("Calculated A_tilde (D^-1/2 * A_hat * D^-1/2)")

    # 7. Convert from SciPy sparse matrix to PyTorch sparse tensor
    
    # Convert to COO format (Coordinate format), which PyTorch understands
    coo = A_tilde.tocoo()
    
    indices = torch.LongTensor(np.vstack((coo.row, coo.col)))
    values = torch.FloatTensor(coo.data)
    shape = torch.Size(coo.shape)
    
    sparse_norm_adj = torch.sparse_coo_tensor(indices, values, shape)
    
    print(f"Converted A_tilde to PyTorch sparse tensor with {sparse_norm_adj._nnz()} non-zero entries.")

    # 8. Save the final tensor
    graph_path = os.path.join(data_dir, 'sparse_norm_adj.pt')
    torch.save(sparse_norm_adj, graph_path)
    
    print(f"--- Graph building complete. Saved to '{graph_path}' ---")

if __name__ == "__main__":
    data_directory = "./data"
    build_normalized_adjacency_matrix(data_directory)