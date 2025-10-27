import os
os.environ['TORCH_COMPILE_DISABLE'] = '1'


import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import pandas as pd
import numpy as np
from tqdm import tqdm
import random

# --- 1. Define Model ---
class LightGCN(nn.Module):
    """
    The "from scratch" LightGCN model.
    """
    def __init__(self, num_users, num_items, embed_dim, num_layers, sparse_norm_adj):
        super(LightGCN, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        
        # This is the (N x d) embedding matrix E^(0)
        self.embedding_user = nn.Embedding(num_users, embed_dim)
        self.embedding_item = nn.Embedding(num_items, embed_dim)
        
        # Initialize embeddings as per the paper (Xavier uniform)
        nn.init.xavier_uniform_(self.embedding_user.weight)
        nn.init.xavier_uniform_(self.embedding_item.weight)
        
        self.sparse_norm_adj = sparse_norm_adj
        
        # This is for the final layer combination
        self.alpha = 1.0 / (self.num_layers + 1)

    def propagate_embeddings(self):
        """
        Performs the propagation logic: E^(k+1) = A_tilde * E^(k)
        """
        emb_0 = torch.cat([self.embedding_user.weight, self.embedding_item.weight])
        
        all_embeddings = [emb_0]
        current_embeddings = emb_0
        
        for k in range(self.num_layers):
            next_embeddings = torch.sparse.mm(self.sparse_norm_adj, current_embeddings)
            all_embeddings.append(next_embeddings)
            current_embeddings = next_embeddings
            
        # Stack all embeddings (E^0 to E^K) and take the mean
        all_embeddings = torch.stack(all_embeddings, dim=0)
        final_embeddings = torch.mean(all_embeddings, dim=0)
        
        final_user_emb, final_item_emb = torch.split(final_embeddings, [self.num_users, self.num_items])
        
        return final_user_emb, final_item_emb

    def forward(self, user_indices, pos_item_indices, neg_item_indices):
        """
        The forward pass for training.
        """
        # Get the final, propagated embeddings
        final_user_emb, final_item_emb = self.propagate_embeddings()
        
        # Look up the specific embeddings for this batch
        batch_user_emb = final_user_emb[user_indices]
        batch_pos_item_emb = final_item_emb[pos_item_indices]
        batch_neg_item_emb = final_item_emb[neg_item_indices]
        
        return batch_user_emb, batch_pos_item_emb, batch_neg_item_emb

# --- 2. Define BPR Loss ---
class BPRLoss(nn.Module):
    """
    Bayesian Personalized Ranking (BPR) Loss.
    """
    def __init__(self, l2_reg=1e-5):
        super(BPRLoss, self).__init__()
        self.l2_reg = l2_reg

    def forward(self, user_emb, pos_item_emb, neg_item_emb):
        # Calculate scores
        pos_scores = torch.sum(user_emb * pos_item_emb, dim=1)
        neg_scores = torch.sum(user_emb * neg_item_emb, dim=1)
        
        # BPR loss
        loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-9))
        
        # L2 regularization (weight decay)
        reg_loss = (torch.norm(user_emb, p=2).pow(2) +
                    torch.norm(pos_item_emb, p=2).pow(2) +
                    torch.norm(neg_item_emb, p=2).pow(2)) / len(user_emb)
        
        return loss + self.l2_reg * reg_loss

# --- 3. Define Dataset and Sampler ---
class RecDataset(data.Dataset):
    def __init__(self, user_item_pairs):
        self.user_item_pairs = user_item_pairs

    def __len__(self):
        return len(self.user_item_pairs)

    def __getitem__(self, idx):
        return self.user_item_pairs[idx]

def get_sampler(train_df, num_items):
    """
    Creates a sampler that returns (user, pos_item, neg_item) triplets.
    """
    # Create a dict of users and their positive items
    user_pos_items = train_df.groupby('user_index')['item_index'].apply(set).to_dict()
    all_items = set(np.arange(num_items))
    
    triplets = []
    print("Sampling negative items...")
    for user, pos_items in tqdm(user_pos_items.items()):
        for pos_item in pos_items:
            # Sample a negative item
            neg_item = random.choice(list(all_items - pos_items))
            triplets.append((user, pos_item, neg_item))
            
    print(f"Generated {len(triplets)} (u, i, j) triplets.")
    return triplets

# --- 4. Define Evaluation Metrics ---
def get_user_positive_items(test_df):
    """
    Get a dict of users and their *actual* positive items from the test set.
    """
    return test_df.groupby('user_index')['item_index'].apply(set).to_dict()

def evaluate(model, test_user_pos_items, num_users, num_items, k, device):
    """
    Calculate Recall@K and NDCG@K.
    """
    model.eval()  # Set model to evaluation mode
    
    # Get the final, propagated embeddings for ALL users and items
    with torch.no_grad():
        final_user_emb, final_item_emb = model.propagate_embeddings()
        final_user_emb = final_user_emb.to(device)
        final_item_emb = final_item_emb.to(device)

    recalls = []
    ndcgs = []
    
    print(f"Evaluating model for Recall@{k} and NDCG@{k}...")
    
    # We'll evaluate user by user (or in small batches if memory is an issue)
    for user_id in tqdm(list(test_user_pos_items.keys())):
        # 1. Get all item scores for this user
        user_embedding = final_user_emb[user_id]
        
        # Score = U_i * I^T
        all_item_scores = torch.matmul(user_embedding, final_item_emb.T)
        
        # 2. Get the ground truth positive items
        true_pos_items = test_user_pos_items[user_id]
        if not true_pos_items:
            continue

        # 3. Get the Top K recommended items
        _, top_k_indices = torch.topk(all_item_scores, k=k, largest=True)
        top_k_indices = top_k_indices.cpu().numpy() # Move to CPU for set operations

        # 4. Calculate metrics
        num_hits = sum(1 for item_id in top_k_indices if item_id in true_pos_items)
        recall = num_hits / len(true_pos_items)
        recalls.append(recall)
        
        # Calculate NDCG
        dcg = sum(1 / np.log2(i + 2) for i, item_id in enumerate(top_k_indices) if item_id in true_pos_items)
        idcg = sum(1 / np.log2(i + 2) for i in range(min(len(true_pos_items), k)))
        ndcg = dcg / (idcg + 1e-9)
        ndcgs.append(ndcg)

    mean_recall = np.mean(recalls)
    mean_ndcg = np.mean(ndcgs)
    
    return mean_recall, mean_ndcg

# --- 5. Main Training Function ---
def main():
    # --- Config ---
    DATA_DIR = "./data"
    EMBED_DIM = 64
    NUM_LAYERS = 3
    BATCH_SIZE = 2048
    LEARNING_RATE = 1e-3
    L2_REG = 1e-5
    NUM_EPOCHS = 100  # Start with 20, can increase later
    EVAL_K = 20
    
  
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    # --- 1. Load All Data ---
    print("Loading data...")
    # Fix for new PyTorch: weights_only=False
    data_info = torch.load(os.path.join(DATA_DIR, 'data_info.pt'), weights_only=False) 
    num_users = data_info['num_users']
    num_items = data_info['num_items']
    
    train_df = pd.read_csv(os.path.join(DATA_DIR, 'train_df.csv'))
    test_df = pd.read_csv(os.path.join(DATA_DIR, 'test_df.csv'))
    
    # Load the graph
    # Fix for new PyTorch: weights_only=False
    sparse_norm_adj = torch.load(os.path.join(DATA_DIR, 'sparse_norm_adj.pt'), weights_only=False)
    sparse_norm_adj = sparse_norm_adj.to(DEVICE)
    print("Data loaded successfully.")

    # --- 2. Create Dataset and DataLoader ---
    triplets = get_sampler(train_df, num_items)
    dataset = RecDataset(triplets)
    dataloader = data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Get test data for evaluation
    test_user_pos_items = get_user_positive_items(test_df)

    # --- 3. Initialize Model, Loss, and Optimizer ---
    model = LightGCN(num_users, num_items, EMBED_DIM, NUM_LAYERS, sparse_norm_adj).to(DEVICE)
    criterion = BPRLoss(l2_reg=L2_REG)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- 4. Training Loop ---
    print("--- Starting Training ---")
    for epoch in range(NUM_EPOCHS):
        model.train()  # Set model to training mode
        epoch_loss = 0.0
        
        # We use tqdm for a nice progress bar
        for user, pos_item, neg_item in tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
            user = user.to(DEVICE)
            pos_item = pos_item.to(DEVICE)
            neg_item = neg_item.to(DEVICE)
            
            optimizer.zero_grad()
            
            user_emb, pos_item_emb, neg_item_emb = model(user, pos_item, neg_item)
            loss = criterion(user_emb, pos_item_emb, neg_item_emb)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Average BPR Loss: {avg_epoch_loss:.4f}")
        
        # --- 5. Evaluation ---
        # We evaluate after each epoch
        recall, ndcg = evaluate(model, test_user_pos_items, num_users, num_items, k=EVAL_K, device=DEVICE)
        print(f"*** Epoch {epoch+1} Evaluation ***")
        print(f"Recall@{EVAL_K}: {recall:.4f}")
        print(f"NDCG@{EVAL_K}:   {ndcg:.4f}")
        print("****************************")
        
    print("--- Training Complete ---")
    
    # --- 6. Save the Final Model (Optional) ---
    # We save just the embeddings, as that's all we need for inference
    final_user_emb, final_item_emb = model.propagate_embeddings()
    
    torch.save(final_user_emb.cpu(), os.path.join(DATA_DIR, "final_user_embeddings.pt"))
    torch.save(final_item_emb.cpu(), os.path.join(DATA_DIR, "final_item_embeddings.pt"))
    print("Saved final embeddings to './data'")

if __name__ == "__main__":
    main()
