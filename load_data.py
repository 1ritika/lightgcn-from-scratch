import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import os

def load_and_process_data(data_dir, test_size=0.2):
    """
    Loads the MovieLens 1M data, processes it, and saves train/test splits
    and mappings.
    """
    print("--- Starting data loading and processing ---")
    
    # 1. Define file paths
    ratings_file = os.path.join(data_dir, 'ml-1m/ratings.dat')
    
    # 2. Load the data
    # The file is '::' separated and has no header
    ratings_df = pd.read_csv(
        ratings_file,
        sep='::',
        engine='python',
        header=None,
        names=['UserID', 'MovieID', 'Rating', 'Timestamp']
    )
    
    # We only care about interactions, not the rating value, for this model
    ratings_df = ratings_df[['UserID', 'MovieID']]
    print(f"Loaded {len(ratings_df)} total interactions.")
    
    # 3. Re-index users and items to be 0-indexed and continuous
    
    # Create a mapping from original UserID to new index
    unique_users = ratings_df['UserID'].unique()
    user_to_index = {old_id: new_id for new_id, old_id in enumerate(unique_users)}
    num_users = len(unique_users)
    
    # Create a mapping from original MovieID to new index
    unique_items = ratings_df['MovieID'].unique()
    item_to_index = {old_id: new_id for new_id, old_id in enumerate(unique_items)}
    num_items = len(unique_items)
    
    # Apply the new mappings
    ratings_df['user_index'] = ratings_df['UserID'].map(user_to_index)
    ratings_df['item_index'] = ratings_df['MovieID'].map(item_to_index)
    
    print(f"Found {num_users} unique users and {num_items} unique items.")
    
    # 4. Create train/test split
    # We split the *interactions* (the rows in the dataframe)
    train_df, test_df = train_test_split(
        ratings_df,
        test_size=test_size,
        random_state=42,
        stratify=ratings_df['user_index']  # Ensure users are in both sets
    )
    
    print(f"Created train split with {len(train_df)} interactions.")
    print(f"Created test split with {len(test_df)} interactions.")
    
    # 5. Save the processed data for later use
    # We will save the dataframes and the number of users/items
    
    # Create paths in our data directory
    train_path = os.path.join(data_dir, 'train_df.csv')
    test_path = os.path.join(data_dir, 'test_df.csv')
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    # Save the all-important counts and mappings
    data_info = {
        'num_users': num_users,
        'num_items': num_items,
        'train_interactions': len(train_df),
        'test_interactions': len(test_df),
        'user_to_index': user_to_index,
        'item_to_index': item_to_index
    }
    
    # Save as a .pt file
    info_path = os.path.join(data_dir, 'data_info.pt')
    torch.save(data_info, info_path)
    
    print(f"Saved processed data and info to '{data_dir}'")
    print("--- Data processing complete ---")
    return data_info

if __name__ == "__main__":
    # Define the path to our data directory
    data_directory = "./data"
    
    # Run the function
    load_and_process_data(data_directory)