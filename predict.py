import torch
import pandas as pd
import os

# --- 1. Configuration ---
DATA_DIR = "./data"
USER_ID_TO_RECOMMEND = 12  #  CAN CHANGE THIS
TOP_K = 10

# We are only doing inference, so let's not track gradients
torch.set_grad_enabled(False)

# --- 2. Load All Necessary Data ---
print("Loading model and data...")

# Load the trained embeddings (making sure to load on CPU)
try:
    user_embeddings = torch.load(os.path.join(DATA_DIR, "final_user_embeddings.pt"), map_location="cpu")
    item_embeddings = torch.load(os.path.join(DATA_DIR, "final_item_embeddings.pt"), map_location="cpu")
except FileNotFoundError:
    print("Error: Model embeddings not found.")
    print("Please run 'train.py' first to generate them.")
    exit()

# Load the data_info file to get the mappings
try:
    data_info = torch.load(os.path.join(DATA_DIR, 'data_info.pt'), weights_only=False)
except FileNotFoundError:
    print("Error: data_info.pt not found.")
    print("Please run 'load_data.py' first.")
    exit()
    
user_to_index = data_info['user_to_index']
item_to_index = data_info['item_to_index']

# We need to create a reverse mapping to get movie titles
# { 'index': 'original_movie_id' }
index_to_item = {v: k for k, v in item_to_index.items()}

# Load the movie titles from movies.dat
movies_file = os.path.join(DATA_DIR, 'ml-1m/movies.dat')
movies_df = pd.read_csv(
    movies_file,
    sep='::',
    engine='python',
    header=None,
    names=['MovieID', 'Title', 'Genres'],
    encoding='ISO-8859-1' # This encoding is needed for MovieLens
)
# Create a simple { 'MovieID': 'Title' } mapping
movie_id_to_title = movies_df.set_index('MovieID')['Title'].to_dict()

# Load the ratings file to show what the user has *already* watched
ratings_file = os.path.join(DATA_DIR, 'ml-1m/ratings.dat')
ratings_df = pd.read_csv(
    ratings_file,
    sep='::',
    engine='python',
    header=None,
    names=['UserID', 'MovieID', 'Rating', 'Timestamp']
)

# --- 3. Get User's Ground Truth ---

# Find what the user has already rated (and liked, e.g., 4-5 stars)
user_rated_movies = ratings_df[ratings_df['UserID'] == USER_ID_TO_RECOMMEND]
user_liked_movies = user_rated_movies[user_rated_movies['Rating'] >= 4]['MovieID']
user_liked_titles = [movie_id_to_title.get(movie_id, "Unknown Movie") for movie_id in user_liked_movies]

if not user_liked_titles:
    print(f"Warning: User {USER_ID_TO_RECOMMEND} has no liked movies (>= 4 stars) in the dataset.")
else:
    print(f"--- User {USER_ID_TO_RECOMMEND} likes: ---")
    for title in user_liked_titles[:10]: # Show up to 10
        print(f"  - {title}")
    print("---------------------------------")


# --- 4. Perform Inference ---

# Check if user is in our training data
if USER_ID_TO_RECOMMEND not in user_to_index:
    print(f"Error: UserID {USER_ID_TO_RECOMMEND} was not part of the training set.")
    exit()

# Get the internal index for our user
user_index = user_to_index[USER_ID_TO_RECOMMEND]

# Get the embedding vector for this user
user_vector = user_embeddings[user_index]

# Calculate the scores (dot product) between this user and ALL items
# This is the "inference" step
scores = torch.matmul(user_vector, item_embeddings.T)

# --- 5. Get Top K Recommendations ---

# Get the indices of the top K scores
top_scores, top_indices = torch.topk(scores, k=TOP_K + len(user_liked_movies)) # Get extra in case of overlap

# Convert to a set for easy filtering
user_rated_movie_indices = set(user_rated_movies['MovieID'].map(item_to_index.get).dropna().astype(int))

# Filter out movies the user has *already* rated
recommendations = []
for item_index in top_indices:
    if item_index.item() not in user_rated_movie_indices:
        recommendations.append(item_index.item())
    
    if len(recommendations) == TOP_K:
        break

# --- 6. Print Results ---
print(f"*** Top {TOP_K} Recommendations for User {USER_ID_TO_RECOMMEND} ***")
for i, item_index in enumerate(recommendations):
    # Map internal index back to original MovieID
    original_movie_id = index_to_item[item_index]
    # Get the title
    title = movie_id_to_title[original_movie_id]
    print(f"  {i+1}. {title}")