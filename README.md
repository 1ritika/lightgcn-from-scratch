# LightGCN Recommendation System (from Scratch)

This repository contains a PyTorch implementation of the LightGCN paper ("LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation") built from scratch. The model is trained and evaluated on the MovieLens 1M dataset.

## Project Structure

- `load_data.py`: Loads the raw MovieLens 1M data, processes user/item IDs, splits into train/test, and saves processed files.
- `build_graph.py`: Constructs the normalized sparse adjacency matrix required by LightGCN using the training data.
- `train.py`: Defines the LightGCN model architecture, BPR loss, training loop, and evaluation metrics (Recall@K, NDCG@K). Trains the model and saves final embeddings.
- `predict.py`: Loads the trained embeddings and generates Top-K movie recommendations for a specified user.
- `data/`: Directory containing the raw (after download) and processed data files.
- `.gitignore`: Specifies intentionally untracked files that Git should ignore.
- `requirements.txt`: (Currently empty) Lists project dependencies.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd lightgcn-from-scratch # Or your repo name
    ```
2.  **Create a virtual environment (Recommended):**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt # (Populate this file later!)
    # Or manually install: pip install torch torchvision torchaudio pandas numpy scipy tqdm scikit-learn matplotlib
    ```
4.  **Download Data:** Run the download commands inside `load_data.py` or manually place the `ml-1m.zip` file in the `data/` directory and unzip it.

## Usage

1.  **Process Data:**
    ```bash
    python3 load_data.py
    ```
2.  **Build Graph:**
    ```bash
    python3 build_graph.py
    ```
3.  **Train Model:**
    ```bash
    python3 train.py
    ```
    *(Note: Training was performed on CPU due to server driver issues. Modify `DEVICE` in `train.py` for GPU training.)*
4.  **Generate Predictions:**
    ```bash
    python3 predict.py
    ```
    *(Modify `USER_ID_TO_RECOMMEND` in `predict.py` to get recommendations for different users.)*

## Results (MovieLens 1M, CPU, 100 Epochs)

- Peak Recall@20: ~0.1175
- Peak NDCG@20: ~0.1396

