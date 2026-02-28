"""
Train Node2Vec Embeddings on Item Co-occurrence Graph

Learns dense vector representations for items using random walks
on the co-occurrence graph. Embeddings capture item similarity based
on sequential cart behavior.

Uses gensim's Word2Vec for efficient training without torch-cluster dependency.

Output:
    data/processed/item_embeddings.npy - (num_items, embedding_dim) array
    data/processed/item_id_mapping.pkl - item_id <-> index mapping

Usage:
    python -m src.graph.train_node2vec
    python -m src.graph.train_node2vec --embedding-dim 128 --epochs 10
"""

import argparse
import logging
import pickle
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
from gensim.models import Word2Vec
from tqdm import tqdm

from src.graph.build_cooccurrence import CooccurrenceGraph

# -----------------------------------------------------------------------------
# Logging Configuration
# -----------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

DEFAULT_GRAPH_PATH = Path("data/processed/item_cooccurrence.pkl")
DEFAULT_EMBEDDINGS_PATH = Path("data/processed/item_embeddings.npy")
DEFAULT_MAPPING_PATH = Path("data/processed/item_id_mapping.pkl")

# Node2Vec defaults
DEFAULT_EMBEDDING_DIM = 64
DEFAULT_WALK_LENGTH = 20
DEFAULT_CONTEXT_SIZE = 10
DEFAULT_WALKS_PER_NODE = 10
DEFAULT_P = 1.0  # Return parameter
DEFAULT_Q = 1.0  # In-out parameter
DEFAULT_EPOCHS = 5


# -----------------------------------------------------------------------------
# Graph Conversion
# -----------------------------------------------------------------------------

def load_cooccurrence_graph(path: Path) -> CooccurrenceGraph:
    """Load pickled co-occurrence graph."""
    logger.info(f"Loading co-occurrence graph from {path}")
    with open(path, "rb") as f:
        graph = pickle.load(f)
    logger.info(f"Loaded graph with {graph.num_items:,} items, {graph.num_edges:,} edges")
    return graph


def build_adjacency_dict(
    graph: CooccurrenceGraph,
) -> Tuple[Dict[str, List[Tuple[str, float]]], Dict[str, int], Dict[int, str]]:
    """
    Build adjacency dictionary with transition probabilities.
    
    Args:
        graph: CooccurrenceGraph instance
        
    Returns:
        adjacency: Dict mapping item_id -> List[(neighbor_id, probability)]
        item_to_idx: item_id -> index mapping
        idx_to_item: index -> item_id mapping
    """
    logger.info("Building adjacency dictionary...")
    
    # Create item ID mappings
    all_items = sorted(graph.item_counts.keys())
    item_to_idx = {item_id: idx for idx, item_id in enumerate(all_items)}
    idx_to_item = {idx: item_id for item_id, idx in item_to_idx.items()}
    
    # Build symmetric adjacency for undirected walks
    adjacency: Dict[str, List[Tuple[str, float]]] = {item: [] for item in all_items}
    
    for item_id, neighbors in graph.adjacency_list.items():
        for neighbor_id, prob in neighbors:
            if neighbor_id in item_to_idx:
                adjacency[item_id].append((neighbor_id, prob))
                # Add reverse edge for undirected behavior
                adjacency[neighbor_id].append((item_id, prob))
    
    # Remove duplicates and normalize
    for item_id in adjacency:
        seen = {}
        for neighbor_id, prob in adjacency[item_id]:
            if neighbor_id not in seen:
                seen[neighbor_id] = prob
            else:
                seen[neighbor_id] = max(seen[neighbor_id], prob)
        adjacency[item_id] = list(seen.items())
    
    num_edges = sum(len(neighbors) for neighbors in adjacency.values())
    logger.info(f"Adjacency dict: {len(adjacency):,} nodes, {num_edges:,} edges")
    
    return adjacency, item_to_idx, idx_to_item


# -----------------------------------------------------------------------------
# Random Walk Generation
# -----------------------------------------------------------------------------

def node2vec_walk(
    adjacency: Dict[str, List[Tuple[str, float]]],
    start_node: str,
    walk_length: int,
    p: float = 1.0,
    q: float = 1.0,
) -> List[str]:
    """
    Perform single Node2Vec random walk.
    
    Args:
        adjacency: Adjacency dictionary
        start_node: Starting node
        walk_length: Length of walk
        p: Return parameter
        q: In-out parameter
        
    Returns:
        List of node IDs in the walk
    """
    walk = [start_node]
    
    if not adjacency.get(start_node):
        return walk
    
    # First step (uniform)
    neighbors = adjacency[start_node]
    if not neighbors:
        return walk
    
    probs = [prob for _, prob in neighbors]
    total = sum(probs)
    probs = [p / total for p in probs]
    next_node = random.choices([n for n, _ in neighbors], weights=probs, k=1)[0]
    walk.append(next_node)
    
    # Subsequent steps with Node2Vec bias
    for _ in range(walk_length - 2):
        cur = walk[-1]
        prev = walk[-2]
        
        neighbors = adjacency.get(cur, [])
        if not neighbors:
            break
        
        prev_neighbors = set(n for n, _ in adjacency.get(prev, []))
        
        # Compute unnormalized probabilities with Node2Vec bias
        unnorm_probs = []
        for neighbor, prob in neighbors:
            if neighbor == prev:
                # Return to previous node
                unnorm_probs.append(prob / p)
            elif neighbor in prev_neighbors:
                # Neighbor of previous (BFS-like)
                unnorm_probs.append(prob)
            else:
                # Not neighbor of previous (DFS-like)
                unnorm_probs.append(prob / q)
        
        total = sum(unnorm_probs)
        if total == 0:
            break
        
        probs = [up / total for up in unnorm_probs]
        next_node = random.choices([n for n, _ in neighbors], weights=probs, k=1)[0]
        walk.append(next_node)
    
    return walk


def generate_walks(
    adjacency: Dict[str, List[Tuple[str, float]]],
    walks_per_node: int,
    walk_length: int,
    p: float = 1.0,
    q: float = 1.0,
    seed: int = None,
) -> List[List[str]]:
    """
    Generate random walks for all nodes.
    
    Args:
        adjacency: Adjacency dictionary
        walks_per_node: Number of walks starting from each node
        walk_length: Length of each walk
        p: Return parameter
        q: In-out parameter
        seed: Random seed
        
    Returns:
        List of walks (each walk is a list of node IDs)
    """
    if seed is not None:
        random.seed(seed)
    
    nodes = list(adjacency.keys())
    walks = []
    
    logger.info(f"Generating {walks_per_node} walks per node (walk_length={walk_length}, p={p}, q={q})")
    
    for walk_num in range(walks_per_node):
        random.shuffle(nodes)
        for node in tqdm(nodes, desc=f"Walk {walk_num + 1}/{walks_per_node}", leave=False):
            if adjacency.get(node):  # Only walk from nodes with neighbors
                walk = node2vec_walk(adjacency, node, walk_length, p, q)
                if len(walk) > 1:
                    walks.append(walk)
    
    logger.info(f"Generated {len(walks):,} walks")
    return walks


# -----------------------------------------------------------------------------
# Node2Vec Training with Gensim
# -----------------------------------------------------------------------------

def train_node2vec(
    adjacency: Dict[str, List[Tuple[str, float]]],
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    walk_length: int = DEFAULT_WALK_LENGTH,
    context_size: int = DEFAULT_CONTEXT_SIZE,
    walks_per_node: int = DEFAULT_WALKS_PER_NODE,
    p: float = DEFAULT_P,
    q: float = DEFAULT_Q,
    epochs: int = DEFAULT_EPOCHS,
    min_count: int = 1,
    num_workers: int = 4,
    seed: int = 42,
) -> Tuple[Word2Vec, List[str]]:
    """
    Train Node2Vec embeddings using gensim Word2Vec.
    
    Args:
        adjacency: Adjacency dictionary
        embedding_dim: Dimension of embeddings
        walk_length: Length of random walks
        context_size: Context window size
        walks_per_node: Number of walks per node
        p: Return parameter (controls likelihood of revisiting)
        q: In-out parameter (controls BFS vs DFS behavior)
        epochs: Training epochs
        min_count: Minimum word frequency
        num_workers: Number of workers
        seed: Random seed
        
    Returns:
        model: Trained Word2Vec model
        vocab: List of item IDs in model vocabulary
    """
    # Generate random walks
    walk_start = time.time()
    walks = generate_walks(
        adjacency=adjacency,
        walks_per_node=walks_per_node,
        walk_length=walk_length,
        p=p,
        q=q,
        seed=seed,
    )
    walk_time = time.time() - walk_start
    logger.info(f"Walk generation time: {walk_time:.1f}s")
    
    # Train Word2Vec
    logger.info(f"Training Word2Vec (dim={embedding_dim}, window={context_size}, epochs={epochs})")
    
    train_start = time.time()
    
    model = Word2Vec(
        sentences=walks,
        vector_size=embedding_dim,
        window=context_size,
        min_count=min_count,
        sg=1,  # Skip-gram
        hs=0,  # Negative sampling
        negative=5,
        workers=num_workers,
        epochs=epochs,
        seed=seed,
        compute_loss=True,
    )
    
    train_time = time.time() - train_start
    total_time = walk_time + train_time
    
    # Get final loss
    final_loss = model.get_latest_training_loss()
    
    logger.info(f"Walk generation: {walk_time:.1f}s")
    logger.info(f"Training time: {train_time:.1f}s")
    logger.info(f"Total time: {total_time:.1f}s")
    logger.info(f"Final loss: {final_loss:.2f}")
    logger.info(f"Vocabulary size: {len(model.wv):,}")
    
    return model, list(model.wv.index_to_key)


# -----------------------------------------------------------------------------
# Save Functions
# -----------------------------------------------------------------------------

def save_embeddings(
    embeddings: np.ndarray,
    item_to_idx: Dict[str, int],
    idx_to_item: Dict[int, str],
    embeddings_path: Path,
    mapping_path: Path,
    metadata: Dict = None,
) -> None:
    """
    Save embeddings and ID mappings.
    
    Args:
        embeddings: (num_items, embedding_dim) array
        item_to_idx: item_id -> index mapping
        idx_to_item: index -> item_id mapping
        embeddings_path: Path to save embeddings .npy
        mapping_path: Path to save mappings .pkl
        metadata: Optional metadata dict
    """
    # Ensure directories exist
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save embeddings
    np.save(embeddings_path, embeddings)
    logger.info(f"Saved embeddings to {embeddings_path}")
    
    # Save mappings with metadata
    mapping_data = {
        "item_to_idx": item_to_idx,
        "idx_to_item": idx_to_item,
        "metadata": metadata or {},
    }
    with open(mapping_path, "wb") as f:
        pickle.dump(mapping_data, f)
    logger.info(f"Saved ID mappings to {mapping_path}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train Node2Vec embeddings on co-occurrence graph"
    )
    parser.add_argument(
        "--graph-path",
        type=Path,
        default=DEFAULT_GRAPH_PATH,
        help="Path to co-occurrence graph pickle"
    )
    parser.add_argument(
        "--embeddings-path",
        type=Path,
        default=DEFAULT_EMBEDDINGS_PATH,
        help="Output path for embeddings .npy"
    )
    parser.add_argument(
        "--mapping-path",
        type=Path,
        default=DEFAULT_MAPPING_PATH,
        help="Output path for ID mappings .pkl"
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=DEFAULT_EMBEDDING_DIM,
        help="Embedding dimension"
    )
    parser.add_argument(
        "--walk-length",
        type=int,
        default=DEFAULT_WALK_LENGTH,
        help="Random walk length"
    )
    parser.add_argument(
        "--context-size",
        type=int,
        default=DEFAULT_CONTEXT_SIZE,
        help="Context window size"
    )
    parser.add_argument(
        "--walks-per-node",
        type=int,
        default=DEFAULT_WALKS_PER_NODE,
        help="Number of walks per node"
    )
    parser.add_argument(
        "--p",
        type=float,
        default=DEFAULT_P,
        help="Return parameter p"
    )
    parser.add_argument(
        "--q",
        type=float,
        default=DEFAULT_Q,
        help="In-out parameter q"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of workers for training"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Node2Vec Embedding Training (gensim)")
    logger.info("=" * 60)
    
    # Load graph
    graph = load_cooccurrence_graph(args.graph_path)
    
    # Build adjacency dictionary
    adjacency, item_to_idx, idx_to_item = build_adjacency_dict(graph)
    
    # Train Node2Vec
    model, vocab = train_node2vec(
        adjacency=adjacency,
        embedding_dim=args.embedding_dim,
        walk_length=args.walk_length,
        context_size=args.context_size,
        walks_per_node=args.walks_per_node,
        p=args.p,
        q=args.q,
        epochs=args.epochs,
        num_workers=args.workers,
        seed=args.seed,
    )
    
    # Extract embeddings as numpy array
    # Create aligned embeddings matrix
    embeddings = np.zeros((len(item_to_idx), args.embedding_dim), dtype=np.float32)
    items_with_embeddings = 0
    
    for item_id, idx in item_to_idx.items():
        if item_id in model.wv:
            embeddings[idx] = model.wv[item_id]
            items_with_embeddings += 1
    
    logger.info(f"Items with embeddings: {items_with_embeddings:,}/{len(item_to_idx):,}")
    
    # Prepare metadata
    metadata = {
        "created_at": datetime.now().isoformat(),
        "source_graph": str(args.graph_path),
        "embedding_dim": args.embedding_dim,
        "walk_length": args.walk_length,
        "context_size": args.context_size,
        "walks_per_node": args.walks_per_node,
        "p": args.p,
        "q": args.q,
        "epochs": args.epochs,
        "num_items": len(item_to_idx),
        "items_with_embeddings": items_with_embeddings,
        "final_loss": model.get_latest_training_loss(),
    }
    
    # Save
    save_embeddings(
        embeddings=embeddings,
        item_to_idx=item_to_idx,
        idx_to_item=idx_to_item,
        embeddings_path=args.embeddings_path,
        mapping_path=args.mapping_path,
        metadata=metadata,
    )
    
    # Save gensim model for similarity queries
    gensim_model_path = args.embeddings_path.parent / "node2vec_model.bin"
    model.save(str(gensim_model_path))
    logger.info(f"Saved gensim model to {gensim_model_path}")
    
    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info(f"Embeddings: {args.embeddings_path}")
    logger.info(f"Mappings: {args.mapping_path}")
    logger.info(f"Gensim model: {gensim_model_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
