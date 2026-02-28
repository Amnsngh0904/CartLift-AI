"""
Build Item Co-occurrence Graph from Cart Sequences

Computes conditional probabilities P(j | i) based on sequential cart additions.
Creates an adjacency list with top-k neighbors per item for efficient lookup.

Key Concepts:
- P(j | i) = count(i → j) / count(i)
- Sequential: considers items added after item i in the same session
- Directional: P(j|i) ≠ P(i|j) 

Output:
    data/processed/item_cooccurrence.pkl containing:
    - adjacency_list: Dict[item_id, List[Tuple[neighbor_id, probability]]]
    - item_counts: Dict[item_id, int]
    - pair_counts: Dict[Tuple[item_i, item_j], int]
    - metadata: generation info

Usage:
    python -m src.graph.build_cooccurrence
    python -m src.graph.build_cooccurrence --input data/processed/cart_sequences.csv --top-k 50
"""

import argparse
import logging
import pickle
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd

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

DEFAULT_SEQUENCES_PATH = Path("data/processed/cart_sequences.csv")
DEFAULT_OUTPUT_PATH = Path("data/processed/item_cooccurrence.pkl")
DEFAULT_TOP_K = 50
MIN_PAIR_COUNT = 2  # Minimum co-occurrence count to include


# -----------------------------------------------------------------------------
# Data Classes
# -----------------------------------------------------------------------------

@dataclass
class CooccurrenceGraph:
    """
    Item co-occurrence graph with conditional probabilities.
    
    Attributes:
        adjacency_list: Dict mapping item_id -> List[(neighbor_id, P(neighbor|item))]
                       Sorted by probability descending, limited to top_k
        item_counts: Dict mapping item_id -> total occurrences
        pair_counts: Dict mapping (item_i, item_j) -> co-occurrence count
        top_k: Number of neighbors stored per item
        metadata: Generation metadata
    """
    adjacency_list: Dict[str, List[Tuple[str, float]]]
    item_counts: Dict[str, int]
    pair_counts: Dict[Tuple[str, str], int]
    top_k: int
    metadata: Dict = field(default_factory=dict)
    
    def get_neighbors(
        self,
        item_id: str,
        top_n: int = None
    ) -> List[Tuple[str, float]]:
        """
        Get top neighbors for an item.
        
        Args:
            item_id: Source item ID
            top_n: Number of neighbors to return (default: all stored)
            
        Returns:
            List of (neighbor_id, probability) tuples
        """
        neighbors = self.adjacency_list.get(item_id, [])
        if top_n is not None:
            return neighbors[:top_n]
        return neighbors
    
    def get_probability(self, item_i: str, item_j: str) -> float:
        """
        Get P(j | i) - probability of j given i was added.
        
        Args:
            item_i: Source item (condition)
            item_j: Target item
            
        Returns:
            Conditional probability or 0.0 if not found
        """
        for neighbor_id, prob in self.adjacency_list.get(item_i, []):
            if neighbor_id == item_j:
                return prob
        return 0.0
    
    def get_item_count(self, item_id: str) -> int:
        """Get occurrence count for an item."""
        return self.item_counts.get(item_id, 0)
    
    @property
    def num_items(self) -> int:
        """Total number of unique items."""
        return len(self.item_counts)
    
    @property
    def num_edges(self) -> int:
        """Total number of edges (neighbor relationships)."""
        return sum(len(neighbors) for neighbors in self.adjacency_list.values())
    
    def summary(self) -> str:
        """Return summary statistics."""
        if not self.adjacency_list:
            return "Empty graph"
        
        neighbor_counts = [len(n) for n in self.adjacency_list.values()]
        avg_neighbors = sum(neighbor_counts) / len(neighbor_counts)
        
        return (
            f"Items: {self.num_items:,}\n"
            f"Edges: {self.num_edges:,}\n"
            f"Avg neighbors/item: {avg_neighbors:.1f}\n"
            f"Top-K: {self.top_k}"
        )


# -----------------------------------------------------------------------------
# Graph Building Functions
# -----------------------------------------------------------------------------

def count_cooccurrences(
    sequences_df: pd.DataFrame,
    directional: bool = True
) -> Tuple[Dict[str, int], Dict[Tuple[str, str], int]]:
    """
    Count item occurrences and co-occurrences from sequences.
    
    For sequential co-occurrence:
    - If directional=True: count (i, j) where j appears AFTER i in sequence
    - If directional=False: count all pairs (i, j) where i != j
    
    Args:
        sequences_df: DataFrame with item_sequence column
        directional: Whether to count directional pairs only
        
    Returns:
        Tuple of (item_counts, pair_counts)
    """
    item_counts: Dict[str, int] = defaultdict(int)
    pair_counts: Dict[Tuple[str, str], int] = defaultdict(int)
    
    total_sequences = len(sequences_df)
    log_interval = max(1, total_sequences // 10)
    
    for idx, row in sequences_df.iterrows():
        sequence = row["item_sequence"].split("|")
        
        # Count individual items
        for item in sequence:
            item_counts[item] += 1
        
        # Count co-occurrences
        n = len(sequence)
        for i in range(n):
            item_i = sequence[i]
            
            if directional:
                # Only count items that appear AFTER item_i
                for j in range(i + 1, n):
                    item_j = sequence[j]
                    if item_i != item_j:
                        pair_counts[(item_i, item_j)] += 1
            else:
                # Count all pairs (both directions)
                for j in range(n):
                    if i != j:
                        item_j = sequence[j]
                        if item_i != item_j:
                            pair_counts[(item_i, item_j)] += 1
        
        if (idx + 1) % log_interval == 0:
            pct = 100 * (idx + 1) / total_sequences
            logger.info(f"  Processed {idx + 1:,}/{total_sequences:,} sequences ({pct:.0f}%)")
    
    return dict(item_counts), dict(pair_counts)


def build_adjacency_list(
    item_counts: Dict[str, int],
    pair_counts: Dict[Tuple[str, str], int],
    top_k: int = DEFAULT_TOP_K,
    min_count: int = MIN_PAIR_COUNT
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Build adjacency list with conditional probabilities.
    
    P(j | i) = count(i, j) / count(i)
    
    Args:
        item_counts: Item occurrence counts
        pair_counts: Pair co-occurrence counts
        top_k: Number of top neighbors to keep per item
        min_count: Minimum pair count to include
        
    Returns:
        Adjacency list: Dict[item_id, List[(neighbor_id, probability)]]
    """
    logger.info(f"Building adjacency list (top-{top_k}, min_count={min_count})...")
    
    # Group pairs by source item
    neighbors_by_item: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
    
    for (item_i, item_j), count in pair_counts.items():
        if count >= min_count:
            neighbors_by_item[item_i].append((item_j, count))
    
    # Convert to probabilities and sort
    adjacency_list: Dict[str, List[Tuple[str, float]]] = {}
    
    items_processed = 0
    items_with_neighbors = 0
    
    for item_id, item_count in item_counts.items():
        if item_id not in neighbors_by_item:
            adjacency_list[item_id] = []
            continue
        
        neighbors = neighbors_by_item[item_id]
        
        # Convert counts to probabilities
        neighbor_probs = [
            (neighbor_id, count / item_count)
            for neighbor_id, count in neighbors
        ]
        
        # Sort by probability descending
        neighbor_probs.sort(key=lambda x: x[1], reverse=True)
        
        # Keep top-k
        adjacency_list[item_id] = neighbor_probs[:top_k]
        
        items_processed += 1
        if neighbor_probs:
            items_with_neighbors += 1
    
    logger.info(f"  Items processed: {items_processed:,}")
    logger.info(f"  Items with neighbors: {items_with_neighbors:,}")
    
    return adjacency_list


def build_cooccurrence_graph(
    sequences_path: Path,
    output_path: Path,
    top_k: int = DEFAULT_TOP_K,
    directional: bool = True,
    min_count: int = MIN_PAIR_COUNT
) -> CooccurrenceGraph:
    """
    Build complete co-occurrence graph from cart sequences.
    
    Args:
        sequences_path: Path to cart_sequences.csv
        output_path: Path for output pickle file
        top_k: Number of top neighbors per item
        directional: Whether to use directional co-occurrence
        min_count: Minimum pair count to include
        
    Returns:
        CooccurrenceGraph object
    """
    logger.info("=" * 60)
    logger.info("BUILDING ITEM CO-OCCURRENCE GRAPH")
    logger.info("=" * 60)
    logger.info(f"Input: {sequences_path}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Top-K neighbors: {top_k}")
    logger.info(f"Directional: {directional}")
    logger.info(f"Min pair count: {min_count}")
    
    # Load sequences
    logger.info("\nLoading cart sequences...")
    sequences_df = pd.read_csv(sequences_path)
    logger.info(f"Loaded {len(sequences_df):,} sequences")
    
    # Calculate sequence length stats
    seq_lengths = sequences_df["item_sequence"].str.count(r"\|") + 1
    logger.info(f"Sequence length: min={seq_lengths.min()}, "
               f"max={seq_lengths.max()}, mean={seq_lengths.mean():.2f}")
    
    # Count co-occurrences
    logger.info("\nCounting co-occurrences...")
    item_counts, pair_counts = count_cooccurrences(sequences_df, directional)
    
    logger.info(f"\nCounting complete:")
    logger.info(f"  Unique items: {len(item_counts):,}")
    logger.info(f"  Unique pairs: {len(pair_counts):,}")
    
    # Item count statistics
    counts_list = list(item_counts.values())
    logger.info(f"  Item count range: {min(counts_list)} - {max(counts_list)}")
    logger.info(f"  Mean item count: {sum(counts_list)/len(counts_list):.1f}")
    
    # Pair count statistics
    if pair_counts:
        pair_counts_list = list(pair_counts.values())
        logger.info(f"  Pair count range: {min(pair_counts_list)} - {max(pair_counts_list)}")
        logger.info(f"  Mean pair count: {sum(pair_counts_list)/len(pair_counts_list):.1f}")
        
        # Pairs meeting threshold
        pairs_above_min = sum(1 for c in pair_counts_list if c >= min_count)
        logger.info(f"  Pairs with count >= {min_count}: {pairs_above_min:,}")
    
    # Build adjacency list
    logger.info("\nBuilding adjacency list...")
    adjacency_list = build_adjacency_list(
        item_counts, pair_counts,
        top_k=top_k, min_count=min_count
    )
    
    # Create graph object
    metadata = {
        "created_at": datetime.now().isoformat(),
        "source_file": str(sequences_path),
        "num_sequences": len(sequences_df),
        "directional": directional,
        "min_pair_count": min_count,
        "top_k": top_k,
    }
    
    graph = CooccurrenceGraph(
        adjacency_list=adjacency_list,
        item_counts=item_counts,
        pair_counts=pair_counts,
        top_k=top_k,
        metadata=metadata,
    )
    
    # Save to pickle
    logger.info("\nSaving graph...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "wb") as f:
        pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"Saved to: {output_path} ({file_size_mb:.1f} MB)")
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("CO-OCCURRENCE GRAPH SUMMARY")
    logger.info("=" * 60)
    logger.info(f"\n{graph.summary()}")
    
    # Show example neighbors
    logger.info("\nExample neighbors (top 5 items by count):")
    top_items = sorted(item_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    for item_id, count in top_items:
        neighbors = graph.get_neighbors(item_id, top_n=3)
        neighbor_str = ", ".join(
            f"{n_id[:16]}... ({prob:.2%})" 
            for n_id, prob in neighbors
        ) if neighbors else "No neighbors"
        logger.info(f"  {item_id[:20]}... (count={count}): {neighbor_str}")
    
    # Probability distribution stats
    all_probs = []
    for neighbors in adjacency_list.values():
        all_probs.extend([p for _, p in neighbors])
    
    if all_probs:
        logger.info(f"\nProbability statistics:")
        logger.info(f"  Min P(j|i): {min(all_probs):.4f}")
        logger.info(f"  Max P(j|i): {max(all_probs):.4f}")
        logger.info(f"  Mean P(j|i): {sum(all_probs)/len(all_probs):.4f}")
    
    logger.info("=" * 60)
    
    return graph


def load_cooccurrence_graph(path: Path) -> CooccurrenceGraph:
    """
    Load co-occurrence graph from pickle file.
    
    Args:
        path: Path to pickle file
        
    Returns:
        CooccurrenceGraph object
    """
    with open(path, "rb") as f:
        return pickle.load(f)


# -----------------------------------------------------------------------------
# CLI Entry Point
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Build item co-occurrence graph from cart sequences",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=DEFAULT_SEQUENCES_PATH,
        help="Path to cart_sequences.csv",
    )
    
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path for output pickle file",
    )
    
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of top neighbors to store per item",
    )
    
    parser.add_argument(
        "--min-count",
        type=int,
        default=MIN_PAIR_COUNT,
        help="Minimum pair count to include",
    )
    
    parser.add_argument(
        "--bidirectional",
        action="store_true",
        help="Use bidirectional co-occurrence (default: directional)",
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        build_cooccurrence_graph(
            sequences_path=args.input,
            output_path=args.output,
            top_k=args.top_k,
            directional=not args.bidirectional,
            min_count=args.min_count,
        )
        return 0
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return 99


if __name__ == "__main__":
    sys.exit(main())
