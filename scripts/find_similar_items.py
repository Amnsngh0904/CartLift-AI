"""
Find similar items using Node2Vec embeddings.

Usage:
    python -m scripts.find_similar_items
"""
import pickle
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gensim.models import Word2Vec
import pandas as pd

# Import before pickle to register class
from src.graph.build_cooccurrence import CooccurrenceGraph


def main():
    # Load model and item data
    print("Loading model and data...")
    model = Word2Vec.load(str(PROJECT_ROOT / "data/processed/node2vec_model.bin"))
    items_df = pd.read_csv(str(PROJECT_ROOT / "data/processed/menu_items_simulation.csv"))
    
    # Load co-occurrence graph to find popular items
    with open(PROJECT_ROOT / "data/processed/item_cooccurrence.pkl", "rb") as f:
        graph: CooccurrenceGraph = pickle.load(f)
    
    # Find top 5 most frequent items
    top_items = sorted(graph.item_counts.items(), key=lambda x: -x[1])[:5]
    
    print("=" * 70)
    print("TOP 10 COSINE SIMILAR ITEMS FOR POPULAR ITEMS")
    print("=" * 70)
    
    for item_id, count in top_items:
        # Get item name from items_df
        item_row = items_df[items_df["item_id"] == item_id]
        item_name = item_row["item_name"].values[0] if len(item_row) > 0 else "Unknown"
        
        print(f"\n📌 {item_name} (ID: {item_id}, count: {count:,})")
        print("-" * 60)
        
        if item_id in model.wv:
            similar = model.wv.most_similar(item_id, topn=10)
            for i, (sim_id, score) in enumerate(similar, 1):
                sim_row = items_df[items_df["item_id"] == sim_id]
                sim_name = sim_row["item_name"].values[0] if len(sim_row) > 0 else "Unknown"
                print(f"  {i:2d}. {sim_name[:40]:<40} (sim: {score:.4f})")
        else:
            print("  [Item not in vocabulary]")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
