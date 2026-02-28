"""
Embedding Alignment and Feature Integrity Validation

Validates:
1. Candidate item ID → embedding_idx → item_id mapping
2. Cart embedding indices mapping
3. Baseline AUC with simple heuristic
4. Category distribution for positives/negatives
"""

import pickle
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# Paths
TRAINING_DATASET_PATH = Path("data/processed/training_dataset.parquet")
ITEM_MAPPING_PATH = Path("data/processed/item_id_mapping.pkl")
ITEM_FEATURES_PATH = Path("data/processed/item_features.parquet")
MENU_ITEMS_PATH = Path("data/processed/menu_items_enriched.csv")


def load_item_mapping() -> tuple:
    """Load item_id <-> idx mapping."""
    with open(ITEM_MAPPING_PATH, "rb") as f:
        data = pickle.load(f)
    
    if isinstance(data, dict) and "item_to_idx" in data:
        item_to_idx = data["item_to_idx"]
        idx_to_item = {v: k for k, v in item_to_idx.items()}
    elif isinstance(data, list) and len(data) >= 2 and isinstance(data[0], dict):
        idx_to_item = data[0]
        idx_to_item = {int(k): v for k, v in idx_to_item.items()}
        item_to_idx = {v: k for k, v in idx_to_item.items()}
    else:
        idx_to_item = data if isinstance(data, dict) else {}
        idx_to_item = {int(k): v for k, v in idx_to_item.items()}
        item_to_idx = {v: k for k, v in idx_to_item.items()}
    
    return item_to_idx, idx_to_item


def validate_embedding_alignment():
    """
    Validate embedding alignment for 5 random training samples.
    """
    print("=" * 60)
    print("1. VALIDATING EMBEDDING ALIGNMENT (5 random samples)")
    print("=" * 60)
    
    # Load item mapping
    item_to_idx, idx_to_item = load_item_mapping()
    print(f"   Loaded mapping: {len(item_to_idx):,} items")
    
    # Load training data
    df = pd.read_parquet(TRAINING_DATASET_PATH)
    print(f"   Loaded training data: {len(df):,} rows")
    
    # Sample 5 random rows
    random.seed(42)
    sample_indices = random.sample(range(len(df)), 5)
    
    print("\n   Sample Validation:")
    print("-" * 60)
    
    all_match = True
    for i, idx in enumerate(sample_indices):
        row = df.iloc[idx]
        candidate_item = row["candidate_item"]
        embedding_idx = row["embedding_idx"]
        
        # Map embedding_idx back to item_id
        mapped_item = idx_to_item.get(embedding_idx, "NOT_FOUND")
        
        match = "✓ MATCH" if mapped_item == candidate_item else "✗ MISMATCH"
        if mapped_item != candidate_item:
            all_match = False
        
        print(f"   Sample {i+1}:")
        print(f"      candidate_item:  {candidate_item}")
        print(f"      embedding_idx:   {embedding_idx}")
        print(f"      mapped_item:     {mapped_item}")
        print(f"      Result: {match}")
        print()
    
    print(f"   Overall: {'ALL MATCH ✓' if all_match else 'MISMATCHES FOUND ✗'}")
    return all_match


def validate_cart_embedding_indices():
    """
    Validate cart_embedding_indices map correctly to item IDs.
    """
    print("\n" + "=" * 60)
    print("2. VALIDATING CART EMBEDDING INDICES")
    print("=" * 60)
    
    # Load item mapping
    item_to_idx, idx_to_item = load_item_mapping()
    
    # Load training data (just a sample)
    df = pd.read_parquet(TRAINING_DATASET_PATH)
    
    # Find rows with non-empty cart_embedding_indices
    non_empty_carts = df[df["cart_embedding_indices"].apply(lambda x: len(x) > 0 if isinstance(x, list) else False)]
    
    print(f"   Rows with non-empty cart_embedding_indices: {len(non_empty_carts):,}")
    
    if len(non_empty_carts) == 0:
        print("   No cart embedding indices to validate")
        return True
    
    # Sample 3 rows
    random.seed(123)
    sample_indices = random.sample(range(len(non_empty_carts)), min(3, len(non_empty_carts)))
    
    print("\n   Sample Cart Validation:")
    print("-" * 60)
    
    all_valid = True
    for i, idx in enumerate(sample_indices):
        row = non_empty_carts.iloc[idx]
        cart_indices = row["cart_embedding_indices"]
        
        print(f"   Sample {i+1}: session={row['session_id'][:20]}...")
        print(f"      cart_embedding_indices: {cart_indices}")
        
        mapped_items = []
        for emb_idx in cart_indices:
            mapped = idx_to_item.get(emb_idx, "NOT_FOUND")
            mapped_items.append(mapped)
            if mapped == "NOT_FOUND":
                all_valid = False
        
        print(f"      mapped_item_ids: {mapped_items}")
        print(f"      all valid: {'✓' if 'NOT_FOUND' not in mapped_items else '✗'}")
        print()
    
    print(f"   Overall: {'ALL VALID ✓' if all_valid else 'INVALID INDICES FOUND ✗'}")
    return all_valid


def compute_baseline_auc():
    """
    Compute baseline AUC using simple heuristic:
    if missing_beverage == 1 and candidate_category == beverage: score = 1 else 0
    """
    print("\n" + "=" * 60)
    print("3. BASELINE AUC (Heuristic: missing_beverage + beverage candidate)")
    print("=" * 60)
    
    # Load training data
    df = pd.read_parquet(TRAINING_DATASET_PATH)
    
    # Load item features to get categories
    item_features = pd.read_parquet(ITEM_FEATURES_PATH)
    
    # Create category lookup (category_index: 3 = beverage based on CATEGORY_MAP)
    CATEGORY_MAP = {
        "main": 0,
        "starter": 1,
        "dessert": 2,
        "beverage": 3,
        "side": 4,
        "snack": 5,
        "combo": 6,
        "other": 7,
    }
    
    item_to_category_idx = dict(zip(item_features["item_id"], item_features["category_index"]))
    
    # Get candidate category for each row
    df["candidate_category_idx"] = df["candidate_item"].map(item_to_category_idx)
    
    # Compute heuristic score
    # Score = 1 if missing_beverage=1 AND candidate is beverage (category_idx=3)
    df["heuristic_score"] = ((df["missing_beverage"] == 1) & (df["candidate_category_idx"] == 3)).astype(int)
    
    # Get labels
    y_true = df["label"].values
    y_score = df["heuristic_score"].values
    
    # Check if we have both classes
    if len(np.unique(y_true)) < 2:
        print("   Cannot compute AUC: only one class present")
        return None
    
    # Compute AUC
    auc = roc_auc_score(y_true, y_score)
    
    print(f"   Total samples: {len(df):,}")
    print(f"   Heuristic fires (score=1): {df['heuristic_score'].sum():,} ({100*df['heuristic_score'].mean():.2f}%)")
    print(f"   Positives (label=1): {y_true.sum():,}")
    print(f"   Heuristic hits on positives: {df[(df['heuristic_score'] == 1) & (df['label'] == 1)]['heuristic_score'].sum():,}")
    print(f"\n   BASELINE AUC: {auc:.4f}")
    
    return auc


def analyze_category_distribution():
    """
    Print candidate category distribution for positives and negatives.
    """
    print("\n" + "=" * 60)
    print("4. CANDIDATE CATEGORY DISTRIBUTION")
    print("=" * 60)
    
    # Load training data
    df = pd.read_parquet(TRAINING_DATASET_PATH)
    
    # Load item features
    item_features = pd.read_parquet(ITEM_FEATURES_PATH)
    
    CATEGORY_MAP = {
        "main": 0,
        "starter": 1,
        "dessert": 2,
        "beverage": 3,
        "side": 4,
        "snack": 5,
        "combo": 6,
        "other": 7,
    }
    IDX_TO_CATEGORY = {v: k for k, v in CATEGORY_MAP.items()}
    
    item_to_category_idx = dict(zip(item_features["item_id"], item_features["category_index"]))
    
    df["candidate_category_idx"] = df["candidate_item"].map(item_to_category_idx)
    df["candidate_category"] = df["candidate_category_idx"].map(IDX_TO_CATEGORY)
    
    # Split by label
    positives = df[df["label"] == 1]
    negatives = df[df["label"] == 0]
    
    print(f"\n   POSITIVES (n={len(positives):,}):")
    print("-" * 40)
    pos_dist = positives["candidate_category"].value_counts()
    for cat, count in pos_dist.items():
        pct = 100 * count / len(positives)
        print(f"      {cat:12s}: {count:>8,} ({pct:>5.1f}%)")
    
    print(f"\n   NEGATIVES (n={len(negatives):,}):")
    print("-" * 40)
    neg_dist = negatives["candidate_category"].value_counts()
    for cat, count in neg_dist.items():
        pct = 100 * count / len(negatives)
        print(f"      {cat:12s}: {count:>8,} ({pct:>5.1f}%)")
    
    # Compute lift (positive rate by category)
    print(f"\n   CATEGORY LIFT (Positive Rate):")
    print("-" * 40)
    for cat in CATEGORY_MAP.keys():
        cat_rows = df[df["candidate_category"] == cat]
        if len(cat_rows) > 0:
            pos_rate = cat_rows["label"].mean()
            print(f"      {cat:12s}: {100*pos_rate:>5.1f}%")
    
    return pos_dist, neg_dist


def main():
    print("\n" + "=" * 70)
    print("  EMBEDDING ALIGNMENT AND FEATURE INTEGRITY VALIDATION")
    print("=" * 70 + "\n")
    
    # 1. Validate embedding alignment
    alignment_ok = validate_embedding_alignment()
    
    # 2. Validate cart embedding indices
    cart_ok = validate_cart_embedding_indices()
    
    # 3. Compute baseline AUC
    baseline_auc = compute_baseline_auc()
    
    # 4. Category distribution
    pos_dist, neg_dist = analyze_category_distribution()
    
    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"   Embedding alignment: {'PASS ✓' if alignment_ok else 'FAIL ✗'}")
    print(f"   Cart indices valid:  {'PASS ✓' if cart_ok else 'FAIL ✗'}")
    print(f"   Baseline AUC:        {baseline_auc:.4f}" if baseline_auc else "   Baseline AUC:        N/A")
    print("=" * 60)
    
    return 0 if (alignment_ok and cart_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
