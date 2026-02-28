"""
Feature Engineering Module for Add-On Recommendation System

Builds training features from cart events, user data, restaurant data,
and item embeddings. Processes large files in chunks to fit in memory.

Key Features:
- Handles zero embeddings via category centroid imputation
- Modular feature tables (user, restaurant, item)
- Chunked processing for 15M+ row cart_events.csv
- Memory-efficient: stores embedding indices, not vectors

Usage:
    python -m src.features.feature_builder
    python -m src.features.feature_builder --chunk-size 250000
"""

import argparse
import gc
import logging
import pickle
import psutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

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

# Input paths
CART_EVENTS_PATH = Path("data/synthetic/cart_events.csv")
RESTAURANTS_PATH = Path("data/processed/restaurants_cleaned.csv")
MENU_ITEMS_PATH = Path("data/processed/menu_items_enriched.csv")
EMBEDDINGS_PATH = Path("data/processed/item_embeddings.npy")
MAPPING_PATH = Path("data/processed/item_id_mapping.pkl")
USERS_PATH = Path("data/synthetic/users.csv")

# Output paths
OUTPUT_DIR = Path("data/processed")
TRAINING_DATASET_PATH = OUTPUT_DIR / "training_dataset.parquet"
USER_FEATURES_PATH = OUTPUT_DIR / "user_features.parquet"
RESTAURANT_FEATURES_PATH = OUTPUT_DIR / "restaurant_features.parquet"
ITEM_FEATURES_PATH = OUTPUT_DIR / "item_features.parquet"
FIXED_EMBEDDINGS_PATH = OUTPUT_DIR / "item_embeddings_fixed.npy"
FEATURE_SCALERS_PATH = OUTPUT_DIR / "feature_scalers.pkl"

# Processing
DEFAULT_CHUNK_SIZE = 500_000

# Category mappings
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


# -----------------------------------------------------------------------------
# Memory Utilities
# -----------------------------------------------------------------------------

def log_memory_usage(stage: str = "") -> float:
    """Log current memory usage and return GB used."""
    process = psutil.Process()
    mem_info = process.memory_info()
    mem_gb = mem_info.rss / (1024 ** 3)
    logger.info(f"Memory [{stage}]: {mem_gb:.2f} GB")
    return mem_gb


def force_gc():
    """Force garbage collection."""
    gc.collect()


# -----------------------------------------------------------------------------
# PART 1: Handle Zero Embeddings
# -----------------------------------------------------------------------------

def fix_zero_embeddings(
    embeddings: np.ndarray,
    item_to_idx: Dict[str, int],
    items_df: pd.DataFrame,
) -> Tuple[np.ndarray, Dict]:
    """
    Replace zero embeddings with category centroid embeddings.
    
    Args:
        embeddings: (num_items, dim) embedding matrix
        item_to_idx: item_id -> index mapping
        items_df: DataFrame with item_id, item_category columns
        
    Returns:
        fixed_embeddings: Fixed embedding matrix
        stats: Dictionary with fix statistics
    """
    logger.info("=" * 60)
    logger.info("PART 1: Fixing Zero Embeddings")
    logger.info("=" * 60)
    
    num_items, dim = embeddings.shape
    fixed_embeddings = embeddings.copy()
    
    # Identify zero embedding rows
    zero_mask = np.all(embeddings == 0, axis=1)
    num_zero = zero_mask.sum()
    logger.info(f"Zero embeddings detected: {num_zero:,} ({100*num_zero/num_items:.1f}%)")
    
    if num_zero == 0:
        logger.info("No zero embeddings to fix")
        return fixed_embeddings, {"zero_count": 0, "replaced_count": 0}
    
    # Build category -> indices mapping
    category_indices: Dict[str, List[int]] = defaultdict(list)
    
    for _, row in items_df.iterrows():
        item_id = row["item_id"]
        category = row.get("item_category", "other")
        if item_id in item_to_idx:
            idx = item_to_idx[item_id]
            category_indices[category].append(idx)
    
    # Compute category centroids from non-zero embeddings
    category_centroids: Dict[str, np.ndarray] = {}
    
    for category, indices in category_indices.items():
        indices_arr = np.array(indices)
        non_zero_mask = ~zero_mask[indices_arr]
        non_zero_indices = indices_arr[non_zero_mask]
        
        if len(non_zero_indices) > 0:
            centroid = embeddings[non_zero_indices].mean(axis=0)
            category_centroids[category] = centroid
            logger.info(f"  Category '{category}': {len(non_zero_indices):,} non-zero items")
    
    # Global centroid fallback
    global_non_zero = embeddings[~zero_mask]
    global_centroid = global_non_zero.mean(axis=0) if len(global_non_zero) > 0 else np.zeros(dim)
    
    # Replace zero embeddings with category centroids
    replaced_count = 0
    idx_to_category: Dict[int, str] = {}
    
    for category, indices in category_indices.items():
        for idx in indices:
            idx_to_category[idx] = category
    
    for idx in np.where(zero_mask)[0]:
        category = idx_to_category.get(idx, "other")
        centroid = category_centroids.get(category, global_centroid)
        fixed_embeddings[idx] = centroid
        replaced_count += 1
    
    logger.info(f"Replaced {replaced_count:,} zero embeddings with category centroids")
    
    # Normalize embeddings (L2 normalization)
    norms = np.linalg.norm(fixed_embeddings, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1)  # Avoid division by zero
    fixed_embeddings = fixed_embeddings / norms
    logger.info("Applied L2 normalization to all embeddings")
    
    # Verify
    new_zero_mask = np.all(fixed_embeddings == 0, axis=1)
    remaining_zero = new_zero_mask.sum()
    logger.info(f"Remaining zero embeddings: {remaining_zero:,}")
    
    stats = {
        "zero_count": int(num_zero),
        "replaced_count": replaced_count,
        "remaining_zero": int(remaining_zero),
        "categories_used": list(category_centroids.keys()),
    }
    
    return fixed_embeddings, stats


# -----------------------------------------------------------------------------
# Feature Normalization Utilities
# -----------------------------------------------------------------------------

def compute_scaler_stats(
    user_df: pd.DataFrame,
    restaurant_df: pd.DataFrame,
) -> Dict:
    """
    Compute normalization statistics from feature DataFrames.
    
    Strategy:
    1. Apply log1p to skewed features (recency, frequency, monetary, votes, price, menu_size)
    2. Compute mean/std for z-score standardization
    3. Store min/max for bounded features (rating, price_band)
    
    Returns:
        Dictionary with scaler statistics for each feature
    """
    logger.info("Computing scaler statistics...")
    
    scalers = {
        "user": {},
        "restaurant": {},
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "user_count": len(user_df),
            "restaurant_count": len(restaurant_df),
        }
    }
    
    # User features requiring log1p + z-score
    user_log_features = ["recency_days", "frequency", "monetary_avg"]
    for feat in user_log_features:
        if feat in user_df.columns:
            values = user_df[feat].fillna(0).clip(lower=0)
            log_values = np.log1p(values)
            scalers["user"][feat] = {
                "transform": "log1p_zscore",
                "mean": float(log_values.mean()),
                "std": float(log_values.std()) if log_values.std() > 0 else 1.0,
            }
            logger.info(f"  user.{feat}: log1p -> mean={scalers['user'][feat]['mean']:.4f}, std={scalers['user'][feat]['std']:.4f}")
    
    # User features already in 0-1 range (keep as-is)
    user_ratio_features = ["cuisine_entropy", "dessert_ratio", "beverage_ratio"]
    for feat in user_ratio_features:
        if feat in user_df.columns:
            scalers["user"][feat] = {
                "transform": "passthrough",
            }
    
    # avg_cart_size: light normalization (typically 1-6)
    if "avg_cart_size" in user_df.columns:
        values = user_df["avg_cart_size"].fillna(2).clip(lower=0)
        log_values = np.log1p(values)
        scalers["user"]["avg_cart_size"] = {
            "transform": "log1p_zscore",
            "mean": float(log_values.mean()),
            "std": float(log_values.std()) if log_values.std() > 0 else 1.0,
        }
        logger.info(f"  user.avg_cart_size: log1p -> mean={scalers['user']['avg_cart_size']['mean']:.4f}")
    
    # Restaurant features requiring log1p + z-score
    restaurant_log_features = ["delivery_votes", "avg_item_price", "menu_size"]
    for feat in restaurant_log_features:
        if feat in restaurant_df.columns:
            values = restaurant_df[feat].fillna(0).clip(lower=0)
            log_values = np.log1p(values)
            scalers["restaurant"][feat] = {
                "transform": "log1p_zscore",
                "mean": float(log_values.mean()),
                "std": float(log_values.std()) if log_values.std() > 0 else 1.0,
            }
            logger.info(f"  restaurant.{feat}: log1p -> mean={scalers['restaurant'][feat]['mean']:.4f}, std={scalers['restaurant'][feat]['std']:.4f}")
    
    # smoothed_rating: scale to 0-1 (typical range 3.0-4.5)
    if "smoothed_rating" in restaurant_df.columns:
        values = restaurant_df["smoothed_rating"].fillna(3.5)
        scalers["restaurant"]["smoothed_rating"] = {
            "transform": "minmax",
            "min": 3.0,  # Fixed bounds for rating
            "max": 5.0,
        }
        logger.info(f"  restaurant.smoothed_rating: minmax [3.0, 5.0] -> [0, 1]")
    
    # price_band_index: divide by 4 (range 0-4)
    if "price_band_index" in restaurant_df.columns:
        scalers["restaurant"]["price_band_index"] = {
            "transform": "divide",
            "divisor": 4.0,
        }
        logger.info(f"  restaurant.price_band_index: divide by 4 -> [0, 1]")
    
    return scalers


def normalize_user_features(
    df: pd.DataFrame,
    scalers: Dict,
) -> pd.DataFrame:
    """
    Apply normalization to user features using pre-computed scalers.
    
    Args:
        df: User features DataFrame with raw values
        scalers: Dictionary with scaler statistics
    
    Returns:
        DataFrame with normalized features (suffixed with _norm)
    """
    result = df.copy()
    user_scalers = scalers.get("user", {})
    
    for feat, config in user_scalers.items():
        if feat not in result.columns:
            continue
        
        transform = config.get("transform", "passthrough")
        col_name = f"{feat}_norm"
        
        if transform == "log1p_zscore":
            values = result[feat].fillna(0).clip(lower=0)
            log_values = np.log1p(values)
            mean = config["mean"]
            std = config["std"]
            result[col_name] = (log_values - mean) / std
        elif transform == "passthrough":
            result[col_name] = result[feat].fillna(0.5)  # Default for ratios
        elif transform == "minmax":
            min_val = config["min"]
            max_val = config["max"]
            values = result[feat].fillna((min_val + max_val) / 2)
            result[col_name] = (values - min_val) / (max_val - min_val)
        elif transform == "divide":
            divisor = config["divisor"]
            result[col_name] = result[feat].fillna(0) / divisor
    
    # Ensure no NaN/inf values
    norm_cols = [c for c in result.columns if c.endswith("_norm")]
    result[norm_cols] = result[norm_cols].replace([np.inf, -np.inf], 0).fillna(0)
    
    return result


def normalize_restaurant_features(
    df: pd.DataFrame,
    scalers: Dict,
) -> pd.DataFrame:
    """
    Apply normalization to restaurant features using pre-computed scalers.
    
    Args:
        df: Restaurant features DataFrame with raw values
        scalers: Dictionary with scaler statistics
    
    Returns:
        DataFrame with normalized features (suffixed with _norm)
    """
    result = df.copy()
    restaurant_scalers = scalers.get("restaurant", {})
    
    for feat, config in restaurant_scalers.items():
        if feat not in result.columns:
            continue
        
        transform = config.get("transform", "passthrough")
        col_name = f"{feat}_norm"
        
        if transform == "log1p_zscore":
            values = result[feat].fillna(0).clip(lower=0)
            log_values = np.log1p(values)
            mean = config["mean"]
            std = config["std"]
            result[col_name] = (log_values - mean) / std
        elif transform == "passthrough":
            result[col_name] = result[feat].fillna(0)
        elif transform == "minmax":
            min_val = config["min"]
            max_val = config["max"]
            values = result[feat].fillna((min_val + max_val) / 2)
            result[col_name] = (values - min_val) / (max_val - min_val)
        elif transform == "divide":
            divisor = config["divisor"]
            result[col_name] = result[feat].fillna(0) / divisor
    
    # Ensure no NaN/inf values
    norm_cols = [c for c in result.columns if c.endswith("_norm")]
    result[norm_cols] = result[norm_cols].replace([np.inf, -np.inf], 0).fillna(0)
    
    return result


# -----------------------------------------------------------------------------
# PART 2: Build Feature Tables
# -----------------------------------------------------------------------------

def build_user_features(
    cart_events_path: Path,
    users_df: pd.DataFrame,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> pd.DataFrame:
    """
    Build user feature table from cart events and user data.
    
    Features:
    - recency_days: days since last order
    - frequency: total sessions
    - monetary_avg: average cart total
    - cuisine_entropy: diversity of cuisines ordered
    - avg_cart_size: average items per cart
    - dessert_ratio: fraction of dessert items
    - beverage_ratio: fraction of beverage items
    """
    logger.info("-" * 40)
    logger.info("Building User Features")
    logger.info("-" * 40)
    
    # Aggregators
    user_stats: Dict[str, Dict] = defaultdict(lambda: {
        "sessions": set(),
        "cart_totals": [],
        "cart_sizes": [],
        "timestamps": [],
        "categories": defaultdict(int),
        "cuisines": set(),
    })
    
    # Process cart events in chunks
    chunk_num = 0
    for chunk in pd.read_csv(cart_events_path, chunksize=chunk_size):
        chunk_num += 1
        if chunk_num % 10 == 1:
            logger.info(f"  Processing chunk {chunk_num}...")
        
        # Only process positive labels (actual cart additions)
        positive = chunk[chunk["label"] == 1]
        
        for _, row in positive.iterrows():
            user_id = row["user_id"]
            stats = user_stats[user_id]
            stats["sessions"].add(row["session_id"])
            stats["timestamps"].append(row["timestamp"])
            
            # Cart info (from last step of session)
            if pd.notna(row.get("cart_total")):
                stats["cart_totals"].append(float(row["cart_total"]))
            
            # Count categories
            cart_cats = row.get("cart_categories", "")
            if pd.notna(cart_cats) and cart_cats:
                for cat in str(cart_cats).split("|"):
                    cat = cat.strip().lower()
                    if cat:
                        stats["categories"][cat] += 1
    
    logger.info(f"  Processed {chunk_num} chunks, {len(user_stats):,} users")
    
    # Compute features
    reference_date = datetime(2026, 2, 28)
    user_features = []
    
    for user_id, stats in user_stats.items():
        # Recency
        if stats["timestamps"]:
            try:
                last_ts = max(pd.to_datetime(ts) for ts in stats["timestamps"])
                recency_days = (reference_date - last_ts.to_pydatetime().replace(tzinfo=None)).days
            except:
                recency_days = 30
        else:
            recency_days = 30
        
        # Frequency
        frequency = len(stats["sessions"])
        
        # Monetary
        monetary_avg = np.mean(stats["cart_totals"]) if stats["cart_totals"] else 0.0
        
        # Category ratios
        total_items = sum(stats["categories"].values())
        dessert_ratio = stats["categories"].get("dessert", 0) / max(total_items, 1)
        beverage_ratio = stats["categories"].get("beverage", 0) / max(total_items, 1)
        
        # Cuisine entropy (from user data)
        user_row = users_df[users_df["user_id"] == user_id]
        if len(user_row) > 0:
            cuisines = str(user_row["preferred_cuisines"].values[0]).split("|")
            cuisine_entropy = min(len(cuisines), 5) / 5.0  # Normalized
        else:
            cuisine_entropy = 0.5
        
        # Average cart size (estimate from cart_totals count per session)
        avg_cart_size = len(stats["cart_totals"]) / max(frequency, 1)
        
        user_features.append({
            "user_id": user_id,
            "recency_days": recency_days,
            "frequency": frequency,
            "monetary_avg": round(monetary_avg, 2),
            "cuisine_entropy": round(cuisine_entropy, 4),
            "avg_cart_size": round(avg_cart_size, 2),
            "dessert_ratio": round(dessert_ratio, 4),
            "beverage_ratio": round(beverage_ratio, 4),
        })
    
    df = pd.DataFrame(user_features)
    
    # Clip extreme values
    df["recency_days"] = df["recency_days"].clip(lower=0, upper=365)
    df["frequency"] = df["frequency"].clip(lower=1, upper=1000)
    df["monetary_avg"] = df["monetary_avg"].clip(lower=0, upper=10000)
    df["avg_cart_size"] = df["avg_cart_size"].clip(lower=0, upper=20)
    
    logger.info(f"  User features: {len(df):,} rows, {len(df.columns)} columns")
    logger.info(f"  Feature stats:")
    for col in ["recency_days", "frequency", "monetary_avg", "avg_cart_size"]:
        logger.info(f"    {col}: mean={df[col].mean():.2f}, std={df[col].std():.2f}, min={df[col].min():.2f}, max={df[col].max():.2f}")
    
    return df


def build_restaurant_features(restaurants_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build restaurant feature table.
    
    Features:
    - smoothed_rating: Bayesian smoothed rating
    - delivery_votes: Total delivery votes
    - avg_item_price: Average price of items
    - price_band_index: Price tier (0-4)
    """
    logger.info("-" * 40)
    logger.info("Building Restaurant Features")
    logger.info("-" * 40)
    
    # Compute avg item price per restaurant
    item_prices = items_df.groupby("restaurant_id")["price"].agg(["mean", "std", "count"]).reset_index()
    item_prices.columns = ["restaurant_id", "avg_item_price", "price_std", "menu_size"]
    
    # Merge with restaurant data
    df = restaurants_df[["restaurant_id", "smoothed_rating", "delivery_votes"]].copy()
    df = df.merge(item_prices, on="restaurant_id", how="left")
    
    # Fill missing values
    df["avg_item_price"] = df["avg_item_price"].fillna(df["avg_item_price"].median())
    df["price_std"] = df["price_std"].fillna(0)
    df["menu_size"] = df["menu_size"].fillna(0).astype(int)
    
    # Price band (quintiles)
    df["price_band_index"] = pd.qcut(
        df["avg_item_price"].clip(lower=0),
        q=5,
        labels=[0, 1, 2, 3, 4],
        duplicates="drop"
    ).astype(int)
    
    # Clip extreme values
    df["delivery_votes"] = df["delivery_votes"].clip(lower=0, upper=100000)
    df["avg_item_price"] = df["avg_item_price"].clip(lower=0, upper=5000)
    df["menu_size"] = df["menu_size"].clip(lower=0, upper=500)
    df["smoothed_rating"] = df["smoothed_rating"].clip(lower=1.0, upper=5.0)
    df["price_band_index"] = df["price_band_index"].clip(lower=0, upper=4)
    
    # Select final columns (raw values - normalization applied separately)
    df = df[[
        "restaurant_id",
        "smoothed_rating",
        "delivery_votes",
        "avg_item_price",
        "price_band_index",
        "menu_size",
    ]]
    
    logger.info(f"  Restaurant features: {len(df):,} rows, {len(df.columns)} columns")
    logger.info(f"  Feature stats:")
    for col in ["smoothed_rating", "delivery_votes", "avg_item_price", "menu_size"]:
        logger.info(f"    {col}: mean={df[col].mean():.2f}, std={df[col].std():.2f}, min={df[col].min():.2f}, max={df[col].max():.2f}")
    
    return df


def build_item_features(
    items_df: pd.DataFrame,
    item_to_idx: Dict[str, int],
) -> pd.DataFrame:
    """
    Build item feature table.
    
    Features:
    - embedding_idx: Index into embedding matrix
    - normalized_price: Z-score normalized price
    - best_seller: Best seller flag
    - category_index: Numeric category index
    - veg_flag: Vegetarian indicator
    """
    logger.info("-" * 40)
    logger.info("Building Item Features")
    logger.info("-" * 40)
    
    # Build features
    item_features = []
    
    price_mean = items_df["price"].mean()
    price_std = items_df["price"].std()
    
    for _, row in items_df.iterrows():
        item_id = row["item_id"]
        
        # Embedding index
        embedding_idx = item_to_idx.get(item_id, -1)
        
        # Normalized price
        price = row.get("price", 0)
        normalized_price = (price - price_mean) / max(price_std, 1)
        
        # Category index
        category = str(row.get("item_category", "other")).lower()
        category_index = CATEGORY_MAP.get(category, CATEGORY_MAP["other"])
        
        item_features.append({
            "item_id": item_id,
            "restaurant_id": row["restaurant_id"],
            "embedding_idx": embedding_idx,
            "price": round(price, 2),
            "normalized_price": round(normalized_price, 4),
            "best_seller": int(row.get("best_seller", 0)),
            "category_index": category_index,
            "veg_flag": int(row.get("veg_flag", 0)),
        })
    
    df = pd.DataFrame(item_features)
    
    # Log coverage
    has_embedding = (df["embedding_idx"] >= 0).sum()
    logger.info(f"  Items with embeddings: {has_embedding:,}/{len(df):,}")
    logger.info(f"  Item features: {len(df):,} rows, {len(df.columns)} columns")
    
    return df


# -----------------------------------------------------------------------------
# PART 3: Cart Dynamic Features (Chunked Processing)
# -----------------------------------------------------------------------------

def compute_cart_dynamic_features(
    row: pd.Series,
    item_categories: Dict[str, str],
    item_embedding_idx: Dict[str, int],
) -> Dict:
    """Compute dynamic features for a single cart row including cart embedding indices."""
    # Parse cart items
    cart_items_str = row.get("cart_items", "")
    if pd.isna(cart_items_str) or not cart_items_str:
        cart_items = []
    else:
        cart_items = [item.strip() for item in str(cart_items_str).split("|") if item.strip()]
    
    cart_size = len(cart_items)
    cart_value = float(row.get("cart_total", 0))
    
    # Get embedding indices for cart items (for sequential modeling)
    # Returns list of ints, empty list for empty cart
    cart_embedding_indices = []
    for item_id in cart_items:
        idx = item_embedding_idx.get(item_id, -1)
        if idx >= 0:  # Only include valid indices
            cart_embedding_indices.append(idx)
    
    # Get categories of cart items
    cart_categories = set()
    for item_id in cart_items:
        cat = item_categories.get(item_id, "other")
        cart_categories.add(cat)
    
    # Missing flags
    missing_beverage = 1 if "beverage" not in cart_categories else 0
    missing_dessert = 1 if "dessert" not in cart_categories else 0
    
    # Heavy meal flag (multiple mains or large cart)
    main_count = sum(1 for item_id in cart_items 
                     if item_categories.get(item_id, "other") == "main")
    heavy_meal = 1 if main_count >= 2 or cart_size >= 4 else 0
    
    return {
        "cart_size": cart_size,
        "cart_value": round(cart_value, 2),
        "missing_beverage": missing_beverage,
        "missing_dessert": missing_dessert,
        "heavy_meal": heavy_meal,
        "cart_embedding_indices": cart_embedding_indices,  # List[int] for sequential modeling
    }


def process_cart_events_chunked(
    cart_events_path: Path,
    item_features_df: pd.DataFrame,
    user_features_df: pd.DataFrame,
    restaurant_features_df: pd.DataFrame,
    output_path: Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> Dict:
    """
    Process cart events in chunks and build training dataset.
    
    Outputs parquet with:
    - session_id, user_id, restaurant_id, candidate_item, label
    - user_id (for feature lookup)
    - restaurant_id (for feature lookup)
    - item_id (embedding_idx lookup)
    - cart dynamic features
    - context features
    """
    logger.info("=" * 60)
    logger.info("PART 3 & 4: Processing Cart Events (Chunked)")
    logger.info("=" * 60)
    
    # Build lookup dictionaries
    item_categories = dict(zip(
        item_features_df["item_id"],
        item_features_df["category_index"].map({v: k for k, v in CATEGORY_MAP.items()})
    ))
    
    item_embedding_idx = dict(zip(
        item_features_df["item_id"],
        item_features_df["embedding_idx"]
    ))
    
    user_ids_set = set(user_features_df["user_id"])
    restaurant_ids_set = set(restaurant_features_df["restaurant_id"])
    
    # Process in chunks
    chunk_num = 0
    total_rows = 0
    first_chunk = True
    stats = {
        "total_rows": 0,
        "positive_rows": 0,
        "negative_rows": 0,
        "skipped_rows": 0,
    }
    
    log_memory_usage("Before chunked processing")
    
    for chunk in pd.read_csv(cart_events_path, chunksize=chunk_size):
        chunk_num += 1
        chunk_start = (chunk_num - 1) * chunk_size
        
        if chunk_num % 5 == 1:
            logger.info(f"  Processing chunk {chunk_num} (rows {chunk_start:,}-{chunk_start + len(chunk):,})...")
            log_memory_usage(f"Chunk {chunk_num}")
        
        # Build output rows
        output_rows = []
        
        for _, row in chunk.iterrows():
            # Basic validation
            user_id = row["user_id"]
            restaurant_id = row["restaurant_id"]
            candidate_item = row["candidate_item"]
            
            if pd.isna(candidate_item):
                stats["skipped_rows"] += 1
                continue
            
            # Compute cart dynamic features (pass item_embedding_idx for cart sequence)
            cart_features = compute_cart_dynamic_features(row, item_categories, item_embedding_idx)
            
            # Get embedding index for candidate item
            embedding_idx = item_embedding_idx.get(candidate_item, -1)
            
            # Context features
            hour = int(row.get("hour", 12))
            meal_type = str(row.get("meal_type", "lunch"))
            user_type = str(row.get("user_type", "moderate"))
            
            # Encode context
            meal_type_idx = {"breakfast": 0, "lunch": 1, "dinner": 2, "snack": 3}.get(meal_type, 1)
            user_type_idx = {"budget": 0, "moderate": 1, "premium": 2}.get(user_type, 1)
            
            output_rows.append({
                "session_id": row["session_id"],
                "step_number": int(row["step_number"]),
                "user_id": user_id,
                "restaurant_id": restaurant_id,
                "candidate_item": candidate_item,
                "label": int(row["label"]),
                # Cart dynamic features
                "cart_size": cart_features["cart_size"],
                "cart_value": cart_features["cart_value"],
                "missing_beverage": cart_features["missing_beverage"],
                "missing_dessert": cart_features["missing_dessert"],
                "heavy_meal": cart_features["heavy_meal"],
                # Cart sequence (for sequential modeling)
                "cart_embedding_indices": cart_features["cart_embedding_indices"],
                # Item reference
                "embedding_idx": embedding_idx,
                # Context features
                "hour": hour,
                "meal_type_idx": meal_type_idx,
                "user_type_idx": user_type_idx,
            })
            
            if row["label"] == 1:
                stats["positive_rows"] += 1
            else:
                stats["negative_rows"] += 1
        
        # Convert to DataFrame and write to parquet with Arrow list support
        chunk_df = pd.DataFrame(output_rows)
        total_rows += len(chunk_df)
        
        # Convert to PyArrow Table to properly handle list column
        table = pa.Table.from_pandas(chunk_df, preserve_index=False)
        
        if first_chunk:
            pq.write_table(table, output_path)
            first_chunk = False
        else:
            # Write to separate files and concatenate at end
            chunk_path = output_path.parent / f"_chunk_{chunk_num}.parquet"
            pq.write_table(table, chunk_path)
        
        # Clear memory
        del table
        del chunk_df
        del output_rows
        force_gc()
    
    stats["total_rows"] = total_rows
    
    # Concatenate all chunk files using PyArrow for memory efficiency
    logger.info("Concatenating chunk files...")
    chunk_files = sorted(output_path.parent.glob("_chunk_*.parquet"))
    
    if chunk_files:
        # Read all tables
        tables = [pq.read_table(output_path)]
        for cf in chunk_files:
            tables.append(pq.read_table(cf))
            cf.unlink()  # Delete chunk file
        
        # Concatenate and write
        final_table = pa.concat_tables(tables)
        pq.write_table(final_table, output_path)
        
        del tables
        del final_table
        force_gc()
    
    log_memory_usage("After chunked processing")
    
    return stats


# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------

def validate_output(
    training_path: Path,
    user_features_path: Path,
    restaurant_features_path: Path,
    item_features_path: Path,
) -> Dict:
    """Validate final outputs for null values and consistency."""
    logger.info("=" * 60)
    logger.info("Validating Outputs")
    logger.info("=" * 60)
    
    results = {}
    
    # Check training dataset
    logger.info("Checking training dataset...")
    train_df = pd.read_parquet(training_path)
    null_counts = train_df.isnull().sum()
    has_nulls = null_counts.sum() > 0
    
    results["training"] = {
        "rows": len(train_df),
        "columns": len(train_df.columns),
        "null_values": int(null_counts.sum()),
        "has_nulls": has_nulls,
        "column_nulls": {col: int(v) for col, v in null_counts.items() if v > 0},
    }
    
    if has_nulls:
        logger.warning(f"  Training dataset has null values: {dict(null_counts[null_counts > 0])}")
    else:
        logger.info(f"  Training dataset: {len(train_df):,} rows, no nulls")
    
    # Validate cart_embedding_indices column
    if "cart_embedding_indices" in train_df.columns:
        null_lists = train_df["cart_embedding_indices"].isnull().sum()
        empty_lists = (train_df["cart_embedding_indices"].apply(len) == 0).sum()
        non_empty = len(train_df) - empty_lists
        logger.info(f"  cart_embedding_indices: {null_lists} nulls, {empty_lists:,} empty, {non_empty:,} non-empty")
    
    # Print example row
    logger.info("  Example row:")
    example = train_df.iloc[0].to_dict()
    for k, v in example.items():
        logger.info(f"    {k}: {v}")
    
    # Print example with non-empty cart
    non_empty_mask = train_df["cart_embedding_indices"].apply(len) > 0
    if non_empty_mask.any():
        logger.info("  Example row with cart items:")
        example_cart = train_df[non_empty_mask].iloc[0].to_dict()
        for k, v in example_cart.items():
            logger.info(f"    {k}: {v}")
    
    del train_df
    force_gc()
    
    # Check feature tables
    for name, path in [
        ("user_features", user_features_path),
        ("restaurant_features", restaurant_features_path),
        ("item_features", item_features_path),
    ]:
        df = pd.read_parquet(path)
        null_count = df.isnull().sum().sum()
        results[name] = {
            "rows": len(df),
            "columns": len(df.columns),
            "null_values": int(null_count),
        }
        logger.info(f"  {name}: {len(df):,} rows, {null_count} nulls")
        del df
    
    force_gc()
    
    return results


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build features for add-on recommendation")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="Chunk size for processing")
    parser.add_argument("--skip-embeddings", action="store_true", help="Skip embedding fix step")
    args = parser.parse_args()
    
    start_time = datetime.now()
    
    logger.info("=" * 60)
    logger.info("Feature Engineering Pipeline")
    logger.info("=" * 60)
    logger.info(f"Chunk size: {args.chunk_size:,}")
    log_memory_usage("Start")
    
    # ---------------------------------------------------------------------
    # Load base data
    # ---------------------------------------------------------------------
    logger.info("Loading base data...")
    
    # Load embeddings and mappings
    embeddings = np.load(EMBEDDINGS_PATH)
    with open(MAPPING_PATH, "rb") as f:
        mapping_data = pickle.load(f)
    item_to_idx = mapping_data["item_to_idx"]
    idx_to_item = mapping_data["idx_to_item"]
    
    logger.info(f"  Embeddings shape: {embeddings.shape}")
    
    # Load DataFrames
    items_df = pd.read_csv(MENU_ITEMS_PATH)
    restaurants_df = pd.read_csv(RESTAURANTS_PATH)
    users_df = pd.read_csv(USERS_PATH)
    
    logger.info(f"  Items: {len(items_df):,}")
    logger.info(f"  Restaurants: {len(restaurants_df):,}")
    logger.info(f"  Users: {len(users_df):,}")
    
    log_memory_usage("After loading base data")
    
    # ---------------------------------------------------------------------
    # PART 1: Fix Zero Embeddings
    # ---------------------------------------------------------------------
    if not args.skip_embeddings:
        fixed_embeddings, embedding_stats = fix_zero_embeddings(embeddings, item_to_idx, items_df)
        np.save(FIXED_EMBEDDINGS_PATH, fixed_embeddings)
        logger.info(f"Saved fixed embeddings to {FIXED_EMBEDDINGS_PATH}")
        
        # Use fixed embeddings going forward
        embeddings = fixed_embeddings
        del fixed_embeddings
        force_gc()
    
    log_memory_usage("After embedding fix")
    
    # ---------------------------------------------------------------------
    # PART 2: Build Feature Tables
    # ---------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("PART 2: Building Feature Tables")
    logger.info("=" * 60)
    
    # User features (raw)
    user_features_df = build_user_features(CART_EVENTS_PATH, users_df, args.chunk_size)
    log_memory_usage("After user features")
    
    # Restaurant features (raw)
    restaurant_features_df = build_restaurant_features(restaurants_df, items_df)
    
    # Item features
    item_features_df = build_item_features(items_df, item_to_idx)
    item_features_df.to_parquet(ITEM_FEATURES_PATH, engine="pyarrow", index=False)
    logger.info(f"Saved item features to {ITEM_FEATURES_PATH}")
    
    # ---------------------------------------------------------------------
    # PART 2.5: Compute & Apply Feature Normalization
    # ---------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("PART 2.5: Computing Feature Normalization")
    logger.info("=" * 60)
    
    # Compute scaler statistics from full dataset
    scalers = compute_scaler_stats(user_features_df, restaurant_features_df)
    
    # Save scalers for inference
    with open(FEATURE_SCALERS_PATH, "wb") as f:
        pickle.dump(scalers, f)
    logger.info(f"Saved feature scalers to {FEATURE_SCALERS_PATH}")
    
    # Apply normalization to user features
    logger.info("Applying normalization to user features...")
    user_features_df = normalize_user_features(user_features_df, scalers)
    
    # Validate no nulls in normalized columns
    user_norm_cols = [c for c in user_features_df.columns if c.endswith("_norm")]
    user_null_count = user_features_df[user_norm_cols].isnull().sum().sum()
    if user_null_count > 0:
        logger.warning(f"  User normalized features have {user_null_count} null values!")
    else:
        logger.info(f"  User normalized features: {len(user_norm_cols)} columns, no nulls")
    
    # Log normalized feature stats
    logger.info("  Normalized user feature stats:")
    for col in user_norm_cols:
        logger.info(f"    {col}: mean={user_features_df[col].mean():.4f}, std={user_features_df[col].std():.4f}, min={user_features_df[col].min():.4f}, max={user_features_df[col].max():.4f}")
    
    user_features_df.to_parquet(USER_FEATURES_PATH, engine="pyarrow", index=False)
    logger.info(f"Saved user features to {USER_FEATURES_PATH}")
    
    # Apply normalization to restaurant features
    logger.info("Applying normalization to restaurant features...")
    restaurant_features_df = normalize_restaurant_features(restaurant_features_df, scalers)
    
    # Validate no nulls in normalized columns
    restaurant_norm_cols = [c for c in restaurant_features_df.columns if c.endswith("_norm")]
    restaurant_null_count = restaurant_features_df[restaurant_norm_cols].isnull().sum().sum()
    if restaurant_null_count > 0:
        logger.warning(f"  Restaurant normalized features have {restaurant_null_count} null values!")
    else:
        logger.info(f"  Restaurant normalized features: {len(restaurant_norm_cols)} columns, no nulls")
    
    # Log normalized feature stats
    logger.info("  Normalized restaurant feature stats:")
    for col in restaurant_norm_cols:
        logger.info(f"    {col}: mean={restaurant_features_df[col].mean():.4f}, std={restaurant_features_df[col].std():.4f}, min={restaurant_features_df[col].min():.4f}, max={restaurant_features_df[col].max():.4f}")
    
    restaurant_features_df.to_parquet(RESTAURANT_FEATURES_PATH, engine="pyarrow", index=False)
    logger.info(f"Saved restaurant features to {RESTAURANT_FEATURES_PATH}")
    
    log_memory_usage("After all feature tables")
    
    # Free base dataframes
    del items_df
    del restaurants_df
    del users_df
    del embeddings
    force_gc()
    
    # ---------------------------------------------------------------------
    # PART 3 & 4: Process Cart Events and Build Training Dataset
    # ---------------------------------------------------------------------
    processing_stats = process_cart_events_chunked(
        cart_events_path=CART_EVENTS_PATH,
        item_features_df=item_features_df,
        user_features_df=user_features_df,
        restaurant_features_df=restaurant_features_df,
        output_path=TRAINING_DATASET_PATH,
        chunk_size=args.chunk_size,
    )
    
    logger.info(f"Saved training dataset to {TRAINING_DATASET_PATH}")
    logger.info(f"  Total rows: {processing_stats['total_rows']:,}")
    logger.info(f"  Positive: {processing_stats['positive_rows']:,}")
    logger.info(f"  Negative: {processing_stats['negative_rows']:,}")
    logger.info(f"  Skipped: {processing_stats['skipped_rows']:,}")
    
    # ---------------------------------------------------------------------
    # Validation
    # ---------------------------------------------------------------------
    validation_results = validate_output(
        TRAINING_DATASET_PATH,
        USER_FEATURES_PATH,
        RESTAURANT_FEATURES_PATH,
        ITEM_FEATURES_PATH,
    )
    
    # ---------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Duration: {duration:.1f} seconds")
    log_memory_usage("Final")
    
    logger.info("\nOutput Files:")
    logger.info(f"  - {TRAINING_DATASET_PATH}")
    logger.info(f"  - {USER_FEATURES_PATH}")
    logger.info(f"  - {RESTAURANT_FEATURES_PATH}")
    logger.info(f"  - {ITEM_FEATURES_PATH}")
    logger.info(f"  - {FIXED_EMBEDDINGS_PATH}")
    
    logger.info("\nSummary Stats:")
    for name, stats in validation_results.items():
        logger.info(f"  {name}: {stats['rows']:,} rows, {stats['columns']} cols, {stats['null_values']} nulls")


if __name__ == "__main__":
    main()
