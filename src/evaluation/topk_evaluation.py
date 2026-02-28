"""
Top-K Evaluation Module for Cart Add-to-Cart Recommendation

Evaluates ranking quality of the recommendation model using:
- Precision@K: Fraction of top-K that are relevant
- Recall@K: Fraction of relevant items in top-K
- NDCG@K: Normalized Discounted Cumulative Gain
- Add-on Acceptance Rate@K: Rate of accepted add-ons in top-K

Compares model performance against random ranking baseline.

Usage:
    python -m src.evaluation.topk_evaluation
    python -m src.evaluation.topk_evaluation --checkpoint checkpoints/best_model_final.pt

Author: ZOMATHON Team
Date: February 2026
"""

import argparse
import gc
import logging
import pickle
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.model.cart_transformer import CartAddToCartModel, create_cart_mask

# -----------------------------------------------------------------------------
# Logging Configuration
# -----------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/topk_evaluation.log", mode="a")
    ]
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Constants & Paths
# -----------------------------------------------------------------------------

CART_EVENTS_PATH = Path("data/synthetic/cart_events.csv")
EMBEDDINGS_PATH = Path("data/processed/item_embeddings_fixed.npy")
ITEM_MAPPING_PATH = Path("data/processed/item_id_mapping.pkl")
USER_FEATURES_PATH = Path("data/processed/user_features.parquet")
RESTAURANT_FEATURES_PATH = Path("data/processed/restaurant_features.parquet")
MENU_ITEMS_PATH = Path("data/processed/menu_items_simulation.csv")
DEFAULT_CHECKPOINT = Path("checkpoints/best_model_final.pt")

MAX_CART_SIZE = 10
MEAL_TYPE_MAP = {"breakfast": 0, "lunch": 1, "snack": 2, "dinner": 3, "late_night": 4}
USER_TYPE_MAP = {"budget": 0, "moderate": 1, "premium": 2, "luxury": 3}


# -----------------------------------------------------------------------------
# Data Structures
# -----------------------------------------------------------------------------

@dataclass
class SessionStep:
    """Represents a single cart state with multiple candidate items."""
    session_id: str
    step_number: int
    user_id: str
    restaurant_id: str
    cart_items: List[str]
    cart_indices: List[int]
    candidates: List[str]           # List of candidate item IDs
    candidate_indices: List[int]    # Embedding indices for candidates
    labels: List[int]               # 1 if added, 0 otherwise
    user_features: List[float]
    restaurant_features: List[float]
    cart_dynamic_features: List[float]
    context_features: List[float]


# -----------------------------------------------------------------------------
# Ranking Metrics
# -----------------------------------------------------------------------------

def precision_at_k(ranked_labels: List[int], k: int) -> float:
    """
    Precision@K: Fraction of top-K items that are relevant.
    
    Args:
        ranked_labels: Labels sorted by predicted score (descending)
        k: Number of top items to consider
    
    Returns:
        Precision value in [0, 1]
    """
    if k == 0:
        return 0.0
    top_k = ranked_labels[:k]
    return sum(top_k) / k


def recall_at_k(ranked_labels: List[int], k: int) -> float:
    """
    Recall@K: Fraction of relevant items that appear in top-K.
    
    Args:
        ranked_labels: Labels sorted by predicted score (descending)
        k: Number of top items to consider
    
    Returns:
        Recall value in [0, 1]
    """
    total_relevant = sum(ranked_labels)
    if total_relevant == 0:
        return 0.0
    top_k = ranked_labels[:k]
    return sum(top_k) / total_relevant


def dcg_at_k(ranked_labels: List[int], k: int) -> float:
    """
    Discounted Cumulative Gain at K.
    
    Uses log2(pos+2) as discount factor (pos is 0-indexed).
    """
    top_k = ranked_labels[:k]
    dcg = 0.0
    for i, label in enumerate(top_k):
        dcg += label / np.log2(i + 2)  # +2 because log2(1)=0
    return dcg


def ndcg_at_k(ranked_labels: List[int], k: int) -> float:
    """
    Normalized Discounted Cumulative Gain at K.
    
    NDCG = DCG / IDCG, where IDCG is the ideal DCG (perfect ranking).
    
    Args:
        ranked_labels: Labels sorted by predicted score (descending)
        k: Number of top items to consider
    
    Returns:
        NDCG value in [0, 1]
    """
    # Compute actual DCG
    dcg = dcg_at_k(ranked_labels, k)
    
    # Compute ideal DCG (perfect ranking: all 1s first)
    ideal_labels = sorted(ranked_labels, reverse=True)
    idcg = dcg_at_k(ideal_labels, k)
    
    if idcg == 0:
        return 0.0
    return dcg / idcg


def acceptance_rate_at_k(ranked_labels: List[int], k: int) -> float:
    """
    Add-on acceptance rate at K.
    
    Returns 1 if any of the top-K items was accepted (label=1), else 0.
    This represents whether the user would likely accept a recommendation.
    """
    top_k = ranked_labels[:k]
    return 1.0 if any(top_k) else 0.0


def mean_reciprocal_rank(ranked_labels: List[int]) -> float:
    """
    Mean Reciprocal Rank (MRR).
    
    Returns 1/rank of the first relevant item, or 0 if none found.
    """
    for i, label in enumerate(ranked_labels):
        if label == 1:
            return 1.0 / (i + 1)
    return 0.0


# -----------------------------------------------------------------------------
# Data Loading & Preprocessing
# -----------------------------------------------------------------------------

def load_item_mapping() -> Dict[str, int]:
    """Load item ID to embedding index mapping."""
    with open(ITEM_MAPPING_PATH, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict) and "item_to_idx" in data:
        return data["item_to_idx"]
    elif isinstance(data, list):
        idx_to_item = data[0]
        return {item_id: idx for idx, item_id in idx_to_item.items()}
    else:
        return {item_id: idx for idx, item_id in data.items()}


def load_auxiliary_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load user features, restaurant features, and menu items."""
    users_df = pd.read_parquet(USER_FEATURES_PATH)
    restaurants_df = pd.read_parquet(RESTAURANT_FEATURES_PATH)
    menu_items_df = pd.read_csv(MENU_ITEMS_PATH)
    return users_df, restaurants_df, menu_items_df


def build_feature_lookups(
    users_df: pd.DataFrame,
    restaurants_df: pd.DataFrame,
    menu_items_df: pd.DataFrame
) -> Tuple[Dict, Dict, Dict, Dict]:
    """Build lookup dictionaries for features."""
    
    # User features lookup
    user_features_lookup = {}
    for _, row in users_df.iterrows():
        user_features_lookup[row["user_id"]] = [
            float(row.get("recency_days_norm", 0.0)),
            float(row.get("frequency_norm", 0.0)),
            float(row.get("monetary_avg_norm", 0.0)),
            float(row.get("cuisine_entropy_norm", 0.5)),
            float(row.get("avg_cart_size_norm", 0.0)),
            float(row.get("dessert_ratio_norm", 0.1)),
            float(row.get("beverage_ratio_norm", 0.15)),
        ]
    
    # Restaurant features lookup
    restaurant_features_lookup = {}
    for _, row in restaurants_df.iterrows():
        restaurant_features_lookup[row["restaurant_id"]] = [
            float(row.get("smoothed_rating_norm", 0.25)),
            float(row.get("delivery_votes_norm", 0.0)),
            float(row.get("avg_item_price_norm", 0.0)),
            float(row.get("price_band_index_norm", 0.5)),
            float(row.get("menu_size_norm", 0.0)),
        ]
    
    # Item prices lookup
    item_prices = {}
    item_categories = {}
    cat_map = {"main": 0, "starter": 1, "dessert": 2, "beverage": 3, "side": 4, "snack": 5, "combo": 6, "other": 7}
    for _, row in menu_items_df.iterrows():
        item_prices[row["item_id"]] = float(row.get("price", 100))
        cat = str(row.get("item_category", "other")).lower()
        item_categories[row["item_id"]] = cat_map.get(cat, 7)
    
    return user_features_lookup, restaurant_features_lookup, item_prices, item_categories


def load_validation_sessions(
    val_ratio: float = 0.1,
    max_sessions: int = 50000
) -> Tuple[List[SessionStep], Dict]:
    """
    Load validation set grouped by session+step.
    
    Args:
        val_ratio: Last X% of data is validation (time-based split)
        max_sessions: Maximum number of session-steps to evaluate
    
    Returns:
        List of SessionStep objects, feature lookups
    """
    logger.info("Loading validation data...")
    
    # Load mappings and features
    item_to_idx = load_item_mapping()
    users_df, restaurants_df, menu_items_df = load_auxiliary_data()
    user_lookup, restaurant_lookup, item_prices, item_categories = build_feature_lookups(
        users_df, restaurants_df, menu_items_df
    )
    
    # Default features
    default_user = [0.0, 0.0, 0.0, 0.5, 0.0, 0.1, 0.15]
    default_restaurant = [0.25, 0.0, 0.0, 0.5, 0.0]
    
    # Load cart events
    logger.info("Loading cart events...")
    df = pd.read_csv(CART_EVENTS_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    # Time-based split - get validation portion
    split_idx = int(len(df) * (1 - val_ratio))
    val_df = df.iloc[split_idx:].copy()
    logger.info(f"Validation set: {len(val_df):,} rows")
    
    # Group by session_id + step_number
    logger.info("Grouping into session-steps...")
    grouped = val_df.groupby(["session_id", "step_number"])
    
    sessions = []
    for (session_id, step_number), group in grouped:
        if len(sessions) >= max_sessions:
            break
        
        # Get shared attributes from first row
        first_row = group.iloc[0]
        user_id = first_row["user_id"]
        restaurant_id = first_row["restaurant_id"]
        
        # Parse cart items
        cart_items_str = str(first_row.get("cart_items", ""))
        if cart_items_str and cart_items_str != "nan":
            cart_items = [x.strip() for x in cart_items_str.split("|") if x.strip()]
        else:
            cart_items = []
        
        # Convert cart to indices
        cart_indices = []
        for item_id in cart_items[:MAX_CART_SIZE]:
            idx = item_to_idx.get(item_id, -1)
            if idx >= 0:
                cart_indices.append(idx + 1)  # +1 for padding
        
        # Get all candidates and their labels
        candidates = []
        candidate_indices = []
        labels = []
        
        for _, row in group.iterrows():
            candidate_id = row["candidate_item"]
            label = int(row.get("label", 0))
            
            idx = item_to_idx.get(candidate_id, -1)
            if idx >= 0:
                candidates.append(candidate_id)
                candidate_indices.append(idx + 1)
                labels.append(label)
        
        if not candidates:
            continue
        
        # User features
        user_features = user_lookup.get(user_id, default_user.copy())
        
        # Restaurant features
        restaurant_features = restaurant_lookup.get(restaurant_id, default_restaurant.copy())
        
        # Cart dynamic features
        cart_total = float(first_row.get("cart_total", 0))
        cart_size = len(cart_indices)
        if cart_items:
            prices = [item_prices.get(item_id, 100) for item_id in cart_items]
            avg_cart_price = np.mean(prices)
        else:
            avg_cart_price = 0.0
        
        cart_categories_str = str(first_row.get("cart_categories", ""))
        cart_categories = set(cart_categories_str.split("|")) if cart_categories_str and cart_categories_str != "nan" else set()
        missing_beverage = 1.0 if "beverage" not in cart_categories else 0.0
        missing_dessert = 1.0 if "dessert" not in cart_categories else 0.0
        main_count = cart_categories_str.count("main") if cart_categories_str else 0
        side_count = cart_categories_str.count("side") if cart_categories_str else 0
        heavy_meal = 1.0 if (main_count >= 2 or (main_count >= 1 and side_count >= 1)) else 0.0
        
        cart_dynamic_features = [
            cart_total / 1000.0,
            cart_size / MAX_CART_SIZE,
            avg_cart_price / 500.0,
            missing_beverage,
            missing_dessert,
            heavy_meal
        ]
        
        # Context features (base)
        hour = int(first_row.get("hour", 12))
        meal_type = str(first_row.get("meal_type", "lunch"))
        user_type = str(first_row.get("user_type", "moderate"))
        
        context_features = [
            hour / 24.0,
            float(MEAL_TYPE_MAP.get(meal_type, 1)),
            float(USER_TYPE_MAP.get(user_type, 1))
        ]
        
        sessions.append(SessionStep(
            session_id=session_id,
            step_number=step_number,
            user_id=user_id,
            restaurant_id=restaurant_id,
            cart_items=cart_items,
            cart_indices=cart_indices,
            candidates=candidates,
            candidate_indices=candidate_indices,
            labels=labels,
            user_features=user_features,
            restaurant_features=restaurant_features,
            cart_dynamic_features=cart_dynamic_features,
            context_features=context_features
        ))
    
    logger.info(f"Loaded {len(sessions):,} session-steps for evaluation")
    
    # Return feature lookups for candidate category features
    return sessions, {
        "item_to_idx": item_to_idx,
        "item_categories": item_categories
    }


# -----------------------------------------------------------------------------
# Model Inference
# -----------------------------------------------------------------------------

def load_model(checkpoint_path: Path, device: str = "cpu") -> CartAddToCartModel:
    """Load trained model from checkpoint."""
    logger.info(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    config = checkpoint.get("config", {})
    
    model = CartAddToCartModel(
        item_embeddings_path=EMBEDDINGS_PATH,
        user_feature_dim=config.get("user_feature_dim", 7),
        restaurant_feature_dim=config.get("restaurant_feature_dim", 5),
        cart_dynamic_feature_dim=config.get("cart_dynamic_feature_dim", 6),
        context_feature_dim=config.get("context_feature_dim", 7),
        freeze_embeddings=True,
        disable_transformer=True  # Mean pooling baseline
    )
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    logger.info(f"Model loaded: {model.count_parameters():,} parameters")
    return model


@torch.no_grad()
def rank_candidates(
    model: CartAddToCartModel,
    session: SessionStep,
    item_categories: Dict[str, int],
    device: str = "cpu"
) -> Tuple[List[int], List[float]]:
    """
    Rank all candidates for a session by model score.
    
    Returns:
        ranked_labels: Labels sorted by descending predicted score
        ranked_scores: Corresponding prediction scores
    """
    n_candidates = len(session.candidates)
    if n_candidates == 0:
        return [], []
    
    # Prepare batch input (one row per candidate)
    max_cart_len = max(len(session.cart_indices), 1)
    
    # Cart indices (same for all candidates)
    cart_indices = torch.zeros(n_candidates, max_cart_len, dtype=torch.long, device=device)
    if session.cart_indices:
        for i in range(n_candidates):
            cart_indices[i, :len(session.cart_indices)] = torch.tensor(session.cart_indices, dtype=torch.long)
    
    # Cart mask
    cart_mask = torch.ones(n_candidates, max_cart_len, dtype=torch.bool, device=device)
    if session.cart_indices:
        cart_mask[:, :len(session.cart_indices)] = False
    
    # Candidate indices
    candidate_indices = torch.tensor(session.candidate_indices, dtype=torch.long, device=device)
    
    # User features (same for all)
    user_features = torch.tensor(session.user_features, dtype=torch.float32, device=device)
    user_features = user_features.unsqueeze(0).expand(n_candidates, -1)
    
    # Restaurant features (same for all)
    restaurant_features = torch.tensor(session.restaurant_features, dtype=torch.float32, device=device)
    restaurant_features = restaurant_features.unsqueeze(0).expand(n_candidates, -1)
    
    # Cart dynamic features (same for all)
    cart_dynamic_features = torch.tensor(session.cart_dynamic_features, dtype=torch.float32, device=device)
    cart_dynamic_features = cart_dynamic_features.unsqueeze(0).expand(n_candidates, -1)
    
    # Context features (different per candidate due to category one-hot)
    context_batch = []
    for candidate_id in session.candidates:
        ctx = session.context_features.copy()
        cat_idx = item_categories.get(candidate_id, 7)
        is_main = 1.0 if cat_idx == 0 else 0.0
        is_dessert = 1.0 if cat_idx == 2 else 0.0
        is_beverage = 1.0 if cat_idx == 3 else 0.0
        is_side = 1.0 if cat_idx == 4 else 0.0
        ctx.extend([is_main, is_dessert, is_beverage, is_side])
        context_batch.append(ctx)
    
    context_features = torch.tensor(context_batch, dtype=torch.float32, device=device)
    
    # Forward pass
    logits = model(
        cart_indices, candidate_indices,
        user_features, restaurant_features,
        cart_dynamic_features, context_features,
        cart_mask
    )
    
    scores = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
    
    # Sort by score (descending)
    sorted_indices = np.argsort(-scores)
    ranked_labels = [session.labels[i] for i in sorted_indices]
    ranked_scores = [scores[i] for i in sorted_indices]
    
    return ranked_labels, ranked_scores


def random_rank_candidates(session: SessionStep) -> List[int]:
    """Random baseline: shuffle labels randomly."""
    labels_copy = session.labels.copy()
    np.random.shuffle(labels_copy)
    return labels_copy


# -----------------------------------------------------------------------------
# Evaluation Pipeline
# -----------------------------------------------------------------------------

def evaluate_topk(
    model: CartAddToCartModel,
    sessions: List[SessionStep],
    item_categories: Dict[str, int],
    ks: List[int] = [5, 8],
    device: str = "cpu"
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate model on top-K ranking metrics.
    
    Args:
        model: Trained model
        sessions: List of session-steps to evaluate
        item_categories: Item category lookup
        ks: List of K values for metrics
        device: Computation device
    
    Returns:
        Dictionary with metrics for model and random baseline
    """
    logger.info(f"Evaluating {len(sessions):,} session-steps...")
    
    # Initialize metric accumulators
    model_metrics = {f"precision@{k}": [] for k in ks}
    model_metrics.update({f"recall@{k}": [] for k in ks})
    model_metrics.update({f"ndcg@{k}": [] for k in ks})
    model_metrics.update({f"acceptance@{k}": [] for k in ks})
    model_metrics["mrr"] = []
    
    random_metrics = {f"precision@{k}": [] for k in ks}
    random_metrics.update({f"recall@{k}": [] for k in ks})
    random_metrics.update({f"ndcg@{k}": [] for k in ks})
    random_metrics.update({f"acceptance@{k}": [] for k in ks})
    random_metrics["mrr"] = []
    
    # Track sessions with positives
    sessions_with_positives = 0
    total_candidates = 0
    total_positives = 0
    
    start_time = time.time()
    
    for i, session in enumerate(sessions):
        if (i + 1) % 5000 == 0:
            logger.info(f"  Processed {i+1:,}/{len(sessions):,} sessions...")
        
        n_candidates = len(session.candidates)
        n_positives = sum(session.labels)
        total_candidates += n_candidates
        total_positives += n_positives
        
        if n_positives > 0:
            sessions_with_positives += 1
        
        # Skip sessions with no candidates
        if n_candidates == 0:
            continue
        
        # Model ranking
        model_ranked_labels, model_scores = rank_candidates(
            model, session, item_categories, device
        )
        
        # Random baseline
        random_ranked_labels = random_rank_candidates(session)
        
        # Compute metrics for each K
        for k in ks:
            # Model metrics
            model_metrics[f"precision@{k}"].append(precision_at_k(model_ranked_labels, k))
            model_metrics[f"recall@{k}"].append(recall_at_k(model_ranked_labels, k))
            model_metrics[f"ndcg@{k}"].append(ndcg_at_k(model_ranked_labels, k))
            model_metrics[f"acceptance@{k}"].append(acceptance_rate_at_k(model_ranked_labels, k))
            
            # Random metrics
            random_metrics[f"precision@{k}"].append(precision_at_k(random_ranked_labels, k))
            random_metrics[f"recall@{k}"].append(recall_at_k(random_ranked_labels, k))
            random_metrics[f"ndcg@{k}"].append(ndcg_at_k(random_ranked_labels, k))
            random_metrics[f"acceptance@{k}"].append(acceptance_rate_at_k(random_ranked_labels, k))
        
        # MRR
        model_metrics["mrr"].append(mean_reciprocal_rank(model_ranked_labels))
        random_metrics["mrr"].append(mean_reciprocal_rank(random_ranked_labels))
    
    eval_time = time.time() - start_time
    
    # Compute means
    model_results = {k: np.mean(v) for k, v in model_metrics.items()}
    random_results = {k: np.mean(v) for k, v in random_metrics.items()}
    
    # Log summary stats
    logger.info(f"Evaluation completed in {eval_time:.1f}s")
    logger.info(f"Sessions evaluated: {len(sessions):,}")
    logger.info(f"Sessions with positives: {sessions_with_positives:,} ({100*sessions_with_positives/len(sessions):.1f}%)")
    logger.info(f"Total candidates: {total_candidates:,}")
    logger.info(f"Total positives: {total_positives:,} ({100*total_positives/total_candidates:.2f}%)")
    logger.info(f"Avg candidates per session: {total_candidates/len(sessions):.1f}")
    
    return {
        "model": model_results,
        "random": random_results,
        "stats": {
            "n_sessions": len(sessions),
            "sessions_with_positives": sessions_with_positives,
            "total_candidates": total_candidates,
            "total_positives": total_positives,
            "eval_time_seconds": eval_time
        }
    }


def print_metrics_table(results: Dict) -> str:
    """Print a formatted metrics comparison table."""
    
    model_metrics = results["model"]
    random_metrics = results["random"]
    stats = results["stats"]
    
    # Build table
    lines = []
    lines.append("=" * 70)
    lines.append("TOP-K EVALUATION RESULTS")
    lines.append("=" * 70)
    lines.append(f"Sessions evaluated: {stats['n_sessions']:,}")
    lines.append(f"Sessions with positive labels: {stats['sessions_with_positives']:,}")
    lines.append(f"Total candidates: {stats['total_candidates']:,}")
    lines.append(f"Total positive labels: {stats['total_positives']:,}")
    lines.append(f"Evaluation time: {stats['eval_time_seconds']:.1f}s")
    lines.append("-" * 70)
    lines.append("")
    lines.append(f"{'Metric':<20} {'Model':>12} {'Random':>12} {'Improvement':>15}")
    lines.append("-" * 70)
    
    # Sort metrics by K value
    metric_order = [
        "precision@5", "precision@8",
        "recall@5", "recall@8",
        "ndcg@5", "ndcg@8",
        "acceptance@5", "acceptance@8",
        "mrr"
    ]
    
    for metric in metric_order:
        if metric in model_metrics:
            model_val = model_metrics[metric]
            random_val = random_metrics[metric]
            
            if random_val > 0:
                improvement = (model_val - random_val) / random_val * 100
                imp_str = f"+{improvement:.1f}%"
            else:
                improvement = float('inf') if model_val > 0 else 0
                imp_str = "+∞%" if model_val > 0 else "0%"
            
            lines.append(f"{metric:<20} {model_val:>12.4f} {random_val:>12.4f} {imp_str:>15}")
    
    lines.append("=" * 70)
    
    table = "\n".join(lines)
    return table


# -----------------------------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Top-K Evaluation for Cart Recommendations")
    parser.add_argument(
        "--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT),
        help=f"Path to model checkpoint (default: {DEFAULT_CHECKPOINT})"
    )
    parser.add_argument(
        "--max-sessions", type=int, default=50000,
        help="Maximum number of session-steps to evaluate (default: 50000)"
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.1,
        help="Validation set ratio (default: 0.1, last 10%% of data)"
    )
    parser.add_argument(
        "--device", type=str, default="mps",
        choices=["cpu", "cuda", "mps"],
        help="Computation device (default: mps)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Validate device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"
    if device == "mps" and not torch.backends.mps.is_available():
        logger.warning("MPS not available, falling back to CPU")
        device = "cpu"
    
    logger.info("=" * 70)
    logger.info("Top-K Evaluation Module")
    logger.info("=" * 70)
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Max sessions: {args.max_sessions:,}")
    logger.info(f"Device: {device}")
    logger.info("")
    
    # Load model
    model = load_model(Path(args.checkpoint), device)
    
    # Load validation sessions
    sessions, lookups = load_validation_sessions(
        val_ratio=args.val_ratio,
        max_sessions=args.max_sessions
    )
    
    # Run evaluation
    results = evaluate_topk(
        model, sessions, lookups["item_categories"],
        ks=[5, 8], device=device
    )
    
    # Print results
    table = print_metrics_table(results)
    print("\n" + table)
    logger.info("\n" + table)
    
    # Save results
    results_path = Path("logs/topk_evaluation_results.pkl")
    with open(results_path, "wb") as f:
        pickle.dump(results, f)
    logger.info(f"Results saved to {results_path}")
    
    return results


if __name__ == "__main__":
    main()
