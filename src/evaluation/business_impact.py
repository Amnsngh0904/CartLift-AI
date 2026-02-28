"""
Business Impact Simulation Module for Cart Add-to-Cart Recommendation

Compares model recommendations against popularity baseline to estimate:
- Add-on acceptance rate
- Attach rate improvement
- Average Order Value (AOV) lift  
- Average items per order lift

Usage:
    python -m src.evaluation.business_impact
    python -m src.evaluation.business_impact --checkpoint checkpoints/best_model_final.pt

Author: ZOMATHON Team
Date: February 2026
"""

import argparse
import logging
import pickle
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

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
        logging.FileHandler("logs/business_impact.log", mode="a")
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
    candidates: List[str]
    candidate_indices: List[int]
    labels: List[int]
    user_features: List[float]
    restaurant_features: List[float]
    cart_dynamic_features: List[float]
    context_features: List[float]
    cart_total: float = 0.0


@dataclass
class SimulationResult:
    """Result of simulating a recommendation strategy."""
    addon_accepted: bool           # Whether user accepted an addon
    addon_price: float             # Price of accepted addon (0 if none)
    num_addons_accepted: int       # Number of addons user would accept
    total_addon_value: float       # Total value of accepted addons
    accept_position: int           # Position in ranking where first acceptance occurred (-1 if none)


# -----------------------------------------------------------------------------
# Data Loading
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
    
    restaurant_features_lookup = {}
    for _, row in restaurants_df.iterrows():
        restaurant_features_lookup[row["restaurant_id"]] = [
            float(row.get("smoothed_rating_norm", 0.25)),
            float(row.get("delivery_votes_norm", 0.0)),
            float(row.get("avg_item_price_norm", 0.0)),
            float(row.get("price_band_index_norm", 0.5)),
            float(row.get("menu_size_norm", 0.0)),
        ]
    
    item_prices = {}
    item_categories = {}
    cat_map = {"main": 0, "starter": 1, "dessert": 2, "beverage": 3, "side": 4, "snack": 5, "combo": 6, "other": 7}
    for _, row in menu_items_df.iterrows():
        item_prices[row["item_id"]] = float(row.get("price", 100))
        cat = str(row.get("item_category", "other")).lower()
        item_categories[row["item_id"]] = cat_map.get(cat, 7)
    
    return user_features_lookup, restaurant_features_lookup, item_prices, item_categories


def build_popularity_baseline(menu_items_df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Build popularity baseline: top items per restaurant sorted by item_votes.
    
    Returns:
        Dict mapping restaurant_id -> list of item_ids sorted by popularity (descending)
    """
    popularity = defaultdict(list)
    
    # Sort by restaurant_id and item_votes (descending)
    sorted_df = menu_items_df.sort_values(
        ["restaurant_id", "item_votes"], 
        ascending=[True, False]
    )
    
    for _, row in sorted_df.iterrows():
        restaurant_id = row["restaurant_id"]
        item_id = row["item_id"]
        popularity[restaurant_id].append(item_id)
    
    logger.info(f"Built popularity baseline for {len(popularity)} restaurants")
    return dict(popularity)


def load_validation_sessions(
    val_ratio: float = 0.1,
    max_sessions: int = 50000
) -> Tuple[List[SessionStep], Dict]:
    """Load validation set grouped by session+step."""
    logger.info("Loading validation data...")
    
    item_to_idx = load_item_mapping()
    users_df, restaurants_df, menu_items_df = load_auxiliary_data()
    user_lookup, restaurant_lookup, item_prices, item_categories = build_feature_lookups(
        users_df, restaurants_df, menu_items_df
    )
    
    # Build popularity baseline
    popularity_baseline = build_popularity_baseline(menu_items_df)
    
    default_user = [0.0, 0.0, 0.0, 0.5, 0.0, 0.1, 0.15]
    default_restaurant = [0.25, 0.0, 0.0, 0.5, 0.0]
    
    logger.info("Loading cart events...")
    df = pd.read_csv(CART_EVENTS_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    split_idx = int(len(df) * (1 - val_ratio))
    val_df = df.iloc[split_idx:].copy()
    logger.info(f"Validation set: {len(val_df):,} rows")
    
    logger.info("Grouping into session-steps...")
    grouped = val_df.groupby(["session_id", "step_number"])
    
    sessions = []
    for (session_id, step_number), group in grouped:
        if len(sessions) >= max_sessions:
            break
        
        first_row = group.iloc[0]
        user_id = first_row["user_id"]
        restaurant_id = first_row["restaurant_id"]
        
        cart_items_str = str(first_row.get("cart_items", ""))
        if cart_items_str and cart_items_str != "nan":
            cart_items = [x.strip() for x in cart_items_str.split("|") if x.strip()]
        else:
            cart_items = []
        
        cart_indices = []
        for item_id in cart_items[:MAX_CART_SIZE]:
            idx = item_to_idx.get(item_id, -1)
            if idx >= 0:
                cart_indices.append(idx + 1)
        
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
        
        user_features = user_lookup.get(user_id, default_user.copy())
        restaurant_features = restaurant_lookup.get(restaurant_id, default_restaurant.copy())
        
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
            context_features=context_features,
            cart_total=cart_total
        ))
    
    logger.info(f"Loaded {len(sessions):,} session-steps for evaluation")
    
    return sessions, {
        "item_to_idx": item_to_idx,
        "item_categories": item_categories,
        "item_prices": item_prices,
        "popularity_baseline": popularity_baseline
    }


# -----------------------------------------------------------------------------
# Model Loading
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
        disable_transformer=True
    )
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    logger.info(f"Model loaded: {model.count_parameters():,} parameters")
    return model


# -----------------------------------------------------------------------------
# Ranking Functions
# -----------------------------------------------------------------------------

@torch.no_grad()
def rank_by_model(
    model: CartAddToCartModel,
    session: SessionStep,
    item_categories: Dict[str, int],
    device: str = "cpu"
) -> List[str]:
    """Rank candidates by model score (descending)."""
    n_candidates = len(session.candidates)
    if n_candidates == 0:
        return []
    
    max_cart_len = max(len(session.cart_indices), 1)
    
    cart_indices = torch.zeros(n_candidates, max_cart_len, dtype=torch.long, device=device)
    if session.cart_indices:
        for i in range(n_candidates):
            cart_indices[i, :len(session.cart_indices)] = torch.tensor(session.cart_indices, dtype=torch.long)
    
    cart_mask = torch.ones(n_candidates, max_cart_len, dtype=torch.bool, device=device)
    if session.cart_indices:
        cart_mask[:, :len(session.cart_indices)] = False
    
    candidate_indices = torch.tensor(session.candidate_indices, dtype=torch.long, device=device)
    
    user_features = torch.tensor(session.user_features, dtype=torch.float32, device=device)
    user_features = user_features.unsqueeze(0).expand(n_candidates, -1)
    
    restaurant_features = torch.tensor(session.restaurant_features, dtype=torch.float32, device=device)
    restaurant_features = restaurant_features.unsqueeze(0).expand(n_candidates, -1)
    
    cart_dynamic_features = torch.tensor(session.cart_dynamic_features, dtype=torch.float32, device=device)
    cart_dynamic_features = cart_dynamic_features.unsqueeze(0).expand(n_candidates, -1)
    
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
    
    logits = model(
        cart_indices, candidate_indices,
        user_features, restaurant_features,
        cart_dynamic_features, context_features,
        cart_mask
    )
    
    scores = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
    sorted_indices = np.argsort(-scores)
    
    return [session.candidates[i] for i in sorted_indices]


def rank_by_popularity(
    session: SessionStep,
    popularity_baseline: Dict[str, List[str]]
) -> List[str]:
    """
    Rank candidates by global popularity within the restaurant.
    
    Returns candidates sorted by their position in the restaurant popularity list.
    Items not in the popularity list are ranked last.
    """
    restaurant_id = session.restaurant_id
    pop_order = popularity_baseline.get(restaurant_id, [])
    
    # Create position lookup (lower = more popular)
    position_lookup = {item_id: idx for idx, item_id in enumerate(pop_order)}
    max_pos = len(pop_order)
    
    # Sort candidates by popularity position
    def get_popularity_rank(item_id):
        return position_lookup.get(item_id, max_pos)
    
    sorted_candidates = sorted(session.candidates, key=get_popularity_rank)
    return sorted_candidates


# -----------------------------------------------------------------------------
# Simulation Logic
# -----------------------------------------------------------------------------

def simulate_recommendation(
    ranked_items: List[str],
    session: SessionStep,
    item_prices: Dict[str, float],
    k: int = 8
) -> SimulationResult:
    """
    Simulate user behavior given a ranked list of recommendations.
    
    Assumes user sees top-K items and may accept those that match their intent
    (indicated by label=1 in the ground truth).
    
    Args:
        ranked_items: Items ranked by recommendation strategy
        session: Session with ground truth labels
        item_prices: Price lookup for items
        k: Number of items shown to user
    
    Returns:
        SimulationResult with acceptance metrics
    """
    # Map candidate to label
    candidate_to_label = dict(zip(session.candidates, session.labels))
    
    top_k = ranked_items[:k]
    
    addon_accepted = False
    first_accept_position = -1
    num_accepted = 0
    total_value = 0.0
    
    for i, item_id in enumerate(top_k):
        label = candidate_to_label.get(item_id, 0)
        if label == 1:
            if not addon_accepted:
                addon_accepted = True
                first_accept_position = i + 1  # 1-indexed
            num_accepted += 1
            total_value += item_prices.get(item_id, 100)
    
    addon_price = total_value / num_accepted if num_accepted > 0 else 0.0
    
    return SimulationResult(
        addon_accepted=addon_accepted,
        addon_price=addon_price,
        num_addons_accepted=num_accepted,
        total_addon_value=total_value,
        accept_position=first_accept_position
    )


# -----------------------------------------------------------------------------
# Business Metrics Computation
# -----------------------------------------------------------------------------

@dataclass
class BusinessMetrics:
    """Aggregated business metrics from simulation."""
    name: str
    n_sessions: int
    
    # Add-on acceptance
    addon_acceptance_rate: float        # % of sessions with at least one addon accepted
    avg_addons_per_accepting_session: float  # Avg addons when user accepts
    
    # Revenue metrics
    avg_addon_value: float              # Average price of accepted addons
    total_addon_revenue: float          # Total revenue from addons
    avg_revenue_per_session: float      # Total addon revenue / n_sessions
    
    # Order completion
    attach_rate: float                  # % of sessions with addon (same as acceptance)
    avg_items_added: float              # Avg items added across all sessions
    
    # Position metrics
    avg_accept_position: float          # Avg position of first acceptance (lower = better)


def compute_business_metrics(
    results: List[SimulationResult],
    name: str
) -> BusinessMetrics:
    """Compute aggregated business metrics from simulation results."""
    n_sessions = len(results)
    
    if n_sessions == 0:
        return BusinessMetrics(
            name=name, n_sessions=0,
            addon_acceptance_rate=0.0, avg_addons_per_accepting_session=0.0,
            avg_addon_value=0.0, total_addon_revenue=0.0, avg_revenue_per_session=0.0,
            attach_rate=0.0, avg_items_added=0.0, avg_accept_position=0.0
        )
    
    # Count acceptances
    accepting_sessions = [r for r in results if r.addon_accepted]
    n_accepting = len(accepting_sessions)
    
    addon_acceptance_rate = n_accepting / n_sessions
    
    # Addons per accepting session
    if n_accepting > 0:
        avg_addons_per_accepting = sum(r.num_addons_accepted for r in accepting_sessions) / n_accepting
        avg_accept_position = sum(r.accept_position for r in accepting_sessions) / n_accepting
    else:
        avg_addons_per_accepting = 0.0
        avg_accept_position = 0.0
    
    # Revenue
    total_addon_revenue = sum(r.total_addon_value for r in results)
    avg_addon_value = total_addon_revenue / sum(r.num_addons_accepted for r in results) if sum(r.num_addons_accepted for r in results) > 0 else 0.0
    avg_revenue_per_session = total_addon_revenue / n_sessions
    
    # Items added
    avg_items_added = sum(r.num_addons_accepted for r in results) / n_sessions
    
    return BusinessMetrics(
        name=name,
        n_sessions=n_sessions,
        addon_acceptance_rate=addon_acceptance_rate,
        avg_addons_per_accepting_session=avg_addons_per_accepting,
        avg_addon_value=avg_addon_value,
        total_addon_revenue=total_addon_revenue,
        avg_revenue_per_session=avg_revenue_per_session,
        attach_rate=addon_acceptance_rate,  # Same as acceptance rate
        avg_items_added=avg_items_added,
        avg_accept_position=avg_accept_position
    )


def run_simulation(
    model: CartAddToCartModel,
    sessions: List[SessionStep],
    lookups: Dict,
    k: int = 8,
    device: str = "cpu"
) -> Tuple[BusinessMetrics, BusinessMetrics]:
    """
    Run full business impact simulation.
    
    Returns:
        (model_metrics, popularity_metrics)
    """
    logger.info(f"Running business simulation on {len(sessions):,} sessions (K={k})...")
    
    item_categories = lookups["item_categories"]
    item_prices = lookups["item_prices"]
    popularity_baseline = lookups["popularity_baseline"]
    
    model_results = []
    popularity_results = []
    
    start_time = time.time()
    
    for i, session in enumerate(sessions):
        if (i + 1) % 10000 == 0:
            logger.info(f"  Processed {i+1:,}/{len(sessions):,} sessions...")
        
        # Model ranking
        model_ranking = rank_by_model(model, session, item_categories, device)
        model_sim = simulate_recommendation(model_ranking, session, item_prices, k)
        model_results.append(model_sim)
        
        # Popularity ranking
        pop_ranking = rank_by_popularity(session, popularity_baseline)
        pop_sim = simulate_recommendation(pop_ranking, session, item_prices, k)
        popularity_results.append(pop_sim)
    
    elapsed = time.time() - start_time
    logger.info(f"Simulation completed in {elapsed:.1f}s")
    
    # Compute metrics
    model_metrics = compute_business_metrics(model_results, "Model (MLP)")
    popularity_metrics = compute_business_metrics(popularity_results, "Popularity Baseline")
    
    return model_metrics, popularity_metrics


# -----------------------------------------------------------------------------
# Output Formatting
# -----------------------------------------------------------------------------

def print_comparison_table(
    model_metrics: BusinessMetrics,
    popularity_metrics: BusinessMetrics,
    k: int
) -> str:
    """Print formatted comparison table."""
    
    lines = []
    lines.append("=" * 80)
    lines.append(f"BUSINESS IMPACT SIMULATION RESULTS (Top-{k} Recommendations)")
    lines.append("=" * 80)
    lines.append(f"Sessions evaluated: {model_metrics.n_sessions:,}")
    lines.append("")
    lines.append(f"{'Metric':<40} {'Model':>15} {'Popularity':>15} {'Lift':>10}")
    lines.append("-" * 80)
    
    def compute_lift(model_val, baseline_val):
        if baseline_val == 0:
            return "+∞%" if model_val > 0 else "0%"
        lift = (model_val - baseline_val) / baseline_val * 100
        return f"+{lift:.1f}%" if lift >= 0 else f"{lift:.1f}%"
    
    # Add-on Acceptance Rate
    lines.append(f"{'Add-on Acceptance Rate':<40} {model_metrics.addon_acceptance_rate*100:>14.2f}% {popularity_metrics.addon_acceptance_rate*100:>14.2f}% {compute_lift(model_metrics.addon_acceptance_rate, popularity_metrics.addon_acceptance_rate):>10}")
    
    # Attach Rate (same as acceptance)
    lines.append(f"{'Attach Rate':<40} {model_metrics.attach_rate*100:>14.2f}% {popularity_metrics.attach_rate*100:>14.2f}% {compute_lift(model_metrics.attach_rate, popularity_metrics.attach_rate):>10}")
    
    # Avg Items Added (per session)
    lines.append(f"{'Avg Items Added per Session':<40} {model_metrics.avg_items_added:>15.3f} {popularity_metrics.avg_items_added:>15.3f} {compute_lift(model_metrics.avg_items_added, popularity_metrics.avg_items_added):>10}")
    
    # Avg Addons per Accepting Session
    lines.append(f"{'Avg Addons (accepting sessions)':<40} {model_metrics.avg_addons_per_accepting_session:>15.3f} {popularity_metrics.avg_addons_per_accepting_session:>15.3f} {compute_lift(model_metrics.avg_addons_per_accepting_session, popularity_metrics.avg_addons_per_accepting_session):>10}")
    
    lines.append("-" * 80)
    lines.append("REVENUE IMPACT")
    lines.append("-" * 80)
    
    # Avg Addon Value (price)
    lines.append(f"{'Avg Addon Price (₹)':<40} {model_metrics.avg_addon_value:>15.2f} {popularity_metrics.avg_addon_value:>15.2f} {compute_lift(model_metrics.avg_addon_value, popularity_metrics.avg_addon_value):>10}")
    
    # Revenue per Session (AOV lift proxy)
    lines.append(f"{'Addon Revenue per Session (₹)':<40} {model_metrics.avg_revenue_per_session:>15.2f} {popularity_metrics.avg_revenue_per_session:>15.2f} {compute_lift(model_metrics.avg_revenue_per_session, popularity_metrics.avg_revenue_per_session):>10}")
    
    # Total Revenue
    lines.append(f"{'Total Addon Revenue (₹)':<40} {model_metrics.total_addon_revenue:>15,.0f} {popularity_metrics.total_addon_revenue:>15,.0f} {compute_lift(model_metrics.total_addon_revenue, popularity_metrics.total_addon_revenue):>10}")
    
    lines.append("-" * 80)
    lines.append("USER EXPERIENCE")
    lines.append("-" * 80)
    
    # Average Accept Position (lower = better)
    lines.append(f"{'Avg Position of First Accept':<40} {model_metrics.avg_accept_position:>15.2f} {popularity_metrics.avg_accept_position:>15.2f} {'Better' if model_metrics.avg_accept_position < popularity_metrics.avg_accept_position else 'Worse':>10}")
    
    lines.append("=" * 80)
    
    # Summary
    aov_lift = (model_metrics.avg_revenue_per_session - popularity_metrics.avg_revenue_per_session) / popularity_metrics.avg_revenue_per_session * 100 if popularity_metrics.avg_revenue_per_session > 0 else 0
    items_lift = (model_metrics.avg_items_added - popularity_metrics.avg_items_added) / popularity_metrics.avg_items_added * 100 if popularity_metrics.avg_items_added > 0 else 0
    
    lines.append("")
    lines.append("SUMMARY:")
    lines.append(f"  • Attach Rate Improvement: {compute_lift(model_metrics.attach_rate, popularity_metrics.attach_rate)}")
    lines.append(f"  • AOV Lift (addon revenue): {aov_lift:+.1f}%")
    lines.append(f"  • Avg Items per Order Lift: {items_lift:+.1f}%")
    lines.append(f"  • Better User Experience: Users find relevant items {popularity_metrics.avg_accept_position - model_metrics.avg_accept_position:.2f} positions earlier")
    lines.append("=" * 80)
    
    table = "\n".join(lines)
    return table


# -----------------------------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Business Impact Simulation")
    parser.add_argument(
        "--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT),
        help=f"Path to model checkpoint (default: {DEFAULT_CHECKPOINT})"
    )
    parser.add_argument(
        "--max-sessions", type=int, default=50000,
        help="Maximum number of sessions to simulate (default: 50000)"
    )
    parser.add_argument(
        "--k", type=int, default=8,
        help="Number of recommendations shown (default: 8)"
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.1,
        help="Validation set ratio (default: 0.1)"
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
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"
    if device == "mps" and not torch.backends.mps.is_available():
        logger.warning("MPS not available, falling back to CPU")
        device = "cpu"
    
    logger.info("=" * 80)
    logger.info("Business Impact Simulation")
    logger.info("=" * 80)
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Max sessions: {args.max_sessions:,}")
    logger.info(f"K (recommendations shown): {args.k}")
    logger.info(f"Device: {device}")
    logger.info("")
    
    # Load model
    model = load_model(Path(args.checkpoint), device)
    
    # Load sessions
    sessions, lookups = load_validation_sessions(
        val_ratio=args.val_ratio,
        max_sessions=args.max_sessions
    )
    
    # Run simulation
    model_metrics, popularity_metrics = run_simulation(
        model, sessions, lookups, k=args.k, device=device
    )
    
    # Print results
    table = print_comparison_table(model_metrics, popularity_metrics, args.k)
    print("\n" + table)
    logger.info("\n" + table)
    
    # Save results
    results = {
        "model": model_metrics,
        "popularity": popularity_metrics,
        "k": args.k,
        "n_sessions": len(sessions)
    }
    
    results_path = Path("logs/business_impact_results.pkl")
    with open(results_path, "wb") as f:
        pickle.dump(results, f)
    logger.info(f"Results saved to {results_path}")
    
    return results


if __name__ == "__main__":
    main()
