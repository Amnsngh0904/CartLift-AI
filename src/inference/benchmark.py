"""
Benchmarking Utility for Cart Add-on Recommendation Service

Measures inference latency and throughput for:
- Single request latency (p50, p95, p99, max)
- Batch request throughput

Usage:
    python -m src.inference.benchmark
    python -m src.inference.benchmark --requests 100 --device mps

Author: ZOMATHON Team
Date: February 2026
"""

import argparse
import logging
import pickle
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

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
# Constants & Paths
# -----------------------------------------------------------------------------

DATA_DIR = Path("data/processed")
CHECKPOINTS_DIR = Path("checkpoints")

MODEL_PATH = CHECKPOINTS_DIR / "best_model_final.pt"
EMBEDDINGS_PATH = DATA_DIR / "item_embeddings_fixed.npy"
ITEM_MAPPING_PATH = DATA_DIR / "item_id_mapping.pkl"
USER_FEATURES_PATH = DATA_DIR / "user_features.parquet"
RESTAURANT_FEATURES_PATH = DATA_DIR / "restaurant_features.parquet"
MENU_ITEMS_PATH = DATA_DIR / "menu_items_enriched.csv"

MAX_CART_SIZE = 10
EMBEDDING_DIM = 64
MAX_CANDIDATES = 50

MEAL_TYPES = ["breakfast", "lunch", "snack", "dinner", "late_night"]
USER_TYPES = ["budget", "moderate", "premium", "luxury"]
CATEGORY_MAP = {"main": 0, "starter": 1, "dessert": 2, "beverage": 3, "side": 4, "snack": 5, "combo": 6, "other": 7}
MEAL_TYPE_MAP = {"breakfast": 0, "lunch": 1, "snack": 2, "dinner": 3, "late_night": 4}
USER_TYPE_MAP = {"budget": 0, "moderate": 1, "premium": 2, "luxury": 3}


# -----------------------------------------------------------------------------
# Benchmark Model (Standalone - doesn't require FastAPI)
# -----------------------------------------------------------------------------

class BenchmarkModel:
    """
    Standalone model for benchmarking inference latency.
    
    Does not require FastAPI server - directly loads and runs model.
    """
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.model = None
        self.item_embeddings = None
        self.item_to_idx: Dict[str, int] = {}
        
        # Feature lookups
        self.user_features: Dict[str, List[float]] = {}
        self.restaurant_features: Dict[str, List[float]] = {}
        self.item_prices: Dict[str, float] = {}
        self.item_categories: Dict[str, str] = {}
        
        # Restaurant items
        self.restaurant_items: Dict[str, List[str]] = {}
        self.all_users: List[str] = []
        self.all_restaurants: List[str] = []
        
        # Default features
        self.default_user_features = [0.0, 0.0, 0.0, 0.5, 0.0, 0.1, 0.15]
        self.default_restaurant_features = [0.25, 0.0, 0.0, 0.5, 0.0]
    
    def load(self):
        """Load model and data."""
        logger.info(f"Loading benchmark model on device: {self.device}")
        
        # Load embeddings
        embeddings_np = np.load(str(EMBEDDINGS_PATH))
        padding = np.zeros((1, EMBEDDING_DIM), dtype=np.float32)
        embeddings_np = np.vstack([padding, embeddings_np])
        self.item_embeddings = torch.from_numpy(embeddings_np).to(self.device)
        
        # Load item mapping
        with open(ITEM_MAPPING_PATH, "rb") as f:
            mapping_data = pickle.load(f)
        self.item_to_idx = mapping_data["item_to_idx"]
        
        # Load model
        checkpoint = torch.load(MODEL_PATH, map_location=self.device, weights_only=False)
        config = checkpoint.get("config", {})
        
        from src.model.cart_transformer import CartAddToCartModel
        self.model = CartAddToCartModel(
            item_embeddings=self.item_embeddings,
            user_feature_dim=config.get("user_feature_dim", 7),
            restaurant_feature_dim=config.get("restaurant_feature_dim", 5),
            cart_dynamic_feature_dim=config.get("cart_dynamic_feature_dim", 6),
            context_feature_dim=config.get("context_feature_dim", 7),
            freeze_embeddings=True,
            disable_transformer=True
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        
        # Load features
        users_df = pd.read_parquet(USER_FEATURES_PATH)
        for _, row in users_df.iterrows():
            self.user_features[row["user_id"]] = [
                float(row.get("recency_days_norm", 0.0)),
                float(row.get("frequency_norm", 0.0)),
                float(row.get("monetary_avg_norm", 0.0)),
                float(row.get("cuisine_entropy_norm", 0.5)),
                float(row.get("avg_cart_size_norm", 0.0)),
                float(row.get("dessert_ratio_norm", 0.1)),
                float(row.get("beverage_ratio_norm", 0.15)),
            ]
        self.all_users = list(self.user_features.keys())
        
        restaurants_df = pd.read_parquet(RESTAURANT_FEATURES_PATH)
        for _, row in restaurants_df.iterrows():
            self.restaurant_features[row["restaurant_id"]] = [
                float(row.get("smoothed_rating_norm", 0.25)),
                float(row.get("delivery_votes_norm", 0.0)),
                float(row.get("avg_item_price_norm", 0.0)),
                float(row.get("price_band_index_norm", 0.5)),
                float(row.get("menu_size_norm", 0.0)),
            ]
        
        # Load menu items
        menu_df = pd.read_csv(MENU_ITEMS_PATH)
        menu_df_sorted = menu_df.sort_values(["restaurant_id", "item_votes"], ascending=[True, False])
        for _, row in menu_df_sorted.iterrows():
            item_id = row["item_id"]
            restaurant_id = row["restaurant_id"]
            
            self.item_prices[item_id] = float(row.get("price", 100))
            self.item_categories[item_id] = str(row.get("item_category", "other")).lower()
            
            if restaurant_id not in self.restaurant_items:
                self.restaurant_items[restaurant_id] = []
            self.restaurant_items[restaurant_id].append(item_id)
        
        self.all_restaurants = list(self.restaurant_items.keys())
        
        logger.info(f"Loaded {len(self.item_to_idx)} items, {len(self.all_restaurants)} restaurants, {len(self.all_users)} users")
    
    def generate_random_request(self) -> Dict:
        """Generate a random request for benchmarking."""
        restaurant_id = random.choice(self.all_restaurants)
        user_id = random.choice(self.all_users)
        
        # Random cart size (0-5 items)
        restaurant_items = self.restaurant_items.get(restaurant_id, [])
        cart_size = random.randint(0, min(5, len(restaurant_items) - 1))
        cart_items = random.sample(restaurant_items, cart_size) if cart_size > 0 else []
        
        return {
            "user_id": user_id,
            "restaurant_id": restaurant_id,
            "cart_item_ids": cart_items,
            "hour": random.randint(6, 23),
            "meal_type": random.choice(MEAL_TYPES),
            "user_type": random.choice(USER_TYPES)
        }
    
    @torch.no_grad()
    def infer(self, request: Dict) -> Tuple[List[Tuple[str, float]], float, float, float]:
        """
        Run inference on a single request.
        
        Returns:
            recommendations: List of (item_id, score)
            total_ms: Total latency
            feature_ms: Feature construction time
            model_ms: Model forward pass time
        """
        total_start = time.time()
        
        # Get candidates
        restaurant_id = request["restaurant_id"]
        cart_items = request["cart_item_ids"]
        
        all_items = self.restaurant_items.get(restaurant_id, [])
        cart_set = set(cart_items)
        
        candidates = []
        for item_id in all_items:
            if item_id in cart_set:
                continue
            if item_id not in self.item_to_idx:
                continue
            candidates.append(item_id)
            if len(candidates) >= MAX_CANDIDATES:
                break
        
        if not candidates:
            return [], 0.0, 0.0, 0.0
        
        n_candidates = len(candidates)
        
        # Build features
        feature_start = time.time()
        
        # Cart embedding (mean pooling)
        if cart_items:
            cart_indices = [self.item_to_idx.get(i, 0) + 1 for i in cart_items if i in self.item_to_idx]
            if cart_indices:
                cart_idx_tensor = torch.tensor(cart_indices, dtype=torch.long, device=self.device)
                cart_emb = self.item_embeddings[cart_idx_tensor].mean(dim=0, keepdim=True)
            else:
                cart_emb = torch.zeros(1, EMBEDDING_DIM, device=self.device)
        else:
            cart_emb = torch.zeros(1, EMBEDDING_DIM, device=self.device)
        cart_emb = cart_emb.expand(n_candidates, -1)
        
        # Cart dynamic features
        if cart_items:
            prices = [self.item_prices.get(i, 100) for i in cart_items]
            cart_total = sum(prices) / 1000.0
            cart_size_norm = len(cart_items) / MAX_CART_SIZE
            avg_price = np.mean(prices) / 500.0
            
            categories = set(self.item_categories.get(i, "other") for i in cart_items)
            missing_bev = 1.0 if "beverage" not in categories else 0.0
            missing_des = 1.0 if "dessert" not in categories else 0.0
            
            main_count = sum(1 for i in cart_items if self.item_categories.get(i) == "main")
            side_count = sum(1 for i in cart_items if self.item_categories.get(i) == "side")
            heavy_meal = 1.0 if (main_count >= 2 or (main_count >= 1 and side_count >= 1)) else 0.0
            
            cart_dyn = [cart_total, cart_size_norm, avg_price, missing_bev, missing_des, heavy_meal]
        else:
            cart_dyn = [0.0, 0.0, 0.0, 1.0, 1.0, 0.0]
        
        cart_dyn_tensor = torch.tensor([cart_dyn] * n_candidates, dtype=torch.float32, device=self.device)
        
        # User features
        user_feat = self.user_features.get(request["user_id"], self.default_user_features)
        user_tensor = torch.tensor([user_feat] * n_candidates, dtype=torch.float32, device=self.device)
        
        # Restaurant features
        rest_feat = self.restaurant_features.get(request["restaurant_id"], self.default_restaurant_features)
        rest_tensor = torch.tensor([rest_feat] * n_candidates, dtype=torch.float32, device=self.device)
        
        # Candidate embeddings
        cand_indices = [self.item_to_idx.get(i, 0) + 1 for i in candidates]
        cand_tensor = torch.tensor(cand_indices, dtype=torch.long, device=self.device)
        cand_emb = self.item_embeddings[cand_tensor]
        
        # Context features (with category one-hot)
        hour_norm = request["hour"] / 24.0
        meal_idx = float(MEAL_TYPE_MAP.get(request["meal_type"], 1))
        user_type_idx = float(USER_TYPE_MAP.get(request["user_type"], 1))
        
        context_batch = []
        for item_id in candidates:
            cat = self.item_categories.get(item_id, "other")
            cat_idx = CATEGORY_MAP.get(cat.lower(), 7)
            ctx = [
                hour_norm, meal_idx, user_type_idx,
                1.0 if cat_idx == 0 else 0.0,
                1.0 if cat_idx == 2 else 0.0,
                1.0 if cat_idx == 3 else 0.0,
                1.0 if cat_idx == 4 else 0.0,
            ]
            context_batch.append(ctx)
        context_tensor = torch.tensor(context_batch, dtype=torch.float32, device=self.device)
        
        # Interaction features
        interaction_dot = (cart_emb * cand_emb).sum(dim=1, keepdim=True)
        interaction_abs = torch.abs(cart_emb - cand_emb)
        
        feature_ms = (time.time() - feature_start) * 1000
        
        # Model forward pass
        model_start = time.time()
        
        combined = torch.cat([
            cart_emb, cand_emb, interaction_dot, interaction_abs,
            user_tensor, rest_tensor, cart_dyn_tensor, context_tensor
        ], dim=1)
        
        logits = self.model.mlp(combined)
        scores = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
        
        model_ms = (time.time() - model_start) * 1000
        total_ms = (time.time() - total_start) * 1000
        
        # Sort and return top-8
        sorted_indices = np.argsort(-scores)[:8]
        recommendations = [(candidates[i], float(scores[i])) for i in sorted_indices]
        
        return recommendations, total_ms, feature_ms, model_ms


# -----------------------------------------------------------------------------
# Benchmark Functions
# -----------------------------------------------------------------------------

@dataclass
class LatencyStats:
    """Latency statistics."""
    count: int
    mean_ms: float
    std_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float
    min_ms: float


def compute_stats(latencies: List[float]) -> LatencyStats:
    """Compute latency statistics."""
    arr = np.array(latencies)
    return LatencyStats(
        count=len(arr),
        mean_ms=np.mean(arr),
        std_ms=np.std(arr),
        p50_ms=np.percentile(arr, 50),
        p95_ms=np.percentile(arr, 95),
        p99_ms=np.percentile(arr, 99),
        max_ms=np.max(arr),
        min_ms=np.min(arr)
    )


def benchmark_single_request(model: BenchmarkModel, num_requests: int = 100) -> Dict[str, LatencyStats]:
    """
    Benchmark single request latency.
    
    Args:
        model: Loaded benchmark model
        num_requests: Number of requests to simulate
    
    Returns:
        Dict with latency stats for total, feature, and model times
    """
    logger.info(f"Benchmarking {num_requests} single requests...")
    
    # Warmup
    logger.info("  Warmup (10 requests)...")
    for _ in range(10):
        request = model.generate_random_request()
        model.infer(request)
    
    # Benchmark
    total_latencies = []
    feature_latencies = []
    model_latencies = []
    
    logger.info("  Running benchmark...")
    for i in range(num_requests):
        request = model.generate_random_request()
        _, total_ms, feature_ms, model_ms = model.infer(request)
        
        total_latencies.append(total_ms)
        feature_latencies.append(feature_ms)
        model_latencies.append(model_ms)
        
        if (i + 1) % 25 == 0:
            logger.info(f"    Completed {i+1}/{num_requests}")
    
    return {
        "total": compute_stats(total_latencies),
        "feature_build": compute_stats(feature_latencies),
        "model_forward": compute_stats(model_latencies)
    }


def benchmark_batch_requests(model: BenchmarkModel, batch_sizes: List[int] = [100, 1000]) -> Dict[int, float]:
    """
    Benchmark batch request throughput.
    
    Args:
        model: Loaded benchmark model
        batch_sizes: List of batch sizes to test
    
    Returns:
        Dict mapping batch_size -> requests_per_second
    """
    results = {}
    
    for batch_size in batch_sizes:
        logger.info(f"Benchmarking batch of {batch_size} requests...")
        
        # Generate all requests upfront
        requests = [model.generate_random_request() for _ in range(batch_size)]
        
        # Time the batch
        start = time.time()
        for request in requests:
            model.infer(request)
        elapsed = time.time() - start
        
        throughput = batch_size / elapsed
        results[batch_size] = throughput
        
        logger.info(f"  Batch {batch_size}: {elapsed:.2f}s, {throughput:.1f} req/s")
    
    return results


def detect_bottlenecks(stats: Dict[str, LatencyStats]) -> List[str]:
    """Detect potential bottlenecks."""
    bottlenecks = []
    
    total_mean = stats["total"].mean_ms
    feature_mean = stats["feature_build"].mean_ms
    model_mean = stats["model_forward"].mean_ms
    
    # Check if total exceeds target
    if total_mean > 200:
        bottlenecks.append(f"⚠️ Total latency ({total_mean:.1f}ms) exceeds 200ms target")
    elif total_mean > 50:
        bottlenecks.append(f"⚡ Total latency ({total_mean:.1f}ms) above 50ms ideal")
    else:
        bottlenecks.append(f"✅ Total latency ({total_mean:.1f}ms) meets <50ms target")
    
    # Check component breakdown
    if feature_mean > model_mean * 2:
        bottlenecks.append(f"🔧 Feature construction ({feature_mean:.1f}ms) dominates model forward ({model_mean:.1f}ms)")
    
    # Check variance
    if stats["total"].p95_ms > stats["total"].mean_ms * 2:
        bottlenecks.append(f"📊 High latency variance (p95={stats['total'].p95_ms:.1f}ms vs mean={total_mean:.1f}ms)")
    
    return bottlenecks


def print_benchmark_results(
    single_stats: Dict[str, LatencyStats],
    batch_results: Dict[int, float],
    device: str,
    bottlenecks: List[str]
):
    """Print formatted benchmark results."""
    
    lines = []
    lines.append("=" * 70)
    lines.append("INFERENCE BENCHMARK RESULTS")
    lines.append("=" * 70)
    lines.append(f"Device: {device}")
    lines.append("")
    
    lines.append("-" * 70)
    lines.append("SINGLE REQUEST LATENCY (100 requests)")
    lines.append("-" * 70)
    lines.append(f"{'Component':<20} {'Mean':>10} {'P50':>10} {'P95':>10} {'P99':>10} {'Max':>10}")
    lines.append("-" * 70)
    
    for name, stats in single_stats.items():
        lines.append(
            f"{name:<20} {stats.mean_ms:>9.2f}ms {stats.p50_ms:>9.2f}ms "
            f"{stats.p95_ms:>9.2f}ms {stats.p99_ms:>9.2f}ms {stats.max_ms:>9.2f}ms"
        )
    
    lines.append("")
    lines.append("-" * 70)
    lines.append("BATCH THROUGHPUT")
    lines.append("-" * 70)
    
    for batch_size, throughput in batch_results.items():
        lines.append(f"Batch {batch_size}: {throughput:.1f} requests/second")
    
    lines.append("")
    lines.append("-" * 70)
    lines.append("ANALYSIS")
    lines.append("-" * 70)
    
    for b in bottlenecks:
        lines.append(b)
    
    lines.append("=" * 70)
    
    result = "\n".join(lines)
    print(result)
    logger.info("\n" + result)
    
    return result


# -----------------------------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Benchmark Inference Service")
    parser.add_argument(
        "--requests", type=int, default=100,
        help="Number of single requests to benchmark (default: 100)"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["cpu", "cuda", "mps", "auto"],
        help="Device for inference (default: auto)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    args = parser.parse_args()
    
    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Determine device
    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    else:
        device = args.device
    
    logger.info("=" * 70)
    logger.info("Cart Add-on Recommendation Inference Benchmark")
    logger.info("=" * 70)
    logger.info(f"Device: {device}")
    logger.info(f"Requests: {args.requests}")
    logger.info("")
    
    # Load model
    model = BenchmarkModel(device=device)
    model.load()
    
    # Run benchmarks
    single_stats = benchmark_single_request(model, num_requests=args.requests)
    batch_results = benchmark_batch_requests(model, batch_sizes=[100, 1000])
    
    # Detect bottlenecks
    bottlenecks = detect_bottlenecks(single_stats)
    
    # Print results
    print_benchmark_results(single_stats, batch_results, device, bottlenecks)
    
    return {
        "single_stats": single_stats,
        "batch_results": batch_results,
        "device": device,
        "bottlenecks": bottlenecks
    }


if __name__ == "__main__":
    main()
