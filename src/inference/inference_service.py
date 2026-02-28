"""
Production-Ready Inference Service for Cart Add-on Recommendations

FastAPI service that provides real-time add-on recommendations based on:
- Current cart contents
- User preferences
- Restaurant context
- Temporal features

Endpoints:
    POST /recommend - Get top-K recommendations for a cart

Usage:
    uvicorn src.inference.inference_service:app --host 0.0.0.0 --port 8000
    
    # Or for development:
    python -m src.inference.inference_service

Author: ZOMATHON Team
Date: February 2026
"""

import logging
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

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
FEATURE_SCALERS_PATH = DATA_DIR / "feature_scalers.pkl"
USER_FEATURES_PATH = DATA_DIR / "user_features.parquet"
RESTAURANT_FEATURES_PATH = DATA_DIR / "restaurant_features.parquet"
MENU_ITEMS_PATH = DATA_DIR / "menu_items_enriched.csv"

# Model constants
MAX_CART_SIZE = 10
EMBEDDING_DIM = 64
TOP_K_RECOMMENDATIONS = 8
MAX_CANDIDATES = 50

# Feature mappings
MEAL_TYPE_MAP = {"breakfast": 0, "lunch": 1, "snack": 2, "dinner": 3, "late_night": 4}
USER_TYPE_MAP = {"budget": 0, "moderate": 1, "premium": 2, "luxury": 3}
CATEGORY_MAP = {"main": 0, "starter": 1, "dessert": 2, "beverage": 3, "side": 4, "snack": 5, "combo": 6, "other": 7}


# -----------------------------------------------------------------------------
# Pydantic Models
# -----------------------------------------------------------------------------

class RecommendRequest(BaseModel):
    """Input schema for recommendation request."""
    user_id: str = Field(..., description="User identifier (e.g., USER_xxx)")
    restaurant_id: str = Field(..., description="Restaurant identifier (e.g., REST_xxx)")
    cart_item_ids: List[str] = Field(default=[], description="List of item IDs already in cart")
    hour: int = Field(default=12, ge=0, le=23, description="Hour of day (0-23)")
    meal_type: str = Field(default="lunch", description="Meal type: breakfast, lunch, snack, dinner, late_night")
    user_type: str = Field(default="moderate", description="User spending type: budget, moderate, premium, luxury")


class RecommendationItem(BaseModel):
    """Single recommendation item."""
    item_id: str
    item_name: str
    price: float
    category: str
    score: float


class RecommendResponse(BaseModel):
    """Output schema for recommendation response."""
    recommendations: List[RecommendationItem]
    latency_ms: float
    feature_build_ms: float
    model_forward_ms: float
    candidate_count: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    device: str
    num_items: int
    num_restaurants: int
    num_users: int


# -----------------------------------------------------------------------------
# Model Wrapper
# -----------------------------------------------------------------------------

class InferenceModel:
    """
    Wrapper for the CartAddToCartModel for inference.
    
    Handles:
    - Model loading with proper device placement
    - Embedding lookups
    - Feature construction
    - Candidate ranking
    """
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.model = None
        self.item_embeddings = None
        self.item_to_idx: Dict[str, int] = {}
        self.idx_to_item: Dict[int, str] = {}
        
        # Feature lookups
        self.user_features: Dict[str, List[float]] = {}
        self.restaurant_features: Dict[str, List[float]] = {}
        self.item_prices: Dict[str, float] = {}
        self.item_categories: Dict[str, str] = {}
        self.item_names: Dict[str, str] = {}
        
        # Restaurant candidate pools (sorted by popularity)
        self.restaurant_items: Dict[str, List[str]] = {}
        
        # Default features
        self.default_user_features = [0.0, 0.0, 0.0, 0.5, 0.0, 0.1, 0.15]
        self.default_restaurant_features = [0.25, 0.0, 0.0, 0.5, 0.0]
        
        self._loaded = False
    
    def load(self):
        """Load all model components and data."""
        start_time = time.time()
        logger.info("Loading inference model components...")
        
        # 1. Load item embeddings
        logger.info(f"  Loading embeddings from {EMBEDDINGS_PATH}...")
        embeddings_np = np.load(str(EMBEDDINGS_PATH))
        # Add padding embedding at index 0
        padding = np.zeros((1, EMBEDDING_DIM), dtype=np.float32)
        embeddings_np = np.vstack([padding, embeddings_np])
        self.item_embeddings = torch.from_numpy(embeddings_np).to(self.device)
        logger.info(f"    Loaded {embeddings_np.shape[0]} embeddings")
        
        # 2. Load item mapping
        logger.info(f"  Loading item mapping from {ITEM_MAPPING_PATH}...")
        with open(ITEM_MAPPING_PATH, "rb") as f:
            mapping_data = pickle.load(f)
        self.item_to_idx = mapping_data["item_to_idx"]
        self.idx_to_item = mapping_data["idx_to_item"]
        logger.info(f"    Loaded {len(self.item_to_idx)} items")
        
        # 3. Load model
        logger.info(f"  Loading model from {MODEL_PATH}...")
        checkpoint = torch.load(MODEL_PATH, map_location=self.device, weights_only=False)
        config = checkpoint.get("config", {})
        
        # Build model architecture
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
        logger.info(f"    Model loaded: {self.model.count_parameters():,} parameters")
        
        # 4. Load user features
        logger.info(f"  Loading user features from {USER_FEATURES_PATH}...")
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
        logger.info(f"    Loaded {len(self.user_features)} users")
        
        # 5. Load restaurant features
        logger.info(f"  Loading restaurant features from {RESTAURANT_FEATURES_PATH}...")
        restaurants_df = pd.read_parquet(RESTAURANT_FEATURES_PATH)
        for _, row in restaurants_df.iterrows():
            self.restaurant_features[row["restaurant_id"]] = [
                float(row.get("smoothed_rating_norm", 0.25)),
                float(row.get("delivery_votes_norm", 0.0)),
                float(row.get("avg_item_price_norm", 0.0)),
                float(row.get("price_band_index_norm", 0.5)),
                float(row.get("menu_size_norm", 0.0)),
            ]
        logger.info(f"    Loaded {len(self.restaurant_features)} restaurants")
        
        # 6. Load menu items
        logger.info(f"  Loading menu items from {MENU_ITEMS_PATH}...")
        menu_df = pd.read_csv(MENU_ITEMS_PATH)
        
        # Build restaurant -> items mapping (sorted by popularity)
        menu_df_sorted = menu_df.sort_values(["restaurant_id", "item_votes"], ascending=[True, False])
        for _, row in menu_df_sorted.iterrows():
            item_id = row["item_id"]
            restaurant_id = row["restaurant_id"]
            
            self.item_prices[item_id] = float(row.get("price", 100))
            self.item_categories[item_id] = str(row.get("item_category", "other")).lower()
            self.item_names[item_id] = str(row.get("item_name", item_id))
            
            if restaurant_id not in self.restaurant_items:
                self.restaurant_items[restaurant_id] = []
            self.restaurant_items[restaurant_id].append(item_id)
        
        logger.info(f"    Loaded {len(self.item_prices)} items from {len(self.restaurant_items)} restaurants")
        
        self._loaded = True
        elapsed = time.time() - start_time
        logger.info(f"Model loading complete in {elapsed:.2f}s")
    
    def is_loaded(self) -> bool:
        return self._loaded
    
    def get_candidate_pool(
        self,
        restaurant_id: str,
        cart_item_ids: List[str],
        max_candidates: int = MAX_CANDIDATES
    ) -> List[str]:
        """
        Get candidate pool for a restaurant.
        
        - Fetches top items by popularity
        - Excludes items already in cart
        - Filters to items with valid embeddings
        """
        # Get restaurant items sorted by popularity
        all_items = self.restaurant_items.get(restaurant_id, [])
        
        if not all_items:
            logger.warning(f"No items found for restaurant {restaurant_id}")
            return []
        
        # Exclude cart items
        cart_set = set(cart_item_ids)
        
        candidates = []
        for item_id in all_items:
            if item_id in cart_set:
                continue
            if item_id not in self.item_to_idx:
                continue
            candidates.append(item_id)
            if len(candidates) >= max_candidates:
                break
        
        return candidates
    
    def build_cart_embedding(self, cart_item_ids: List[str]) -> torch.Tensor:
        """
        Build cart embedding via mean pooling.
        
        Returns: (1, EMBEDDING_DIM) tensor
        """
        if not cart_item_ids:
            return torch.zeros(1, EMBEDDING_DIM, device=self.device)
        
        indices = []
        for item_id in cart_item_ids[:MAX_CART_SIZE]:
            idx = self.item_to_idx.get(item_id, -1)
            if idx >= 0:
                indices.append(idx + 1)  # +1 for padding offset
        
        if not indices:
            return torch.zeros(1, EMBEDDING_DIM, device=self.device)
        
        indices_tensor = torch.tensor(indices, dtype=torch.long, device=self.device)
        embeddings = self.item_embeddings[indices_tensor]  # (n_items, 64)
        cart_embedding = embeddings.mean(dim=0, keepdim=True)  # (1, 64)
        
        return cart_embedding
    
    def build_cart_dynamic_features(self, cart_item_ids: List[str]) -> List[float]:
        """
        Build cart dynamic features.
        
        Returns: [cart_total_norm, cart_size_norm, avg_price_norm, 
                  missing_beverage, missing_dessert, heavy_meal]
        """
        if not cart_item_ids:
            return [0.0, 0.0, 0.0, 1.0, 1.0, 0.0]
        
        # Compute cart total and avg price
        prices = [self.item_prices.get(item_id, 100) for item_id in cart_item_ids]
        cart_total = sum(prices)
        avg_price = np.mean(prices)
        cart_size = len(cart_item_ids)
        
        # Check cart categories
        categories = set(self.item_categories.get(item_id, "other") for item_id in cart_item_ids)
        missing_beverage = 1.0 if "beverage" not in categories else 0.0
        missing_dessert = 1.0 if "dessert" not in categories else 0.0
        
        # Heavy meal detection
        main_count = sum(1 for item_id in cart_item_ids if self.item_categories.get(item_id) == "main")
        side_count = sum(1 for item_id in cart_item_ids if self.item_categories.get(item_id) == "side")
        heavy_meal = 1.0 if (main_count >= 2 or (main_count >= 1 and side_count >= 1)) else 0.0
        
        return [
            cart_total / 1000.0,
            cart_size / MAX_CART_SIZE,
            avg_price / 500.0,
            missing_beverage,
            missing_dessert,
            heavy_meal
        ]
    
    def build_context_features(
        self,
        hour: int,
        meal_type: str,
        user_type: str,
        candidate_category: str
    ) -> List[float]:
        """
        Build context features including candidate category one-hot.
        
        Returns: [hour_norm, meal_type_idx, user_type_idx, 
                  is_main, is_dessert, is_beverage, is_side]
        """
        cat_idx = CATEGORY_MAP.get(candidate_category.lower(), 7)
        
        return [
            hour / 24.0,
            float(MEAL_TYPE_MAP.get(meal_type, 1)),
            float(USER_TYPE_MAP.get(user_type, 1)),
            1.0 if cat_idx == 0 else 0.0,  # is_main
            1.0 if cat_idx == 2 else 0.0,  # is_dessert
            1.0 if cat_idx == 3 else 0.0,  # is_beverage
            1.0 if cat_idx == 4 else 0.0,  # is_side
        ]
    
    @torch.no_grad()
    def rank_candidates(
        self,
        request: RecommendRequest,
        candidates: List[str]
    ) -> Tuple[List[Tuple[str, float]], float, float]:
        """
        Rank candidates using the model.
        
        Returns:
            ranked_items: List of (item_id, score) sorted by score descending
            feature_build_ms: Time to build features
            model_forward_ms: Time for model forward pass
        """
        n_candidates = len(candidates)
        if n_candidates == 0:
            return [], 0.0, 0.0
        
        feature_start = time.time()
        
        # Build cart embedding (shared across candidates)
        cart_embedding = self.build_cart_embedding(request.cart_item_ids)  # (1, 64)
        cart_embedding = cart_embedding.expand(n_candidates, -1)  # (n, 64)
        
        # Build cart dynamic features (shared)
        cart_dyn = self.build_cart_dynamic_features(request.cart_item_ids)
        cart_dyn_tensor = torch.tensor([cart_dyn] * n_candidates, dtype=torch.float32, device=self.device)
        
        # User features
        user_feat = self.user_features.get(request.user_id, self.default_user_features)
        user_tensor = torch.tensor([user_feat] * n_candidates, dtype=torch.float32, device=self.device)
        
        # Restaurant features  
        rest_feat = self.restaurant_features.get(request.restaurant_id, self.default_restaurant_features)
        rest_tensor = torch.tensor([rest_feat] * n_candidates, dtype=torch.float32, device=self.device)
        
        # Candidate embeddings
        candidate_indices = []
        for item_id in candidates:
            idx = self.item_to_idx.get(item_id, 0)
            candidate_indices.append(idx + 1)  # +1 for padding offset
        candidate_tensor = torch.tensor(candidate_indices, dtype=torch.long, device=self.device)
        candidate_embeddings = self.item_embeddings[candidate_tensor]  # (n, 64)
        
        # Context features (per candidate due to category)
        context_batch = []
        for item_id in candidates:
            cat = self.item_categories.get(item_id, "other")
            ctx = self.build_context_features(request.hour, request.meal_type, request.user_type, cat)
            context_batch.append(ctx)
        context_tensor = torch.tensor(context_batch, dtype=torch.float32, device=self.device)
        
        # Build interaction features
        interaction_dot = (cart_embedding * candidate_embeddings).sum(dim=1, keepdim=True)  # (n, 1)
        interaction_abs = torch.abs(cart_embedding - candidate_embeddings)  # (n, 64)
        
        feature_time = (time.time() - feature_start) * 1000
        
        # Concatenate all features for MLP
        model_start = time.time()
        
        combined = torch.cat([
            cart_embedding,      # (n, 64)
            candidate_embeddings,  # (n, 64)
            interaction_dot,     # (n, 1)
            interaction_abs,     # (n, 64)
            user_tensor,         # (n, 7)
            rest_tensor,         # (n, 5)
            cart_dyn_tensor,     # (n, 6)
            context_tensor       # (n, 7)
        ], dim=1)
        
        # Forward pass through MLP
        logits = self.model.mlp(combined)  # (n, 1)
        scores = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
        
        model_time = (time.time() - model_start) * 1000
        
        # Rank by score
        sorted_indices = np.argsort(-scores)
        ranked_items = [(candidates[i], float(scores[i])) for i in sorted_indices]
        
        return ranked_items, feature_time, model_time


# -----------------------------------------------------------------------------
# FastAPI Application
# -----------------------------------------------------------------------------

app = FastAPI(
    title="Cart Add-on Recommendation Service",
    description="Real-time add-on recommendations for food delivery carts",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
inference_model: Optional[InferenceModel] = None


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    global inference_model
    
    # Determine device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    logger.info(f"Initializing inference model on device: {device}")
    inference_model = InferenceModel(device=device)
    inference_model.load()


@app.get("/")
async def root():
    """Root endpoint with usage instructions."""
    return {
        "service": "Cart Add-on Recommendation Service",
        "version": "1.0.0",
        "endpoints": {
            "POST /recommend": "Get add-on recommendations (see /docs for schema)",
            "GET /health": "Health check",
            "GET /docs": "Interactive API documentation (Swagger UI)",
            "GET /restaurants": "List available restaurants",
            "GET /restaurant/{id}/items": "List items for a restaurant"
        },
        "example_curl": 'curl -X POST http://localhost:8000/recommend -H "Content-Type: application/json" -d \'{"user_id": "USER_A3C31A078DC3", "restaurant_id": "REST_5C6535B5E80E", "cart_item_ids": [], "hour": 19, "meal_type": "dinner", "user_type": "premium"}\''
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    if inference_model is None or not inference_model.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        device=inference_model.device,
        num_items=len(inference_model.item_to_idx),
        num_restaurants=len(inference_model.restaurant_items),
        num_users=len(inference_model.user_features)
    )


@app.post("/recommend", response_model=RecommendResponse)
async def recommend(request: RecommendRequest):
    """
    Get add-on recommendations for a cart.
    
    Returns top-8 recommended items ranked by predicted add-to-cart probability.
    """
    if inference_model is None or not inference_model.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    total_start = time.time()
    
    # 1. Get candidate pool
    candidates = inference_model.get_candidate_pool(
        request.restaurant_id,
        request.cart_item_ids,
        max_candidates=MAX_CANDIDATES
    )
    
    if not candidates:
        return RecommendResponse(
            recommendations=[],
            latency_ms=0.0,
            feature_build_ms=0.0,
            model_forward_ms=0.0,
            candidate_count=0
        )
    
    # 2. Rank candidates
    ranked_items, feature_ms, model_ms = inference_model.rank_candidates(request, candidates)
    
    # 3. Build response
    recommendations = []
    for item_id, score in ranked_items[:TOP_K_RECOMMENDATIONS]:
        recommendations.append(RecommendationItem(
            item_id=item_id,
            item_name=inference_model.item_names.get(item_id, item_id),
            price=inference_model.item_prices.get(item_id, 0.0),
            category=inference_model.item_categories.get(item_id, "other"),
            score=round(score, 4)
        ))
    
    total_ms = (time.time() - total_start) * 1000
    
    # Log latency
    logger.info(
        f"Request: user={request.user_id}, restaurant={request.restaurant_id}, "
        f"cart_size={len(request.cart_item_ids)}, candidates={len(candidates)}, "
        f"latency={total_ms:.1f}ms (feature={feature_ms:.1f}ms, model={model_ms:.1f}ms)"
    )
    
    return RecommendResponse(
        recommendations=recommendations,
        latency_ms=round(total_ms, 2),
        feature_build_ms=round(feature_ms, 2),
        model_forward_ms=round(model_ms, 2),
        candidate_count=len(candidates)
    )


@app.get("/restaurants")
async def list_restaurants():
    """List available restaurants."""
    if inference_model is None or not inference_model.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    restaurants = list(inference_model.restaurant_items.keys())[:100]
    return {"restaurants": restaurants, "total": len(inference_model.restaurant_items)}


@app.get("/restaurant/{restaurant_id}/items")
async def list_restaurant_items(restaurant_id: str, limit: int = 20):
    """List items for a restaurant."""
    if inference_model is None or not inference_model.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    items = inference_model.restaurant_items.get(restaurant_id, [])[:limit]
    
    result = []
    for item_id in items:
        result.append({
            "item_id": item_id,
            "item_name": inference_model.item_names.get(item_id, item_id),
            "price": inference_model.item_prices.get(item_id, 0.0),
            "category": inference_model.item_categories.get(item_id, "other")
        })
    
    return {"items": result, "total": len(inference_model.restaurant_items.get(restaurant_id, []))}


# -----------------------------------------------------------------------------
# CLI Entry Point (for development)
# -----------------------------------------------------------------------------

def main():
    """Run development server."""
    import uvicorn
    
    logger.info("Starting development server...")
    uvicorn.run(
        "src.inference.inference_service:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )


if __name__ == "__main__":
    main()
