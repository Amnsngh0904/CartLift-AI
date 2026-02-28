"""
Cart Session Simulator

Generates realistic cart events using real restaurant and menu item data.
Creates training data for addon recommendation models.

Features:
- Weighted restaurant selection (rating, votes, cuisine, price)
- Sequential item selection with realistic dependencies
- Negative sampling for contrastive learning
- Time-of-day context simulation

Usage:
    python -m src.data_generation.session_generator
    python -m src.data_generation.session_generator --num-sessions 1000000 --neg-samples 5
"""

import argparse
import hashlib
import logging
import random
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
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
# Constants & Configuration
# -----------------------------------------------------------------------------

# Default paths
DEFAULT_RESTAURANTS_PATH = Path("data/processed/restaurants_cleaned.csv")
DEFAULT_MENU_ITEMS_PATH = Path("data/processed/menu_items_simulation.csv")
DEFAULT_USERS_PATH = Path("data/synthetic/users.csv")
DEFAULT_OUTPUT_PATH = Path("data/synthetic/cart_events.csv")

# Session configuration
DEFAULT_NUM_SESSIONS = 1_000_000
DEFAULT_NEG_SAMPLES = 5
MIN_CART_SIZE = 1
MAX_CART_SIZE = 6

# User generation
NUM_USERS = 50_000

# Time distribution (hour weights for order times)
HOUR_WEIGHTS = {
    # Breakfast (7-10 AM)
    7: 0.5, 8: 1.0, 9: 1.2, 10: 0.8,
    # Lunch (11 AM - 2 PM)
    11: 1.5, 12: 3.0, 13: 2.5, 14: 1.5,
    # Snack time (3-5 PM)
    15: 0.8, 16: 1.0, 17: 1.2,
    # Dinner (6-10 PM)
    18: 1.5, 19: 2.5, 20: 3.0, 21: 2.0, 22: 1.0,
    # Late night (11 PM - 12 AM)
    23: 0.5, 0: 0.2,
}

# Category transition probabilities
CATEGORY_TRANSITIONS = {
    # After main dish
    "main": {
        "side": 0.60,
        "beverage": 0.55,
        "dessert": 0.25,
        "main": 0.15,  # Another main (rare)
    },
    # After side dish
    "side": {
        "main": 0.40,
        "beverage": 0.50,
        "dessert": 0.30,
        "side": 0.20,
    },
    # After beverage
    "beverage": {
        "dessert": 0.35,
        "side": 0.25,
        "main": 0.20,
        "beverage": 0.10,
    },
    # After dessert
    "dessert": {
        "beverage": 0.40,
        "dessert": 0.15,
        "side": 0.10,
        "main": 0.05,
    },
}

# Price brackets for user budget compatibility
PRICE_BRACKETS = {
    "budget": (0, 150),
    "moderate": (100, 350),
    "premium": (250, 800),
    "luxury": (500, float("inf")),
}

# Cuisine preferences (for user-restaurant matching)
CUISINE_GROUPS = {
    "indian": ["North Indian", "South Indian", "Mughlai", "Biryani", "Street Food"],
    "chinese": ["Chinese", "Indo-Chinese", "Asian", "Thai", "Japanese", "Korean"],
    "western": ["Italian", "Continental", "American", "Mexican", "Mediterranean"],
    "fast_food": ["Fast Food", "Pizza", "Burger", "Rolls", "Wraps", "Sandwiches"],
    "healthy": ["Healthy Food", "Salads", "Juices", "Smoothies"],
    "dessert": ["Desserts", "Ice Cream", "Bakery", "Cafe", "Coffee"],
}


# -----------------------------------------------------------------------------
# Data Classes
# -----------------------------------------------------------------------------

class UserType(str, Enum):
    """User spending type."""
    BUDGET = "budget"
    MODERATE = "moderate"
    PREMIUM = "premium"
    LUXURY = "luxury"


class MealType(str, Enum):
    """Meal time type based on hour."""
    BREAKFAST = "breakfast"
    LUNCH = "lunch"
    SNACK = "snack"
    DINNER = "dinner"
    LATE_NIGHT = "late_night"


@dataclass
class User:
    """User profile for session simulation."""
    user_id: str
    user_type: UserType
    preferred_cuisines: List[str]
    avg_order_value: float
    order_frequency: int  # orders per month
    city: str
    
    def get_price_range(self) -> Tuple[float, float]:
        """Get acceptable price range based on user type."""
        return PRICE_BRACKETS[self.user_type.value]


@dataclass
class CartState:
    """Current state of a shopping cart."""
    session_id: str
    user: User
    restaurant_id: str
    items: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    total_price: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    
    def has_main(self) -> bool:
        return "main" in self.categories
    
    def has_beverage(self) -> bool:
        return "beverage" in self.categories
    
    def has_dessert(self) -> bool:
        return "dessert" in self.categories
    
    def is_heavy_meal(self) -> bool:
        """Check if cart contains heavy meal (multiple mains or main+sides)."""
        main_count = self.categories.count("main")
        side_count = self.categories.count("side")
        return main_count >= 2 or (main_count >= 1 and side_count >= 1)
    
    def get_last_category(self) -> Optional[str]:
        return self.categories[-1] if self.categories else None


@dataclass
class CartEvent:
    """Single cart event (add item decision)."""
    session_id: str
    step_number: int
    user_id: str
    restaurant_id: str
    cart_items: str  # Pipe-separated item IDs
    candidate_item: str
    label: int  # 1 = added, 0 = not added
    timestamp: str
    city: str
    hour: int
    meal_type: str
    user_type: str
    cart_total: float
    cart_categories: str  # Pipe-separated categories


# -----------------------------------------------------------------------------
# User Generation
# -----------------------------------------------------------------------------

def generate_users(
    num_users: int,
    cities: List[str],
    output_path: Path
) -> pd.DataFrame:
    """
    Generate synthetic user profiles.
    
    Args:
        num_users: Number of users to generate
        cities: List of available cities
        output_path: Path to save users CSV
        
    Returns:
        DataFrame with user profiles
    """
    logger.info(f"Generating {num_users:,} synthetic users...")
    
    # User type distribution (realistic bell curve)
    user_type_weights = {
        UserType.BUDGET: 0.30,
        UserType.MODERATE: 0.45,
        UserType.PREMIUM: 0.20,
        UserType.LUXURY: 0.05,
    }
    
    user_types = random.choices(
        list(user_type_weights.keys()),
        weights=list(user_type_weights.values()),
        k=num_users
    )
    
    # City distribution (weighted by size)
    city_weights = {city: 1.0 for city in cities}
    # Boost metro cities
    for metro in ["Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai", "Kolkata"]:
        if metro in city_weights:
            city_weights[metro] = 3.0
    
    city_choices = random.choices(
        list(city_weights.keys()),
        weights=list(city_weights.values()),
        k=num_users
    )
    
    # Generate users
    users = []
    cuisine_groups_list = list(CUISINE_GROUPS.keys())
    
    for i in range(num_users):
        user_type = user_types[i]
        
        # Average order value based on user type
        if user_type == UserType.BUDGET:
            avg_order = np.random.normal(200, 50)
        elif user_type == UserType.MODERATE:
            avg_order = np.random.normal(400, 100)
        elif user_type == UserType.PREMIUM:
            avg_order = np.random.normal(700, 150)
        else:  # LUXURY
            avg_order = np.random.normal(1200, 300)
        
        avg_order = max(100, avg_order)
        
        # Order frequency (orders per month)
        order_freq = int(np.random.exponential(5) + 1)
        order_freq = min(30, order_freq)
        
        # Cuisine preferences (1-3 preferred groups)
        num_pref = random.randint(1, 3)
        preferred_cuisines = random.sample(cuisine_groups_list, num_pref)
        
        user_id = f"USER_{hashlib.md5(f'{i}'.encode()).hexdigest()[:12].upper()}"
        
        users.append({
            "user_id": user_id,
            "user_type": user_type.value,
            "preferred_cuisines": "|".join(preferred_cuisines),
            "avg_order_value": round(avg_order, 2),
            "order_frequency": order_freq,
            "city": city_choices[i],
        })
    
    df = pd.DataFrame(users)
    
    # Save to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(df):,} users to {output_path}")
    
    # Print distribution
    logger.info("User type distribution:")
    for ut, count in df["user_type"].value_counts().items():
        logger.info(f"  {ut}: {count:,} ({100*count/len(df):.1f}%)")
    
    return df


def load_users(users_path: Path, cities: List[str]) -> List[User]:
    """Load or generate users."""
    if not users_path.exists():
        logger.info(f"Users file not found at {users_path}, generating...")
        df = generate_users(NUM_USERS, cities, users_path)
    else:
        logger.info(f"Loading users from {users_path}")
        df = pd.read_csv(users_path)
        logger.info(f"Loaded {len(df):,} users")
    
    users = []
    for _, row in df.iterrows():
        users.append(User(
            user_id=row["user_id"],
            user_type=UserType(row["user_type"]),
            preferred_cuisines=row["preferred_cuisines"].split("|"),
            avg_order_value=row["avg_order_value"],
            order_frequency=row["order_frequency"],
            city=row["city"],
        ))
    
    return users


# -----------------------------------------------------------------------------
# Restaurant Selection
# -----------------------------------------------------------------------------

class RestaurantSelector:
    """
    Selects restaurants with weighted probability.
    
    Weights based on:
    - Smoothed rating
    - Delivery votes (popularity)
    - Cuisine match with user preference
    - Price compatibility with user budget
    """
    
    def __init__(self, restaurants_df: pd.DataFrame, menu_items_df: pd.DataFrame):
        self.restaurants = restaurants_df
        self.menu_items = menu_items_df
        
        # Pre-compute restaurant average prices
        self.restaurant_avg_price = menu_items_df.groupby("restaurant_id")["price"].mean()
        
        # Pre-compute restaurant cuisine groups
        self.restaurant_cuisine_group = {}
        for _, row in restaurants_df.iterrows():
            rest_id = row["restaurant_id"]
            cuisine = row["cuisine"]
            self.restaurant_cuisine_group[rest_id] = self._get_cuisine_group(cuisine)
        
        # Index restaurants by city for efficient lookup
        self.restaurants_by_city: Dict[str, pd.DataFrame] = {}
        for city in restaurants_df["city"].unique():
            self.restaurants_by_city[city] = restaurants_df[
                restaurants_df["city"] == city
            ].copy()
        
        logger.info(f"RestaurantSelector initialized with {len(restaurants_df)} restaurants")
    
    def _get_cuisine_group(self, cuisine: str) -> str:
        """Map cuisine to cuisine group."""
        cuisine_lower = cuisine.lower()
        for group, cuisines in CUISINE_GROUPS.items():
            for c in cuisines:
                if c.lower() in cuisine_lower or cuisine_lower in c.lower():
                    return group
        return "indian"  # Default
    
    def _compute_weights(
        self,
        restaurants: pd.DataFrame,
        user: User
    ) -> np.ndarray:
        """
        Compute selection weights for restaurants.
        
        Formula:
            weight = rating_score * popularity_score * cuisine_score * price_score
        """
        weights = np.ones(len(restaurants))
        
        for i, (_, row) in enumerate(restaurants.iterrows()):
            rest_id = row["restaurant_id"]
            
            # Rating score (normalize to 0-1, smoothed rating ~3.1-4.5)
            rating_score = (row["smoothed_rating"] - 3.0) / 1.5
            rating_score = max(0.1, min(1.0, rating_score))
            
            # Popularity score (log-scaled delivery votes)
            votes = row["delivery_votes"]
            if votes > 0:
                pop_score = np.log1p(votes) / np.log1p(1000)  # Normalize to ~1 at 1000 votes
                pop_score = min(1.5, pop_score)
            else:
                pop_score = 0.5  # Base score for zero-vote restaurants
            
            # Cuisine match score
            rest_cuisine_group = self.restaurant_cuisine_group.get(rest_id, "indian")
            if rest_cuisine_group in user.preferred_cuisines:
                cuisine_score = 1.5  # Boost for preferred cuisine
            else:
                cuisine_score = 0.8
            
            # Price compatibility score
            avg_price = self.restaurant_avg_price.get(rest_id, 200)
            user_price_range = user.get_price_range()
            if user_price_range[0] <= avg_price <= user_price_range[1]:
                price_score = 1.2
            elif avg_price < user_price_range[0]:
                price_score = 0.7  # Too cheap
            else:
                price_score = 0.5  # Too expensive
            
            # Combined weight
            weights[i] = rating_score * pop_score * cuisine_score * price_score
        
        # Normalize to probabilities
        weights = weights / weights.sum()
        
        return weights
    
    def select_restaurant(self, user: User) -> Optional[str]:
        """
        Select a restaurant for the user.
        
        Args:
            user: User profile
            
        Returns:
            Restaurant ID or None if no suitable restaurants
        """
        # Get restaurants in user's city (or all if city not found)
        if user.city in self.restaurants_by_city:
            candidates = self.restaurants_by_city[user.city]
        else:
            candidates = self.restaurants
        
        if len(candidates) == 0:
            return None
        
        # Compute selection weights
        weights = self._compute_weights(candidates, user)
        
        # Sample restaurant
        idx = np.random.choice(len(candidates), p=weights)
        return candidates.iloc[idx]["restaurant_id"]


# -----------------------------------------------------------------------------
# Item Selection
# -----------------------------------------------------------------------------

class ItemSelector:
    """
    Selects menu items with realistic probability.
    
    First item selection:
        P(item) ∝ 0.5 * normalized_votes + 0.3 * best_seller + 0.2 * price_compat
    
    Sequential selection:
        Based on category transitions and cart state.
    """
    
    def __init__(self, menu_items_df: pd.DataFrame):
        self.menu_items = menu_items_df
        
        # Index items by restaurant
        self.items_by_restaurant: Dict[str, pd.DataFrame] = {}
        for rest_id in menu_items_df["restaurant_id"].unique():
            self.items_by_restaurant[rest_id] = menu_items_df[
                menu_items_df["restaurant_id"] == rest_id
            ].copy()
        
        # Pre-compute normalized votes per restaurant
        self.normalized_votes: Dict[str, np.ndarray] = {}
        for rest_id, items in self.items_by_restaurant.items():
            votes = items["item_votes"].values.astype(float)
            max_votes = votes.max()
            if max_votes > 0:
                self.normalized_votes[rest_id] = votes / max_votes
            else:
                self.normalized_votes[rest_id] = np.ones(len(items)) / len(items)
        
        logger.info(f"ItemSelector initialized with {len(menu_items_df)} items")
    
    def _compute_first_item_weights(
        self,
        items: pd.DataFrame,
        user: User,
        meal_type: MealType
    ) -> np.ndarray:
        """
        Compute weights for first item selection.
        
        Formula:
            weight = 0.5 * norm_votes + 0.3 * best_seller + 0.2 * price_compat
        """
        n = len(items)
        rest_id = items.iloc[0]["restaurant_id"]
        
        # Normalized votes
        norm_votes = self.normalized_votes.get(rest_id, np.ones(n) / n)
        
        # Best seller flag
        best_seller = items["best_seller"].values.astype(float)
        
        # Price compatibility
        user_range = user.get_price_range()
        prices = items["price"].values
        price_compat = np.ones(n)
        for i, price in enumerate(prices):
            if user_range[0] <= price <= user_range[1]:
                price_compat[i] = 1.0
            elif price < user_range[0]:
                price_compat[i] = 0.6
            else:
                price_compat[i] = 0.3
        
        # Meal type boost (breakfast -> lighter items, dinner -> heavier)
        category_boost = np.ones(n)
        categories = items["item_category"].values
        
        if meal_type == MealType.BREAKFAST:
            # Prefer beverages and light items
            for i, cat in enumerate(categories):
                if cat == "beverage":
                    category_boost[i] = 1.5
                elif cat == "side":
                    category_boost[i] = 1.2
                elif cat == "main":
                    category_boost[i] = 0.8
        elif meal_type in [MealType.LUNCH, MealType.DINNER]:
            # Prefer mains
            for i, cat in enumerate(categories):
                if cat == "main":
                    category_boost[i] = 1.5
        elif meal_type == MealType.SNACK:
            # Prefer desserts and beverages
            for i, cat in enumerate(categories):
                if cat in ["dessert", "beverage"]:
                    category_boost[i] = 1.3
        
        # Combined weight
        weights = (
            0.5 * norm_votes +
            0.3 * best_seller +
            0.2 * price_compat
        ) * category_boost
        
        # Handle NaN or zero weights
        weights = np.nan_to_num(weights, nan=0.0)
        total = weights.sum()
        if total > 0:
            weights = weights / total
        else:
            # Fallback: uniform distribution
            weights = np.ones(n) / n
        
        return weights
    
    def _compute_sequential_weights(
        self,
        items: pd.DataFrame,
        cart_state: CartState,
        user: User,
        meal_type: Optional[MealType] = None
    ) -> np.ndarray:
        """
        Compute weights for sequential item selection.
        
        Based on:
        - Category transitions from last item
        - Cart state (main present, heavy meal, etc.)
        - User type (premium users like desserts)
        - Meal type (lunch/dinner boosts beverages)
        
        Behavioral signals:
        - Cart has main but no beverage → beverages get 6x weight
        - Cart has heavy meal → desserts get 4x weight
        - Premium user → dessert probability ×3
        - Lunch/dinner → beverage probability ×2
        - After first main → reduce main probability ×0.5
        """
        n = len(items)
        rest_id = items.iloc[0]["restaurant_id"]
        
        # Base weights from votes (reduced randomness)
        norm_votes = self.normalized_votes.get(rest_id, np.ones(n) / n)
        # Reduced noise: lower weight on votes, higher on best_seller (more signal)
        base_weights = 0.15 * norm_votes + 0.35 * items["best_seller"].values.astype(float) + 0.5
        
        # Category transition weights
        last_cat = cart_state.get_last_category()
        categories = items["item_category"].values
        
        category_weights = np.ones(n)
        
        if last_cat and last_cat in CATEGORY_TRANSITIONS:
            transitions = CATEGORY_TRANSITIONS[last_cat]
            for i, cat in enumerate(categories):
                category_weights[i] = transitions.get(cat, 0.1)
        
        # Cart state adjustments - STRENGTHENED SIGNALS
        state_boost = np.ones(n)
        
        # 1. If cart has main but no beverage → beverages get 6x weight
        if cart_state.has_main() and not cart_state.has_beverage():
            for i, cat in enumerate(categories):
                if cat == "beverage":
                    state_boost[i] = 6.0  # Increased from 3.0
        
        # 2. If cart has heavy meal → desserts get 4x weight
        if cart_state.is_heavy_meal():
            for i, cat in enumerate(categories):
                if cat == "dessert":
                    state_boost[i] = 4.0  # Increased from 2.5
        
        # 3. Premium/luxury users → dessert probability ×3
        if user.user_type in [UserType.PREMIUM, UserType.LUXURY]:
            for i, cat in enumerate(categories):
                if cat == "dessert":
                    state_boost[i] *= 3.0  # Increased from 2.0
        
        # 4. Lunch/dinner → beverage probability ×2
        if meal_type in [MealType.LUNCH, MealType.DINNER]:
            for i, cat in enumerate(categories):
                if cat == "beverage":
                    state_boost[i] *= 2.0
        
        # 5. Reduce probability of selecting another main after first main (×0.5)
        if cart_state.has_main():
            for i, cat in enumerate(categories):
                if cat == "main":
                    state_boost[i] *= 0.5
        
        # Avoid duplicates (reduce weight for categories already in cart)
        for i, cat in enumerate(categories):
            if cart_state.categories.count(cat) >= 2:
                state_boost[i] *= 0.3
        
        # Price compatibility
        user_range = user.get_price_range()
        remaining_budget = user.avg_order_value - cart_state.total_price
        remaining_budget = max(50, remaining_budget)
        
        prices = items["price"].values
        price_compat = np.ones(n)
        for i, price in enumerate(prices):
            if price <= remaining_budget:
                price_compat[i] = 1.0
            else:
                price_compat[i] = 0.3  # Over budget
        
        # Combined weight
        weights = base_weights * category_weights * state_boost * price_compat
        
        # Filter out items already in cart
        for i, item_id in enumerate(items["item_id"].values):
            if item_id in cart_state.items:
                weights[i] = 0.0
        
        # Handle NaN values
        weights = np.nan_to_num(weights, nan=0.0)
        
        # Normalize (handle all zeros)
        total = weights.sum()
        if total > 0:
            weights = weights / total
        else:
            # Fallback: uniform over non-cart items
            fallback_weights = np.zeros(len(items))
            for i, item_id in enumerate(items["item_id"].values):
                if item_id not in cart_state.items:
                    fallback_weights[i] = 1.0
            fallback_total = fallback_weights.sum()
            if fallback_total > 0:
                weights = fallback_weights / fallback_total
            else:
                # All items in cart - return uniform (this shouldn't happen)
                weights = np.ones(len(items)) / len(items)
        
        return weights
    
    def select_first_item(
        self,
        restaurant_id: str,
        user: User,
        meal_type: MealType
    ) -> Optional[Tuple[str, str, float]]:
        """
        Select first item for cart.
        
        Returns:
            Tuple of (item_id, category, price) or None
        """
        if restaurant_id not in self.items_by_restaurant:
            return None
        
        items = self.items_by_restaurant[restaurant_id]
        if len(items) == 0:
            return None
        
        weights = self._compute_first_item_weights(items, user, meal_type)
        idx = np.random.choice(len(items), p=weights)
        
        row = items.iloc[idx]
        return (row["item_id"], row["item_category"], row["price"])
    
    def select_next_item(
        self,
        cart_state: CartState,
        user: User,
        meal_type: Optional[MealType] = None
    ) -> Optional[Tuple[str, str, float]]:
        """
        Select next item based on cart state.
        
        Returns:
            Tuple of (item_id, category, price) or None
        """
        rest_id = cart_state.restaurant_id
        if rest_id not in self.items_by_restaurant:
            return None
        
        items = self.items_by_restaurant[rest_id]
        if len(items) == 0:
            return None
        
        weights = self._compute_sequential_weights(items, cart_state, user, meal_type)
        idx = np.random.choice(len(items), p=weights)
        
        row = items.iloc[idx]
        return (row["item_id"], row["item_category"], row["price"])
    
    def get_negative_samples(
        self,
        restaurant_id: str,
        positive_item: str,
        cart_items: List[str],
        num_samples: int = 5
    ) -> List[str]:
        """
        Generate negative samples for contrastive learning.
        
        Selects items that are:
        - Not the positive item
        - Not already in cart
        - From the same restaurant
        
        Sampling weighted by:
        - Category diversity (different category preferred)
        - Votes (some popular items as hard negatives)
        """
        if restaurant_id not in self.items_by_restaurant:
            return []
        
        items = self.items_by_restaurant[restaurant_id]
        
        # Filter out positive and cart items
        excluded = set(cart_items) | {positive_item}
        candidates = items[~items["item_id"].isin(excluded)]
        
        if len(candidates) == 0:
            return []
        
        if len(candidates) <= num_samples:
            return candidates["item_id"].tolist()
        
        # Weight by votes (harder negatives)
        votes = candidates["item_votes"].values.astype(float) + 1
        weights = np.log1p(votes)
        weights = weights / weights.sum()
        
        indices = np.random.choice(
            len(candidates),
            size=min(num_samples, len(candidates)),
            replace=False,
            p=weights
        )
        
        return candidates.iloc[indices]["item_id"].tolist()


# -----------------------------------------------------------------------------
# Session Generation
# -----------------------------------------------------------------------------

def get_meal_type(hour: int) -> MealType:
    """Determine meal type from hour."""
    if 7 <= hour <= 10:
        return MealType.BREAKFAST
    elif 11 <= hour <= 14:
        return MealType.LUNCH
    elif 15 <= hour <= 17:
        return MealType.SNACK
    elif 18 <= hour <= 22:
        return MealType.DINNER
    else:
        return MealType.LATE_NIGHT


def generate_timestamp() -> datetime:
    """Generate realistic order timestamp."""
    # Random date within last 90 days
    days_ago = random.randint(0, 90)
    base_date = datetime.now() - timedelta(days=days_ago)
    
    # Weight hours by order patterns
    hours = list(HOUR_WEIGHTS.keys())
    weights = list(HOUR_WEIGHTS.values())
    hour = random.choices(hours, weights=weights)[0]
    
    minute = random.randint(0, 59)
    second = random.randint(0, 59)
    
    return base_date.replace(hour=hour, minute=minute, second=second, microsecond=0)


def should_stop_adding(cart_size: int, user: User) -> bool:
    """
    Determine if user should stop adding items.
    
    Stop probability increases after 3 items.
    Premium users tend to have larger carts.
    """
    if cart_size >= MAX_CART_SIZE:
        return True
    
    if cart_size < MIN_CART_SIZE:
        return False
    
    # Base stop probability
    base_prob = 0.0
    
    if cart_size == 1:
        base_prob = 0.15
    elif cart_size == 2:
        base_prob = 0.30
    elif cart_size == 3:
        base_prob = 0.50
    elif cart_size == 4:
        base_prob = 0.70
    else:
        base_prob = 0.85
    
    # Adjust for user type
    if user.user_type == UserType.BUDGET:
        base_prob += 0.10  # Budget users stop earlier
    elif user.user_type == UserType.PREMIUM:
        base_prob -= 0.10  # Premium users add more
    elif user.user_type == UserType.LUXURY:
        base_prob -= 0.15
    
    base_prob = max(0.0, min(0.95, base_prob))
    
    return random.random() < base_prob


def generate_session(
    session_id: str,
    user: User,
    restaurant_selector: RestaurantSelector,
    item_selector: ItemSelector,
    neg_samples: int = 5
) -> List[CartEvent]:
    """
    Generate a single cart session with positive and negative samples.
    
    Returns:
        List of CartEvent objects
    """
    events = []
    
    # Generate timestamp and meal type
    timestamp = generate_timestamp()
    hour = timestamp.hour
    meal_type = get_meal_type(hour)
    
    # Select restaurant
    restaurant_id = restaurant_selector.select_restaurant(user)
    if restaurant_id is None:
        return []
    
    # Get restaurant info
    rest_info = restaurant_selector.restaurants[
        restaurant_selector.restaurants["restaurant_id"] == restaurant_id
    ]
    if len(rest_info) == 0:
        return []
    
    city = rest_info.iloc[0]["city"]
    
    # Initialize cart state
    cart_state = CartState(
        session_id=session_id,
        user=user,
        restaurant_id=restaurant_id,
        timestamp=timestamp,
    )
    
    step = 0
    
    # Select first item
    first_item = item_selector.select_first_item(restaurant_id, user, meal_type)
    if first_item is None:
        return []
    
    item_id, category, price = first_item
    
    # Add first item to cart
    cart_state.items.append(item_id)
    cart_state.categories.append(category)
    cart_state.total_price += price
    
    # Create positive event for first item
    events.append(CartEvent(
        session_id=session_id,
        step_number=step,
        user_id=user.user_id,
        restaurant_id=restaurant_id,
        cart_items="",  # Empty cart before first item
        candidate_item=item_id,
        label=1,
        timestamp=timestamp.isoformat(),
        city=city,
        hour=hour,
        meal_type=meal_type.value,
        user_type=user.user_type.value,
        cart_total=0.0,
        cart_categories="",
    ))
    
    # Generate negative samples for first item
    negatives = item_selector.get_negative_samples(
        restaurant_id, item_id, [], neg_samples
    )
    for neg_item in negatives:
        events.append(CartEvent(
            session_id=session_id,
            step_number=step,
            user_id=user.user_id,
            restaurant_id=restaurant_id,
            cart_items="",
            candidate_item=neg_item,
            label=0,
            timestamp=timestamp.isoformat(),
            city=city,
            hour=hour,
            meal_type=meal_type.value,
            user_type=user.user_type.value,
            cart_total=0.0,
            cart_categories="",
        ))
    
    step += 1
    
    # Sequential item selection
    while not should_stop_adding(len(cart_state.items), user):
        # Select next item (with meal_type for context-aware selection)
        next_item = item_selector.select_next_item(cart_state, user, meal_type)
        if next_item is None:
            break
        
        item_id, category, price = next_item
        
        # Skip if already in cart
        if item_id in cart_state.items:
            continue
        
        # Create event with current cart state
        cart_items_str = "|".join(cart_state.items)
        cart_categories_str = "|".join(cart_state.categories)
        
        # Positive event
        events.append(CartEvent(
            session_id=session_id,
            step_number=step,
            user_id=user.user_id,
            restaurant_id=restaurant_id,
            cart_items=cart_items_str,
            candidate_item=item_id,
            label=1,
            timestamp=timestamp.isoformat(),
            city=city,
            hour=hour,
            meal_type=meal_type.value,
            user_type=user.user_type.value,
            cart_total=cart_state.total_price,
            cart_categories=cart_categories_str,
        ))
        
        # Negative samples
        negatives = item_selector.get_negative_samples(
            restaurant_id, item_id, cart_state.items, neg_samples
        )
        for neg_item in negatives:
            events.append(CartEvent(
                session_id=session_id,
                step_number=step,
                user_id=user.user_id,
                restaurant_id=restaurant_id,
                cart_items=cart_items_str,
                candidate_item=neg_item,
                label=0,
                timestamp=timestamp.isoformat(),
                city=city,
                hour=hour,
                meal_type=meal_type.value,
                user_type=user.user_type.value,
                cart_total=cart_state.total_price,
                cart_categories=cart_categories_str,
            ))
        
        # Update cart state
        cart_state.items.append(item_id)
        cart_state.categories.append(category)
        cart_state.total_price += price
        
        step += 1
    
    return events


def run_session_generation(
    restaurants_path: Path,
    menu_items_path: Path,
    users_path: Path,
    output_path: Path,
    num_sessions: int = DEFAULT_NUM_SESSIONS,
    neg_samples: int = DEFAULT_NEG_SAMPLES,
    seed: int = 42
) -> pd.DataFrame:
    """
    Run the complete session generation pipeline.
    
    Args:
        restaurants_path: Path to restaurants CSV
        menu_items_path: Path to menu items CSV
        users_path: Path to users CSV (generated if not exists)
        output_path: Path for output cart events CSV
        num_sessions: Number of sessions to generate
        neg_samples: Number of negative samples per positive
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with all cart events
    """
    logger.info("=" * 60)
    logger.info("CART SESSION GENERATION")
    logger.info("=" * 60)
    
    # Set seed
    random.seed(seed)
    np.random.seed(seed)
    
    # Load data
    logger.info(f"Loading restaurants from {restaurants_path}")
    restaurants_df = pd.read_csv(restaurants_path)
    logger.info(f"Loaded {len(restaurants_df):,} restaurants")
    
    logger.info(f"Loading menu items from {menu_items_path}")
    menu_items_df = pd.read_csv(menu_items_path)
    logger.info(f"Loaded {len(menu_items_df):,} menu items")
    
    # Load or generate users
    cities = restaurants_df["city"].unique().tolist()
    users = load_users(users_path, cities)
    
    # Initialize selectors
    restaurant_selector = RestaurantSelector(restaurants_df, menu_items_df)
    item_selector = ItemSelector(menu_items_df)
    
    # Generate sessions
    logger.info(f"\nGenerating {num_sessions:,} sessions with {neg_samples} negative samples each...")
    
    all_events = []
    sessions_generated = 0
    empty_sessions = 0
    
    # Progress logging
    log_interval = max(1, num_sessions // 20)
    
    for i in range(num_sessions):
        # Select random user
        user = random.choice(users)
        
        # Generate session ID
        session_id = f"SESS_{uuid.uuid4().hex[:16].upper()}"
        
        # Generate session events
        events = generate_session(
            session_id, user,
            restaurant_selector, item_selector,
            neg_samples
        )
        
        if events:
            all_events.extend(events)
            sessions_generated += 1
        else:
            empty_sessions += 1
        
        # Progress logging
        if (i + 1) % log_interval == 0:
            pct = 100 * (i + 1) / num_sessions
            logger.info(f"Progress: {i + 1:,}/{num_sessions:,} ({pct:.0f}%) - "
                       f"{len(all_events):,} events generated")
    
    logger.info(f"\nGeneration complete!")
    logger.info(f"Sessions generated: {sessions_generated:,}")
    logger.info(f"Empty sessions (skipped): {empty_sessions:,}")
    logger.info(f"Total events: {len(all_events):,}")
    
    # Convert to DataFrame
    df = pd.DataFrame([
        {
            "session_id": e.session_id,
            "step_number": e.step_number,
            "user_id": e.user_id,
            "restaurant_id": e.restaurant_id,
            "cart_items": e.cart_items,
            "candidate_item": e.candidate_item,
            "label": e.label,
            "timestamp": e.timestamp,
            "city": e.city,
            "hour": e.hour,
            "meal_type": e.meal_type,
            "user_type": e.user_type,
            "cart_total": e.cart_total,
            "cart_categories": e.cart_categories,
        }
        for e in all_events
    ])
    
    # Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved cart events to {output_path}")
    
    # Print summary statistics
    print_session_summary(df)
    
    return df


def print_session_summary(df: pd.DataFrame) -> None:
    """Print summary statistics of generated sessions."""
    logger.info("\n" + "=" * 60)
    logger.info("SESSION GENERATION SUMMARY")
    logger.info("=" * 60)
    
    # Session statistics
    num_sessions = df["session_id"].nunique()
    logger.info(f"\nSessions: {num_sessions:,}")
    logger.info(f"Total events: {len(df):,}")
    
    # Positive vs negative
    pos_count = (df["label"] == 1).sum()
    neg_count = (df["label"] == 0).sum()
    logger.info(f"Positive samples: {pos_count:,} ({100*pos_count/len(df):.1f}%)")
    logger.info(f"Negative samples: {neg_count:,} ({100*neg_count/len(df):.1f}%)")
    
    # Cart size distribution
    cart_sizes = df[df["label"] == 1].groupby("session_id")["step_number"].max() + 1
    logger.info(f"\nCart Size Distribution:")
    logger.info(f"  Min: {cart_sizes.min()}")
    logger.info(f"  Median: {cart_sizes.median():.0f}")
    logger.info(f"  Mean: {cart_sizes.mean():.2f}")
    logger.info(f"  Max: {cart_sizes.max()}")
    
    # Cart size breakdown
    size_counts = cart_sizes.value_counts().sort_index()
    for size, count in size_counts.items():
        logger.info(f"  Size {size}: {count:,} sessions ({100*count/num_sessions:.1f}%)")
    
    # Meal type distribution
    logger.info(f"\nMeal Type Distribution:")
    for meal, count in df[df["label"] == 1]["meal_type"].value_counts().items():
        logger.info(f"  {meal}: {count:,} ({100*count/pos_count:.1f}%)")
    
    # User type distribution
    logger.info(f"\nUser Type Distribution:")
    for ut, count in df[df["label"] == 1]["user_type"].value_counts().items():
        logger.info(f"  {ut}: {count:,} ({100*count/pos_count:.1f}%)")
    
    # City distribution
    logger.info(f"\nTop 5 Cities:")
    for city, count in df[df["label"] == 1]["city"].value_counts().head(5).items():
        logger.info(f"  {city}: {count:,} ({100*count/pos_count:.1f}%)")
    
    # Hour distribution
    logger.info(f"\nPeak Hours:")
    hour_counts = df[df["label"] == 1]["hour"].value_counts().sort_values(ascending=False)
    for hour, count in hour_counts.head(5).items():
        logger.info(f"  {hour}:00: {count:,} ({100*count/pos_count:.1f}%)")
    
    # Category Attach Rates
    logger.info(f"\nCategory Attach Rates (per session):")
    positive_df = df[df["label"] == 1].copy()
    
    # Get all categories per session by looking at cart_categories at last step
    session_final = positive_df.groupby("session_id").last()
    
    # Count category occurrences across all sessions
    all_categories = []
    for cats in session_final["cart_categories"]:
        if cats and isinstance(cats, str):
            all_categories.extend(cats.split("|"))
    
    # Also add the final item's category (from candidate_item)
    # Need to load menu items to get category mapping
    try:
        menu_df = pd.read_csv(DEFAULT_MENU_ITEMS_PATH)
        item_to_cat = dict(zip(menu_df["item_id"], menu_df["item_category"]))
        
        for _, row in session_final.iterrows():
            item_id = row["candidate_item"]
            if item_id in item_to_cat:
                all_categories.append(item_to_cat[item_id])
    except Exception:
        pass
    
    # Calculate attach rates
    category_counts = {}
    for cat in all_categories:
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    for cat in ["main", "side", "beverage", "dessert"]:
        count = category_counts.get(cat, 0)
        rate = 100 * count / num_sessions if num_sessions > 0 else 0
        logger.info(f"  {cat}: {count:,} ({rate:.1f}% of sessions)")
    
    # Calculate behavioral signal rates
    logger.info(f"\nBehavioral Signal Verification:")
    
    # Sessions with main but no beverage initially -> did they add beverage?
    # This would require more complex analysis, so we'll show category co-occurrence
    main_sessions = 0
    main_with_beverage = 0
    heavy_meal_sessions = 0
    heavy_with_dessert = 0
    
    for _, row in session_final.iterrows():
        cats_str = row["cart_categories"]
        if cats_str and isinstance(cats_str, str):
            cats = cats_str.split("|")
            # Add the final item's category
            final_item = row["candidate_item"]
            if final_item in item_to_cat:
                cats.append(item_to_cat[final_item])
            
            has_main = "main" in cats
            has_beverage = "beverage" in cats
            has_dessert = "dessert" in cats
            main_count = cats.count("main")
            side_count = cats.count("side")
            is_heavy = main_count >= 2 or (main_count >= 1 and side_count >= 1)
            
            if has_main:
                main_sessions += 1
                if has_beverage:
                    main_with_beverage += 1
            
            if is_heavy:
                heavy_meal_sessions += 1
                if has_dessert:
                    heavy_with_dessert += 1
    
    if main_sessions > 0:
        logger.info(f"  Main→Beverage attach: {main_with_beverage:,}/{main_sessions:,} ({100*main_with_beverage/main_sessions:.1f}%)")
    if heavy_meal_sessions > 0:
        logger.info(f"  Heavy meal→Dessert attach: {heavy_with_dessert:,}/{heavy_meal_sessions:,} ({100*heavy_with_dessert/heavy_meal_sessions:.1f}%)")
    
    # Premium user dessert rate
    premium_sessions = positive_df[positive_df["user_type"].isin(["premium", "luxury"])]["session_id"].unique()
    non_premium_sessions = positive_df[~positive_df["user_type"].isin(["premium", "luxury"])]["session_id"].unique()
    
    premium_desserts = 0
    premium_total = len(premium_sessions)
    non_premium_desserts = 0
    non_premium_total = len(non_premium_sessions)
    
    for sess_id in premium_sessions:
        sess_data = session_final.loc[sess_id] if sess_id in session_final.index else None
        if sess_data is not None:
            cats = str(sess_data["cart_categories"]).split("|") if sess_data["cart_categories"] else []
            if sess_data["candidate_item"] in item_to_cat:
                cats.append(item_to_cat[sess_data["candidate_item"]])
            if "dessert" in cats:
                premium_desserts += 1
    
    for sess_id in non_premium_sessions:
        sess_data = session_final.loc[sess_id] if sess_id in session_final.index else None
        if sess_data is not None:
            cats = str(sess_data["cart_categories"]).split("|") if sess_data["cart_categories"] else []
            if sess_data["candidate_item"] in item_to_cat:
                cats.append(item_to_cat[sess_data["candidate_item"]])
            if "dessert" in cats:
                non_premium_desserts += 1
    
    if premium_total > 0:
        logger.info(f"  Premium user dessert rate: {100*premium_desserts/premium_total:.1f}%")
    if non_premium_total > 0:
        logger.info(f"  Non-premium user dessert rate: {100*non_premium_desserts/non_premium_total:.1f}%")
    
    # Lunch/dinner beverage rate
    lunch_dinner_sessions = positive_df[positive_df["meal_type"].isin(["lunch", "dinner"])]["session_id"].unique()
    other_meal_sessions = positive_df[~positive_df["meal_type"].isin(["lunch", "dinner"])]["session_id"].unique()
    
    ld_beverages = 0
    ld_total = len(lunch_dinner_sessions)
    other_beverages = 0
    other_total = len(other_meal_sessions)
    
    for sess_id in lunch_dinner_sessions:
        sess_data = session_final.loc[sess_id] if sess_id in session_final.index else None
        if sess_data is not None:
            cats = str(sess_data["cart_categories"]).split("|") if sess_data["cart_categories"] else []
            if sess_data["candidate_item"] in item_to_cat:
                cats.append(item_to_cat[sess_data["candidate_item"]])
            if "beverage" in cats:
                ld_beverages += 1
    
    for sess_id in other_meal_sessions:
        sess_data = session_final.loc[sess_id] if sess_id in session_final.index else None
        if sess_data is not None:
            cats = str(sess_data["cart_categories"]).split("|") if sess_data["cart_categories"] else []
            if sess_data["candidate_item"] in item_to_cat:
                cats.append(item_to_cat[sess_data["candidate_item"]])
            if "beverage" in cats:
                other_beverages += 1
    
    if ld_total > 0:
        logger.info(f"  Lunch/dinner beverage rate: {100*ld_beverages/ld_total:.1f}%")
    if other_total > 0:
        logger.info(f"  Other meal beverage rate: {100*other_beverages/other_total:.1f}%")
    
    logger.info("=" * 60)


# -----------------------------------------------------------------------------
# CLI Entry Point
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic cart sessions for addon recommendation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--restaurants", "-r",
        type=Path,
        default=DEFAULT_RESTAURANTS_PATH,
        help="Path to restaurants CSV",
    )
    
    parser.add_argument(
        "--menu-items", "-m",
        type=Path,
        default=DEFAULT_MENU_ITEMS_PATH,
        help="Path to menu items CSV",
    )
    
    parser.add_argument(
        "--users", "-u",
        type=Path,
        default=DEFAULT_USERS_PATH,
        help="Path to users CSV (generated if not exists)",
    )
    
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path for output cart events CSV",
    )
    
    parser.add_argument(
        "--num-sessions", "-n",
        type=int,
        default=DEFAULT_NUM_SESSIONS,
        help="Number of sessions to generate",
    )
    
    parser.add_argument(
        "--neg-samples",
        type=int,
        default=DEFAULT_NEG_SAMPLES,
        help="Number of negative samples per positive",
    )
    
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="Random seed for reproducibility",
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
        run_session_generation(
            restaurants_path=args.restaurants,
            menu_items_path=args.menu_items,
            users_path=args.users,
            output_path=args.output,
            num_sessions=args.num_sessions,
            neg_samples=args.neg_samples,
            seed=args.seed,
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
