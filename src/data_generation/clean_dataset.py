"""
Data Cleaning Script for Zomato Restaurant & Menu Item Dataset

This script processes the raw Zomato dataset and produces two clean, normalized tables:
1. restaurants_cleaned.csv - Restaurant-level information
2. menu_items_cleaned.csv - Item-level information with restaurant references

Dataset source: Kaggle Zomato dataset
Expected columns:
- Restaurant Name, Dining Rating, Delivery Rating, Dining Votes, Delivery Votes
- Cuisine, Place Name, City
- Item Name, Best Seller, Votes, Prices

Usage:
    python -m src.data_generation.clean_dataset --input data/raw/zomato_dataset.csv
    python -m src.data_generation.clean_dataset --input data/raw/zomato_dataset.csv --output-dir data/processed
"""

import argparse
import hashlib
import logging
import re
import sys
from pathlib import Path
from typing import Tuple

import pandas as pd
import numpy as np

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
# Constants & Mappings
# -----------------------------------------------------------------------------

# Expected raw column names (from Kaggle Zomato dataset)
RAW_COLUMN_MAPPING = {
    "Restaurant Name": "restaurant_name",
    "Dining Rating": "dining_rating",
    "Delivery Rating": "delivery_rating",
    "Dining Votes": "dining_votes",
    "Delivery Votes": "delivery_votes",
    "Cuisine": "cuisine",
    "Cuisine ": "cuisine",  # Handle trailing space
    "Place Name": "place",
    "Place": "place",
    "City": "city",
    "City Name": "city",
    "Item Name": "item_name",
    "Best Seller": "best_seller",
    "Votes": "item_votes",
    "Prices": "price",
}

# City name standardization mapping
CITY_STANDARDIZATION = {
    "bengaluru": "Bangalore",
    "bangalore": "Bangalore",
    "blr": "Bangalore",
    "mumbai": "Mumbai",
    "bombay": "Mumbai",
    "delhi": "Delhi",
    "new delhi": "Delhi",
    "ncr": "Delhi",
    "chennai": "Chennai",
    "madras": "Chennai",
    "kolkata": "Kolkata",
    "calcutta": "Kolkata",
    "hyderabad": "Hyderabad",
    "hyd": "Hyderabad",
    "pune": "Pune",
    "ahmedabad": "Ahmedabad",
    "ahemdabad": "Ahmedabad",
    "jaipur": "Jaipur",
    "lucknow": "Lucknow",
    "gurgaon": "Gurugram",
    "gurugram": "Gurugram",
    "noida": "Noida",
    "ghaziabad": "Ghaziabad",
    "chandigarh": "Chandigarh",
    "kochi": "Kochi",
    "cochin": "Kochi",
    "indore": "Indore",
    "nagpur": "Nagpur",
    "coimbatore": "Coimbatore",
    "vizag": "Visakhapatnam",
    "visakhapatnam": "Visakhapatnam",
    "surat": "Surat",
    "vadodara": "Vadodara",
    "baroda": "Vadodara",
    "bhopal": "Bhopal",
    "patna": "Patna",
    "thiruvananthapuram": "Thiruvananthapuram",
    "trivandrum": "Thiruvananthapuram",
    "mysore": "Mysuru",
    "mysuru": "Mysuru",
}

# Cuisine name standardization
CUISINE_STANDARDIZATION = {
    "north indian": "North Indian",
    "northindian": "North Indian",
    "south indian": "South Indian",
    "southindian": "South Indian",
    "chinese": "Chinese",
    "indo-chinese": "Indo-Chinese",
    "indo chinese": "Indo-Chinese",
    "fast food": "Fast Food",
    "fastfood": "Fast Food",
    "italian": "Italian",
    "continental": "Continental",
    "mughlai": "Mughlai",
    "biryani": "Biryani",
    "street food": "Street Food",
    "cafe": "Cafe",
    "bakery": "Bakery",
    "desserts": "Desserts",
    "beverages": "Beverages",
    "pizza": "Pizza",
    "burger": "Burger",
    "healthy food": "Healthy Food",
    "seafood": "Seafood",
    "thai": "Thai",
    "mexican": "Mexican",
    "japanese": "Japanese",
    "korean": "Korean",
    "arabian": "Arabian",
    "lebanese": "Lebanese",
    "mediterranean": "Mediterranean",
    "american": "American",
    "european": "European",
    "asian": "Asian",
    "ice cream": "Ice Cream",
    "juices": "Juices",
    "tea": "Tea",
    "coffee": "Coffee",
}

# Best seller tag mapping
BESTSELLER_TAGS = {
    "bestseller": 1,
    "best seller": 1,
    "must try": 1,
    "chef's special": 1,
    "chefs special": 1,
    "popular": 1,
    "recommended": 1,
    "top rated": 1,
}


# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------

def load_raw_data(file_path: Path) -> pd.DataFrame:
    """
    Load raw dataset from CSV file.
    
    Args:
        file_path: Path to the raw CSV file
        
    Returns:
        DataFrame with raw data
        
    Raises:
        FileNotFoundError: If input file doesn't exist
        ValueError: If file is empty or unreadable
    """
    logger.info(f"Loading raw data from: {file_path}")
    
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")
    
    try:
        df = pd.read_csv(file_path, encoding="utf-8")
    except UnicodeDecodeError:
        logger.warning("UTF-8 decoding failed, trying latin-1 encoding")
        df = pd.read_csv(file_path, encoding="latin-1")
    
    if df.empty:
        raise ValueError("Input file is empty")
    
    logger.info(f"Loaded {len(df):,} rows with {len(df.columns)} columns")
    logger.info(f"Raw columns: {list(df.columns)}")
    
    return df


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names to snake_case.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with standardized column names
    """
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()
    
    # Build mapping for existing columns
    actual_mapping = {}
    for raw_col in df.columns:
        # Check direct match first
        if raw_col in RAW_COLUMN_MAPPING:
            actual_mapping[raw_col] = RAW_COLUMN_MAPPING[raw_col]
        else:
            # Try case-insensitive match
            raw_lower = raw_col.lower().strip()
            for expected, target in RAW_COLUMN_MAPPING.items():
                if expected.lower().strip() == raw_lower:
                    actual_mapping[raw_col] = target
                    break
    
    df = df.rename(columns=actual_mapping)
    
    # Log mapping results
    mapped_cols = list(actual_mapping.values())
    logger.info(f"Mapped {len(actual_mapping)} columns: {mapped_cols}")
    
    unmapped = [c for c in df.columns if c not in mapped_cols and c not in RAW_COLUMN_MAPPING.values()]
    if unmapped:
        logger.warning(f"Unmapped columns (will be dropped): {unmapped}")
    
    return df


# -----------------------------------------------------------------------------
# Cleaning Functions
# -----------------------------------------------------------------------------

def clean_city(city: str) -> str:
    """Standardize city name."""
    if pd.isna(city):
        return "Unknown"
    
    # Clean and normalize
    city_clean = str(city).strip()
    city_lower = city_clean.lower()
    
    # Remove leading/trailing punctuation and spaces
    city_lower = re.sub(r'^[\s,]+|[\s,]+$', '', city_lower)
    
    # Check mapping
    if city_lower in CITY_STANDARDIZATION:
        return CITY_STANDARDIZATION[city_lower]
    
    # Title case if not mapped
    return city_clean.strip().title()


def clean_cuisine(cuisine: str) -> str:
    """Standardize cuisine name, taking primary cuisine if multiple."""
    if pd.isna(cuisine):
        return "Other"
    
    # Take primary cuisine (first one if comma-separated)
    primary = str(cuisine).split(",")[0].strip()
    primary_lower = primary.lower()
    
    # Check mapping
    if primary_lower in CUISINE_STANDARDIZATION:
        return CUISINE_STANDARDIZATION[primary_lower]
    
    # Title case if not mapped
    return primary.title()


def clean_rating(rating) -> float:
    """Convert rating to float, handling various formats."""
    if pd.isna(rating):
        return np.nan
    
    rating_str = str(rating).strip()
    
    # Handle invalid markers
    if rating_str.lower() in ["--", "-", "new", "n/a", "na", "", "none"]:
        return np.nan
    
    # Extract numeric value
    match = re.search(r'(\d+\.?\d*)', rating_str)
    if match:
        value = float(match.group(1))
        # Validate range (0-5)
        if 0 <= value <= 5:
            return round(value, 2)
    
    return np.nan


def clean_votes(votes) -> int:
    """Convert votes to integer, handling various formats."""
    if pd.isna(votes):
        return 0
    
    votes_str = str(votes).strip().lower()
    
    # Handle invalid markers
    if votes_str in ["--", "-", "new", "n/a", "na", "", "none"]:
        return 0
    
    # Remove commas and spaces
    votes_str = votes_str.replace(",", "").replace(" ", "")
    
    # Handle K/M suffixes
    multiplier = 1
    if votes_str.endswith("k"):
        multiplier = 1000
        votes_str = votes_str[:-1]
    elif votes_str.endswith("m"):
        multiplier = 1000000
        votes_str = votes_str[:-1]
    
    # Extract numeric value
    match = re.search(r'(\d+\.?\d*)', votes_str)
    if match:
        return int(float(match.group(1)) * multiplier)
    
    return 0


def clean_price(price) -> float:
    """Convert price to float, removing currency symbols."""
    if pd.isna(price):
        return np.nan
    
    price_str = str(price).strip()
    
    # Handle invalid markers
    if price_str.lower() in ["--", "-", "n/a", "na", "", "free", "none"]:
        return np.nan
    
    # Remove currency symbols and commas
    price_str = re.sub(r'[₹$€£,\s]', '', price_str)
    
    # Handle price ranges (take lower bound)
    if "-" in price_str and not price_str.startswith("-"):
        price_str = price_str.split("-")[0]
    
    # Extract numeric value
    match = re.search(r'(\d+\.?\d*)', price_str)
    if match:
        value = float(match.group(1))
        if value >= 0:
            return round(value, 2)
    
    return np.nan


def clean_item_name(item_name: str) -> str:
    """Normalize item name: lowercase, strip spaces."""
    if pd.isna(item_name):
        return ""
    
    # Lowercase and strip
    name_clean = str(item_name).lower().strip()
    
    # Remove extra whitespace
    name_clean = re.sub(r'\s+', ' ', name_clean)
    
    # Remove leading/trailing special chars
    name_clean = name_clean.strip(".-_,;:'\"")
    
    return name_clean


def clean_bestseller(value) -> int:
    """Convert best seller tag to binary (0/1)."""
    if pd.isna(value):
        return 0
    
    value_str = str(value).lower().strip()
    
    # Check if any bestseller tag matches
    for tag in BESTSELLER_TAGS:
        if tag in value_str:
            return 1
    
    return 0


def generate_restaurant_id(restaurant_name: str, city: str, place: str) -> str:
    """Generate unique restaurant ID based on name, city, and place."""
    key_parts = [
        str(restaurant_name).lower().strip(),
        str(city).lower().strip(),
        str(place).lower().strip() if pd.notna(place) else "",
    ]
    key_string = "|".join(key_parts)
    hash_value = hashlib.md5(key_string.encode()).hexdigest()[:12]
    return f"REST_{hash_value.upper()}"


def generate_item_id(restaurant_id: str, item_name: str, price: float) -> str:
    """Generate unique item ID based on restaurant, name, and price."""
    key_parts = [
        str(restaurant_id).lower(),
        str(item_name).lower().strip(),
        str(price) if pd.notna(price) else "0",
    ]
    key_string = "|".join(key_parts)
    hash_value = hashlib.md5(key_string.encode()).hexdigest()[:12]
    return f"ITEM_{hash_value.upper()}"


# -----------------------------------------------------------------------------
# Main Processing
# -----------------------------------------------------------------------------

def clean_restaurant_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract and clean restaurant-level data.
    
    Args:
        df: Raw DataFrame with standardized column names
        
    Returns:
        Cleaned restaurant DataFrame with unique restaurants
    """
    logger.info("Extracting restaurant-level data...")
    
    # Required restaurant columns
    restaurant_cols = ["restaurant_name", "city", "place", "cuisine",
                       "delivery_rating", "delivery_votes", 
                       "dining_rating", "dining_votes"]
    
    # Check available columns
    available_cols = [c for c in restaurant_cols if c in df.columns]
    missing_cols = set(restaurant_cols) - set(available_cols)
    if missing_cols:
        logger.warning(f"Missing columns (will use defaults): {missing_cols}")
    
    # Extract restaurant data
    restaurants_df = df[available_cols].copy()
    
    # Clean city
    if "city" in restaurants_df.columns:
        restaurants_df["city"] = restaurants_df["city"].apply(clean_city)
    else:
        restaurants_df["city"] = "Unknown"
    
    # Clean cuisine
    if "cuisine" in restaurants_df.columns:
        restaurants_df["cuisine"] = restaurants_df["cuisine"].apply(clean_cuisine)
    else:
        restaurants_df["cuisine"] = "Other"
    
    # Clean ratings
    for col in ["delivery_rating", "dining_rating"]:
        if col in restaurants_df.columns:
            restaurants_df[col] = restaurants_df[col].apply(clean_rating)
    
    # Clean votes
    for col in ["delivery_votes", "dining_votes"]:
        if col in restaurants_df.columns:
            restaurants_df[col] = restaurants_df[col].apply(clean_votes)
    
    # Clean restaurant name
    if "restaurant_name" in restaurants_df.columns:
        restaurants_df["restaurant_name"] = restaurants_df["restaurant_name"].str.strip()
        # Remove rows with missing restaurant name
        initial_count = len(restaurants_df)
        restaurants_df = restaurants_df[restaurants_df["restaurant_name"].notna()]
        restaurants_df = restaurants_df[restaurants_df["restaurant_name"].str.len() > 0]
        removed = initial_count - len(restaurants_df)
        if removed > 0:
            logger.info(f"Removed {removed:,} rows with missing restaurant name")
    
    # Clean place
    if "place" in restaurants_df.columns:
        restaurants_df["place"] = restaurants_df["place"].str.strip()
    
    # Generate restaurant IDs
    restaurants_df["restaurant_id"] = restaurants_df.apply(
        lambda row: generate_restaurant_id(
            row.get("restaurant_name", ""),
            row.get("city", ""),
            row.get("place", "")
        ), axis=1
    )
    
    # Remove duplicates (keep first occurrence with most votes)
    if "dining_votes" in restaurants_df.columns:
        restaurants_df = restaurants_df.sort_values("dining_votes", ascending=False)
    
    initial_count = len(restaurants_df)
    restaurants_df = restaurants_df.drop_duplicates(subset=["restaurant_id"], keep="first")
    duplicates_removed = initial_count - len(restaurants_df)
    logger.info(f"Removed {duplicates_removed:,} duplicate restaurant entries")
    
    # Fill missing ratings with median
    for col in ["delivery_rating", "dining_rating"]:
        if col in restaurants_df.columns:
            median_val = restaurants_df[col].median()
            na_count = restaurants_df[col].isna().sum()
            if na_count > 0:
                restaurants_df[col] = restaurants_df[col].fillna(median_val)
                logger.info(f"Filled {na_count:,} missing {col} with median {median_val:.2f}")
    
    # Fill missing votes with 0
    for col in ["delivery_votes", "dining_votes"]:
        if col in restaurants_df.columns:
            restaurants_df[col] = restaurants_df[col].fillna(0).astype(int)
    
    # Select and order final columns (drop 'place' from final output as per spec)
    final_cols = ["restaurant_id", "restaurant_name", "city", "cuisine",
                  "delivery_rating", "delivery_votes", "dining_rating", "dining_votes"]
    final_cols = [c for c in final_cols if c in restaurants_df.columns]
    restaurants_df = restaurants_df[final_cols].reset_index(drop=True)
    
    logger.info(f"Restaurant table: {len(restaurants_df):,} unique restaurants")
    
    return restaurants_df


def clean_menu_data(df: pd.DataFrame, restaurants_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract and clean menu item-level data.
    
    Args:
        df: Raw DataFrame with standardized column names
        restaurants_df: Cleaned restaurant DataFrame
        
    Returns:
        Cleaned menu items DataFrame
    """
    logger.info("Extracting menu item-level data...")
    
    # Create working copy with all needed columns
    menu_df = df.copy()
    
    # Generate restaurant_id for each row
    menu_df["restaurant_id"] = menu_df.apply(
        lambda row: generate_restaurant_id(
            row.get("restaurant_name", ""),
            row.get("city", ""),
            row.get("place", "")
        ), axis=1
    )
    
    # Clean item name
    if "item_name" in menu_df.columns:
        menu_df["item_name"] = menu_df["item_name"].apply(clean_item_name)
    else:
        logger.error("Missing 'item_name' column!")
        raise ValueError("Dataset must contain 'Item Name' column")
    
    # Clean price
    if "price" in menu_df.columns:
        menu_df["price"] = menu_df["price"].apply(clean_price)
    else:
        logger.error("Missing 'price' column!")
        raise ValueError("Dataset must contain 'Prices' column")
    
    # Remove items with missing price
    initial_count = len(menu_df)
    menu_df = menu_df[menu_df["price"].notna()]
    menu_df = menu_df[menu_df["price"] > 0]
    removed_price = initial_count - len(menu_df)
    logger.info(f"Removed {removed_price:,} items with missing/invalid price")
    
    # Remove items with empty names
    initial_count = len(menu_df)
    menu_df = menu_df[menu_df["item_name"].str.len() > 0]
    removed_name = initial_count - len(menu_df)
    if removed_name > 0:
        logger.info(f"Removed {removed_name:,} items with empty name")
    
    # Clean item votes
    if "item_votes" in menu_df.columns:
        menu_df["item_votes"] = menu_df["item_votes"].apply(clean_votes)
    else:
        menu_df["item_votes"] = 0
    
    # Clean best seller flag
    if "best_seller" in menu_df.columns:
        menu_df["best_seller"] = menu_df["best_seller"].apply(clean_bestseller)
    else:
        menu_df["best_seller"] = 0
    
    # Filter to only include items from known restaurants
    valid_restaurant_ids = set(restaurants_df["restaurant_id"])
    initial_count = len(menu_df)
    menu_df = menu_df[menu_df["restaurant_id"].isin(valid_restaurant_ids)]
    orphan_count = initial_count - len(menu_df)
    if orphan_count > 0:
        logger.warning(f"Removed {orphan_count:,} items with unknown restaurant_id")
    
    # Generate item IDs
    menu_df["item_id"] = menu_df.apply(
        lambda row: generate_item_id(
            row["restaurant_id"],
            row["item_name"],
            row["price"]
        ), axis=1
    )
    
    # Remove duplicate items (same restaurant + item + price)
    initial_count = len(menu_df)
    menu_df = menu_df.drop_duplicates(subset=["item_id"], keep="first")
    duplicates_removed = initial_count - len(menu_df)
    if duplicates_removed > 0:
        logger.info(f"Removed {duplicates_removed:,} duplicate menu items")
    
    # Select and order final columns
    final_cols = ["item_id", "restaurant_id", "item_name", "price", "item_votes", "best_seller"]
    menu_df = menu_df[final_cols].reset_index(drop=True)
    
    # Ensure correct dtypes
    menu_df["price"] = menu_df["price"].astype(float)
    menu_df["item_votes"] = menu_df["item_votes"].astype(int)
    menu_df["best_seller"] = menu_df["best_seller"].astype(int)
    
    logger.info(f"Menu table: {len(menu_df):,} items")
    
    return menu_df


def save_cleaned_data(
    restaurants_df: pd.DataFrame,
    menu_df: pd.DataFrame,
    output_dir: Path
) -> Tuple[Path, Path]:
    """Save cleaned DataFrames to CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save restaurants
    restaurants_path = output_dir / "restaurants_cleaned.csv"
    restaurants_df.to_csv(restaurants_path, index=False)
    logger.info(f"Saved: {restaurants_path} ({len(restaurants_df):,} rows)")
    
    # Save menu items
    menu_path = output_dir / "menu_items_cleaned.csv"
    menu_df.to_csv(menu_path, index=False)
    logger.info(f"Saved: {menu_path} ({len(menu_df):,} rows)")
    
    return restaurants_path, menu_path


def print_summary(
    raw_df: pd.DataFrame,
    restaurants_df: pd.DataFrame,
    menu_df: pd.DataFrame
) -> None:
    """Print comprehensive cleaning summary statistics."""
    logger.info("=" * 70)
    logger.info("DATA CLEANING SUMMARY")
    logger.info("=" * 70)
    
    # Raw data stats
    logger.info(f"RAW DATA:")
    logger.info(f"  - Total rows: {len(raw_df):,}")
    logger.info(f"  - Columns: {len(raw_df.columns)}")
    
    # Restaurant statistics
    logger.info(f"\nRESTAURANT TABLE:")
    logger.info(f"  - Unique restaurants: {len(restaurants_df):,}")
    logger.info(f"  - Unique cities: {restaurants_df['city'].nunique()}")
    logger.info(f"  - Unique cuisines: {restaurants_df['cuisine'].nunique()}")
    
    if "delivery_rating" in restaurants_df.columns:
        logger.info(f"  - Delivery rating: min={restaurants_df['delivery_rating'].min():.1f}, "
                   f"max={restaurants_df['delivery_rating'].max():.1f}, "
                   f"mean={restaurants_df['delivery_rating'].mean():.2f}")
    if "dining_rating" in restaurants_df.columns:
        logger.info(f"  - Dining rating: min={restaurants_df['dining_rating'].min():.1f}, "
                   f"max={restaurants_df['dining_rating'].max():.1f}, "
                   f"mean={restaurants_df['dining_rating'].mean():.2f}")
    
    # Menu statistics
    logger.info(f"\nMENU TABLE:")
    logger.info(f"  - Total items: {len(menu_df):,}")
    logger.info(f"  - Avg items/restaurant: {len(menu_df) / len(restaurants_df):.1f}")
    logger.info(f"  - Unique item names: {menu_df['item_name'].nunique():,}")
    logger.info(f"  - Price range: ₹{menu_df['price'].min():.0f} - ₹{menu_df['price'].max():.0f}")
    logger.info(f"  - Mean price: ₹{menu_df['price'].mean():.0f}")
    logger.info(f"  - Median price: ₹{menu_df['price'].median():.0f}")
    logger.info(f"  - Best sellers: {menu_df['best_seller'].sum():,} "
               f"({100 * menu_df['best_seller'].mean():.1f}%)")
    
    # Top 5 cities
    logger.info("\nTOP 5 CITIES:")
    for city, count in restaurants_df["city"].value_counts().head(5).items():
        logger.info(f"  - {city}: {count:,} restaurants")
    
    # Top 5 cuisines
    logger.info("\nTOP 5 CUISINES:")
    for cuisine, count in restaurants_df["cuisine"].value_counts().head(5).items():
        logger.info(f"  - {cuisine}: {count:,} restaurants")
    
    # Data quality metrics
    logger.info("\nDATA QUALITY:")
    logger.info(f"  - Compression: {len(raw_df):,} → {len(restaurants_df):,} restaurants " 
               f"({100 * len(restaurants_df) / len(raw_df):.2f}%)")
    logger.info(f"  - Items retained: {len(menu_df):,} / {len(raw_df):,} "
               f"({100 * len(menu_df) / len(raw_df):.1f}%)")
    
    logger.info("=" * 70)


def run_cleaning_pipeline(
    input_path: Path,
    output_dir: Path
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the complete data cleaning pipeline.
    
    Args:
        input_path: Path to raw input CSV
        output_dir: Directory for output files
        
    Returns:
        Tuple of (restaurants_df, menu_df)
    """
    logger.info("Starting data cleaning pipeline...")
    logger.info(f"Input: {input_path}")
    logger.info(f"Output directory: {output_dir}")
    
    # Load raw data
    raw_df = load_raw_data(input_path)
    
    # Standardize column names
    df = standardize_column_names(raw_df)
    
    # Clean restaurant data
    restaurants_df = clean_restaurant_data(df)
    
    # Clean menu data
    menu_df = clean_menu_data(df, restaurants_df)
    
    # Save cleaned data
    save_cleaned_data(restaurants_df, menu_df, output_dir)
    
    # Print summary
    print_summary(raw_df, restaurants_df, menu_df)
    
    logger.info("Data cleaning pipeline completed successfully!")
    
    return restaurants_df, menu_df


# -----------------------------------------------------------------------------
# CLI Entry Point
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Clean Zomato restaurant and menu item dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=Path("data/raw/zomato_dataset.csv"),
        help="Path to raw input CSV file",
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path("data/processed"),
        help="Output directory for cleaned CSV files",
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose (DEBUG) logging",
    )
    
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        run_cleaning_pipeline(args.input, args.output_dir)
        return 0
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except ValueError as e:
        logger.error(f"Value error: {e}")
        return 2
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return 99


if __name__ == "__main__":
    sys.exit(main())
