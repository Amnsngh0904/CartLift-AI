"""
Session Preprocessing Module

Prepares menu data for session simulation by:
- Capping menu items per restaurant to top-N ranked items
- Ranking based on item_votes and best_seller status
- Creating simulation-ready dataset

Usage:
    python -m src.data_generation.session_preprocessing
    python -m src.data_generation.session_preprocessing --max-items 100
"""

import argparse
import logging
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
# Constants
# -----------------------------------------------------------------------------

DEFAULT_MAX_ITEMS_PER_RESTAURANT = 120


# -----------------------------------------------------------------------------
# Menu Size Capping
# -----------------------------------------------------------------------------

def rank_menu_items(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rank menu items within each restaurant.
    
    Ranking criteria (in order of priority):
    1. item_votes (descending) - popular items first
    2. best_seller (descending) - best sellers prioritized
    3. price (descending) - higher priced items tend to be mains
    
    Args:
        df: Menu items DataFrame
        
    Returns:
        DataFrame with 'item_rank' column added
    """
    df = df.copy()
    
    # Create composite ranking score
    # Normalize votes to 0-1 range within each restaurant
    df['_vote_score'] = df.groupby('restaurant_id')['item_votes'].transform(
        lambda x: (x - x.min()) / (x.max() - x.min() + 1e-10)
    )
    
    # Best seller gets bonus
    df['_bestseller_score'] = df['best_seller'].fillna(0)
    
    # Combine scores (votes weighted higher than bestseller)
    df['_rank_score'] = df['_vote_score'] * 0.7 + df['_bestseller_score'] * 0.3
    
    # Rank within restaurant (1 = best)
    df['item_rank'] = df.groupby('restaurant_id')['_rank_score'].rank(
        method='first',
        ascending=False
    ).astype(int)
    
    # Clean up temporary columns
    df = df.drop(columns=['_vote_score', '_bestseller_score', '_rank_score'])
    
    return df


def cap_menu_items(
    df: pd.DataFrame,
    max_items: int = DEFAULT_MAX_ITEMS_PER_RESTAURANT,
) -> pd.DataFrame:
    """
    Cap menu items per restaurant to top-N ranked items.
    
    For restaurants with more than max_items:
    - Keeps only the top-ranked items
    
    For restaurants with fewer items:
    - Keeps all items unchanged
    
    Args:
        df: Menu items DataFrame with item_rank column
        max_items: Maximum items to keep per restaurant
        
    Returns:
        Filtered DataFrame
    """
    if 'item_rank' not in df.columns:
        df = rank_menu_items(df)
    
    logger.info(f"Capping menu items to top {max_items} per restaurant...")
    
    # Filter to top-N items per restaurant
    df_capped = df[df['item_rank'] <= max_items].copy()
    
    return df_capped


def print_capping_summary(
    df_original: pd.DataFrame,
    df_capped: pd.DataFrame,
    max_items: int
) -> None:
    """
    Print summary of menu capping operation.
    
    Args:
        df_original: Original DataFrame before capping
        df_capped: DataFrame after capping
        max_items: Maximum items threshold used
    """
    logger.info("=" * 60)
    logger.info("MENU SIZE CAPPING SUMMARY")
    logger.info("=" * 60)
    
    # Restaurant item counts
    original_counts = df_original.groupby('restaurant_id').size()
    capped_counts = df_capped.groupby('restaurant_id').size()
    
    logger.info(f"\nConfiguration:")
    logger.info(f"  Max items per restaurant: {max_items}")
    
    logger.info(f"\nBefore Capping:")
    logger.info(f"  Total menu items: {len(df_original):,}")
    logger.info(f"  Restaurants: {len(original_counts):,}")
    logger.info(f"  Items per restaurant:")
    logger.info(f"    Min: {original_counts.min()}")
    logger.info(f"    Median: {original_counts.median():.0f}")
    logger.info(f"    Mean: {original_counts.mean():.1f}")
    logger.info(f"    Max: {original_counts.max()}")
    
    # Affected restaurants (those with >max_items)
    affected = original_counts[original_counts > max_items]
    logger.info(f"\nRestaurants Affected:")
    logger.info(f"  Count: {len(affected)} ({100*len(affected)/len(original_counts):.1f}%)")
    
    if len(affected) > 0:
        items_removed = df_original.groupby('restaurant_id').size() - \
                       df_capped.groupby('restaurant_id').size()
        items_removed = items_removed[items_removed > 0]
        logger.info(f"  Items removed: {items_removed.sum():,}")
        logger.info(f"  Avg items removed per affected restaurant: {items_removed.mean():.1f}")
        
        # Show top 5 most affected restaurants
        logger.info("\n  Top 5 most affected restaurants:")
        for rest_id, count in items_removed.nlargest(5).items():
            orig_count = original_counts[rest_id]
            rest_name = "Unknown"
            # Try to get restaurant name from data
            sample = df_original[df_original['restaurant_id'] == rest_id].head(1)
            if 'restaurant_name' in sample.columns:
                rest_name = sample['restaurant_name'].values[0]
            logger.info(f"    {rest_name[:30]}: {orig_count} → {max_items} "
                       f"(removed {int(count)})")
    
    logger.info(f"\nAfter Capping:")
    logger.info(f"  Total menu items: {len(df_capped):,}")
    logger.info(f"  Restaurants: {len(capped_counts):,}")
    logger.info(f"  Items per restaurant:")
    logger.info(f"    Min: {capped_counts.min()}")
    logger.info(f"    Median: {capped_counts.median():.0f}")
    logger.info(f"    Mean: {capped_counts.mean():.1f}")
    logger.info(f"    Max: {capped_counts.max()}")
    
    # Reduction statistics
    reduction_pct = 100 * (1 - len(df_capped) / len(df_original))
    logger.info(f"\nReduction:")
    logger.info(f"  Items removed: {len(df_original) - len(df_capped):,} ({reduction_pct:.1f}%)")
    
    # Category distribution comparison
    if 'item_category' in df_original.columns:
        logger.info("\nCategory Distribution Comparison:")
        cat_before = df_original['item_category'].value_counts(normalize=True) * 100
        cat_after = df_capped['item_category'].value_counts(normalize=True) * 100
        
        for cat in cat_before.index:
            before_pct = cat_before.get(cat, 0)
            after_pct = cat_after.get(cat, 0)
            diff = after_pct - before_pct
            logger.info(f"  {cat}: {before_pct:.1f}% → {after_pct:.1f}% ({diff:+.1f}%)")
    
    logger.info("=" * 60)


def run_preprocessing_pipeline(
    input_path: Path,
    output_path: Path,
    max_items: int = DEFAULT_MAX_ITEMS_PER_RESTAURANT,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the complete menu preprocessing pipeline.
    
    Args:
        input_path: Path to menu_items_enriched.csv
        output_path: Path for output menu_items_simulation.csv
        max_items: Maximum items per restaurant
        
    Returns:
        Tuple of (original DataFrame, capped DataFrame)
    """
    logger.info("Starting session preprocessing pipeline...")
    
    # Load data
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    df_original = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df_original):,} menu items from enriched dataset")
    
    # Rank items within each restaurant
    df_ranked = rank_menu_items(df_original)
    
    # Cap menu items
    df_capped = cap_menu_items(df_ranked, max_items=max_items)
    
    # Print summary
    print_capping_summary(df_original, df_capped, max_items)
    
    # Save results (without item_rank column in output)
    output_df = df_capped.drop(columns=['item_rank'], errors='ignore')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    logger.info(f"Saved simulation dataset to: {output_path}")
    
    logger.info("Session preprocessing pipeline completed successfully!")
    logger.info("Note: Original enriched dataset remains unchanged.")
    
    return df_original, df_capped


# -----------------------------------------------------------------------------
# CLI Entry Point
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Preprocess menu data for session simulation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=Path("data/processed/menu_items_enriched.csv"),
        help="Path to input enriched menu items CSV",
    )
    
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("data/processed/menu_items_simulation.csv"),
        help="Path for output simulation-ready CSV",
    )
    
    parser.add_argument(
        "--max-items", "-n",
        type=int,
        default=DEFAULT_MAX_ITEMS_PER_RESTAURANT,
        help="Maximum items to keep per restaurant",
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
        run_preprocessing_pipeline(
            input_path=args.input,
            output_path=args.output,
            max_items=args.max_items,
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
