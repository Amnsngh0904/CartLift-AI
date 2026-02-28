"""
Restaurant Utility Functions

Includes:
- Bayesian rating smoothing for delivery ratings
- Utility functions for restaurant data processing

Usage:
    python -m src.data_generation.restaurant_utils
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

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
# Bayesian Rating Smoothing
# -----------------------------------------------------------------------------

def compute_smoothed_rating(
    rating: float,
    votes: int,
    global_mean: float,
    smoothing_constant: int = 50
) -> float:
    """
    Compute Bayesian smoothed rating for a single restaurant.
    
    Formula:
        smoothed_rating = (rating * votes + global_mean * m) / (votes + m)
    
    Where:
        - rating: The restaurant's raw delivery rating
        - votes: Number of delivery votes
        - global_mean: Average rating across all restaurants
        - m: Smoothing constant (prior strength)
    
    The smoothing constant 'm' determines how much to trust the global mean:
    - Low m (e.g., 10): Trust restaurant's own rating more
    - High m (e.g., 100): Trust global mean more
    - Default 50: Balanced approach
    
    Args:
        rating: Restaurant's delivery rating
        votes: Number of delivery votes
        global_mean: Global average delivery rating
        smoothing_constant: Prior strength (default: 50)
        
    Returns:
        Smoothed rating value
    """
    if pd.isna(rating) or pd.isna(votes):
        return global_mean
    
    numerator = (rating * votes) + (global_mean * smoothing_constant)
    denominator = votes + smoothing_constant
    
    return numerator / denominator


def add_smoothed_rating_column(
    df: pd.DataFrame,
    rating_col: str = "delivery_rating",
    votes_col: str = "delivery_votes",
    output_col: str = "smoothed_rating",
    smoothing_constant: int = 50,
    global_mean: Optional[float] = None
) -> pd.DataFrame:
    """
    Add Bayesian smoothed rating column to restaurant DataFrame.
    
    This addresses the cold-start problem where:
    - Restaurants with zero votes get the global mean
    - Restaurants with few votes get pulled toward global mean
    - Restaurants with many votes keep their original rating
    
    Args:
        df: Restaurant DataFrame with rating and votes columns
        rating_col: Column name for delivery rating
        votes_col: Column name for delivery votes
        output_col: Output column name for smoothed rating
        smoothing_constant: Bayesian prior strength (default: 50)
        global_mean: Optional pre-computed global mean. If None, computed from data.
        
    Returns:
        DataFrame with new smoothed_rating column
    """
    df = df.copy()
    
    # Compute global mean if not provided
    if global_mean is None:
        # Use weighted mean (by votes) for more accurate global average
        # But also include zero-vote restaurants to not bias toward high-activity
        global_mean = df[rating_col].mean()
    
    logger.info(f"Computing smoothed ratings with:")
    logger.info(f"  Global mean: {global_mean:.3f}")
    logger.info(f"  Smoothing constant (m): {smoothing_constant}")
    
    # Apply smoothing to each restaurant
    df[output_col] = df.apply(
        lambda row: compute_smoothed_rating(
            rating=row[rating_col],
            votes=row[votes_col],
            global_mean=global_mean,
            smoothing_constant=smoothing_constant
        ),
        axis=1
    )
    
    # Round to 3 decimal places for cleaner output
    df[output_col] = df[output_col].round(3)
    
    return df


def print_smoothing_summary(
    df: pd.DataFrame,
    rating_col: str = "delivery_rating",
    votes_col: str = "delivery_votes",
    smoothed_col: str = "smoothed_rating"
) -> None:
    """
    Print summary statistics comparing raw and smoothed ratings.
    
    Args:
        df: DataFrame with both raw and smoothed ratings
        rating_col: Raw rating column name
        votes_col: Votes column name
        smoothed_col: Smoothed rating column name
    """
    logger.info("=" * 60)
    logger.info("BAYESIAN RATING SMOOTHING SUMMARY")
    logger.info("=" * 60)
    
    # Global statistics
    global_mean = df[rating_col].mean()
    logger.info(f"\nGlobal mean rating: {global_mean:.3f}")
    
    # Zero-vote restaurants
    zero_votes = df[df[votes_col] == 0]
    logger.info(f"Zero-vote restaurants: {len(zero_votes)} ({100*len(zero_votes)/len(df):.1f}%)")
    
    # Rating distribution comparison
    logger.info("\nRaw Rating Distribution:")
    logger.info(f"  Min: {df[rating_col].min():.2f}")
    logger.info(f"  Mean: {df[rating_col].mean():.3f}")
    logger.info(f"  Median: {df[rating_col].median():.2f}")
    logger.info(f"  Max: {df[rating_col].max():.2f}")
    logger.info(f"  Std: {df[rating_col].std():.3f}")
    
    logger.info("\nSmoothed Rating Distribution:")
    logger.info(f"  Min: {df[smoothed_col].min():.3f}")
    logger.info(f"  Mean: {df[smoothed_col].mean():.3f}")
    logger.info(f"  Median: {df[smoothed_col].median():.3f}")
    logger.info(f"  Max: {df[smoothed_col].max():.3f}")
    logger.info(f"  Std: {df[smoothed_col].std():.3f}")
    
    # Examples: Before/After for different vote counts
    logger.info("\nExample Before/After Comparisons:")
    
    # Zero votes examples
    zero_sample = zero_votes.head(3)
    if len(zero_sample) > 0:
        logger.info("  Zero-vote restaurants (smoothed → global mean):")
        for _, row in zero_sample.iterrows():
            logger.info(f"    {row['restaurant_name'][:30]}: "
                       f"{row[rating_col]:.1f} → {row[smoothed_col]:.3f}")
    
    # Low votes (1-20)
    low_votes = df[(df[votes_col] > 0) & (df[votes_col] <= 20)]
    low_sample = low_votes.head(3)
    if len(low_sample) > 0:
        logger.info("  Low-vote restaurants (1-20 votes):")
        for _, row in low_sample.iterrows():
            logger.info(f"    {row['restaurant_name'][:30]} ({int(row[votes_col])} votes): "
                       f"{row[rating_col]:.1f} → {row[smoothed_col]:.3f}")
    
    # High votes (>100)
    high_votes = df[df[votes_col] > 100]
    high_sample = high_votes.head(3)
    if len(high_sample) > 0:
        logger.info("  High-vote restaurants (>100 votes):")
        for _, row in high_sample.iterrows():
            logger.info(f"    {row['restaurant_name'][:30]} ({int(row[votes_col])} votes): "
                       f"{row[rating_col]:.1f} → {row[smoothed_col]:.3f}")
    
    # Impact statistics
    rating_diff = (df[smoothed_col] - df[rating_col]).abs()
    logger.info("\nSmoothing Impact:")
    logger.info(f"  Mean rating change: {rating_diff.mean():.3f}")
    logger.info(f"  Max rating change: {rating_diff.max():.3f}")
    logger.info(f"  Restaurants with >0.1 change: {(rating_diff > 0.1).sum()}")
    
    logger.info("=" * 60)


def run_smoothing_pipeline(
    input_path: Path,
    output_path: Path,
    smoothing_constant: int = 50
) -> pd.DataFrame:
    """
    Run the complete rating smoothing pipeline.
    
    Args:
        input_path: Path to restaurants_cleaned.csv
        output_path: Path for output (overwrites with new column)
        smoothing_constant: Bayesian prior strength
        
    Returns:
        DataFrame with smoothed ratings
    """
    logger.info("Starting Bayesian rating smoothing pipeline...")
    
    # Load data
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df):,} restaurants")
    
    # Add smoothed rating
    df = add_smoothed_rating_column(
        df,
        smoothing_constant=smoothing_constant
    )
    
    # Print summary
    print_smoothing_summary(df)
    
    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved smoothed ratings to: {output_path}")
    
    logger.info("Rating smoothing pipeline completed successfully!")
    
    return df


# -----------------------------------------------------------------------------
# CLI Entry Point
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Add Bayesian smoothed ratings to restaurant data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=Path("data/processed/restaurants_cleaned.csv"),
        help="Path to input restaurants CSV",
    )
    
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("data/processed/restaurants_cleaned.csv"),
        help="Path for output CSV (default: overwrite input)",
    )
    
    parser.add_argument(
        "--smoothing-constant", "-m",
        type=int,
        default=50,
        help="Bayesian smoothing constant (higher = more trust in global mean)",
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
        run_smoothing_pipeline(
            input_path=args.input,
            output_path=args.output,
            smoothing_constant=args.smoothing_constant,
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
