"""
Extract Cart Sequences from Cart Events

Creates a compact sequences file containing only positive samples.
Groups items by session in their chronological order.

Output:
    data/processed/cart_sequences.csv with columns:
    - session_id
    - restaurant_id  
    - item_sequence (pipe-separated ordered item IDs)

Usage:
    python -m src.data_generation.extract_sequences
    python -m src.data_generation.extract_sequences --input data/synthetic/cart_events.csv
"""

import argparse
import logging
import sys
from pathlib import Path

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
# Constants
# -----------------------------------------------------------------------------

DEFAULT_INPUT_PATH = Path("data/synthetic/cart_events.csv")
DEFAULT_OUTPUT_PATH = Path("data/processed/cart_sequences.csv")
MAX_FILE_SIZE_MB = 500


# -----------------------------------------------------------------------------
# Main Functions
# -----------------------------------------------------------------------------

def extract_sequences(
    input_path: Path,
    output_path: Path,
    chunk_size: int = 500_000
) -> pd.DataFrame:
    """
    Extract cart sequences from cart events file.
    
    Only processes positive samples (label=1) and groups by session
    to create ordered item sequences.
    
    Args:
        input_path: Path to cart_events.csv
        output_path: Path for output cart_sequences.csv
        chunk_size: Rows to process per chunk (for memory efficiency)
        
    Returns:
        DataFrame with cart sequences
    """
    logger.info("=" * 60)
    logger.info("CART SEQUENCE EXTRACTION")
    logger.info("=" * 60)
    logger.info(f"Input: {input_path}")
    logger.info(f"Output: {output_path}")
    
    # Process in chunks for memory efficiency
    logger.info(f"\nReading cart events (chunk size: {chunk_size:,})...")
    
    # Only read columns we need
    usecols = ["session_id", "restaurant_id", "step_number", "candidate_item", "label"]
    
    sequences_data = []
    total_positives = 0
    total_sessions = set()
    
    chunk_num = 0
    for chunk in pd.read_csv(input_path, usecols=usecols, chunksize=chunk_size):
        chunk_num += 1
        
        # Filter to positive samples only
        positives = chunk[chunk["label"] == 1].copy()
        total_positives += len(positives)
        total_sessions.update(positives["session_id"].unique())
        
        if chunk_num % 5 == 0:
            logger.info(f"  Processed chunk {chunk_num}, "
                       f"positives so far: {total_positives:,}, "
                       f"sessions: {len(total_sessions):,}")
        
        # Group by session and create sequences
        for session_id, group in positives.groupby("session_id"):
            # Sort by step number to get correct order
            group_sorted = group.sort_values("step_number")
            
            restaurant_id = group_sorted["restaurant_id"].iloc[0]
            items = group_sorted["candidate_item"].tolist()
            
            sequences_data.append({
                "session_id": session_id,
                "restaurant_id": restaurant_id,
                "item_sequence": "|".join(items),
                "sequence_length": len(items),
            })
    
    logger.info(f"\nTotal positives processed: {total_positives:,}")
    logger.info(f"Total sessions: {len(total_sessions):,}")
    
    # Create DataFrame
    df = pd.DataFrame(sequences_data)
    
    # Sort by session_id for consistency
    df = df.sort_values("session_id").reset_index(drop=True)
    
    # Log statistics
    logger.info(f"\nSequence Statistics:")
    logger.info(f"  Total sequences: {len(df):,}")
    logger.info(f"  Length distribution:")
    for length, count in df["sequence_length"].value_counts().sort_index().items():
        pct = 100 * count / len(df)
        logger.info(f"    Length {length}: {count:,} ({pct:.1f}%)")
    
    # Check estimated file size
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write to temp file to check size
    temp_path = output_path.with_suffix(".tmp")
    
    # Drop sequence_length from output (used only for stats)
    df_output = df[["session_id", "restaurant_id", "item_sequence"]]
    df_output.to_csv(temp_path, index=False)
    
    file_size_mb = temp_path.stat().st_size / (1024 * 1024)
    logger.info(f"\nOutput file size: {file_size_mb:.1f} MB")
    
    if file_size_mb > MAX_FILE_SIZE_MB:
        logger.warning(f"File size exceeds {MAX_FILE_SIZE_MB}MB limit!")
        # Could implement sampling here if needed
    
    # Rename temp to final
    temp_path.rename(output_path)
    logger.info(f"Saved to: {output_path}")
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("EXTRACTION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Sequences: {len(df):,}")
    logger.info(f"Unique restaurants: {df['restaurant_id'].nunique():,}")
    logger.info(f"Mean sequence length: {df['sequence_length'].mean():.2f}")
    logger.info(f"File size: {file_size_mb:.1f} MB")
    
    return df


# -----------------------------------------------------------------------------
# CLI Entry Point
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract cart sequences from cart events",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Path to input cart_events.csv",
    )
    
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path for output cart_sequences.csv",
    )
    
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500_000,
        help="Rows to process per chunk",
    )
    
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()
    
    try:
        extract_sequences(
            input_path=args.input,
            output_path=args.output,
            chunk_size=args.chunk_size,
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
