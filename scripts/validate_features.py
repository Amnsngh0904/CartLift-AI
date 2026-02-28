"""Quick validation of feature pipeline."""
import pandas as pd
import os

# Load feature files
user_df = pd.read_parquet('data/processed/user_features.parquet')
rest_df = pd.read_parquet('data/processed/restaurant_features.parquet')

# Check normalized columns
user_norm_cols = [c for c in user_df.columns if c.endswith('_norm')]
rest_norm_cols = [c for c in rest_df.columns if c.endswith('_norm')]

print('=' * 60)
print('FEATURE PIPELINE VALIDATION')
print('=' * 60)

print(f'\nUser features: {len(user_df):,} rows')
print(f'  Normalized columns: {user_norm_cols}')
user_nulls = user_df[user_norm_cols].isnull().sum().to_dict()
print(f'  Nulls in _norm columns: {user_nulls}')

print(f'\nRestaurant features: {len(rest_df):,} rows')
print(f'  Normalized columns: {rest_norm_cols}')
rest_nulls = rest_df[rest_norm_cols].isnull().sum().to_dict()
print(f'  Nulls in _norm columns: {rest_nulls}')

# Load cart events for positive rate
cart_df = pd.read_csv('data/synthetic/cart_events.csv', nrows=100000)
pos_rate = cart_df['label'].mean()
print(f'\nPositive rate (first 100k): {pos_rate:.4f} ({pos_rate*100:.1f}%)')

# Total samples
cart_size = os.path.getsize('data/synthetic/cart_events.csv')
print(f'Cart events file size: {cart_size / 1e6:.1f} MB')

total_nulls = sum(user_nulls.values()) + sum(rest_nulls.values())
print(f'\nVALIDATION: {"PASS" if total_nulls == 0 else "FAIL"} (total nulls: {total_nulls})')
