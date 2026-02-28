"""Analyze feature correlations and identify missing signals."""
import pandas as pd

# Load data
print("Loading data...")
df = pd.read_parquet('data/processed/training_dataset.parquet')
item_df = pd.read_parquet('data/processed/item_features.parquet')

# Map candidate_item to category
item_to_cat = dict(zip(item_df['item_id'], item_df['category_index']))
df['candidate_category'] = df['candidate_item'].map(item_to_cat)

# Check correlation of interaction: missing_beverage * candidate_is_beverage
df['is_beverage'] = (df['candidate_category'] == 3).astype(int)
df['is_dessert'] = (df['candidate_category'] == 2).astype(int)
df['is_main'] = (df['candidate_category'] == 0).astype(int)

# Interaction features
df['missing_bev_x_is_bev'] = df['missing_beverage'] * df['is_beverage']
df['missing_des_x_is_des'] = df['missing_dessert'] * df['is_dessert']
df['heavy_x_is_des'] = df['heavy_meal'] * df['is_dessert']

print("=" * 60)
print("FEATURE CORRELATION ANALYSIS")
print("=" * 60)

print("\nCorrelation with label:")
print(f"  is_beverage:           {df[['is_beverage', 'label']].corr().iloc[0,1]:.4f}")
print(f"  is_dessert:            {df[['is_dessert', 'label']].corr().iloc[0,1]:.4f}")
print(f"  is_main:               {df[['is_main', 'label']].corr().iloc[0,1]:.4f}")
print(f"  missing_beverage:      {df[['missing_beverage', 'label']].corr().iloc[0,1]:.4f}")
print(f"  missing_dessert:       {df[['missing_dessert', 'label']].corr().iloc[0,1]:.4f}")
print(f"  heavy_meal:            {df[['heavy_meal', 'label']].corr().iloc[0,1]:.4f}")
print(f"  missing_bev_x_is_bev:  {df[['missing_bev_x_is_bev', 'label']].corr().iloc[0,1]:.4f}")
print(f"  missing_des_x_is_des:  {df[['missing_des_x_is_des', 'label']].corr().iloc[0,1]:.4f}")
print(f"  heavy_x_is_des:        {df[['heavy_x_is_des', 'label']].corr().iloc[0,1]:.4f}")

print("\nPositive rate by candidate_category:")
cat_names = {0: 'main', 1: 'starter', 2: 'dessert', 3: 'beverage', 4: 'side'}
for cat in sorted(df['candidate_category'].dropna().unique()):
    cat_data = df[df['candidate_category'] == cat]
    if len(cat_data) > 0:
        name = cat_names.get(int(cat), f'cat_{int(cat)}')
        print(f"  {name:12s} ({int(cat)}): {cat_data['label'].mean():.4f} (n={len(cat_data):,})")

print("\nConclusion:")
print("  The model needs candidate_category as a feature to learn the")
print("  behavioral signals (beverages have higher positive rate).")
