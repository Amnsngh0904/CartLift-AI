#!/usr/bin/env python3
"""Analyze session generation statistics for behavioral signals."""

import pandas as pd

# Load cart events
print("Loading cart events...")
df = pd.read_csv('data/synthetic/cart_events.csv')

print('=== UPDATED STATISTICS ===')
print()

# Positive rate
positive_rate = df['label'].mean()
print(f'Positive Rate: {positive_rate:.4f} ({positive_rate*100:.2f}%)')
print()

# Filter to positive events only
pos_df = df[df['label'] == 1]

# Get candidate item categories
menu_items = pd.read_csv('data/processed/menu_items_simulation.csv')
item_to_cat = dict(zip(menu_items['item_id'], menu_items['item_category']))
pos_df = pos_df.copy()
pos_df['candidate_category'] = pos_df['candidate_item'].map(item_to_cat)

print('Category Distribution in Positive Adds:')
cat_counts = pos_df['candidate_category'].value_counts()
cat_total = len(pos_df)
for cat, count in cat_counts.items():
    print(f'  {cat}: {count:,} ({count/cat_total*100:.2f}%)')

print()

# Beverage attach rate (with main, no beverage)
print('Beverage attach rate (main present, no beverage):')
pos_main_no_bev = pos_df[
    pos_df['cart_categories'].str.contains('main', na=False) & 
    ~pos_df['cart_categories'].str.contains('beverage', na=False)
]
bev_attach = (pos_main_no_bev['candidate_category'] == 'beverage').sum()
bev_rate = bev_attach / len(pos_main_no_bev) if len(pos_main_no_bev) > 0 else 0
print(f'  Beverage: {bev_attach:,} / {len(pos_main_no_bev):,} ({bev_rate*100:.2f}%)')

print()

# Dessert attach rate (with heavy meal)
def is_heavy(cats):
    if pd.isna(cats) or cats == '':
        return False
    cat_list = cats.split('|')
    main_count = cat_list.count('main')
    side_count = cat_list.count('side')
    return main_count >= 2 or (main_count >= 1 and side_count >= 1)

print('Dessert attach rate (heavy meal):')
pos_heavy = pos_df[pos_df['cart_categories'].apply(is_heavy)]
dessert_attach = (pos_heavy['candidate_category'] == 'dessert').sum()
dessert_rate = dessert_attach / len(pos_heavy) if len(pos_heavy) > 0 else 0
print(f'  Dessert: {dessert_attach:,} / {len(pos_heavy):,} ({dessert_rate*100:.2f}%)')

print()

# Premium user dessert rate
print('Premium user dessert rate:')
pos_premium = pos_df[pos_df['user_type'].isin(['premium', 'luxury'])]
premium_dessert = (pos_premium['candidate_category'] == 'dessert').sum()
premium_rate = premium_dessert / len(pos_premium) if len(pos_premium) > 0 else 0
pos_budget = pos_df[pos_df['user_type'].isin(['budget', 'moderate'])]
budget_dessert = (pos_budget['candidate_category'] == 'dessert').sum()
budget_rate = budget_dessert / len(pos_budget) if len(pos_budget) > 0 else 0
print(f'  Premium/Luxury: {premium_rate*100:.2f}%')
print(f'  Budget/Moderate: {budget_rate*100:.2f}%')
print(f'  Ratio: {premium_rate/budget_rate:.2f}x' if budget_rate > 0 else '  N/A')

print()

# Lunch/Dinner beverage rate
print('Lunch/Dinner beverage rate:')
pos_meal_ld = pos_df[pos_df['meal_type'].isin(['lunch', 'dinner'])]
ld_bev = (pos_meal_ld['candidate_category'] == 'beverage').sum()
ld_rate = ld_bev / len(pos_meal_ld) if len(pos_meal_ld) > 0 else 0
pos_meal_other = pos_df[~pos_df['meal_type'].isin(['lunch', 'dinner'])]
other_bev = (pos_meal_other['candidate_category'] == 'beverage').sum()
other_rate = other_bev / len(pos_meal_other) if len(pos_meal_other) > 0 else 0
print(f'  Lunch/Dinner: {ld_rate*100:.2f}%')
print(f'  Other times: {other_rate*100:.2f}%')
print(f'  Ratio: {ld_rate/other_rate:.2f}x' if other_rate > 0 else '  N/A')
