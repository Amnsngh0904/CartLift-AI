#!/usr/bin/env python3
"""Analyze cleaned and enriched data before proceeding."""

import pandas as pd
import numpy as np

def main():
    # Load data
    restaurants = pd.read_csv('data/processed/restaurants_cleaned.csv')
    menu_items = pd.read_csv('data/processed/menu_items_enriched.csv')

    print("=" * 70)
    print("DATA ANALYSIS SUMMARY")
    print("=" * 70)

    # 1. RESTAURANTS ANALYSIS
    print("\n" + "=" * 70)
    print("1. RESTAURANTS ANALYSIS")
    print("=" * 70)

    print(f"\nTotal Restaurants: {len(restaurants)}")

    print("\nDistribution by City:")
    for city, count in restaurants['city'].value_counts().items():
        pct = count / len(restaurants) * 100
        print(f"  {city}: {count} ({pct:.1f}%)")

    print("\nDelivery Rating Distribution:")
    print(f"  Min: {restaurants['delivery_rating'].min():.1f}")
    print(f"  Median: {restaurants['delivery_rating'].median():.1f}")
    print(f"  Mean: {restaurants['delivery_rating'].mean():.2f}")
    print(f"  Max: {restaurants['delivery_rating'].max():.1f}")
    print(f"  Std: {restaurants['delivery_rating'].std():.2f}")

    print("\nDelivery Votes Distribution:")
    print(f"  Min: {int(restaurants['delivery_votes'].min())}")
    print(f"  Median: {int(restaurants['delivery_votes'].median())}")
    print(f"  Mean: {restaurants['delivery_votes'].mean():.1f}")
    print(f"  Max: {int(restaurants['delivery_votes'].max())}")
    print(f"  95th pct: {int(restaurants['delivery_votes'].quantile(0.95))}")
    zero_votes = (restaurants['delivery_votes'] == 0).sum()
    print(f"  Zero votes: {zero_votes} ({zero_votes/len(restaurants)*100:.1f}%)")

    # 2. MENU ITEMS ANALYSIS
    print("\n" + "=" * 70)
    print("2. MENU ITEMS ANALYSIS")
    print("=" * 70)

    print(f"\nTotal Menu Items: {len(menu_items)}")

    # Items per restaurant
    items_per_rest = menu_items.groupby('restaurant_id').size()
    print("\nItems per Restaurant:")
    print(f"  Min: {items_per_rest.min()}")
    print(f"  Median: {items_per_rest.median():.0f}")
    print(f"  Mean: {items_per_rest.mean():.1f}")
    print(f"  Max: {items_per_rest.max()}")

    print("\nPrice Distribution:")
    print(f"  Min: Rs.{menu_items['price'].min():.0f}")
    print(f"  Median: Rs.{menu_items['price'].median():.0f}")
    print(f"  Mean: Rs.{menu_items['price'].mean():.0f}")
    print(f"  95th pct: Rs.{menu_items['price'].quantile(0.95):.0f}")
    print(f"  99th pct: Rs.{menu_items['price'].quantile(0.99):.0f}")
    print(f"  Max: Rs.{menu_items['price'].max():.0f}")

    print("\nCategory Distribution:")
    for cat, count in menu_items['item_category'].value_counts().items():
        pct = count / len(menu_items) * 100
        print(f"  {cat}: {count} ({pct:.1f}%)")

    best_seller_count = (menu_items['best_seller'] == 1).sum()
    print(f"\nBest Seller Percentage: {best_seller_count/len(menu_items)*100:.1f}%")
    print(f"  Best sellers: {best_seller_count}")
    print(f"  Non-best sellers: {len(menu_items) - best_seller_count}")

    print("\nItem Votes Distribution:")
    print(f"  Min: {int(menu_items['item_votes'].min())}")
    print(f"  Median: {int(menu_items['item_votes'].median())}")
    print(f"  Mean: {menu_items['item_votes'].mean():.1f}")
    print(f"  95th pct: {int(menu_items['item_votes'].quantile(0.95))}")
    print(f"  Max: {int(menu_items['item_votes'].max())}")
    zero_item_votes = (menu_items['item_votes'] == 0).sum()
    print(f"  Zero votes: {zero_item_votes} ({zero_item_votes/len(menu_items)*100:.1f}%)")

    # 3. ANOMALIES
    print("\n" + "=" * 70)
    print("3. ANOMALY DETECTION")
    print("=" * 70)

    # Restaurants with <3 items
    rest_few_items = items_per_rest[items_per_rest < 3]
    print(f"\nRestaurants with <3 items: {len(rest_few_items)}")
    if len(rest_few_items) > 0:
        for rest_id, count in rest_few_items.head(5).items():
            rest_name = restaurants[restaurants['restaurant_id'] == rest_id]['restaurant_name'].values
            name = rest_name[0] if len(rest_name) > 0 else "Unknown"
            print(f"  {rest_id}: {count} item(s) - {name}")
        if len(rest_few_items) > 5:
            print(f"  ... and {len(rest_few_items) - 5} more")

    # Restaurants with >100 items
    rest_many_items = items_per_rest[items_per_rest > 100]
    print(f"\nRestaurants with >100 items: {len(rest_many_items)}")
    if len(rest_many_items) > 0:
        for rest_id, count in rest_many_items.sort_values(ascending=False).head(5).items():
            rest_name = restaurants[restaurants['restaurant_id'] == rest_id]['restaurant_name'].values
            name = rest_name[0] if len(rest_name) > 0 else "Unknown"
            print(f"  {rest_id}: {count} items - {name}")
        if len(rest_many_items) > 5:
            print(f"  ... and {len(rest_many_items) - 5} more")

    # Extreme price outliers
    Q1 = menu_items['price'].quantile(0.25)
    Q3 = menu_items['price'].quantile(0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 3 * IQR

    extreme_high = menu_items[menu_items['price'] > upper_bound]
    print(f"\nExtreme Price Outliers (>Rs.{upper_bound:.0f}, using 3x IQR): {len(extreme_high)}")
    
    if len(extreme_high) > 0:
        print("\nTop 10 highest priced items:")
        top_expensive = menu_items.nlargest(10, 'price')[['item_name', 'price', 'restaurant_id']]
        for _, row in top_expensive.iterrows():
            rest_name = restaurants[restaurants['restaurant_id'] == row['restaurant_id']]['restaurant_name'].values
            name = rest_name[0] if len(rest_name) > 0 else "Unknown"
            print(f"  Rs.{row['price']:.0f}: {row['item_name'][:40]} ({name[:20]})")

    # Very low prices
    very_cheap = menu_items[menu_items['price'] <= 10]
    print(f"\nItems with very low price (<=Rs.10): {len(very_cheap)}")

    # Data integrity
    menu_rest_ids = set(menu_items['restaurant_id'].unique())
    rest_ids = set(restaurants['restaurant_id'].unique())
    orphan_items = menu_rest_ids - rest_ids
    missing_menus = rest_ids - menu_rest_ids

    print(f"\nData Integrity:")
    print(f"  Menu items with no restaurant: {len(orphan_items)} restaurant IDs")
    print(f"  Restaurants with no menu items: {len(missing_menus)} restaurants")

    # 4. SUMMARY TABLE
    print("\n" + "=" * 70)
    print("4. SUMMARY TABLE")
    print("=" * 70)

    print(f"""
{'Metric':<30} {'Value':>15}
{'-'*45}
{'Total Restaurants':<30} {len(restaurants):>15}
{'Total Menu Items':<30} {len(menu_items):>15}
{'Avg Items per Restaurant':<30} {items_per_rest.mean():>15.1f}
{'Median Price':<30} {'Rs.' + str(int(menu_items['price'].median())):>15}
{'95th Percentile Price':<30} {'Rs.' + str(int(menu_items['price'].quantile(0.95))):>15}
{'Best Seller %':<30} {best_seller_count/len(menu_items)*100:>14.1f}%
{'Cities Covered':<30} {restaurants['city'].nunique():>15}
{'Categories':<30} {menu_items['item_category'].nunique():>15}
{'Restaurants <3 items':<30} {len(rest_few_items):>15}
{'Restaurants >100 items':<30} {len(rest_many_items):>15}
{'Extreme Price Outliers':<30} {len(extreme_high):>15}
""")

    # 5. RECOMMENDATIONS
    print("=" * 70)
    print("5. RECOMMENDATIONS")
    print("=" * 70)

    recommendations = []

    if len(rest_few_items) > 0:
        recommendations.append(f"- Filter out {len(rest_few_items)} restaurants with <3 menu items (insufficient for addon recommendations)")

    if len(rest_many_items) > 0:
        recommendations.append(f"- Review {len(rest_many_items)} restaurants with >100 items - may need sampling or special handling")

    if len(extreme_high) > 0:
        recommendations.append(f"- Review {len(extreme_high)} extreme price outliers (>Rs.{upper_bound:.0f}) - may be combos or data errors")

    if zero_votes / len(restaurants) > 0.5:
        recommendations.append(f"- {zero_votes/len(restaurants)*100:.0f}% of restaurants have zero delivery votes - may affect recommendation quality")

    unknown_cat = (menu_items['item_category'] == 'unknown').sum()
    if unknown_cat > 0:
        recommendations.append(f"- {unknown_cat} items ({unknown_cat/len(menu_items)*100:.1f}%) have 'unknown' category - consider improving categorization")

    if len(orphan_items) > 0:
        recommendations.append(f"- Fix {len(orphan_items)} menu items referencing non-existent restaurants")

    if len(missing_menus) > 0:
        recommendations.append(f"- {len(missing_menus)} restaurants have no menu items - consider removing from dataset")

    for rec in recommendations:
        print(rec)

    print("\n" + "=" * 70)
    print("Analysis complete. Awaiting confirmation before proceeding.")
    print("=" * 70)

if __name__ == "__main__":
    main()
