#!/usr/bin/env python3
"""
Generate minimal sample Zomato dataset for running the pipeline.

Creates data/raw/zomato_dataset.csv with the format expected by clean_dataset.py.
Use this when you don't have the full Kaggle dataset.

Usage:
    python scripts/generate_sample_data.py
    python scripts/generate_sample_data.py --num-restaurants 100 --num-items 2000
"""

import argparse
import random
from pathlib import Path

import pandas as pd

# Sample data for realistic Indian restaurant/menu generation
RESTAURANT_NAMES = [
    "Taj Mahal Restaurant", "Spice Garden", "Biryani House", "Dosa Corner",
    "Butter Chicken Palace", "South Indian Delights", "North Indian Kitchen",
    "Punjabi Dhaba", "Mughlai Paradise", "Street Food Junction",
    "Curry House", "Tandoori Express", "Chaat House", "Sweets & More",
    "Cafe Coffee Day", "Chai Point", "Pizza Paradise", "Burger King India",
]

CITIES = ["Bangalore", "Mumbai", "Delhi", "Chennai", "Hyderabad", "Pune", "Kolkata"]
PLACES = ["Koramangala", "Indiranagar", "Bandra", "Andheri", "Connaught Place", "Anna Nagar"]

CUISINES = [
    "North Indian", "South Indian", "Chinese", "Mughlai", "Biryani",
    "Street Food", "Bakery", "Desserts", "Beverages", "Fast Food"
]

MAIN_ITEMS = [
    "Butter Chicken", "Chicken Biryani", "Paneer Butter Masala", "Dal Makhani",
    "Veg Biryani", "Fish Curry", "Mutton Rogan Josh", "Chole Bhature",
    "Dosa", "Idli", "Upma", "Pav Bhaji", "Samosa", "Pani Puri",
    "Fried Rice", "Hakka Noodles", "Manchurian", "Kadai Chicken",
    "Palak Paneer", "Rajma Chawal", "Aloo Paratha", "Pav Bhaji",
]

SIDE_ITEMS = ["Naan", "Roti", "Rice", "Jeera Rice", "Raita", "Salad", "Papad"]
DESSERT_ITEMS = ["Gulab Jamun", "Rasmalai", "Kheer", "Ice Cream", "Brownie", "Jalebi"]
BEVERAGE_ITEMS = ["Coke", "Lassi", "Masala Chai", "Cold Coffee", "Fresh Lime", "Water"]


def generate_sample_dataset(
    num_restaurants: int = 80,
    items_per_restaurant: int = 25,
    output_path: Path = Path("data/raw/zomato_dataset.csv"),
) -> None:
    """Generate a minimal Zomato-format CSV for pipeline testing."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for i in range(num_restaurants):
        rest_name = random.choice(RESTAURANT_NAMES) + f" {i % 10}"
        city = random.choice(CITIES)
        place = random.choice(PLACES)
        cuisine = random.choice(CUISINES)
        dining_rating = round(random.uniform(3.5, 4.8), 1)
        delivery_rating = round(random.uniform(3.2, 4.6), 1)
        dining_votes = random.randint(100, 5000)
        delivery_votes = random.randint(50, 3000)

        # Generate menu items for this restaurant
        all_items = (
            random.sample(MAIN_ITEMS, min(8, len(MAIN_ITEMS))) +
            random.sample(SIDE_ITEMS, min(4, len(SIDE_ITEMS))) +
            random.sample(DESSERT_ITEMS, min(3, len(DESSERT_ITEMS))) +
            random.sample(BEVERAGE_ITEMS, min(4, len(BEVERAGE_ITEMS)))
        )
        all_items = all_items[:items_per_restaurant]

        for item_name in all_items:
            price = random.randint(50, 500) if "Biryani" in item_name or "Chicken" in item_name else random.randint(30, 250)
            votes = random.randint(0, 500)
            best_seller = 1 if random.random() < 0.2 else 0

            rows.append({
                "Restaurant Name": rest_name,
                "Dining Rating": dining_rating,
                "Delivery Rating": delivery_rating,
                "Dining Votes": dining_votes,
                "Delivery Votes": delivery_votes,
                "Cuisine": cuisine,
                "Place Name": place,
                "City": city,
                "Item Name": item_name,
                "Best Seller": best_seller,
                "Votes": votes,
                "Prices": price,
            })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df):,} rows ({num_restaurants} restaurants) -> {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-restaurants", type=int, default=80)
    parser.add_argument("--num-items", type=int, default=25, help="Max items per restaurant")
    parser.add_argument("--output", type=Path, default=Path("data/raw/zomato_dataset.csv"))
    args = parser.parse_args()
    generate_sample_dataset(args.num_restaurants, args.num_items, args.output)


if __name__ == "__main__":
    main()
