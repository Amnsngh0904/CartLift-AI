"""Analyze degree distribution of co-occurrence graph."""
import sys
sys.path.insert(0, '.')

import pickle
import statistics
from src.graph.build_cooccurrence import CooccurrenceGraph

with open('data/processed/item_cooccurrence.pkl', 'rb') as f:
    graph = pickle.load(f)

degrees = {item: len(neighbors) for item, neighbors in graph.adjacency_list.items()}
sorted_degrees = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
degree_values = list(degrees.values())
total = len(degree_values)

print('=' * 60)
print('TOP 10 HIGHEST DEGREE ITEMS')
print('=' * 60)
for i, (item_id, deg) in enumerate(sorted_degrees[:10], 1):
    print(f'{i:2}. {item_id} | degree={deg} | occurrences={graph.item_counts.get(item_id,0):,}')

low = sum(1 for d in degree_values if d <= 2)
high = sum(1 for d in degree_values if d >= 20)

print()
print('=' * 60)
print('DEGREE DISTRIBUTION SUMMARY')
print('=' * 60)
print(f'Total items: {total:,}')
print(f'Items with degree <= 2: {low:,} ({100*low/total:.2f}%)')
print(f'Items with degree >= 20: {high:,} ({100*high/total:.2f}%)')
print()
print(f'Mean degree: {statistics.mean(degree_values):.2f}')
print(f'Median degree: {statistics.median(degree_values):.1f}')
print(f'Min: {min(degree_values)}, Max: {max(degree_values)}')
print('=' * 60)
