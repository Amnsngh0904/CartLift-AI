import { useState, useEffect, useCallback } from 'react';
import { api } from '../utils/api';
import type { RestaurantItem } from '../utils/api';

export function useRestaurantItems(restaurantId: string | null) {
  const [items, setItems] = useState<RestaurantItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchItems = useCallback(async () => {
    if (!restaurantId) {
      setItems([]);
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const data = await api.restaurantItems(restaurantId);
      setItems(data.items ?? []);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load menu');
      setItems([]);
    } finally {
      setLoading(false);
    }
  }, [restaurantId]);

  useEffect(() => {
    fetchItems();
  }, [fetchItems]);

  return { items, loading, error, refetch: fetchItems };
}
