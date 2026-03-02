import { useState, useEffect, useCallback } from 'react';
import { api } from '../utils/api';

export function useRestaurants() {
  const [restaurants, setRestaurants] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchRestaurants = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await api.restaurants();
      setRestaurants(data.restaurants ?? []);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load restaurants');
      setRestaurants([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchRestaurants();
  }, [fetchRestaurants]);

  return { restaurants, loading, error, refetch: fetchRestaurants };
}
