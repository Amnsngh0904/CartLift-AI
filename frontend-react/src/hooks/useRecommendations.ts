import { useState, useCallback } from 'react';
import { api } from '../utils/api';
import type { RecommendRequest, RecommendResponse } from '../utils/api';

export function useRecommendations() {
  const [data, setData] = useState<RecommendResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchRecommendations = useCallback(async (request: RecommendRequest) => {
    setLoading(true);
    setError(null);
    setData(null);
    try {
      const res = await api.recommend(request);
      setData(res);
      return res;
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Failed to get recommendations';
      setError(msg);
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  return { data, loading, error, fetchRecommendations };
}
