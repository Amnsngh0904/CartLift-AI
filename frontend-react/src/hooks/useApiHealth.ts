import { useState, useEffect, useCallback } from 'react';
import { api } from '../utils/api';

export function useApiHealth() {
  const [healthy, setHealthy] = useState(false);
  const [loading, setLoading] = useState(true);

  const check = useCallback(async () => {
    try {
      const res = await api.health();
      setHealthy(res.status === 'healthy' && res.model_loaded);
    } catch {
      setHealthy(false);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    check();
    const t = setInterval(check, 30000);
    return () => clearInterval(t);
  }, [check]);

  return { healthy, loading, refetch: check };
}
