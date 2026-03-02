import { API_BASE } from './constants';

export type RecommendRequest = {
  user_id: string;
  restaurant_id: string;
  cart_item_ids: string[];
  hour: number;
  meal_type: string;
  user_type: string;
};

export type RecommendationItem = {
  item_id: string;
  item_name: string;
  price: number;
  category: string;
  score: number;
};

export type RecommendResponse = {
  recommendations: RecommendationItem[];
  latency_ms: number;
  feature_build_ms: number;
  model_forward_ms: number;
  candidate_count: number;
};

export type HealthResponse = {
  status: string;
  model_loaded: boolean;
  device: string;
  num_items: number;
  num_restaurants: number;
  num_users: number;
};

export type RestaurantItem = {
  item_id: string;
  item_name: string;
  price: number;
  category: string;
};

async function fetchApi<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: { 'Content-Type': 'application/json', ...options?.headers },
  });
  if (!res.ok) throw new Error(await res.text().catch(() => res.statusText));
  return res.json();
}

export const api = {
  health: () => fetchApi<HealthResponse>('/health'),
  restaurants: () => fetchApi<{ restaurants: string[]; total: number }>('/restaurants'),
  restaurantItems: (restaurantId: string, limit = 50) =>
    fetchApi<{ items: RestaurantItem[]; total: number }>(`/restaurant/${restaurantId}/items?limit=${limit}`),
  recommend: (body: RecommendRequest) =>
    fetchApi<RecommendResponse>('/recommend', { method: 'POST', body: JSON.stringify(body) }),
};
