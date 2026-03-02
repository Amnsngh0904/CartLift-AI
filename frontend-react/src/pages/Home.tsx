import { useState, useCallback, useEffect, useMemo } from 'react';
import { motion } from 'framer-motion';
import { Navbar } from '../components/layout/Navbar';
import { Hero } from '../components/layout/Hero';
import { Footer } from '../components/layout/Footer';
import { Categories } from '../components/sections/Categories';
import { RestaurantSelector } from '../components/sections/RestaurantSelector';
import { FoodGrid } from '../components/sections/FoodGrid';
import { RecommendedForYou } from '../components/sections/RecommendedForYou';
import { CartSidebar } from '../components/cart/CartSidebar';
import { useRestaurants } from '../hooks/useRestaurants';
import { useRestaurantItems } from '../hooks/useRestaurantItems';
import { useRecommendations } from '../hooks/useRecommendations';
import { useCart } from '../context/CartContext';
import { useApiHealth } from '../hooks/useApiHealth';
import type { RecommendationItem } from '../utils/api';

const DEFAULT_USER = 'USER_0';
const getHour = () => new Date().getHours();

export function Home() {
  const [selectedRestaurant, setSelectedRestaurant] = useState<string | null>(null);
  const [categoryFilter, setCategoryFilter] = useState<string | null>(null);
  const [mealType] = useState<string>('lunch');
  const [userType] = useState<string>('moderate');

  const { healthy: apiHealthy } = useApiHealth();
  const { restaurants, loading: restaurantsLoading, error: restaurantsError } = useRestaurants();
  const { items, loading: itemsLoading } = useRestaurantItems(selectedRestaurant);
  const { data: recommendationsData, loading: recLoading, fetchRecommendations } = useRecommendations();
  const { items: cartItems, setRestaurant, addItem: addToCart } = useCart();

  const cartItemIds = useMemo(() => cartItems.map((i) => i.item_id), [cartItems]);

  const fetchRecs = useCallback(() => {
    if (!selectedRestaurant) return;
    setRestaurant(selectedRestaurant);
    fetchRecommendations({
      user_id: DEFAULT_USER,
      restaurant_id: selectedRestaurant,
      cart_item_ids: cartItemIds,
      hour: getHour(),
      meal_type: mealType,
      user_type: userType,
    });
  }, [selectedRestaurant, cartItemIds, mealType, userType, fetchRecommendations, setRestaurant]);

  useEffect(() => {
    if (selectedRestaurant && apiHealthy) {
      fetchRecs();
    }
  }, [selectedRestaurant, apiHealthy, fetchRecs]);

  const filteredItems = categoryFilter
    ? items.filter((i) => i.category === categoryFilter)
    : items;

  const handleAddRecommendation = (item: RecommendationItem) => {
    addToCart({
      item_id: item.item_id,
      item_name: item.item_name,
      price: item.price,
      category: item.category,
    });
  };

  return (
    <div className="min-h-screen flex flex-col">
      <Navbar location="Bangalore" />
      <main className="flex-1">
        <Hero />
        {!apiHealthy && (
          <div className="max-w-7xl mx-auto px-4 py-3 bg-amber-50 border-b border-amber-200 text-amber-800 text-sm text-center">
            Recommendation API is offline. Menu browsing still works. Start the API to see suggestions.
          </div>
        )}
        <RestaurantSelector
          restaurants={restaurants}
          selectedId={selectedRestaurant}
          onSelect={setSelectedRestaurant}
          loading={restaurantsLoading}
          error={restaurantsError}
        />
        <Categories selectedId={categoryFilter} onSelect={setCategoryFilter} />
        <section className="py-6 md:py-10">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <motion.h2
              initial={{ opacity: 0, y: 10 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              className="text-2xl font-bold text-[var(--color-text)] mb-6"
            >
              {selectedRestaurant ? 'Menu' : 'Select a restaurant to view menu'}
            </motion.h2>
            <FoodGrid
              items={filteredItems}
              loading={itemsLoading}
              emptyMessage={categoryFilter ? `No ${categoryFilter} items` : 'No items to show'}
            />
          </div>
        </section>
        <RecommendedForYou
          items={recommendationsData?.recommendations ?? []}
          loading={recLoading}
          onAdd={handleAddRecommendation}
          latencyMs={recommendationsData?.latency_ms}
        />
        <Footer />
      </main>
      <CartSidebar />
    </div>
  );
}
