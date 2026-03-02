import { useCart } from '../../context/CartContext';
import { FoodCard } from './FoodCard';
import { FoodCardSkeleton } from '../common/Skeleton';
import type { RestaurantItem } from '../../utils/api';

type FoodGridProps = {
  items: RestaurantItem[];
  loading: boolean;
  emptyMessage?: string;
};

export function FoodGrid({ items, loading, emptyMessage = 'No items to show' }: FoodGridProps) {
  const { addItem, setRestaurant, restaurantId } = useCart();

  const handleAdd = (item: RestaurantItem) => {
    if (restaurantId) setRestaurant(restaurantId);
    addItem({
      item_id: item.item_id,
      item_name: item.item_name,
      price: item.price,
      category: item.category,
    });
  };

  if (loading) {
    return (
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
        {Array.from({ length: 8 }).map((_, i) => (
          <FoodCardSkeleton key={i} />
        ))}
      </div>
    );
  }

  if (!items.length) {
    return (
      <div className="text-center py-16 text-[var(--color-text-secondary)]">
        <p className="text-lg">{emptyMessage}</p>
        <p className="text-sm mt-1">Select a restaurant above to see the menu</p>
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
      {items.map((item) => (
        <FoodCard key={item.item_id} item={item} onAddToCart={handleAdd} />
      ))}
    </div>
  );
}
