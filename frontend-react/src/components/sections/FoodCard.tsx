import { motion } from 'framer-motion';
import { Button } from '../common/Button';
import { getItemImageUrl } from '../../utils/constants';
import type { RestaurantItem } from '../../utils/api';

type FoodCardProps = {
  item: RestaurantItem;
  onAddToCart: (item: RestaurantItem) => void;
  loading?: boolean;
  showRating?: boolean;
};

export function FoodCard({ item, onAddToCart, loading, showRating = true }: FoodCardProps) {
  const imageUrl = getItemImageUrl(item.item_name, item.category);

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      whileHover={{ y: -4, boxShadow: 'var(--shadow-card-hover)' }}
      transition={{ duration: 0.2 }}
      className="rounded-[var(--radius-card)] overflow-hidden bg-[var(--color-surface)] border border-[var(--color-border)]"
    >
      <div className="relative aspect-[4/3] overflow-hidden bg-[var(--color-background)]">
        <img
          src={imageUrl}
          alt={item.item_name}
          className="w-full h-full object-cover transition-transform duration-300 hover:scale-105"
        />
        {showRating && (
          <span className="absolute top-2 left-2 flex items-center gap-1 rounded-md bg-black/60 px-2 py-0.5 text-xs font-medium text-white">
            <StarIcon className="w-3.5 h-3.5" />
            4.2
          </span>
        )}
        <span className="absolute top-2 right-2 rounded-md bg-[var(--color-surface)]/90 px-2 py-0.5 text-xs font-medium capitalize text-[var(--color-text-secondary)]">
          {item.category}
        </span>
      </div>
      <div className="p-4">
        <h3 className="font-semibold text-[var(--color-text)] line-clamp-1">{item.item_name}</h3>
        <p className="mt-0.5 text-sm text-[var(--color-text-secondary)] line-clamp-2">
          Delicious {item.category} – add to your order
        </p>
        <div className="mt-3 flex items-center justify-between gap-2">
          <span className="text-lg font-bold text-[var(--color-text)]">
            ₹{Math.round(item.price)}
          </span>
          <Button
            variant="primary"
            onClick={() => onAddToCart(item)}
            disabled={loading}
            loading={loading}
            className="shrink-0"
          >
            Add to cart
          </Button>
        </div>
      </div>
    </motion.div>
  );
}

function StarIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="currentColor" viewBox="0 0 20 20">
      <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
    </svg>
  );
}
