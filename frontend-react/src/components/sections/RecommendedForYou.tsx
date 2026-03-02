import { motion } from 'framer-motion';
import { Button } from '../common/Button';
import { getItemImageUrl } from '../../utils/constants';
import type { RecommendationItem } from '../../utils/api';

type RecommendedForYouProps = {
  items: RecommendationItem[];
  loading: boolean;
  onAdd: (item: RecommendationItem) => void;
  latencyMs?: number;
};

export function RecommendedForYou({ items, loading, onAdd, latencyMs }: RecommendedForYouProps) {
  if (loading && !items.length) {
    return (
      <section className="py-8 md:py-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <h2 className="text-2xl font-bold text-[var(--color-text)] mb-6">Recommended for you</h2>
          <div className="flex gap-4 overflow-x-auto pb-4">
            {Array.from({ length: 4 }).map((_, i) => (
              <div
                key={i}
                className="shrink-0 w-56 rounded-xl bg-[var(--color-border)] animate-pulse h-64"
              />
            ))}
          </div>
        </div>
      </section>
    );
  }

  if (!items.length) return null;

  return (
    <section className="py-8 md:py-12">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex flex-wrap items-center justify-between gap-4 mb-6">
          <motion.h2
            initial={{ opacity: 0, x: -10 }}
            animate={{ opacity: 1, x: 0 }}
            className="text-2xl font-bold text-[var(--color-text)]"
          >
            Recommended for you
          </motion.h2>
          {latencyMs != null && (
            <span className="text-sm text-[var(--color-muted)]">
              Suggestions in {Math.round(latencyMs)}ms
            </span>
          )}
        </div>
        <div className="flex gap-4 overflow-x-auto pb-4 scrollbar-thin">
          {items.map((item, i) => (
            <motion.div
              key={item.item_id}
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: i * 0.05 }}
              className="shrink-0 w-56 rounded-xl overflow-hidden bg-[var(--color-surface)] border border-[var(--color-border)] hover:shadow-[var(--shadow-card-hover)] transition-shadow"
            >
              <div className="aspect-[4/3] overflow-hidden">
                <img
                  src={getItemImageUrl(item.item_name, item.category, 224, 168)}
                  alt={item.item_name}
                  className="w-full h-full object-cover"
                />
              </div>
              <div className="p-3">
                <h3 className="font-semibold text-sm text-[var(--color-text)] line-clamp-1">
                  {item.item_name}
                </h3>
                <div className="mt-2 flex items-center justify-between">
                  <span className="font-bold text-[var(--color-text)]">₹{Math.round(item.price)}</span>
                  <Button
                    variant="primary"
                    className="!py-1.5 !px-3 !text-xs"
                    onClick={() => onAdd(item)}
                  >
                    Add
                  </Button>
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
