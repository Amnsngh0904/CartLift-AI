import { motion } from 'framer-motion';

type RestaurantSelectorProps = {
  restaurants: string[];
  selectedId: string | null;
  onSelect: (id: string | null) => void;
  loading: boolean;
  error: string | null;
};

export function RestaurantSelector({
  restaurants,
  selectedId,
  onSelect,
  loading,
  error,
}: RestaurantSelectorProps) {
  if (error) {
    return (
      <div className="rounded-xl bg-red-50 border border-red-200 p-4 text-red-700 text-sm">
        {error}
        <p className="mt-2 text-xs opacity-80">Ensure the recommendation API is running on port 8000.</p>
      </div>
    );
  }

  return (
    <section className="py-6 md:py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <h2 className="text-xl font-bold text-[var(--color-text)] mb-4">Choose a restaurant</h2>
        {loading ? (
          <div className="w-full max-w-xs">
            <div className="h-12 rounded-lg bg-[var(--color-border)] animate-pulse" />
          </div>
        ) : (
          <motion.div
            initial={{ opacity: 0, y: 4 }}
            animate={{ opacity: 1, y: 0 }}
            className="w-full max-w-md"
          >
            <select
              value={selectedId ?? ''}
              onChange={(e) => onSelect(e.target.value || null)}
              className="w-full rounded-xl border border-[var(--color-border)] bg-[var(--color-surface)] px-4 py-3 text-[var(--color-text)] focus:ring-2 focus:ring-[var(--color-primary)] focus:border-transparent focus:outline-none appearance-none cursor-pointer"
              style={{
                backgroundImage: `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 24 24' stroke='%234a4a4a'%3E%3Cpath stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M19 9l-7 7-7-7'%3E%3C/path%3E%3C/svg%3E")`,
                backgroundRepeat: 'no-repeat',
                backgroundPosition: 'right 0.75rem center',
                backgroundSize: '1.25rem',
                paddingRight: '2.5rem',
              }}
            >
              <option value="">Select a restaurant</option>
              {restaurants.map((id) => (
                <option key={id} value={id}>
                  {id.replace('REST_', 'Restaurant ')}
                </option>
              ))}
            </select>
          </motion.div>
        )}
      </div>
    </section>
  );
}
