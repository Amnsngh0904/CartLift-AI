import { motion } from 'framer-motion';
import { CATEGORIES } from '../../utils/constants';

type CategoriesProps = {
  selectedId: string | null;
  onSelect: (id: string) => void;
};

const container = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: { staggerChildren: 0.05 },
  },
};

const item = {
  hidden: { opacity: 0, y: 12 },
  show: { opacity: 1, y: 0 },
};

export function Categories({ selectedId, onSelect }: CategoriesProps) {
  return (
    <section className="py-8 md:py-12">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <motion.h2
          initial={{ opacity: 0, y: 10 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-2xl font-bold text-[var(--color-text)] mb-6"
        >
          Categories
        </motion.h2>
        <motion.div
          variants={container}
          initial="hidden"
          whileInView="show"
          viewport={{ once: true }}
          className="flex flex-wrap gap-3"
        >
          {CATEGORIES.map((cat) => (
            <motion.button
              key={cat.id}
              variants={item}
              type="button"
              onClick={() => onSelect(cat.id)}
              className={`
                px-5 py-2.5 rounded-xl font-medium text-sm transition-all duration-200
                ${selectedId === cat.id
                  ? 'bg-[var(--color-primary)] text-white shadow-md'
                  : 'bg-[var(--color-surface)] text-[var(--color-text-secondary)] border border-[var(--color-border)] hover:border-[var(--color-primary)] hover:text-[var(--color-primary)]'
                }
              `}
            >
              {cat.label}
            </motion.button>
          ))}
        </motion.div>
      </div>
    </section>
  );
}
