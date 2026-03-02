import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';

export function ComingSoon() {
  return (
    <div className="min-h-screen flex flex-col bg-[var(--color-background)]">
      <header className="border-b border-[var(--color-border)] bg-[var(--color-surface)]">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <Link to="/" className="text-xl font-bold text-[var(--color-primary)]">
            CARTLIFT AI
          </Link>
        </div>
      </header>
      <main className="flex-1 flex items-center justify-center px-4">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4 }}
          className="text-center max-w-md"
        >
          <h1 className="text-3xl font-bold text-[var(--color-text)] mb-2">
            This feature will be built soon
          </h1>
          <p className="text-[var(--color-text-secondary)] mb-8">
            We are working on it. Check back later or go back to continue ordering.
          </p>
          <Link
            to="/"
            className="inline-flex items-center justify-center rounded-lg bg-[var(--color-primary)] text-white font-semibold px-6 py-3 hover:bg-[var(--color-primary-hover)] transition-colors"
          >
            Back to home
          </Link>
        </motion.div>
      </main>
    </div>
  );
}
