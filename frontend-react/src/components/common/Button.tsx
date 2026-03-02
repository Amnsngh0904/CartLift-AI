import { motion } from 'framer-motion';
import type { ButtonHTMLAttributes, ReactNode } from 'react';

type Variant = 'primary' | 'secondary' | 'ghost' | 'outline';

type ButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: Variant;
  children: ReactNode;
  loading?: boolean;
  fullWidth?: boolean;
};

const variants: Record<Variant, string> = {
  primary: 'bg-[var(--color-primary)] text-white hover:bg-[var(--color-primary-hover)] shadow-md',
  secondary: 'bg-[var(--color-text-secondary)] text-white hover:opacity-90',
  ghost: 'bg-transparent text-[var(--color-text)] hover:bg-black/5',
  outline: 'border-2 border-[var(--color-primary)] text-[var(--color-primary)] bg-transparent hover:bg-[var(--color-primary-light)]',
};

export function Button({
  variant = 'primary',
  className = '',
  children,
  loading,
  fullWidth,
  disabled,
  ...props
}: ButtonProps) {
  return (
    <motion.div
      whileHover={{ scale: disabled || loading ? 1 : 1.02 }}
      whileTap={{ scale: disabled || loading ? 1 : 0.98 }}
      className={fullWidth ? 'w-full' : 'inline-block'}
    >
      <button
        type="button"
        className={`
          inline-flex items-center justify-center gap-2 rounded-[var(--radius-button)] px-5 py-2.5
          font-semibold text-sm transition-colors duration-200
          disabled:opacity-50 disabled:cursor-not-allowed w-full
          ${variants[variant]}
          ${className}
        `}
        disabled={disabled || loading}
        {...props}
      >
        {loading ? (
          <span className="h-4 w-4 animate-spin rounded-full border-2 border-current border-t-transparent" />
        ) : (
          children
        )}
      </button>
    </motion.div>
  );
}
