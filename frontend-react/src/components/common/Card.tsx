import { motion } from 'framer-motion';
import type { ReactNode } from 'react';

type CardProps = {
  children: ReactNode;
  className?: string;
  hover?: boolean;
  padding?: 'none' | 'sm' | 'md' | 'lg';
};

export function Card({ children, className = '', hover = true, padding = 'md' }: CardProps) {
  const paddingClass = {
    none: '',
    sm: 'p-3',
    md: 'p-4',
    lg: 'p-6',
  }[padding];

  const Comp = hover ? motion.div : 'div';
  const motionProps = hover
    ? {
        initial: { opacity: 0, y: 8 },
        animate: { opacity: 1, y: 0 },
        whileHover: { y: -4, boxShadow: 'var(--shadow-card-hover)' },
        transition: { duration: 0.2 },
      }
    : {};

  return (
    <Comp
      className={`
        rounded-[var(--radius-card)] bg-[var(--color-surface)]
        shadow-[var(--shadow-card)] border border-[var(--color-border)]
        transition-shadow duration-200
        ${paddingClass}
        ${className}
      `}
      {...motionProps}
    >
      {children}
    </Comp>
  );
}
