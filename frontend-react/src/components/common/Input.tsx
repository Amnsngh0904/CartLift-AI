import type { InputHTMLAttributes } from 'react';
import { forwardRef } from 'react';

export const Input = forwardRef<HTMLInputElement, InputHTMLAttributes<HTMLInputElement>>(
  ({ className = '', ...props }, ref) => (
    <input
      ref={ref}
      className={`
        w-full rounded-[var(--radius-button)] border border-[var(--color-border)]
        px-4 py-2.5 text-[var(--color-text)] placeholder:text-[var(--color-muted)]
        focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)] focus:border-transparent
        transition-colors duration-200
        ${className}
      `}
      {...props}
    />
  )
);
Input.displayName = 'Input';
