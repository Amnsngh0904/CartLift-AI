type SkeletonProps = {
  className?: string;
  variant?: 'text' | 'circular' | 'rectangular';
};

export function Skeleton({ className = '', variant = 'rectangular' }: SkeletonProps) {
  const base = 'animate-pulse bg-[var(--color-border)]';
  const variants = {
    text: 'h-4 rounded',
    circular: 'rounded-full aspect-square',
    rectangular: 'rounded-[var(--radius-card)]',
  };
  return <div className={`${base} ${variants[variant]} ${className}`} />;
}

export function FoodCardSkeleton() {
  return (
    <div className="rounded-[var(--radius-card)] overflow-hidden bg-[var(--color-surface)] border border-[var(--color-border)]">
      <Skeleton className="w-full aspect-[4/3]" variant="rectangular" />
      <div className="p-4 space-y-2">
        <Skeleton className="h-5 w-3/4" variant="text" />
        <Skeleton className="h-3 w-full" variant="text" />
        <div className="flex justify-between items-center pt-2">
          <Skeleton className="h-5 w-16" variant="text" />
          <Skeleton className="h-9 w-24 rounded-lg" variant="rectangular" />
        </div>
      </div>
    </div>
  );
}
