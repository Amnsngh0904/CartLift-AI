import { motion } from 'framer-motion';
import { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useCart } from '../../context/CartContext';
import { Button } from '../common/Button';
import { Input } from '../common/Input';

type NavbarProps = {
  onSearch?: (query: string) => void;
  location?: string;
};

export function Navbar({ onSearch, location = 'Bangalore' }: NavbarProps) {
  const [query, setQuery] = useState('');
  const navigate = useNavigate();
  const { totalItems, toggleCart } = useCart();

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    onSearch?.(query);
    navigate('/coming-soon');
  };

  return (
    <motion.header
      initial={{ y: -20, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.3 }}
      className="sticky top-0 z-50 bg-[var(--color-surface)]/95 backdrop-blur-md border-b border-[var(--color-border)]"
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16 md:h-18 gap-4">
          <Link to="/" className="flex items-center gap-2 shrink-0">
            <span className="text-xl font-bold text-[var(--color-primary)]">CARTLIFT AI</span>
            <span className="hidden sm:inline text-sm text-[var(--color-text-secondary)] font-medium">
            </span>
          </Link>

          <form
            onSubmit={handleSearch}
            className="flex-1 max-w-xl mx-4 opacity-90"
          >
            <div className="relative flex items-center bg-[var(--color-background)] rounded-lg border border-[var(--color-border)] focus-within:ring-2 focus-within:ring-[var(--color-primary)] focus-within:border-transparent">
              <span className="absolute left-3 text-[var(--color-muted)] pointer-events-none">
                <SearchIcon className="w-5 h-5" />
              </span>
              <Input
                type="search"
                placeholder="Search for dishes or restaurants"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                className="pl-10 pr-4 py-2.5 border-0 bg-transparent"
              />
            </div>
          </form>

          <div className="flex items-center gap-2 shrink-0">
            <Link
              to="/coming-soon"
              className="hidden md:flex items-center gap-1.5 px-3 py-2 rounded-lg hover:bg-black/5 text-sm text-[var(--color-text-secondary)]"
            >
              <LocationIcon className="w-4 h-4" />
              <span>{location}</span>
            </Link>
            <Link to="/coming-soon" className="hidden sm:block">
              <Button variant="ghost">Login</Button>
            </Link>
            <Link to="/coming-soon" className="hidden sm:block">
              <Button variant="outline">Sign up</Button>
            </Link>
            <motion.button
              type="button"
              whileTap={{ scale: 0.95 }}
              onClick={toggleCart}
              className="relative flex items-center justify-center w-10 h-10 rounded-full bg-[var(--color-primary-light)] text-[var(--color-primary)] hover:bg-[var(--color-primary)] hover:text-white transition-colors"
            >
              <CartIcon className="w-5 h-5" />
              {totalItems > 0 && (
                <span className="absolute -top-0.5 -right-0.5 min-w-[18px] h-[18px] rounded-full bg-[var(--color-primary)] text-white text-xs font-bold flex items-center justify-center">
                  {totalItems > 99 ? '99+' : totalItems}
                </span>
              )}
            </motion.button>
          </div>
        </div>
      </div>
    </motion.header>
  );
}

function SearchIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
    </svg>
  );
}

function LocationIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
    </svg>
  );
}

function CartIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 3h2l.4 2M7 13h10l4-8H5.4M7 13L5.4 5M7 13l-2.293 2.293c-.63.63-.184 1.707.707 1.707H17m0 0a2 2 0 100 4 2 2 0 000-4zm-8 2a2 2 0 11-4 0 2 2 0 014 0z" />
    </svg>
  );
}
