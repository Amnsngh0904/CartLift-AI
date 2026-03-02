import { motion, AnimatePresence } from 'framer-motion';
import { Link } from 'react-router-dom';
import { useCart } from '../../context/CartContext';
import { Button } from '../common/Button';
import { getItemImageUrl } from '../../utils/constants';

export function CartSidebar() {
  const { items, isOpen, toggleCart, removeItem, updateQuantity, totalAmount, totalItems, clearCart } = useCart();

  return (
    <>
      <AnimatePresence>
        {isOpen && (
          <>
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={toggleCart}
              className="fixed inset-0 z-40 bg-black/50 backdrop-blur-sm"
            />
            <motion.aside
              initial={{ x: '100%' }}
              animate={{ x: 0 }}
              exit={{ x: '100%' }}
              transition={{ type: 'spring', damping: 25, stiffness: 200 }}
              className="fixed top-0 right-0 z-50 w-full max-w-md h-full bg-[var(--color-surface)] shadow-2xl flex flex-col"
            >
              <div className="flex items-center justify-between p-4 border-b border-[var(--color-border)]">
                <h2 className="text-lg font-bold text-[var(--color-text)]">
                  Cart ({totalItems} {totalItems === 1 ? 'item' : 'items'})
                </h2>
                <button
                  type="button"
                  onClick={toggleCart}
                  className="p-2 rounded-lg hover:bg-black/5 text-[var(--color-text-secondary)]"
                  aria-label="Close cart"
                >
                  <CloseIcon className="w-5 h-5" />
                </button>
              </div>
              <div className="flex-1 overflow-y-auto p-4">
                {items.length === 0 ? (
                  <div className="flex flex-col items-center justify-center py-16 text-center">
                    <p className="text-[var(--color-text-secondary)]">Your cart is empty</p>
                    <p className="text-sm text-[var(--color-muted)] mt-1">Add items from the menu</p>
                    <Button variant="outline" className="mt-4" onClick={toggleCart}>
                      Continue browsing
                    </Button>
                  </div>
                ) : (
                  <ul className="space-y-4">
                    {items.map((item) => (
                      <motion.li
                        key={item.item_id}
                        layout
                        initial={{ opacity: 0, x: 10 }}
                        animate={{ opacity: 1, x: 0 }}
                        exit={{ opacity: 0, x: -10 }}
                        className="flex gap-3 p-3 rounded-xl bg-[var(--color-background)] border border-[var(--color-border)]"
                      >
                        <div className="shrink-0 w-20 h-20 rounded-lg overflow-hidden bg-[var(--color-border)]">
                          <img
                            src={getItemImageUrl(item.item_name, item.category, 80, 80)}
                            alt={item.item_name}
                            className="w-full h-full object-cover"
                          />
                        </div>
                        <div className="flex-1 min-w-0">
                          <h3 className="font-medium text-sm text-[var(--color-text)] line-clamp-1">
                            {item.item_name}
                          </h3>
                          <p className="text-[var(--color-primary)] font-semibold mt-0.5">
                            ₹{Math.round(item.price * item.quantity)}
                          </p>
                          <div className="flex items-center gap-2 mt-1">
                            <button
                              type="button"
                              onClick={() => updateQuantity(item.item_id, item.quantity - 1)}
                              className="w-7 h-7 rounded-md border border-[var(--color-border)] flex items-center justify-center text-[var(--color-text)] hover:bg-black/5"
                            >
                              -
                            </button>
                            <span className="text-sm font-medium w-6 text-center">{item.quantity}</span>
                            <button
                              type="button"
                              onClick={() => updateQuantity(item.item_id, item.quantity + 1)}
                              className="w-7 h-7 rounded-md border border-[var(--color-border)] flex items-center justify-center text-[var(--color-text)] hover:bg-black/5"
                            >
                              +
                            </button>
                            <button
                              type="button"
                              onClick={() => removeItem(item.item_id)}
                              className="ml-auto text-red-600 text-xs hover:underline"
                            >
                              Remove
                            </button>
                          </div>
                        </div>
                      </motion.li>
                    ))}
                  </ul>
                )}
              </div>
              {items.length > 0 && (
                <div className="p-4 border-t border-[var(--color-border)] space-y-3">
                  <div className="flex justify-between text-lg font-bold">
                    <span>Subtotal</span>
                    <span>₹{Math.round(totalAmount)}</span>
                  </div>
                  <Link to="/coming-soon" className="block w-full">
                    <Button fullWidth>Proceed to checkout</Button>
                  </Link>
                  <button
                    type="button"
                    onClick={clearCart}
                    className="w-full text-center text-sm text-[var(--color-muted)] hover:text-red-600"
                  >
                    Clear cart
                  </button>
                </div>
              )}
            </motion.aside>
          </>
        )}
      </AnimatePresence>
    </>
  );
}

function CloseIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
    </svg>
  );
}
