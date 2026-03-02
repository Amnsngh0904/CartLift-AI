import { createContext, useCallback, useContext, useReducer } from 'react';
import type { ReactNode } from 'react';

export type CartItem = {
  item_id: string;
  item_name: string;
  price: number;
  category: string;
  quantity: number;
};

type CartState = {
  items: CartItem[];
  restaurantId: string | null;
  isOpen: boolean;
};

type CartAction =
  | { type: 'ADD'; payload: Omit<CartItem, 'quantity'> }
  | { type: 'REMOVE'; payload: string }
  | { type: 'UPDATE_QTY'; itemId: string; quantity: number }
  | { type: 'SET_RESTAURANT'; payload: string | null }
  | { type: 'CLEAR' }
  | { type: 'TOGGLE_OPEN' }
  | { type: 'SET_ITEMS'; payload: CartItem[] };

const initialState: CartState = { items: [], restaurantId: null, isOpen: false };

function cartReducer(state: CartState, action: CartAction): CartState {
  switch (action.type) {
    case 'ADD': {
      const existing = state.items.find((i) => i.item_id === action.payload.item_id);
      if (existing) {
        return {
          ...state,
          items: state.items.map((i) =>
            i.item_id === action.payload.item_id ? { ...i, quantity: i.quantity + 1 } : i
          ),
        };
      }
      return {
        ...state,
        items: [...state.items, { ...action.payload, quantity: 1 }],
        restaurantId: state.restaurantId,
      };
    }
    case 'REMOVE':
      return { ...state, items: state.items.filter((i) => i.item_id !== action.payload) };
    case 'UPDATE_QTY':
      if (action.quantity <= 0) {
        return { ...state, items: state.items.filter((i) => i.item_id !== action.itemId) };
      }
      return {
        ...state,
        items: state.items.map((i) =>
          i.item_id === action.itemId ? { ...i, quantity: action.quantity } : i
        ),
      };
    case 'SET_RESTAURANT':
      return { ...state, restaurantId: action.payload };
    case 'CLEAR':
      return { ...initialState, isOpen: state.isOpen };
    case 'TOGGLE_OPEN':
      return { ...state, isOpen: !state.isOpen };
    case 'SET_ITEMS':
      return { ...state, items: action.payload };
    default:
      return state;
  }
}

type CartContextValue = {
  items: CartItem[];
  restaurantId: string | null;
  isOpen: boolean;
  addItem: (item: Omit<CartItem, 'quantity'>) => void;
  removeItem: (itemId: string) => void;
  updateQuantity: (itemId: string, quantity: number) => void;
  setRestaurant: (id: string | null) => void;
  clearCart: () => void;
  toggleCart: () => void;
  totalItems: number;
  totalAmount: number;
};

const CartContext = createContext<CartContextValue | null>(null);

export function CartProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(cartReducer, initialState);

  const addItem = useCallback((item: Omit<CartItem, 'quantity'>) => {
    dispatch({ type: 'ADD', payload: item });
  }, []);

  const removeItem = useCallback((itemId: string) => {
    dispatch({ type: 'REMOVE', payload: itemId });
  }, []);

  const updateQuantity = useCallback((itemId: string, quantity: number) => {
    dispatch({ type: 'UPDATE_QTY', itemId, quantity });
  }, []);

  const setRestaurant = useCallback((id: string | null) => {
    dispatch({ type: 'SET_RESTAURANT', payload: id });
  }, []);

  const clearCart = useCallback(() => {
    dispatch({ type: 'CLEAR' });
  }, []);

  const toggleCart = useCallback(() => {
    dispatch({ type: 'TOGGLE_OPEN' });
  }, []);

  const totalItems = state.items.reduce((s, i) => s + i.quantity, 0);
  const totalAmount = state.items.reduce((s, i) => s + i.price * i.quantity, 0);

  const value: CartContextValue = {
    ...state,
    addItem,
    removeItem,
    updateQuantity,
    setRestaurant,
    clearCart,
    toggleCart,
    totalItems,
    totalAmount,
  };

  return <CartContext.Provider value={value}>{children}</CartContext.Provider>;
}

export function useCart() {
  const ctx = useContext(CartContext);
  if (!ctx) throw new Error('useCart must be used within CartProvider');
  return ctx;
}
