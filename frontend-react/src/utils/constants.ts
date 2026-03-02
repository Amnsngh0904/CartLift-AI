export const API_BASE = import.meta.env.VITE_API_URL ?? '/api';

export const MEAL_TYPES = ['breakfast', 'lunch', 'snack', 'dinner', 'late_night'] as const;
export const USER_TYPES = ['budget', 'moderate', 'premium', 'luxury'] as const;

export const CATEGORIES = [
  { id: 'main', label: 'Main Course', slug: 'main' },
  { id: 'starter', label: 'Starters', slug: 'starter' },
  { id: 'dessert', label: 'Desserts', slug: 'dessert' },
  { id: 'beverage', label: 'Beverages', slug: 'beverage' },
  { id: 'side', label: 'Sides', slug: 'side' },
  { id: 'snack', label: 'Snacks', slug: 'snack' },
] as const;

/** Base path for local food images (public/images) */
const LOCAL_IMAGES_BASE = '/images';

/**
 * Map dish name (lowercase) to local image path in public/images.
 * Keys are matched against item names (itemName.toLowerCase().includes(key)); first match wins.
 * Use exact filenames as in the folder (case-sensitive on some servers).
 */
const DISH_IMAGE_MAP: Record<string, string> = {
  // Mains
  'butter chicken': `${LOCAL_IMAGES_BASE}/butter-chicken.png`,
  'chicken biryani': `${LOCAL_IMAGES_BASE}/chicken-biryani.png`,
  'paneer butter masala': `${LOCAL_IMAGES_BASE}/paneer-butter-masala.png`,
  'dal makhani': `${LOCAL_IMAGES_BASE}/dal makhani.png`,
  'veg biryani': `${LOCAL_IMAGES_BASE}/veg biryani.png`,
  'fish curry': `${LOCAL_IMAGES_BASE}/Fish Curry.png`,
  'mutton rogan josh': `${LOCAL_IMAGES_BASE}/Mutton.png`,
  'mutton': `${LOCAL_IMAGES_BASE}/Mutton.png`,
  'chole bhature': `${LOCAL_IMAGES_BASE}/chole Bathure.png`,
  'dosa': `${LOCAL_IMAGES_BASE}/dosa.png`,
  'idli': `${LOCAL_IMAGES_BASE}/idli.png`,
  'upma': `${LOCAL_IMAGES_BASE}/upma.png`,
  'pav bhaji': `${LOCAL_IMAGES_BASE}/pav bhaji.png`,
  'samosa': `${LOCAL_IMAGES_BASE}/samosa.png`,
  'pani puri': `${LOCAL_IMAGES_BASE}/ani puri.png`,
  'fried rice': `${LOCAL_IMAGES_BASE}/rice.png`,
  'hakka noodles': `${LOCAL_IMAGES_BASE}/Hakka Noodles.png`,
  'manchurian': `${LOCAL_IMAGES_BASE}/Manchuriyan.png`,
  'kadai chicken': `${LOCAL_IMAGES_BASE}/Kadhai Chicken.png`,
  'kadhai chicken': `${LOCAL_IMAGES_BASE}/Kadhai Chicken.png`,
  'palak paneer': `${LOCAL_IMAGES_BASE}/Palak Paneer.png`,
  'rajma chawal': `${LOCAL_IMAGES_BASE}/Rajma Chawal.png`,
  'aloo paratha': `${LOCAL_IMAGES_BASE}/Allu Parathe.png`,
  'aloo parathe': `${LOCAL_IMAGES_BASE}/Allu Parathe.png`,
  // Sides
  'naan': `${LOCAL_IMAGES_BASE}/naan.png`,
  'roti': `${LOCAL_IMAGES_BASE}/roti.png`,
  'rice': `${LOCAL_IMAGES_BASE}/rice.png`,
  'jeera rice': `${LOCAL_IMAGES_BASE}/rice.png`,
  'raita': `${LOCAL_IMAGES_BASE}/Raita.png`,
  'salad': `${LOCAL_IMAGES_BASE}/Salad.png`,
  'papad': `${LOCAL_IMAGES_BASE}/Papad.png`,
  // Desserts
  'gulab jamun': `${LOCAL_IMAGES_BASE}/gulab Jamun.png`,
  'rasmalai': `${LOCAL_IMAGES_BASE}/ras malai.png`,
  'ras malai': `${LOCAL_IMAGES_BASE}/ras malai.png`,
  'kheer': `${LOCAL_IMAGES_BASE}/kheer.png`,
  'ice cream': `${LOCAL_IMAGES_BASE}/ice cream.png`,
  'brownie': `${LOCAL_IMAGES_BASE}/brownie.png`,
  'jalebi': `${LOCAL_IMAGES_BASE}/jalabi.png`,
  'jalabi': `${LOCAL_IMAGES_BASE}/jalabi.png`,
  'rasgulla': `${LOCAL_IMAGES_BASE}/rasgulla.png`,
  // Beverages
  'coke': `${LOCAL_IMAGES_BASE}/coke.png`,
  'lassi': `${LOCAL_IMAGES_BASE}/lassi.png`,
  'masala chai': `${LOCAL_IMAGES_BASE}/masala chai.png`,
  'cold coffee': `${LOCAL_IMAGES_BASE}/cold coffee.png`,
  'fresh lime': `${LOCAL_IMAGES_BASE}/fresh lime.png`,
  'water': `${LOCAL_IMAGES_BASE}/Water.png`,
};

/** Default image when no dish-specific image is found */
const FALLBACK_IMAGE = `${LOCAL_IMAGES_BASE}/base.png`;

/**
 * Food items that still use the base image (no matching image in public/images).
 * Update DISH_IMAGE_MAP when you add images for these.
 */
export const ITEMS_USING_BASE_IMAGE: string[] = [];

export function getFoodImageUrl(itemName: string, _category: string): string {
  const key = itemName.toLowerCase().trim();
  for (const [dish, path] of Object.entries(DISH_IMAGE_MAP)) {
    if (key.includes(dish)) return path;
  }
  return FALLBACK_IMAGE;
}

export function getItemImageUrl(itemName: string, _category: string, _width = 400, _height = 300): string {
  const key = itemName.toLowerCase().trim();
  for (const [dish, path] of Object.entries(DISH_IMAGE_MAP)) {
    if (key.includes(dish)) return path;
  }
  return FALLBACK_IMAGE;
}
