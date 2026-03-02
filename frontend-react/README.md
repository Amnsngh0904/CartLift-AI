# CSAO React Frontend

Modern, responsive food-ordering UI for the Cart Suggested Add-On recommendation system. Built with React, TypeScript, Tailwind CSS, and Framer Motion.

## Design

- **Design system**: Zomato-inspired palette (light background, red accent `#e23744`), DM Sans typography, card-based layout
- **Sections**: Sticky navbar, hero with search, category filters, restaurant selector, recommended-for-you strip, food grid, cart sidebar, footer
- **Responsive**: Mobile-first grid and layout; cart as slide-over sidebar
- **Animations**: Framer Motion for hover, tap, stagger, and page/section transitions; skeleton loaders for async data

## Run locally

1. **API**: Start the recommendation API (from repo root):
   ```bash
   uvicorn src.inference.inference_service:app --host 0.0.0.0 --port 8000
   ```

2. **Frontend** (with proxy to API):
   ```bash
   cd frontend-react
   npm install
   npm run dev
   ```
   Open http://localhost:3000. The app proxies `/api` to `http://localhost:8000`.

3. **Without API**: You can still run the app; restaurant list and menu will show an error or empty state until the API is up.

## Build

```bash
npm run build
npm run preview   # serve dist/
```

## Env

Optional: create `.env` with `VITE_API_URL=/api` (default) or another base URL for the recommendation API.

## Structure

```
src/
  components/   common (Button, Card, Input, Skeleton), layout (Navbar, Hero, Footer), sections (Categories, FoodGrid, RecommendedForYou, RestaurantSelector), cart (CartSidebar)
  context/      CartContext
  hooks/         useRestaurants, useRestaurantItems, useRecommendations, useApiHealth
  pages/         Home
  utils/         api, constants (incl. food image URLs)
```

## Tech

- **React 19** + **TypeScript**
- **Vite 7**
- **Tailwind CSS v4** (design tokens in `index.css`)
- **Framer Motion** (animations)
- **React Router** (single route; ready for more)
