"""
CSAO Streamlit Frontend Demo

Interactive demo for Cart Suggested Add-On recommendations.

Usage:
    streamlit run frontend/app.py
    
Author: ZOMATHON Team
Date: February 2026
"""

import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

API_BASE_URL = "http://localhost:8000"
DATA_DIR = Path("data/processed")

# Page config
st.set_page_config(
    page_title="CARTLIFT AI - Cart Add-on Recommendations",
    page_icon="🍕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .recommendation-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        color: white !important;
    }
    .score-badge {
        background: #4CAF50;
        padding: 5px 10px;
        border-radius: 15px;
        font-size: 0.9em;
        color: white !important;
    }
    .cart-item-card {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%);
        padding: 15px;
        border-radius: 12px;
        margin: 8px 0;
        color: white !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
    }
    .cart-item-card strong {
        color: #fff !important;
        font-size: 1.1em;
    }
    .cart-item-card small {
        color: #b8d4e8 !important;
        font-size: 0.85em;
    }
    .cart-item-card b {
        color: #4ade80 !important;
        font-size: 1.2em;
    }
    .rec-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 12px;
        margin: 10px 0;
        color: white !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        min-height: 140px;
    }
    .rec-card h4 {
        margin: 0;
        color: white !important;
        font-size: 1.1em;
    }
    .rec-card p {
        margin: 5px 0;
        font-size: 0.9em;
        color: rgba(255,255,255,0.9) !important;
    }
    .rec-card .price {
        font-size: 1.4em;
        font-weight: bold;
        color: white !important;
    }
    .rec-card .score {
        padding: 5px 12px;
        border-radius: 15px;
        font-size: 0.85em;
        color: white !important;
    }
    .metric-card {
        background: #2d3748;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.3);
        text-align: center;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------

@st.cache_data
def load_menu_items():
    """Load menu items for display."""
    try:
        df = pd.read_csv(DATA_DIR / "menu_items_enriched.csv")
        return df
    except Exception as e:
        st.warning(f"Could not load menu items: {e}")
        return pd.DataFrame()

@st.cache_data
def load_restaurants():
    """Load restaurant data."""
    try:
        df = pd.read_csv(DATA_DIR / "restaurants_cleaned.csv")
        return df
    except Exception as e:
        st.warning(f"Could not load restaurants: {e}")
        return pd.DataFrame()

@st.cache_data
def get_restaurant_menu(restaurant_id: str, menu_df: pd.DataFrame):
    """Get menu items for a specific restaurant."""
    return menu_df[menu_df['restaurant_id'] == restaurant_id]

# -----------------------------------------------------------------------------
# API Interaction
# -----------------------------------------------------------------------------

def check_api_health():
    """Check if the inference API is running."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def get_recommendations(user_id: str, restaurant_id: str, cart_item_ids: list,
                        hour: int, meal_type: str, user_type: str, top_k: int = 5):
    """Call the recommendation API."""
    payload = {
        "user_id": user_id,
        "restaurant_id": restaurant_id,
        "cart_item_ids": cart_item_ids,
        "hour": hour,
        "meal_type": meal_type,
        "user_type": user_type
    }
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{API_BASE_URL}/recommend",
            json=payload,
            timeout=5
        )
        total_time = (time.time() - start_time) * 1000  # ms
        
        if response.status_code == 200:
            data = response.json()
            data['client_latency_ms'] = total_time
            return data
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to API. Make sure the inference service is running.")
        st.code("uvicorn src.inference.inference_service:app --port 8000", language="bash")
        return None
    except Exception as e:
        st.error(f"Error: {e}")
        return None

# -----------------------------------------------------------------------------
# UI Components
# -----------------------------------------------------------------------------

def render_sidebar():
    """Render the sidebar with input controls."""
    st.sidebar.header("🛒 Cart Builder")
    
    # Load data
    menu_df = load_menu_items()
    restaurants_df = load_restaurants()
    
    # Restaurant selection
    if not restaurants_df.empty:
        restaurant_ids = restaurants_df['restaurant_id'].tolist()
        restaurant_names = restaurants_df.apply(
            lambda r: f"{r['restaurant_id']} - {r.get('name', 'Unknown')[:30]}", axis=1
        ).tolist()
        selected_idx = st.sidebar.selectbox(
            "🏪 Select Restaurant",
            range(len(restaurant_ids)),
            format_func=lambda x: restaurant_names[x]
        )
        selected_restaurant_id = restaurant_ids[selected_idx]
    else:
        selected_restaurant_id = st.sidebar.text_input("Restaurant ID", "REST_0")
    
    # Get menu for selected restaurant
    restaurant_menu = get_restaurant_menu(selected_restaurant_id, menu_df)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("📋 Add Items to Cart")
    
    # Cart items selection
    cart_items = []
    if not restaurant_menu.empty:
        # Group by category for easier selection
        categories = restaurant_menu['item_category'].unique()
        
        for category in categories:
            cat_items = restaurant_menu[restaurant_menu['item_category'] == category]
            if len(cat_items) > 0:
                item_options = cat_items.apply(
                    lambda r: f"{r['item_name'][:30]} - ₹{r['price']:.0f}", axis=1
                ).tolist()
                item_ids = cat_items['item_id'].tolist()
                
                selected = st.sidebar.multiselect(
                    f"{category.title()}",
                    range(len(item_options)),
                    format_func=lambda x, opts=item_options: opts[x],
                    key=f"cat_{category}"
                )
                
                for idx in selected:
                    cart_items.append(item_ids[idx])
    else:
        # Manual input fallback
        cart_input = st.sidebar.text_area(
            "Enter item IDs (one per line)",
            "ITEM_0_1\nITEM_0_5",
            height=100
        )
        cart_items = [x.strip() for x in cart_input.split('\n') if x.strip()]
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Context Settings")
    
    # Time of day
    hour = st.sidebar.slider("🕐 Hour of Day", 0, 23, datetime.now().hour)
    
    # Meal type
    meal_types = ["breakfast", "lunch", "snack", "dinner", "late_night"]
    # Auto-select based on hour
    if hour < 10:
        default_meal = 0  # breakfast
    elif hour < 14:
        default_meal = 1  # lunch
    elif hour < 17:
        default_meal = 2  # snack
    elif hour < 22:
        default_meal = 3  # dinner
    else:
        default_meal = 4  # late_night
    
    meal_type = st.sidebar.selectbox(
        "🍽️ Meal Type",
        meal_types,
        index=default_meal
    )
    
    # User type
    user_types = ["budget", "moderate", "premium", "luxury"]
    user_type = st.sidebar.selectbox(
        "💰 User Type",
        user_types,
        index=1
    )
    
    # User ID
    user_id = st.sidebar.text_input("👤 User ID", "USER_0")
    
    # Top K
    top_k = st.sidebar.slider("📊 Number of Recommendations", 3, 10, 5)
    
    return {
        "restaurant_id": selected_restaurant_id,
        "cart_items": cart_items,
        "hour": hour,
        "meal_type": meal_type,
        "user_type": user_type,
        "user_id": user_id,
        "top_k": top_k,
        "menu_df": menu_df
    }

def render_cart(cart_items: list, menu_df: pd.DataFrame):
    """Render the current cart contents."""
    st.subheader("🛒 Current Cart")
    
    if not cart_items:
        st.info("Your cart is empty. Add items from the sidebar.")
        return 0.0
    
    total = 0.0
    cols = st.columns(min(len(cart_items), 4))
    
    for i, item_id in enumerate(cart_items):
        with cols[i % 4]:
            item_info = menu_df[menu_df['item_id'] == item_id]
            if not item_info.empty:
                item = item_info.iloc[0]
                st.markdown(f"""
                <div class="cart-item-card">
                    <strong>{item['item_name'][:25]}</strong><br>
                    <small>{item['item_category'].title()}</small><br>
                    <b>₹{item['price']:.0f}</b>
                </div>
                """, unsafe_allow_html=True)
                total += item['price']
            else:
                st.markdown(f"""
                <div class="cart-item-card">
                    <strong>{item_id}</strong><br>
                    <small>Unknown item</small>
                </div>
                """, unsafe_allow_html=True)
    
    return total

def render_recommendations(response: dict, menu_df: pd.DataFrame):
    """Render recommendation results."""
    st.subheader("✨ Suggested Add-ons")
    
    if not response or 'recommendations' not in response:
        st.warning("No recommendations available.")
        return
    
    recommendations = response.get('recommendations', [])
    
    if not recommendations:
        st.info("No recommendations returned for this cart configuration.")
        return
    
    # Metrics row
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("⚡ API Latency", f"{response.get('latency_ms', 0):.1f} ms")
    with col2:
        st.metric("🌐 Total Latency", f"{response.get('client_latency_ms', 0):.1f} ms")
    with col3:
        st.metric("📦 Recommendations", len(recommendations))
    
    st.markdown("---")
    
    # Recommendations grid
    cols = st.columns(min(len(recommendations), 3))
    
    for i, rec in enumerate(recommendations):
        with cols[i % 3]:
            score_color = "#22c55e" if rec['score'] > 0.5 else "#f59e0b" if rec['score'] > 0.3 else "#ef4444"
            
            st.markdown(f"""
            <div class="rec-card">
                <h4>#{i+1} {rec['item_name'][:30]}</h4>
                <p>{rec['category'].title()}</p>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 15px;">
                    <span class="price">₹{rec['price']:.0f}</span>
                    <span class="score" style="background: {score_color};">
                        Score: {rec['score']:.3f}
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    return recommendations

def render_revenue_projection(cart_total: float, recommendations: list, params: dict):
    """Render revenue projection section."""
    st.subheader("💰 Revenue Impact Simulation")
    
    if not recommendations:
        st.info("Get recommendations first to see revenue projections.")
        return
    
    # User parameters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        daily_sessions = st.number_input("Daily Sessions", 100000, 5000000, 960000, step=100000)
    with col2:
        accept_rate_model = st.slider("Model Accept Rate (%)", 0, 100, 91)
    with col3:
        accept_rate_baseline = st.slider("Baseline Accept Rate (%)", 0, 100, 75)
    
    # Calculate projections
    avg_addon_price = sum(r['price'] for r in recommendations) / len(recommendations) if recommendations else 150
    
    model_revenue = daily_sessions * (accept_rate_model / 100) * avg_addon_price
    baseline_revenue = daily_sessions * (accept_rate_baseline / 100) * avg_addon_price
    uplift = model_revenue - baseline_revenue
    
    # Display
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "📈 Model Daily Revenue",
            f"₹{model_revenue/1e6:.1f}M",
            f"+{(accept_rate_model-accept_rate_baseline):.0f}% rate"
        )
    with col2:
        st.metric(
            "📊 Baseline Daily Revenue",
            f"₹{baseline_revenue/1e6:.1f}M",
            ""
        )
    with col3:
        st.metric(
            "🚀 Daily Uplift",
            f"₹{uplift/1e6:.1f}M",
            f"+{(uplift/baseline_revenue*100):.1f}%" if baseline_revenue > 0 else ""
        )
    with col4:
        annual_uplift = uplift * 365
        st.metric(
            "📅 Annual Uplift",
            f"₹{annual_uplift/1e9:.2f}B",
            ""
        )

def render_model_info():
    """Render model information section."""
    with st.expander("📊 Model Performance Metrics"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **Classification Metrics**
            - AUC: **0.7129**
            - PR-AUC: **0.3921**
            - Positive Rate: 16.7%
            """)
        
        with col2:
            st.markdown("""
            **Ranking Metrics (K=5)**
            - NDCG@5: **0.6445** (+30.7%)
            - MRR: **0.5718** (+39.8%)
            - Precision@5: 0.1818
            """)
        
        with col3:
            st.markdown("""
            **Business Impact (K=5)**
            - Attach Rate Lift: **+21.9%**
            - AOV Lift: **+19.7%**
            - Revenue Lift: ₹31/session
            """)

def render_debug_info(response: dict, params: dict):
    """Render debug information."""
    with st.expander("🔧 Debug Information"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Request Parameters:**")
            st.json({
                "user_id": params.get('user_id'),
                "restaurant_id": params.get('restaurant_id'),
                "cart_item_ids": params.get('cart_items'),
                "hour": params.get('hour'),
                "meal_type": params.get('meal_type'),
                "user_type": params.get('user_type')
            })
        
        with col2:
            st.markdown("**Response Stats:**")
            if response:
                st.json({
                    "num_recommendations": len(response.get('recommendations', [])),
                    "api_latency_ms": response.get('latency_ms'),
                    "client_latency_ms": response.get('client_latency_ms')
                })
            else:
                st.info("No response yet")

# -----------------------------------------------------------------------------
# Main App
# -----------------------------------------------------------------------------

def main():
    """Main application."""
    # Header
    st.title("🍕 CSAO - Cart Suggested Add-On")
    st.markdown("*Real-time personalized recommendations for Zomato cart add-ons*")
    
    # API health check
    api_status = check_api_health()
    if api_status:
        st.success("✅ API Connected")
    else:
        st.error("❌ API Not Connected")
        st.info("""
        **To start the API:**
        ```bash
        cd /Users/amansingh/Documents/ML/ZOMATHON
        uvicorn src.inference.inference_service:app --host 0.0.0.0 --port 8000
        ```
        """)
    
    st.markdown("---")
    
    # Sidebar inputs
    params = render_sidebar()
    
    # Main content
    col1, col2 = st.columns([1, 2])
    
    with col1:
        cart_total = render_cart(params['cart_items'], params['menu_df'])
        if cart_total > 0:
            st.metric("Cart Total", f"₹{cart_total:.0f}")
    
    with col2:
        # Get recommendations button
        if st.button("🔮 Get Recommendations", type="primary", use_container_width=True):
            if not params['cart_items']:
                st.warning("Please add at least one item to your cart.")
            elif api_status:
                with st.spinner("Getting recommendations..."):
                    response = get_recommendations(
                        user_id=params['user_id'],
                        restaurant_id=params['restaurant_id'],
                        cart_item_ids=params['cart_items'],
                        hour=params['hour'],
                        meal_type=params['meal_type'],
                        user_type=params['user_type'],
                        top_k=params['top_k']
                    )
                    
                    if response:
                        st.session_state['last_response'] = response
                        st.session_state['last_params'] = params
            else:
                st.error("Please start the API first.")
    
    # Display results if available
    if 'last_response' in st.session_state:
        st.markdown("---")
        recommendations = render_recommendations(
            st.session_state['last_response'],
            params['menu_df']
        )
        
        st.markdown("---")
        render_revenue_projection(
            cart_total,
            st.session_state['last_response'].get('recommendations', []),
            params
        )
    
    # Bottom sections
    st.markdown("---")
    render_model_info()
    
    if 'last_response' in st.session_state:
        render_debug_info(st.session_state['last_response'], params)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9em;">
        <p>Built by <strong>CARTLIFT AI</strong> | 
        <a href="docs/FINAL_SUBMISSION.md">Documentation</a> | 
        <a href="docs/ARCHITECTURE.md">Architecture</a></p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
