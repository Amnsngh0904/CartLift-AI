# CSAO Project Explained

## Cart Suggested Add-On Recommendation System

**A Complete Guide for Everyone**

---

## Table of Contents

1. [What Is This Project?](#1-what-is-this-project)
2. [The Problem We're Solving](#2-the-problem-were-solving)
3. [Our Solution (Simple Explanation)](#3-our-solution-simple-explanation)
4. [How It Works (Technical Deep Dive)](#4-how-it-works-technical-deep-dive)
5. [Project Structure](#5-project-structure)
6. [Step-by-Step: How to Run](#6-step-by-step-how-to-run)
7. [Results & Impact](#7-results--impact)


---

## 1. What Is This Project?

### The One-Liner
> **CSAO predicts which items a customer is most likely to add to their food delivery cart, making suggestions that feel personal rather than generic.**

### For Non-Technical Readers

Imagine you're ordering food on Zomato. You've added Butter Chicken to your cart. Before checkout, Zomato shows you "You might also like..." suggestions.

**Current approach (what most apps do):**
- Show the most popular items from the restaurant
- Same suggestions for everyone

**Our approach (CSAO):**
- Notice you have a main course but no drink → suggest beverages
- Know it's dinner time → suggest desserts
- Remember you often order paneer dishes → suggest similar items
- Learn from millions of orders what items go well together

### For Technical Readers

CSAO is a neural network-based recommendation system that:
- Uses **graph embeddings** (Node2Vec) trained on item co-occurrence patterns
- Combines **user behavioral features** (RFM analysis) with **contextual signals**
- Detects **cart composition gaps** (missing beverage, missing dessert)
- Serves predictions in **<50ms** via a FastAPI endpoint

---

## 2. The Problem We're Solving

### Business Context

Food delivery platforms make significant revenue from add-on items (drinks, desserts, sides). A small increase in "attach rate" (percentage of users who add suggested items) translates to millions in revenue.

### Current Limitations

| Current Approach | Problem |
|------------------|---------|
| Show bestsellers | Same for everyone, ignores individual taste |
| Random suggestions | Often irrelevant, low click-through |
| Rule-based | "If Indian food, suggest lassi" - too simplistic |
| No context awareness | Doesn't consider what's already in cart |

### What Success Looks Like

| Metric | Before (Baseline) | After (CSAO) |
|--------|-------------------|--------------|
| Attach Rate | 75% | 91% (+21.9%) |
| Revenue per Session | ₹160 | ₹191 (+19.7%) |
| User finds relevant item | Position 3.1 | Position 2.3 |

---

## 3. Our Solution (Simple Explanation)

### The Core Idea

Think of it like a smart waiter who:
1. **Sees your table** → knows what you've already ordered
2. **Knows your history** → remembers you love desserts
3. **Checks the clock** → it's dinner, not breakfast
4. **Learned from experience** → knows chai goes with samosa

### How We Built It

```
Step 1: Understand Item Relationships
        - Analyzed 4.6 million past orders
        - Learned "Butter Chicken often ordered with Naan"
        - Created a 'map' of related items (embeddings)

Step 2: Understand Users
        - How often do they order? (Frequency)
        - How much do they spend? (Monetary)
        - Do they prefer desserts? (Dessert ratio)

Step 3: Understand Context
        - What time is it? (Breakfast vs Dinner)
        - What's already in the cart? (Missing a drink?)
        - What type of restaurant? (Fine dining vs QSR)

Step 4: Train A Model
        - Feed all this information to a neural network
        - Show it millions of "did they add this item?" examples
        - It learns to predict: P(user will add item X)

Step 5: Serve Predictions
        - When user is at checkout
        - Score all possible add-ons
        - Show top 5 highest-scoring items
```

---

## 4. How It Works (Technical Deep Dive)

### 4.1 Data Pipeline

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Raw Dataset   │────►│  Data Cleaning  │────►│  Session Sim    │
│  (Zomato CSV)   │     │  & Enrichment   │     │  (500K orders)  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  70K items      │     │  Item categories│     │  4.6M cart      │
│  890 restaurants│     │  Price bands    │     │  events         │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

**Scripts involved:**
- `src/data_generation/clean_dataset.py` - Cleans raw Zomato data
- `src/data_generation/item_categorizer.py` - Classifies items (main/dessert/beverage)
- `src/data_generation/session_generator.py` - Simulates realistic user sessions

### 4.2 Feature Engineering

#### Item Embeddings (Graph-based)

```python
# We build a co-occurrence graph
# Edge: (Item A, Item B) = "ordered together N times"

# Then run Node2Vec: a random walk algorithm
# Items that appear in similar contexts get similar embeddings

# Result: Each item → 64-dimensional vector
# Similar items have similar vectors (cosine similarity)
```

**Why this works:** If Gulab Jamun and Rasgulla are both often ordered after biryani, they'll have similar embeddings even if they're never ordered together directly.

#### User Features (RFM Analysis)

| Feature | What It Measures | Example |
|---------|------------------|---------|
| recency_days | Days since last order | 3 days |
| frequency | Total order count | 47 orders |
| monetary_avg | Average order value | ₹450 |
| cuisine_entropy | Variety in cuisines ordered | 0.8 (high variety) |
| dessert_ratio | % orders with dessert | 0.35 (35%) |
| beverage_ratio | % orders with drink | 0.72 (72%) |

#### Cart Dynamic Features

| Feature | Logic | Purpose |
|---------|-------|---------|
| cart_size | Number of items | Saturation signal |
| cart_total | Total price | Budget awareness |
| missing_beverage | No drink in cart? | Upsell opportunity |
| missing_dessert | No sweet in cart? | Upsell opportunity |
| heavy_meal | 2+ mains? | Less likely to add more |

### 4.3 Model Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                     INPUT FEATURES                              │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Cart Embedding (64d)    ─┐                                    │
│  Candidate Embedding (64d)─┼──► Interaction: dot + diff (65d)  │
│                            │                                    │
│  User Features (7d)       ─┤                                    │
│  Restaurant Features (5d)  ├──► Concatenate ──► MLP ──► Score  │
│  Cart Dynamic (6d)        ─┤         │                          │
│  Context Features (7d)    ─┘         │                          │
│                                      ▼                          │
│                              [218 features]                     │
│                                      │                          │
│                                      ▼                          │
│                         ┌────────────────────┐                  │
│                         │   MLP Network      │                  │
│                         │   218 → 256 → 128  │                  │
│                         │   → 1 (sigmoid)    │                  │
│                         └────────────────────┘                  │
│                                      │                          │
│                                      ▼                          │
│                            P(add to cart)                       │
│                             [0.0 - 1.0]                         │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

**Why MLP over Transformer?**
We tested a Transformer model for intra-cart attention (letting items "communicate" with each other). Results:

| Model | Val AUC | Training Time |
|-------|---------|---------------|
| Transformer | 0.7089 | 85 minutes |
| **MLP (chosen)** | **0.7129** | **10 minutes** |

The simpler model won because:
1. Carts are small (3-5 items) → less need for attention
2. Mean pooling captures "cart style" sufficiently
3. 8x faster training enables more experimentation

### 4.4 Training Process

```python
# Configuration
batch_size = 2048
learning_rate = 0.001
epochs = 12
early_stopping_patience = 2

# Class imbalance handling
# Only 16.7% of shown items are actually added
pos_weight = 5.0  # Upweight positive examples

# Optimization
optimizer = AdamW(weight_decay=0.01)
loss = BCEWithLogitsLoss(pos_weight=5.0)
```

**Training metrics over time:**

| Epoch | Train Loss | Val AUC | Val PR-AUC |
|-------|------------|---------|------------|
| 1 | 1.0524 | 0.6987 | 0.3646 |
| 4 | 1.0258 | 0.7088 | 0.3837 |
| 8 | 1.0200 | 0.7115 | 0.3888 |
| 12 | 1.0169 | **0.7129** | **0.3921** |

### 4.5 Inference Pipeline

```
User Request                Processing                    Response
─────────────────────────────────────────────────────────────────────

POST /recommend    →    1. Load user features (Redis)
{                       2. Load restaurant features
  user_id,              3. Get cart item embeddings
  restaurant_id,        4. Mean pool → cart_embedding
  cart_item_ids,        5. Get all candidates from restaurant
  hour: 19              6. For each candidate:
}                          - Build feature vector (218d)
                           - Forward pass through MLP
                           - Get score P(add)
                        7. Sort by score descending
                        8. Return top K
                                                    →    {
                                                          recommendations: [
                                                            {item_id, name, 
                                                             price, score},
                                                            ...
                                                          ],
                                                          latency_ms: 32
                                                         }
```

---

## 5. Project Structure

```
ZOMATHON/
│
├── 📁 src/                      # Main source code
│   ├── data_generation/         # Data processing scripts
│   │   ├── clean_dataset.py     # Clean raw Zomato CSV
│   │   ├── session_generator.py # Simulate user sessions
│   │   └── item_categorizer.py  # Classify items by type
│   │
│   ├── features/                # Feature engineering
│   │   └── feature_builder.py   # Build training features
│   │
│   ├── graph/                   # Graph-based embeddings
│   │   ├── build_cooccurrence.py # Build item co-occurrence graph
│   │   └── train_node2vec.py    # Train Node2Vec embeddings
│   │
│   ├── model/                   # Neural network model
│   │   ├── cart_transformer.py  # Model architecture
│   │   └── train_model.py       # Training script
│   │
│   ├── evaluation/              # Evaluation modules
│   │   ├── topk_evaluation.py   # Ranking metrics (NDCG, MRR)
│   │   └── business_impact.py   # Revenue simulation
│   │
│   └── inference/               # Production serving
│       ├── inference_service.py # FastAPI service
│       └── benchmark.py         # Latency benchmarks
│
├── 📁 data/                     # Data files
│   ├── raw/                     # Original Zomato dataset
│   ├── processed/               # Cleaned & enriched data
│   └── synthetic/               # Simulated sessions
│
├── 📁 checkpoints/              # Trained models
│   └── best_model_final.pt      # Production model (19.5 MB)
│
├── 📁 frontend/                 # Demo interface
│   └── app.py                   # Streamlit application
│
├── 📁 docs/                     # Documentation
│   ├── FINAL_SUBMISSION.md      # Hackathon submission
│   ├── ARCHITECTURE.md          # Technical architecture
│   ├── DEMO_SCRIPT.md           # Demo talking points
│   └── PROJECT_EXPLAINED.md     # This file
│
├── 📁 scripts/                  # Utility scripts
│   ├── analyze_data.py          # Data exploration
│   └── validate_embeddings.py   # Check embedding quality
│
├── config.yaml                  # Configuration settings
├── requirements.txt             # Python dependencies
└── README.md                    # Project overview
```

---

## 6. Step-by-Step: How to Run

### Prerequisites

- **Python 3.10+** installed
- **8GB+ RAM** (for loading embeddings)
- **macOS/Linux** (Windows may need WSL)

### Step 1: Clone and Setup Environment

```bash
# Navigate to project
cd /Users/amansingh/Documents/ML/ZOMATHON

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Key dependencies:**
- `torch` - Neural network framework
- `fastapi` + `uvicorn` - API server
- `streamlit` - Web interface
- `pandas`, `numpy` - Data processing
- `scikit-learn` - Feature scaling

### Step 2: Verify Data Files Exist

```bash
# Check processed data exists
ls data/processed/
# Should see: menu_items_enriched.csv, item_embeddings_fixed.npy, etc.

# Check model exists
ls checkpoints/
# Should see: best_model_final.pt
```

If files are missing, you need to run the data pipeline first (see Advanced section below).

### Step 3: Start the API Server

```bash
# Start the inference API
uvicorn src.inference.inference_service:app --host 0.0.0.0 --port 8000
```

**Expected output:**
```
INFO     | Initializing inference model on device: mps
INFO     | Loading embeddings... Loaded 70221 embeddings
INFO     | Loading model... 193,985 parameters
INFO     | Model loading complete in 1.9s
INFO     | Uvicorn running on http://0.0.0.0:8000
```

### Step 4: Test the API

**Option A: Using curl (command line)**
```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "USER_0",
    "restaurant_id": "REST_0",
    "cart_item_ids": ["ITEM_0_1", "ITEM_0_5"],
    "hour": 19,
    "meal_type": "dinner",
    "user_type": "moderate"
  }'
```

**Option B: Using browser**
Open http://localhost:8000/docs for interactive API documentation.

**Option C: Using Python**
```python
import requests

response = requests.post(
    "http://localhost:8000/recommend",
    json={
        "user_id": "USER_0",
        "restaurant_id": "REST_0",
        "cart_item_ids": ["ITEM_0_1"],
        "hour": 19,
        "meal_type": "dinner",
        "user_type": "moderate"
    }
)
print(response.json())
```

### Step 5: Launch the Demo UI

Open a **new terminal** (keep API running in the first):

```bash
cd /Users/amansingh/Documents/ML/ZOMATHON
source .venv/bin/activate

# Start Streamlit
streamlit run frontend/app.py
```

**Expected output:**
```
You can now view your Streamlit app in your browser.
Local URL: http://localhost:8501
```

Open http://localhost:8501 in your browser.

### Step 6: Using the Demo

1. **Select a Restaurant** from the dropdown
2. **Add items to cart** by selecting from category dropdowns (Main, Side, Beverage, etc.)
3. **Adjust context** settings:
   - Hour of day (affects meal type recommendations)
   - User type (affects price sensitivity)
4. Click **"Get Recommendations"**
5. See personalized suggestions with scores
6. Try adding a beverage, then click again — notice desserts rise in ranking

---

## 7. Results & Impact

### Model Performance

| Metric | Value | What It Means |
|--------|-------|---------------|
| **Val AUC** | 0.7129 | Model correctly ranks items 71% of time |
| **NDCG@5** | 0.6445 | Relevant items appear high in top 5 |
| **MRR** | 0.5718 | First relevant item at position ~1.75 |
| **Latency** | 32 ms | Fast enough for real-time |

### Business Impact (Simulated)

| Metric | Popularity Baseline | CSAO Model | Improvement |
|--------|---------------------|------------|-------------|
| Attach Rate | 74.6% | 90.9% | **+21.9%** |
| Revenue/Session | ₹160 | ₹191 | **+19.7%** |
| First Hit Position | 3.1 | 2.3 | **0.8 better** |

### Projected Revenue at Zomato Scale

| Assumption | Value |
|------------|-------|
| Daily orders | 2 million |
| Sessions with add-ons | 960,000 |
| Revenue uplift per session | ₹31 |
| **Daily uplift** | **₹29.8 million** |
| **Annual opportunity** | **₹10.9 billion** |

---

### Running Issues

**Q: "API not connected" error in Streamlit**
> Make sure the API is running in another terminal:
> ```bash
> uvicorn src.inference.inference_service:app --port 8000
> ```

**Q: "File not found" errors**
> Check you're in the correct directory with activated environment:
> ```bash
> cd /Users/amansingh/Documents/ML/ZOMATHON
> source .venv/bin/activate
> ```

**Q: Slow first request**
> The first inference request takes longer (~500ms) due to model warm-up. Subsequent requests are <50ms.

**Q: Out of memory**
> The embeddings require ~500MB RAM. Close other applications or use a machine with 8GB+ RAM.

---

## Quick Reference

### One-Command Demo

```bash
# Terminal 1: Start API
uvicorn src.inference.inference_service:app --port 8000

# Terminal 2: Start UI
streamlit run frontend/app.py

# Open browser: http://localhost:8501
```

### API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/recommend` | POST | Get recommendations |
| `/health` | GET | Health check |
| `/docs` | GET | Interactive API docs |
| `/restaurants` | GET | List all restaurants |

### Key Files

| File | Purpose |
|------|---------|
| `checkpoints/best_model_final.pt` | Trained model weights |
| `data/processed/item_embeddings_fixed.npy` | Item vectors |
| `src/inference/inference_service.py` | API server |
| `frontend/app.py` | Demo interface |

---


---

*"The best recommendation is one that feels obvious in hindsight."*
