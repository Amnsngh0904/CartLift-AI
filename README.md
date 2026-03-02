# CartLift-AI : CSAO - Cart Suggested Add-On Recommendation System

A production-grade ML recommendation system that suggests relevant add-on items during checkout based on cart context, user behavior, and item co-occurrence patterns.

## Overview

This system uses **Node2Vec embeddings**, **contextual features**, and an **MLP classifier** to predict which items a user is likely to add to their cart. Built for the ZOMATHON hackathon, it achieves **+21.9% attach rate lift** over popularity baselines.

### Key Results

| Metric | Value |
|--------|-------|
| Validation AUC | 0.7129 |
| NDCG@5 | 0.6445 (+30.7% vs random) |
| Attach Rate | 90.91% (+21.9% vs popularity) |
| Inference Latency | 0.15ms (CPU) |

### Key Features

- **Cart Context Awareness**: Detects missing beverages/desserts for smart upselling
- **Item Embeddings**: Node2Vec on co-occurrence graph captures item relationships
- **Real-time Inference**: FastAPI service with &lt;1ms latency
- **Interactive Demo**: Streamlit frontend for live recommendations

## Dataset

Download the required data files from Google Drive:

**[📁 Dataset Download Link](https://drive.google.com/drive/folders/1UHTxrq5b5009PMo39omF3E4tjkd_rWBt?usp=drive_link)**

Place the downloaded files in the `data/` directory with this structure:
```
data/
├── raw/
│   └── zomato_dataset.csv
├── processed/
│   ├── menu_items_enriched.csv
│   ├── restaurants_cleaned.csv
│   └── item_embeddings_fixed.npy
└── synthetic/
    ├── cart_events.csv
    └── users.csv
```

## Project Structure

```
ZOMATHON/
├── src/                          # Main source code
│   ├── data_generation/          # Data preprocessing & session simulation
│   │   ├── clean_dataset.py      # Raw data cleaning
│   │   ├── item_categorizer.py   # Item category classification
│   │   ├── session_generator.py  # Synthetic session generation
│   │   └── session_preprocessing.py
│   ├── features/                 # Feature engineering
│   │   └── feature_builder.py    # Feature construction & normalization
│   ├── graph/                    # Graph-based embeddings
│   │   ├── build_cooccurrence.py # Co-occurrence graph construction
│   │   └── train_node2vec.py     # Node2Vec embedding training
│   ├── model/                    # Model architecture & training
│   │   ├── cart_transformer.py   # CartAddToCartModel architecture
│   │   └── train_model.py        # Training loop & evaluation
│   ├── evaluation/               # Evaluation metrics
│   │   ├── topk_evaluation.py    # Ranking metrics (NDCG, MRR, etc.)
│   │   └── business_impact.py    # Business simulation
│   └── inference/                # Production serving
│       ├── inference_service.py  # FastAPI REST API
│       └── benchmark.py          # Latency benchmarking
│
├── frontend/                     # Streamlit demo app
│   └── app.py
│
├── frontend-react/               # React frontend (alternative UI)
│
├── scripts/                      # Utility scripts
│   ├── analyze_data.py
│   ├── find_similar_items.py
│   └── validate_embeddings.py
│
├── docs/                         # Documentation
│   ├── TECHNICAL_DOCUMENTATION.txt  # Full technical documentation
│   ├── ARCHITECTURE.md           # System architecture
│   ├── FINAL_SUBMISSION.md       # Hackathon submission
│   └── DEMO_SCRIPT.md            # Demo walkthrough
│
├── data/                         # Data files (download separately)
│   ├── raw/                      # Original Zomato dataset
│   ├── processed/                # Cleaned & enriched data
│   └── synthetic/                # Generated sessions
│
├── checkpoints/                  # Model checkpoints
│   └── best_model_final.pt       # Trained model (19.5 MB)
│
├── config.yaml                   # Configuration file
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Installation

### Prerequisites

- Python 3.10+
- macOS / Linux / Windows

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/ZOMATHON.git
cd ZOMATHON

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download dataset from link above and place in data/
```

## Quick Start

### Option 1: Run Demo (Recommended)

Start the inference API and Streamlit frontend:

```bash
# Terminal 1: Start API server
uvicorn src.inference.inference_service:app --port 8000

# Terminal 2: Start Streamlit UI
streamlit run frontend/app.py
```

Open http://localhost:8501 in your browser.

### Option 2: Full Pipeline

Run the complete training pipeline:

```bash
# Step 1: Clean and preprocess data
python -m src.data_generation.clean_dataset
python -m src.data_generation.item_categorizer

# Step 2: Generate synthetic sessions
python -m src.data_generation.session_generator --num-sessions 500000

# Step 3: Build co-occurrence graph and train embeddings
python -m src.graph.build_cooccurrence
python -m src.graph.train_node2vec

# Step 4: Train the model
python -m src.model.train_model \
    --epochs 12 \
    --batch-size 2048 \
    --disable-transformer

# Step 5: Evaluate
python -m src.evaluation.topk_evaluation --max-sessions 50000
python -m src.evaluation.business_impact --max-sessions 50000
```

## API Usage

### Get Recommendations

```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "restaurant_id": 18363104,
    "cart_item_ids": [101, 102],
    "user_id": 1001,
    "hour": 19,
    "top_k": 5
  }'
```

### Response

```json
{
  "recommendations": [
    {"item_id": 203, "item_name": "Coke", "score": 0.87, "category": "beverage"},
    {"item_id": 305, "item_name": "Gulab Jamun", "score": 0.72, "category": "dessert"}
  ],
  "latency_ms": 0.15
}
```

## Model Architecture

```
Input Features (218 dimensions):
├── Cart Embedding (64d)         # Mean-pooled cart item embeddings
├── Candidate Embedding (64d)    # Target item embedding
├── Dot Product (1d)             # Cart-candidate similarity
├── Absolute Difference (64d)    # Complementarity signal
├── User Features (7d)           # RFM features
├── Restaurant Features (5d)     # Rating, price, menu size
├── Cart Dynamic Features (6d)   # missing_beverage, heavy_meal, etc.
└── Context Features (7d)        # Time, meal type, user type

MLP Head:
├── Linear(218 → 256) + ReLU + Dropout(0.1)
├── Linear(256 → 128) + ReLU + Dropout(0.1)
└── Linear(128 → 1)              # Sigmoid output
```

## Key Files

| File | Description |
|------|-------------|
| `src/model/cart_transformer.py` | Model architecture (CartAddToCartModel) |
| `src/model/train_model.py` | Training script with early stopping |
| `src/inference/inference_service.py` | FastAPI production service |
| `frontend/app.py` | Streamlit interactive demo |
| `docs/TECHNICAL_DOCUMENTATION.txt` | Complete technical documentation |

## Configuration

Edit `config.yaml` to modify:

```yaml
model:
  embedding_dim: 64
  hidden_dims: [256, 128]
  dropout: 0.1

training:
  batch_size: 2048
  epochs: 12
  learning_rate: 0.001
  pos_weight: 5.0

inference:
  device: cpu
  top_k: 5
```

## Performance

### Benchmarks

| Metric | Value |
|--------|-------|
| Training Time | 10 min (Apple M-series) |
| Model Size | 19.5 MB |
| Throughput | 6,755 req/s (single CPU pod) |
| P50 Latency | 0.15 ms |
| P99 Latency | 0.25 ms |

### Business Impact (Simulated)

| Metric | Model | Baseline | Lift |
|--------|-------|----------|------|
| Attach Rate | 90.91% | 74.57% | +21.9% |
| Addon Revenue/Session | ₹191 | ₹160 | +19.7% |

## Documentation

- [Documentation](docs/PROJECT_EXPLAINED.md) - Project Overview
- [Architecture](docs/ARCHITECTURE.md) - System design


## License

MIT License - See LICENSE file for details.

## Team

ZOMATHON Team - March 2026
