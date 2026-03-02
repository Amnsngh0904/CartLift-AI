#!/bin/bash
# =============================================================================
# ZOMATHON - Full Pipeline Run Script
# =============================================================================
# Runs the complete CSAO pipeline: data → features → model → inference
#
# Usage:
#   ./run_all.sh              # Full pipeline with sample data
#   ./run_all.sh --skip-data  # Skip data generation (use existing data)
#   ./run_all.sh --api-only   # Only start API + frontend (requires trained model)
# =============================================================================

set -e
cd "$(dirname "$0")"

PROJECT_ROOT="$(pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# Parse args
SKIP_DATA=false
API_ONLY=false
for arg in "$@"; do
  case $arg in
    --skip-data) SKIP_DATA=true ;;
    --api-only) API_ONLY=true ;;
  esac
done

echo "=============================================="
echo "  ZOMATHON - CSAO Pipeline"
echo "=============================================="

# -----------------------------------------------------------------------------
# Step 0: Setup
# -----------------------------------------------------------------------------
if [ ! -d "venv" ] && [ ! -d ".venv" ]; then
  echo "[0/10] Creating virtual environment..."
  python3 -m venv venv
fi

if [ -d "venv" ]; then
  source venv/bin/activate
elif [ -d ".venv" ]; then
  source .venv/bin/activate
fi

echo "[0/10] Installing dependencies..."
pip install -q -r requirements-minimal.txt

# Create required directories
mkdir -p data/raw data/processed data/synthetic data/cache checkpoints logs outputs

# Copy .env if not exists
if [ ! -f .env ]; then
  cp .env.example .env 2>/dev/null || true
fi

if [ "$API_ONLY" = true ]; then
  echo ""
  echo "Starting API and Frontend only..."
  echo "  API:      uvicorn src.inference.inference_service:app --host 0.0.0.0 --port 8000"
  echo "  Frontend: streamlit run frontend/app.py"
  echo ""
  echo "Run in two terminals:"
  echo "  Terminal 1: uvicorn src.inference.inference_service:app --host 0.0.0.0 --port 8000"
  echo "  Terminal 2: streamlit run frontend/app.py"
  exit 0
fi

# -----------------------------------------------------------------------------
# Step 1: Generate or verify raw data
# -----------------------------------------------------------------------------
if [ "$SKIP_DATA" = false ]; then
  if [ ! -f "data/raw/zomato_dataset.csv" ]; then
    echo "[1/10] Generating sample Zomato dataset..."
    python scripts/generate_sample_data.py
  else
    echo "[1/10] Using existing raw data: data/raw/zomato_dataset.csv"
  fi
else
  echo "[1/10] Skipping data generation (--skip-data)"
fi

# -----------------------------------------------------------------------------
# Step 2: Clean dataset
# -----------------------------------------------------------------------------
echo "[2/10] Cleaning dataset..."
python -m src.data_generation.clean_dataset --input data/raw/zomato_dataset.csv --output-dir data/processed

# -----------------------------------------------------------------------------
# Step 3: Item categorization
# -----------------------------------------------------------------------------
echo "[3/10] Categorizing menu items..."
python -m src.data_generation.item_categorizer \
  --input data/processed/menu_items_cleaned.csv \
  --output data/processed/menu_items_enriched.csv \
  --restaurants data/processed/restaurants_cleaned.csv

# -----------------------------------------------------------------------------
# Step 3.5: Add smoothed ratings to restaurants
# -----------------------------------------------------------------------------
echo "[3.5/10] Adding smoothed ratings to restaurants..."
python -m src.data_generation.restaurant_utils --input data/processed/restaurants_cleaned.csv --output data/processed/restaurants_cleaned.csv

# -----------------------------------------------------------------------------
# Step 4: Session preprocessing
# -----------------------------------------------------------------------------
echo "[4/10] Session preprocessing..."
python -m src.data_generation.session_preprocessing \
  --input data/processed/menu_items_enriched.csv \
  --output data/processed/menu_items_simulation.csv

# -----------------------------------------------------------------------------
# Step 5: Generate cart sessions
# -----------------------------------------------------------------------------
echo "[5/10] Generating cart sessions (this may take a few minutes)..."
python -m src.data_generation.session_generator \
  --restaurants data/processed/restaurants_cleaned.csv \
  --menu-items data/processed/menu_items_simulation.csv \
  --num-sessions 50000 \
  --neg-samples 5

# -----------------------------------------------------------------------------
# Step 6: Extract sequences
# -----------------------------------------------------------------------------
echo "[6/10] Extracting cart sequences..."
python -m src.data_generation.extract_sequences

# -----------------------------------------------------------------------------
# Step 7: Build co-occurrence graph
# -----------------------------------------------------------------------------
echo "[7/10] Building co-occurrence graph..."
python -m src.graph.build_cooccurrence

# -----------------------------------------------------------------------------
# Step 8: Train Node2Vec embeddings
# -----------------------------------------------------------------------------
echo "[8/10] Training Node2Vec embeddings..."
python -m src.graph.train_node2vec

# -----------------------------------------------------------------------------
# Step 9: Build features
# -----------------------------------------------------------------------------
echo "[9/10] Building features..."
python -m src.features.feature_builder

# -----------------------------------------------------------------------------
# Step 10: Train model
# -----------------------------------------------------------------------------
echo "[10/10] Training model..."
python -m src.model.train_model --epochs 5 --batch-size 1024

# -----------------------------------------------------------------------------
# Done
# -----------------------------------------------------------------------------
echo ""
echo "=============================================="
echo "  Pipeline complete!"
echo "=============================================="
echo ""
echo "To run the demo:"
echo "  Terminal 1: uvicorn src.inference.inference_service:app --host 0.0.0.0 --port 8000"
echo "  Terminal 2: streamlit run frontend/app.py"
echo ""
echo "Then open http://localhost:8501 in your browser."
echo ""
