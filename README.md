# Context-Aware Sequential Add-On Recommendation System

A production-grade recommendation system that suggests relevant add-on items based on user context, sequential behavior, and item relationships.

## Overview

This system combines **sequential modeling**, **graph neural networks**, and **contextual awareness** to deliver highly relevant add-on recommendations in real-time.

### Key Features

- **Sequential Understanding**: Captures user journey and purchase patterns
- **Graph-Based Relationships**: Leverages item co-purchase graphs and addon associations
- **Context Awareness**: Incorporates time, device, session, and user context
- **Production Ready**: Includes training pipelines, evaluation, and serving infrastructure

## Project Structure

```
addon_recommendation/
├── data_generation/      # Synthetic data generation & data loading
├── features/             # Feature engineering & preprocessing
├── graph/                # Graph construction & GNN components
├── models/               # Model architectures
├── training/             # Training pipelines & optimization
├── evaluation/           # Metrics & evaluation utilities
├── inference/            # Serving & batch inference
└── utils/                # Common utilities & helpers

config.yaml               # Configuration file
requirements.txt          # Python dependencies
.env.example              # Environment variables template
```

## Installation

### Prerequisites

- Python 3.10+
- CUDA 11.8+ (for GPU support)

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd ZOMATHON

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment variables
cp .env.example .env
```

## Quick Start

### 1. Generate Data

```bash
python -m addon_recommendation.data_generation.generate
```

### 2. Build Features

```bash
python -m addon_recommendation.features.build
```

### 3. Construct Graph

```bash
python -m addon_recommendation.graph.construct
```

### 4. Train Model

```bash
python -m addon_recommendation.training.train --config config.yaml
```

### 5. Evaluate

```bash
python -m addon_recommendation.evaluation.evaluate --checkpoint checkpoints/best.pt
```

### 6. Serve

```bash
python -m addon_recommendation.inference.serve --host 0.0.0.0 --port 8000
```

## Configuration

All configurations are managed through `config.yaml`. Key sections:

| Section | Description |
|---------|-------------|
| `data` | Data paths, splits, and loading parameters |
| `features` | Feature engineering settings |
| `graph` | Graph construction and GNN configuration |
| `model` | Model architecture parameters |
| `training` | Training loop, optimizer, scheduler |
| `evaluation` | Metrics and evaluation settings |
| `inference` | Serving configuration |

## Model Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Item Sequence  │     │   Item Graph    │     │    Context      │
│    Encoder      │     │      GNN        │     │    Encoder      │
└────────┬────────┘     └────────┬────────┘     └────────┬────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │    Context Fusion       │
                    │      (Gated/Concat)     │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   Add-On Prediction     │
                    │        Head             │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   Top-K Add-Ons         │
                    └─────────────────────────┘
```

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| NDCG@K | Normalized Discounted Cumulative Gain |
| Hit Rate@K | Proportion of hits in top-K |
| MRR | Mean Reciprocal Rank |
| Precision@K | Precision at K |
| Recall@K | Recall at K |
| Coverage | Catalog coverage of recommendations |
| Diversity | Intra-list diversity |

## API Reference

### POST /recommend

```json
{
  "user_id": "user_123",
  "session_items": ["item_1", "item_2", "item_3"],
  "context": {
    "time_of_day": "evening",
    "device": "mobile"
  },
  "num_recommendations": 5
}
```

### Response

```json
{
  "recommendations": [
    {"addon_id": "addon_42", "score": 0.95},
    {"addon_id": "addon_17", "score": 0.87}
  ],
  "latency_ms": 12
}
```

## Development

### Code Quality

```bash
# Format code
black addon_recommendation/
isort addon_recommendation/

# Lint
flake8 addon_recommendation/
mypy addon_recommendation/

# Run tests
pytest tests/ -v --cov=addon_recommendation
```

### Pre-commit Hooks

```bash
pre-commit install
pre-commit run --all-files
```

## Experiment Tracking

This project supports Weights & Biases integration:

```bash
wandb login
python -m addon_recommendation.training.train --config config.yaml
```

## License

MIT License

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

## Contact

For questions or issues, please open a GitHub issue.
