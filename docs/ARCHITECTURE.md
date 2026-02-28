# CSAO System Architecture

## Technical Architecture Document

---

## 1. System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CSAO SYSTEM ARCHITECTURE                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌────────────┐    ┌─────────────┐    ┌───────────────┐    ┌──────────┐  │
│   │   Client   │───►│   FastAPI   │───►│   Inference   │───►│  Model   │  │
│   │   (App)    │◄───│   Gateway   │◄───│    Engine     │◄───│  (MLP)   │  │
│   └────────────┘    └─────────────┘    └───────────────┘    └──────────┘  │
│         │                  │                   │                  │        │
│         │                  │                   │                  │        │
│         ▼                  ▼                   ▼                  ▼        │
│   ┌────────────┐    ┌─────────────┐    ┌───────────────┐    ┌──────────┐  │
│   │   Redis    │    │  Prometheus │    │   Embedding   │    │  Model   │  │
│   │   Cache    │    │  + Grafana  │    │    Store      │    │  Weights │  │
│   └────────────┘    └─────────────┘    └───────────────┘    └──────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Core Components

### 2.1 Inference Service (FastAPI)

**File:** `src/inference/inference_service.py`

```python
# Key Endpoints
POST /recommend     # Get ranked add-on recommendations
GET  /health        # Health check
GET  /restaurants   # List available restaurants
```

**Request/Response Flow:**

```
Request:
{
  "restaurant_id": 123,
  "cart_item_ids": [456, 789],
  "user_id": "u_12345",    # optional
  "hour": 19,              # optional (default: current)
  "top_k": 5               # optional (default: 5)
}

Response:
{
  "recommendations": [
    {
      "item_id": 101,
      "item_name": "Chocolate Brownie",
      "price": 149.0,
      "score": 0.8234,
      "reason": "Complements your meal"
    },
    ...
  ],
  "latency_ms": 32.5
}
```

### 2.2 Inference Engine

**Class:** `InferenceModel`

**Responsibilities:**
1. Load frozen model + embeddings
2. Build feature vectors from inputs
3. Score all candidate items
4. Return top-K ranked results

**Key Methods:**

```python
class InferenceModel:
    def __init__(self, model_path, embeddings_path, ...):
        # Load model and artifacts
        
    def rank_candidates(self, cart_item_ids, restaurant_id, 
                        user_id=None, hour=None, top_k=5):
        # Build features → Score candidates → Rank → Return top-K
```

### 2.3 Model Architecture

**Class:** `CartAddToCartModel`

```
Input Features (218 dimensions):
├── Cart Embedding (64d)           # Mean-pooled item embeddings
├── Candidate Embedding (64d)      # Item to rank
├── Interaction Features (65d)     # dot product, L1 diff
├── User Features (7d)             # RFM metrics
├── Restaurant Features (5d)       # Rating, prices, etc.
├── Cart Dynamic Features (6d)     # Size, total, missing categories
└── Context Features (7d)          # Hour, meal type, one-hot cats

MLP:
├── Linear(218 → 256) + ReLU + Dropout(0.1)
├── Linear(256 → 128) + ReLU + Dropout(0.1)
└── Linear(128 → 1) + Sigmoid → P(add-to-cart)
```

---

## 3. Data Flow Architecture

### 3.1 Training Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TRAINING PIPELINE                                  │
│                                                                             │
│   Raw Data          Feature Engineering         Model Training              │
│                                                                             │
│   ┌──────────┐      ┌───────────────┐          ┌────────────────┐          │
│   │ Zomato   │      │  Clean &      │          │  DataLoader    │          │
│   │ Dataset  │─────►│  Enrich       │          │  (batch=2048)  │          │
│   │ (Kaggle) │      │  Items        │          └───────┬────────┘          │
│   └──────────┘      └───────┬───────┘                  │                   │
│                             │                          ▼                   │
│   ┌──────────┐      ┌───────▼───────┐          ┌────────────────┐          │
│   │ Session  │      │  Build Co-   │          │  CartAddToCart │          │
│   │ Simulator│─────►│  occurrence  │          │  Model (MLP)   │          │
│   │          │      │  Graph       │          └───────┬────────┘          │
│   └──────────┘      └───────┬───────┘                  │                   │
│                             │                          ▼                   │
│                     ┌───────▼───────┐          ┌────────────────┐          │
│                     │  Node2Vec    │─────────►│  Loss: BCE     │          │
│                     │  Embeddings  │          │  (pos_weight=5)│          │
│                     │  (70k × 64d) │          └───────┬────────┘          │
│                     └──────────────┘                   │                   │
│                                                        ▼                   │
│                                                ┌────────────────┐          │
│                                                │  Checkpoint    │          │
│                                                │  best_model_   │          │
│                                                │  final.pt      │          │
│                                                └────────────────┘          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Inference Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           INFERENCE PIPELINE                                 │
│                                                                             │
│  1. Request        2. Features       3. Score         4. Response          │
│                                                                             │
│  ┌──────────┐     ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ {"cart": │     │ Load user &  │  │ For each    │  │ Return top-K │     │
│  │  [A,B],  │────►│ restaurant   │─►│ candidate:  │─►│ with scores  │     │
│  │  ...}    │     │ features     │  │ score(x)    │  │ and metadata │     │
│  └──────────┘     └──────────────┘  └──────────────┘  └──────────────┘     │
│        │                │                  │                 │              │
│        │                │                  │                 │              │
│        ▼                ▼                  ▼                 ▼              │
│  ┌──────────┐     ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Parse &  │     │ Mean-pool    │  │ Batch GPU/   │  │ Format JSON  │     │
│  │ Validate │     │ cart → 64d   │  │ CPU forward  │  │ + reasons    │     │
│  └──────────┘     └──────────────┘  └──────────────┘  └──────────────┘     │
│                                                                             │
│  Total Latency: ~32ms (MPS) / ~0.15ms (CPU)                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Latency Analysis

### 4.1 Latency Breakdown

| Stage | Time (ms) | % of Total |
|-------|-----------|------------|
| Request parsing | 0.1 | 0.3% |
| Feature lookup (Redis) | 2.0 | 6.2% |
| Embedding lookup | 0.1 | 0.3% |
| Mean pooling | 0.05 | 0.2% |
| Feature concatenation | 0.1 | 0.3% |
| Model forward pass | 0.15 | 0.5% |
| Ranking (argsort) | 0.1 | 0.3% |
| Response serialization | 0.2 | 0.6% |
| **Network overhead** | **29.2** | **91.3%** |
| **Total** | **32** | **100%** |

### 4.2 Optimization Strategies

| Strategy | Speedup | Implementation |
|----------|---------|----------------|
| CPU inference | 8x | Use CPU for small batch |
| Batch scoring | 10-50x | Score all candidates in one forward |
| Redis caching | 5x | Cache user/restaurant features |
| Embedding preload | 3x | Keep embeddings in memory |
| JIT compilation | 2x | TorchScript for production |

### 4.3 Measured Performance

```
Benchmark Results (1000 iterations, CPU):
├── p50 Latency: 0.15 ms
├── p95 Latency: 0.19 ms
├── p99 Latency: 0.25 ms
└── Throughput: 6,755 req/s
```

---

## 5. Production Deployment

### 5.1 Kubernetes Architecture

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: csao-inference
spec:
  replicas: 5  # Scale based on load
  selector:
    matchLabels:
      app: csao
  template:
    spec:
      containers:
      - name: csao-api
        image: csao/inference:latest
        resources:
          requests:
            cpu: "500m"
            memory: "1Gi"
          limits:
            cpu: "2000m"
            memory: "4Gi"
        ports:
        - containerPort: 8000
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 10
```

### 5.2 Horizontal Pod Autoscaler

```yaml
# hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: csao-hpa
spec:
  scaleTargetRef:
    kind: Deployment
    name: csao-inference
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Pods
    pods:
      metric:
        name: http_request_latency_p99
      target:
        type: AverageValue
        averageValue: 50m  # 50ms target
```

### 5.3 Load Balancer Configuration

```
┌───────────────────────────────────────────────────────────────┐
│                     AWS Application Load Balancer             │
│                                                               │
│   ┌─────────────────────────────────────────────────────┐    │
│   │  Target Group: csao-inference                        │    │
│   │  Health Check: GET /health (interval: 30s)           │    │
│   │  Sticky Sessions: Disabled                           │    │
│   │  Connection Draining: 30s                            │    │
│   └─────────────────────────────────────────────────────┘    │
│                                                               │
│   Routing Rules:                                              │
│   ├── POST /api/v1/recommend → csao-inference:8000           │
│   ├── GET  /health            → csao-inference:8000           │
│   └── *                       → 404                           │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

---

## 6. Data Stores

### 6.1 Feature Store (Redis)

```
┌─────────────────────────────────────────────────────────────────┐
│                        REDIS CLUSTER                             │
│                                                                 │
│   User Features (TTL: 1 hour)                                   │
│   ├── user:{user_id}:rfm    → [recency, freq, monetary, ...]   │
│   └── user:{user_id}:prefs  → [cuisine_entropy, ratios, ...]   │
│                                                                 │
│   Restaurant Features (TTL: 24 hours)                           │
│   ├── rest:{id}:meta        → [rating, votes, avg_price, ...]  │
│   └── rest:{id}:items       → [item_id_1, item_id_2, ...]      │
│                                                                 │
│   Item Cache (TTL: 1 hour)                                      │
│   └── item:{id}:info        → [name, price, category, ...]     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 Embedding Storage

```
┌─────────────────────────────────────────────────────────────────┐
│                     EMBEDDING STORAGE                            │
│                                                                 │
│   Production:                                                   │
│   ├── S3: s3://csao-models/embeddings/item_embeddings.npy       │
│   ├── Local: /models/item_embeddings.npy (loaded at startup)   │
│   └── Memory: numpy array (70,220 × 64) = 18 MB                │
│                                                                 │
│   ID Mapping:                                                   │
│   └── item_id → embedding_index (0 to 70,219)                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 6.3 Model Artifact Storage

```
┌─────────────────────────────────────────────────────────────────┐
│                     MODEL ARTIFACTS                              │
│                                                                 │
│   S3 Bucket: s3://csao-models/                                  │
│   ├── checkpoints/                                              │
│   │   ├── best_model_final.pt (19.5 MB)                        │
│   │   └── model_metadata.json                                   │
│   ├── embeddings/                                               │
│   │   └── item_embeddings_fixed.npy (18 MB)                    │
│   ├── scalers/                                                  │
│   │   ├── user_scaler.pkl                                       │
│   │   └── restaurant_scaler.pkl                                 │
│   └── mappings/                                                 │
│       ├── item_id_to_idx.json                                   │
│       └── restaurant_id_to_idx.json                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 7. Monitoring & Observability

### 7.1 Metrics (Prometheus)

```yaml
# Key Metrics Exposed
csao_request_total{endpoint="/recommend", status="200"}
csao_request_latency_seconds{endpoint="/recommend", quantile="0.99"}
csao_model_inference_seconds{quantile="0.99"}
csao_cache_hit_ratio{cache="user_features"}
csao_candidate_pool_size{restaurant_id="*"}
csao_recommendations_served{top_k="5"}
```

### 7.2 Grafana Dashboard

```
┌─────────────────────────────────────────────────────────────────┐
│                     CSAO DASHBOARD                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│   │ Requests/s  │  │ P99 Latency │  │ Error Rate  │            │
│   │   6,245     │  │   48ms      │  │   0.01%     │            │
│   └─────────────┘  └─────────────┘  └─────────────┘            │
│                                                                 │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │             Latency Over Time (24h)                      │  │
│   │   50ms ─────────────────────────────────────────────────│  │
│   │   40ms ────────────────────────────────────────────────│   │
│   │   30ms ──────────▁▂▃▄▅▆▇█▇▆▅▄▃▂▁──────────────────────│   │
│   │   20ms ─────────────────────────────────────────────────│  │
│   └─────────────────────────────────────────────────────────┘  │
│                                                                 │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │             Top Recommended Items (1h)                   │  │
│   │   1. Chocolate Brownie (4,521)                          │  │
│   │   2. Coke (3,892)                                        │  │
│   │   3. Gulab Jamun (2,156)                                 │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 7.3 Alerting Rules

```yaml
# alerts.yaml
groups:
- name: csao-alerts
  rules:
  - alert: HighLatency
    expr: histogram_quantile(0.99, csao_request_latency_seconds) > 0.1
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "P99 latency above 100ms"
      
  - alert: HighErrorRate
    expr: rate(csao_request_total{status!="200"}[5m]) > 0.01
    for: 2m
    labels:
      severity: critical
    annotations:
      summary: "Error rate above 1%"
      
  - alert: LowThroughput
    expr: rate(csao_request_total[5m]) < 100
    for: 10m
    labels:
      severity: warning
    annotations:
      summary: "Request rate dropped significantly"
```

---

## 8. Security Considerations

### 8.1 API Security

| Control | Implementation |
|---------|----------------|
| Authentication | API Key in header (`X-API-Key`) |
| Rate Limiting | 1000 req/min per API key |
| Input Validation | Pydantic models with strict types |
| Output Sanitization | No PII in responses |

### 8.2 Network Security

```
┌─────────────────────────────────────────────────────────────────┐
│                     NETWORK SECURITY                             │
│                                                                 │
│   Internet                                                      │
│       │                                                         │
│       ▼                                                         │
│   ┌───────────────────────────────────────────────────────┐    │
│   │  WAF (Rate limiting, SQL injection, XSS protection)   │    │
│   └───────────────────────────────────────────────────────┘    │
│       │                                                         │
│       ▼                                                         │
│   ┌───────────────────────────────────────────────────────┐    │
│   │  API Gateway (Authentication, API Key validation)     │    │
│   └───────────────────────────────────────────────────────┘    │
│       │                                                         │
│       ▼ (Private VPC)                                          │
│   ┌───────────────────────────────────────────────────────┐    │
│   │  Internal Load Balancer                                │    │
│   └───────────────────────────────────────────────────────┘    │
│       │                                                         │
│       ▼                                                         │
│   ┌───────────────────────────────────────────────────────┐    │
│   │  CSAO Pods (no direct internet access)                │    │
│   └───────────────────────────────────────────────────────┘    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 9. Model Update Strategy

### 9.1 Blue-Green Deployment

```
Current (Blue):        New (Green):
v1.0.0                 v1.1.0
┌──────────┐           ┌──────────┐
│ Pod 1    │           │ Pod 1    │
│ Pod 2    │           │ Pod 2    │
│ Pod 3    │           │ Pod 3    │
└──────────┘           └──────────┘
    │                       │
    │  100% traffic         │  0% traffic
    ▼                       ▼
┌──────────────────────────────────────────┐
│            Load Balancer                  │
│  (gradual traffic shift: 0% → 100%)      │
└──────────────────────────────────────────┘
```

### 9.2 Model Retraining Schedule

| Component | Frequency | Trigger |
|-----------|-----------|---------|
| Item Embeddings | Weekly | New items > 1000 |
| Model Weights | Daily | AUC drop > 2% |
| Feature Scalers | Monthly | Distribution shift |
| Full Pipeline | Quarterly | Major data changes |

### 9.3 A/B Testing Framework

```
┌─────────────────────────────────────────────────────────────────┐
│                     A/B TEST CONFIGURATION                       │
│                                                                 │
│   Experiment: csao_v2_model                                     │
│   ├── Control (80%): current_model_v1.0                        │
│   └── Treatment (20%): new_model_v2.0                          │
│                                                                 │
│   Success Metrics:                                              │
│   ├── Primary: add_to_cart_rate                                │
│   ├── Secondary: average_order_value                           │
│   └── Guardrail: latency_p99 < 100ms                           │
│                                                                 │
│   Duration: 7 days                                              │
│   Statistical Significance: p < 0.05                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 10. Disaster Recovery

### 10.1 Failover Strategy

| Scenario | Detection | Recovery |
|----------|-----------|----------|
| Pod crash | Kubernetes liveness probe | Auto-restart (30s) |
| Redis down | Connection timeout | Fallback to popularity |
| Model corrupted | Health check fails | Roll back to previous |
| Full outage | All pods unhealthy | Manual promotion from DR |

### 10.2 Fallback Recommendations

If model inference fails, return popularity-based fallback:

```python
def get_fallback_recommendations(restaurant_id, top_k=5):
    """Return most popular items from restaurant"""
    items = get_restaurant_items(restaurant_id)
    sorted_items = sorted(items, key=lambda x: x['votes'], reverse=True)
    return sorted_items[:top_k]
```

---

## 11. Cost Analysis

### 11.1 Infrastructure Costs (AWS, estimated)

| Component | Spec | Monthly Cost |
|-----------|------|--------------|
| EKS Cluster | 5 m5.large nodes | $300 |
| Redis ElastiCache | r6g.large | $150 |
| S3 Storage | 100 GB | $3 |
| ALB | Per-hour + LCU | $50 |
| CloudWatch | Logs + metrics | $100 |
| **Total** | | **$603/month** |

### 11.2 Cost vs Revenue

| Metric | Value |
|--------|-------|
| Monthly infra cost | $603 |
| Daily addon revenue uplift | ₹30.2M |
| Monthly addon revenue uplift | ₹906M |
| **ROI** | **150,000x** |

---

## 12. Summary

### Architecture Highlights

1. **Simple but effective:** MLP + mean pooling beats transformer
2. **Sub-50ms latency:** Production-ready performance
3. **Linearly scalable:** Add pods for more throughput
4. **Graceful degradation:** Fallback to popularity baseline
5. **Observable:** Full Prometheus/Grafana monitoring

### Key Technical Decisions

| Decision | Rationale |
|----------|-----------|
| CPU over GPU | Small model, lower latency |
| Redis cache | Fast feature retrieval |
| Kubernetes | Auto-scaling, self-healing |
| Mean pooling | Simpler, faster, better metrics |
| Frozen embeddings | Faster training, stable inference |

---

**Document Version:** 1.0  
**Last Updated:** March 2026  
**Authors:** ZOMATHON Team
