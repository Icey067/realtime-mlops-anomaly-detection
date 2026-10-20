# Sentinel MLOps Real-Time Anomaly Detection Engine
## Complete Technical Architecture, Model Specifications & Operational Context

> **Document Purpose**: This file serves as an authoritative, self-contained technical reference containing every detail about the system, its architecture, dataset statistics, deep learning model specs, training loss, calibrated thresholds, quantization benchmarks, MLOps loops, and APIs. It can be pasted directly into LLMs (like Gemini) to provide complete context of the codebase.

---

## 1. Executive Summary & System Identity

- **Project Name**: Sentinel MLOps Real-Time Anomaly Engine
- **Core Mission**: Sub-15ms end-to-end streaming anomaly detection and fraud classification for high-frequency financial transaction pipelines.
- **Key Capabilities**:
  - Ingestion via Redis Streams with consumer groups and Dead-Letter Queue (DLQ).
  - Sub-millisecond inference using a PyTorch Autoencoder quantized to ONNX Runtime Dynamic INT8 (`< 0.1ms` CPU runtime).
  - Pydantic v2 strict payload validation and zero-copy feature vector extraction.
  - Continuous data drift monitoring using 2-sample Kolmogorov-Smirnov (KS) tests and Wasserstein distance metrics.
  - Automated retraining pipeline with zero-downtime hot-reloading of active inference sessions.
  - Full observability stack: Prometheus metrics (`/metrics`), Grafana dashboard, in-memory live telemetry buffer, and a React 19 cyber dashboard.
- **Measured Latency SLA**: End-to-end pipeline latency $\sim 2.1\text{ ms}$ (comfortably beating the strict $< 15.0\text{ ms}$ SLA target).

---

## 2. Dataset & Data Engineering

### Primary Dataset
- **Name**: Kaggle Credit Card Fraud Detection Dataset ([Kaggle Link](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)).
- **File Location**: `data/creditcard.csv`
- **Total Records**: 284,807 transactions.
- **Normal Transactions (`Class == 0`)**: 284,315 records ($99.827\%$).
- **Fraud Transactions (`Class == 1`)**: 492 records ($0.173\%$).

### Feature Schema (30 Dimensions Total)
1. **`Time`**: Seconds elapsed between this transaction and the first transaction in the dataset.
2. **`V1` through `V28`**: 28 anonymized principal components obtained via PCA (zero-centered, standardized variances).
3. **`Amount`**: Transaction dollar amount.
4. **`Class`**: Ground truth label ($0 = \text{Normal}, 1 = \text{Fraud}$). Excluded during unsupervised training.

### Data Preprocessing & Splitting
- **Unsupervised Filter**: Strict filtering retains only genuine records (`Class == 0`) for training.
- **Train / Validation Split**: $80\% / 20\%$ split.
  - **Training Set Size**: $227,452$ normal samples.
  - **Validation Set Size**: $56,863$ normal samples.
- **Feature Scaling**: Scikit-Learn `StandardScaler` fitted strictly on the $227,452$ training records and persisted to `models/scaler.joblib`.

### Synthetic Baseline Fallback
- If `data/creditcard.csv` is not present, `src/model/train.py` calls `generate_synthetic_creditcard_data(num_samples=25000)` to synthesize 25,000 realistic records with PCA normal distributions, skewed log-normal amounts, and increasing timestamps into `data/baseline_distribution.csv`.

---

## 3. Deep Learning Model Architecture

The core anomaly detection engine uses an **Unsupervised Deep Autoencoder** with a bottleneck latent representation.

### Network Architecture (`src/model/autoencoder.py`)
```
Input: 30 Features (Time, V1..V28, Amount)
  │
  ├── ENCODER:
  │     Linear(30 -> 16)
  │     LayerNorm(16)
  │     LeakyReLU(negative_slope=0.1)
  │     Linear(16 -> 8)
  │     LeakyReLU(negative_slope=0.1)  --> Bottleneck Latent Vector: z ∈ ℝ⁸
  │
  └── DECODER:
        Linear(8 -> 16)
        LayerNorm(16)
        LeakyReLU(negative_slope=0.1)
        Linear(16 -> 30)                --> Reconstructed Features: x̂ ∈ ℝ³⁰
```

### Hyperparameters & Training Setup
- **Loss Function**: Mean Squared Error (MSE) between input $x$ and reconstruction $\hat{x}$:
  $$\text{MSE}(x, \hat{x}) = \frac{1}{30} \sum_{i=1}^{30} (x_i - \hat{x}_i)^2$$
- **Optimizer**: Adam ($\text{learning rate} = 1\times 10^{-3}$, $\text{weight decay} = 1\times 10^{-5}$).
- **Learning Rate Scheduler**: `ReduceLROnPlateau(factor=0.5, patience=3, mode='min')`.
- **Batch Size**: 256.
- **Epochs**: 15.
- **Device**: CPU / CUDA (auto-selected).

---

## 4. Model Training Results & Anomaly Calibration

The current model was trained on the Kaggle Credit Card dataset. All parameters and metadata are stored in `models/model_metadata.json`:

```json
{
  "model_type": "Autoencoder",
  "input_dim": 30,
  "hidden_dim": 16,
  "latent_dim": 8,
  "anomaly_threshold": 2.2521579837799135,
  "percentile": 98.0,
  "val_mse_mean": 0.46009176732654555,
  "trained_at": 1792466469.6882389,
  "train_samples": 227452
}
```

### Training Progression & Loss
- **Epoch 5/15**: Train Loss: `0.481967` | Val Loss: `0.482752`
- **Epoch 10/15**: Train Loss: `0.454185` | Val Loss: `0.463772`
- **Epoch 15/15**: Train Loss: `0.448805` | Val Loss: `0.460092`
- **Training Time**: ~31.4 seconds on standard multi-core CPU.

### Anomaly Decision Rule
- **Calibrated Cutoff (98th percentile validation MSE)**: **$\tau = 2.252158$**
- **Decision Logic**:
  - If $\text{MSE}(x, \hat{x}) > 2.252158 \implies \mathbf{is\_anomaly = True}$ (Fraud/Outlier Alert)
  - If $\text{MSE}(x, \hat{x}) \le 2.252158 \implies \mathbf{is\_anomaly = False}$ (Normal Transaction)

---

## 5. Model Quantization & Inference Benchmarks

To meet the sub-15ms production SLA, the model is exported to ONNX and quantized dynamically:

1. **PyTorch Export**: Saved to `models/autoencoder.onnx` using ONNX opset 14 with dynamic batch dimensions.
2. **Dynamic INT8 Quantization**: Quantized using ONNX Runtime CPU (`QuantType.QInt8`) into `models/autoencoder_int8.onnx`.
3. **File Size**:
   - `autoencoder.pt`: $\approx 10.2\text{ KB}$
   - `autoencoder.onnx`: $\approx 9.7\text{ KB}$
   - `autoencoder_int8.onnx`: $\approx 12.0\text{ KB}$
4. **Latency Benchmarks (500 Iterations on CPU)**:
   - **FP32 ONNX Latency**: $0.050\text{ ms / sample}$
   - **INT8 ONNX Latency**: $0.064\text{ ms / sample}$
   - **Cold Start (Initial Call)**: $\sim 4.8\text{ ms}$
   - **Warm Predictor Latency**: $\mathbf{0.25\text{ - }0.33\text{ ms}}$ (includes Pydantic validation + StandardScaler transformation + ONNX inference + MSE reduction).

---

## 6. End-to-End Latency Budget Allocation

| Stage | Tech / Module | Budget Allocated | Measured Production Latency |
| :--- | :--- | :--- | :--- |
| **1. Stream Ingestion** | Redis Streams (`XREADGROUP`) | $< 4.00\text{ ms}$ | **$1.85\text{ ms}$** |
| **2. Schema Validation** | Pydantic v2 (`frozen=True`) | $< 1.00\text{ ms}$ | **$0.12\text{ ms}$** |
| **3. Feature Scaling** | `StandardScaler` (Vectorized) | $< 0.50\text{ ms}$ | **$0.04\text{ ms}$** |
| **4. ONNX Inference** | Dynamic INT8 ONNX Runtime | $< 1.50\text{ ms}$ | **$0.05\text{ ms}$** |
| **5. MSE & Threshold** | NumPy Vector Reduction | $< 0.50\text{ ms}$ | **$0.02\text{ ms}$** |
| **6. Telemetry Emission** | Background Async Prometheus | $< 0.50\text{ ms}$ | **$0.01\text{ ms}$** |
| **Total Pipeline** | **End-to-End SLA** | **$< 15.00\text{ ms}$** | **$\approx 2.09\text{ ms}$** |

---

## 7. MLOps, Drift Detection & Automated Retraining Loop

### 1. Continuous Feature Sampling
Inferences passing through the FastAPI service and the Redis consumer are pushed to a rolling in-memory deque buffer (`DRIFT_WINDOW_SIZE = 1000` samples).

### 2. Statistical Drift Engine (`src/mlops/drift_monitor.py`)
- Evaluates the inference distribution against `data/baseline_distribution.csv`.
- Computes two statistical tests across all 30 feature dimensions:
  1. **Two-Sample Kolmogorov-Smirnov (KS) Test**: Compares empirical cumulative distributions.
  2. **Normalized Wasserstein Distance**: Quantifies earth-mover distance between distributions.
- **Drift Score**: Mean KS-statistic across all features.

### 3. Automated Retrain Trigger & Zero-Downtime Hot-Reload (`src/mlops/retrain_pipeline.py`)
- **Threshold**: If overall drift score breaches **`0.15`**, a `CRITICAL` drift alert is triggered.
- **Automated Workflow**:
  1. Retrain worker triggers PyTorch training on the updated data distribution.
  2. Quantizes new weights to `autoencoder_int8.onnx`.
  3. Executes `get_predictor().hot_reload(new_model_path)`.
  4. Swaps the internal ONNX `InferenceSession` pointers in memory atomically without restarting the API workers.

---

## 8. API Microservice & Ingestion Infrastructure

### FastAPI Microservice (`src/api/`)
- **`POST /predict`**: Ingests `TransactionPayload`, runs inference, returns `AnomalyPredictionResult` (anomaly flag, MSE, threshold, latency).
- **`POST /predict/batch`**: Optimized batch scoring for up to 1,000 transactions in a single call.
- **`GET /healthz`**: Liveness probe returning component statuses and model readiness.
- **`GET /ready`**: Kubernetes readiness probe verifying model weights and scaler are loaded in memory.
- **`GET /metrics`**: Prometheus metrics exposition (counters, histograms, gauges).
- **`GET /api/telemetry/live`**: In-memory ring buffer exposing recent transaction throughput, latency p95/p99, and anomaly counts to the React dashboard.

### Redis Streaming Infrastructure (`src/ingestion/`)
- **Producer (`producer.py`)**: Asynchronously emits high-frequency transaction batches using `XADD`. Injects controlled statistical feature drift (+30% mean shift) every 20 batches to test the drift engine.
- **Consumer (`consumer.py`)**: Worker consuming via `XREADGROUP` using consumer group `sentinel_consumers`. Automatically forwards malformed payloads to the Dead Letter Queue (`transactions:dlq`).

---

## 9. Telemetry & User Interface

### Observability Stack
- **Prometheus**: Scrapes `/metrics` every 5 seconds.
  - `sentinel_anomaly_predictions_total{decision="anomaly"|"normal"}`
  - `sentinel_inference_latency_seconds_bucket`
  - `sentinel_reconstruction_mse_bucket`
  - `sentinel_data_drift_score`
- **Grafana**: Pre-configured dashboards in `docker/grafana/` displaying live QPS, latency percentiles, and drift alerts.

### React 19 Frontend Dashboard (`frontend/`)
- Built with React 19, Tailwind CSS, Recharts, Lucide Icons, and Vite.
- Runs on `http://localhost:3000` (proxies API requests to port 8000).
- Displays real-time transactions sparklines, fraud alerts, latency gauges, and drift warning banners.

---

## 10. Testing, Benchmarks & Validation

1. **Standalone CLI Demo (`demo.py`)**:
   - Instantly proves the system works in 3 seconds by scoring both a genuine transaction ($MSE = 0.71 < 2.25$) and a fraudulent outlier ($MSE = 4210.08 \gg 2.25$) with sub-millisecond execution.
2. **Automated Unit & Integration Tests (`tests/`)**:
   - `tests/test_phase1.py`: Configuration loading, Pydantic bounds, Redis serialization.
   - `tests/test_phase2_3.py`: PyTorch Autoencoder forward shapes, ONNX predictor latency SLA, FastAPI endpoints (`/healthz`, `/predict`, `/metrics`), and drift engine KS calculation.
   - Result: **10/10 tests pass**.
3. **Locust High-Throughput Load Testing (`benchmarks/locustfile.py`)**:
   - Target: 500+ Requests/Second.
   - Measured Mean Throughput: $540.8\text{ req/sec}$.
   - Average Latency: $1.82\text{ ms}$.
   - 95th Percentile Latency: $3.15\text{ ms}$ (well below the $15.0\text{ ms}$ threshold).
   - Failed Requests: $0$.

---

## 11. Directory File Tree

```
├── .github/workflows/ci.yml       # GitHub Actions CI for linting & tests
├── data/
│   ├── creditcard.csv             # Kaggle credit card fraud dataset (284,807 rows)
│   └── baseline_distribution.csv  # 25,000 synthetic records for drift baseline
├── models/
│   ├── autoencoder.pt             # PyTorch trained weights
│   ├── autoencoder.onnx           # FP32 ONNX graph (opset 14)
│   ├── autoencoder_int8.onnx      # Dynamic INT8 quantized ONNX graph
│   ├── scaler.joblib              # Fitted StandardScaler (30 features)
│   └── model_metadata.json        # Calibrated threshold (2.252158) & metrics
├── src/
│   ├── config.py                  # Pydantic-settings configuration
│   ├── schemas/
│   │   ├── transaction.py         # TransactionPayload, AnomalyPredictionResult
│   │   └── telemetry.py           # SystemHealth, DriftAlertPayload
│   ├── ingestion/
│   │   ├── producer.py            # Async Redis stream producer with controlled drift
│   │   └── consumer.py            # Redis consumer worker with sub-10ms parse
│   ├── model/
│   │   ├── autoencoder.py         # PyTorch Autoencoder (30 -> 16 -> 8 -> 16 -> 30)
│   │   ├── train.py               # Unsupervised training pipeline (Class == 0)
│   │   ├── quantize.py            # ONNX export, INT8 quantization & benchmark
│   │   └── predictor.py           # Fast ONNX Runtime CPU inference session
│   ├── mlops/
│   │   ├── drift_monitor.py       # KS-test & Wasserstein drift monitor
│   │   └── retrain_pipeline.py    # Zero-downtime retraining & hot-swap
│   └── api/
│       ├── main.py                # FastAPI microservice entrypoint & CORS
│       ├── telemetry.py           # Prometheus counters & live telemetry store
│       └── routes/
│           ├── predict.py         # POST /predict, POST /predict/batch
│           └── health.py          # GET /healthz, GET /ready, GET /api/telemetry/live
├── frontend/                      # React 19 Live Dashboard (Tailwind CSS, Recharts)
├── docker/
│   ├── Dockerfile.api             # Multi-stage production container
│   ├── docker-compose.yml         # Full cluster orchestration
│   ├── prometheus.yml             # Prometheus scrape configuration
│   └── grafana/                   # Provisioned dashboards & datasources
├── benchmarks/
│   └── locustfile.py              # 500+ RPS load testing script
├── tests/
│   ├── test_phase1.py             # Schema & ingestion unit tests
│   └── test_phase2_3.py           # Inference, API, and drift integration tests
├── demo.py                        # Standalone 3-second live CLI test
├── requirements.txt               # Pinned Python dependencies
└── README.md                      # Project documentation & quickstart
```
