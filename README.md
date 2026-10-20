# Sentinel Anomaly Detection

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111%2B-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Autoencoder-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-INT8-00599C?style=flat-square&logo=onnx&logoColor=white)](https://onnxruntime.ai/)
[![Redis](https://img.shields.io/badge/Redis-Streams-DC382D?style=flat-square&logo=redis&logoColor=white)](https://redis.io/)

A streaming anomaly detection service for financial transactions. A PyTorch
autoencoder is trained only on normal transactions, exported to ONNX, and
served through FastAPI. Redis Streams provide the ingestion path, Prometheus
and Grafana provide telemetry, and a drift monitor can trigger retraining.

## Architecture

```
producer -> Redis Stream -> consumer group -> validation + drift sampling
                                     |
                                     +-> drift monitor -> retrain pipeline

client -> FastAPI /predict -> ONNX Runtime
                     |
                     +-> Prometheus /metrics
                     +-> in-memory dashboard buffer
```

The API accepts direct HTTP requests, which is what the load test targets. The
consumer validates stream payloads, records ingestion latency, samples features
for drift detection, publishes the latest drift score to Redis for the
dashboard, and triggers retraining. It does not call the inference endpoint.
The consumer and API are separate processes; they share model artifacts on disk
and the drift score in Redis, but not in-memory state.

## Dataset

The primary dataset is the
[Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
dataset at `data/creditcard.csv` (284,807 rows, 492 frauds). It contains
`Time`, `V1`-`V28`, `Amount`, and `Class`.

- Training uses only `Class == 0` rows with an 80/20 train/validation split
  (227,452 train, 56,863 validation).
- `StandardScaler` is fitted on the training split and saved to
  `models/scaler.joblib`.
- Training writes a 25,000-row sample of normal transactions to
  `data/baseline_distribution.csv` for drift detection.
- If the dataset is missing, `src/model/train.py` generates a synthetic
  fallback so the pipeline can run end to end.

`data/paysim.csv` is present in some checkouts but is not used by the code.

## Model

The autoencoder topology is `30 -> 16 -> 8 -> 16 -> 30` with LayerNorm and
LeakyReLU activations. It minimizes mean squared reconstruction error with Adam
and a `ReduceLROnPlateau` scheduler.

The anomaly threshold is the 98th percentile of validation reconstruction error.
A transaction is flagged when its error is above that value. Training writes
the weights, scaler, threshold, and metadata to `models/`.

Artifacts:

| File | Purpose |
| --- | --- |
| `models/autoencoder.pt` | PyTorch state dict |
| `models/autoencoder.onnx` | FP32 ONNX graph (opset 14) |
| `models/autoencoder.onnx.data` | External ONNX weights when produced |
| `models/autoencoder_int8.onnx` | Dynamic INT8 ONNX graph |
| `models/scaler.joblib` | Fitted `StandardScaler` |
| `models/model_metadata.json` | Threshold, training stats, evaluation metrics |

## Evaluation

`src/model/eval.py` scores all 492 labeled frauds plus the 56,863 held-out
normal validation rows. It computes PR-AUC, ROC-AUC, precision/recall at the
calibrated threshold, and a precision/recall table at the 90th, 95th, 98th, and
99th percentile cutoffs. Results are written to `models/model_metadata.json`.

```bash
python -m src.model.eval --dataset data/creditcard.csv
```

Current results for the INT8 model in this working tree (see
`model_metadata.json`):

| Metric | Value |
| --- | --- |
| PR-AUC | 0.4744 |
| ROC-AUC | 0.9377 |
| Calibrated threshold (p98) | 2.357169 |
| Precision @ threshold | 0.2573 |
| Recall @ threshold | 0.8069 |
| F1 @ threshold | 0.3902 |
| Normal validation MSE mean | 0.496236 |
| Fraud MSE mean | 23.828154 |

| Percentile | Threshold | Precision | Recall | F1 |
| --- | --- | --- | --- | --- |
| 90 | 0.677339 | 0.0698 | 0.8679 | 0.1293 |
| 95 | 1.087482 | 0.1276 | 0.8455 | 0.2217 |
| 98 | 2.391279 | 0.2586 | 0.8069 | 0.3917 |
| 99 | 4.683973 | 0.3747 | 0.6931 | 0.4864 |

The p98 threshold from the evaluation is slightly different from the training
threshold because the evaluation scores the INT8 ONNX graph rather than the
PyTorch checkpoint.

## Setup

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

Optional development tooling:

```bash
pip install ruff
```

Settings are read from environment variables or `.env`; see `.env.example`.
CORS defaults to `http://localhost:3000`. Set `CORS_ALLOW_ORIGINS` to a JSON
array when deploying the dashboard elsewhere.

## Running

### Demo

```bash
python demo.py
```

Scores one normal and one anomalous payload with the loaded ONNX model.

### Tests and lint

```bash
pytest tests -v
ruff check src tests demo.py benchmarks
```

The predictor and API tests require the artifact files in `models/`. Run training
and quantization first if they are absent.

### API

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

- Swagger: `http://localhost:8000/docs`
- Metrics: `http://localhost:8000/metrics`
- Liveness: `http://localhost:8000/healthz`
- Readiness: `http://localhost:8000/ready`

Endpoints:

| Method | Path | Notes |
| --- | --- | --- |
| POST | `/predict` | One transaction |
| POST | `/predict/batch` | Up to 1,000 transactions |
| GET | `/healthz` | Model and Redis status |
| GET | `/ready` | 503 until model and scaler load |
| GET | `/api/telemetry/live` | Recent throughput, latency, and drift score |
| GET | `/metrics` | Prometheus text format |

When model artifacts are missing, `/predict` returns `status: "DEGRADED"` with
`is_anomaly: null` and no reconstruction score. It does not return a fabricated
prediction.

### Redis ingestion

Start Redis on `localhost:6379`, then:

```bash
python -m src.ingestion.consumer
python -m src.ingestion.producer --drift-interval 20
```

The producer replays `data/creditcard.csv` when present, otherwise it generates
synthetic payloads. On every drift interval it scales `Amount` and `V1`-`V28` by
30% to simulate distribution shift. The consumer validates payloads, routes
invalid messages to `transactions:dlq`, samples features for the drift monitor,
writes the drift score to `sentinel:drift_score`, and triggers the retrain
pipeline when the score exceeds `0.15`.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

The Vite dev server runs on port 3000 and proxies API calls to port 8000.

### Docker Compose

```bash
cd docker
docker compose up --build -d
```

Services: Redis (6379), API (8000), producer, consumer, MLflow (5000),
Prometheus (9090), and Grafana (3001, `admin`/`admin`).

The image contains only `src/`; `models/` and `data/` are bind-mounted from the
host. The API runs one Uvicorn worker by default because the live telemetry
buffer and Prometheus registry are per process. Scale with more containers, not
more workers, until the metrics are backed by a shared store.

## Latency benchmarking

Measured latency depends on the host, CPU, Python version, and concurrent load.
Run the benchmark locally rather than relying on numbers from another machine.

The quantization script prints per-sample ONNX latency:

```bash
python -m src.model.quantize
```

For end-to-end HTTP latency, start the API and run Locust:

```bash
locust -f benchmarks/locustfile.py --host http://localhost:8000 --headless -u 200 -r 20 -t 60s
```

The test stops with a summary and prints p95 for `/predict`. Choose the user
count and spawn rate for your environment. The Locust file does not assert a
fixed SLA.

## Project layout

```
.
├── .github/workflows/ci.yml
├── benchmarks/locustfile.py
├── data/                         # creditcard.csv, baseline_distribution.csv
├── docker/
│   ├── Dockerfile.api
│   ├── docker-compose.yml
│   ├── prometheus.yml
│   └── grafana/provisioning/
├── frontend/                     # React dashboard
├── models/                       # trained and quantized artifacts
├── src/
│   ├── api/                      # FastAPI app, routes, metrics, Redis client
│   ├── ingestion/                # Redis producer and consumer
│   ├── mlops/                    # drift monitor and retrain pipeline
│   ├── model/                    # autoencoder, training, quantization, eval
│   ├── schemas/                  # Pydantic request and response models
│   ├── config.py
│   └── logging_config.py
├── tests/
│   ├── test_api.py
│   ├── test_ingestion.py
│   ├── test_predictor.py
│   └── test_schemas.py
├── demo.py
├── pyproject.toml                # project metadata, ruff, pytest
└── requirements.txt
```

## Known limitations and trade-offs

- **Threshold quality.** The 98th percentile cutoff gives high recall but low
  precision at the default operating point. Which threshold to use depends on
  the relative cost of false positives and missed frauds.
- **No labels in training.** The model never sees fraud examples. This keeps the
  setup realistic for unlabeled streams, but a supervised or hybrid model would
  likely separate the classes better.
- **Drift heuristics.** A feature is counted as drifted when `p < 0.01` and
  normalized Wasserstein distance is above `0.20`. The overall score is the
  fraction of drifted features. These cutoffs are not tuned.
- **Retraining is in-process.** The retrain pipeline uses a background thread
  and writes artifacts to disk. With multiple processes there is no
  cross-process lock, so concurrent retrains can race.
- **The API does not reload retrained artifacts.** `hot_reload()` runs in the
  consumer process that triggered the job. Restart the API (or add a reload
  endpoint) to serve the new model.
- **Retraining clears the evaluation block.** `train_autoencoder()` rewrites
  `model_metadata.json` without the previous `evaluation` data. Run
  `python -m src.model.eval` again after a retrain to restore metrics.
- **Stream delivery.** Invalid messages go to a DLQ, but there is no
  `XAUTOCLAIM` loop for messages left pending by a crashed consumer. The
  consumer also exits on Redis connection errors and relies on the container
  restart policy.
- **Telemetry is per process.** The dashboard buffer is in memory, so multiple
  Uvicorn workers or API containers each expose different live numbers.
- **Quantization is not a guaranteed win here.** Dynamic INT8 can reduce compute
  time on larger models; on this ~10 KB network it may be slower and slightly
  larger on disk. Measure on the target host.
- **No authentication or rate limiting.** Add a gateway or middleware before
  exposing the API publicly.
- **Health checks ping Redis on every request.** This is simple and fast on a
  healthy network but gives no protection from a slow or unavailable Redis.
- **Data and model files are gitignored.** A fresh clone needs
  `data/creditcard.csv` or the synthetic fallback to train, and must run
  training plus quantization before the predictor or API tests can pass.
  `models/model_metadata.json` is tracked so the last evaluation is visible.
- **`PROJECT_CONTEXT.md` and `PROJECT_SUMMARY.md` are stale.** Both predate the
  fixes in this repository and contain older claims; treat `README.md` and
  `models/model_metadata.json` as authoritative.

## License

MIT
