# Sentinel MLOps — Complete Project Summary (repo state: commit `a216769`, clean working tree)

Read this fully before changing anything. This document describes the repository **exactly as it exists now**, including quirks, dead code, and artifact caveats.

---

## 1. What this project is

**Sentinel** is an end-to-end MLOps reference project for real-time financial-transaction **anomaly / fraud detection**:

- A **one-class (unsupervised) PyTorch autoencoder** is trained *only on normal transactions* from the Kaggle Credit Card Fraud dataset (`data/creditcard.csv`, 284,807 rows, 492 frauds). Fraud labels are never used in training.
- Anomaly score = **Mean Squared reconstruction error** through an ONNX RUNTIME (CPU, dynamic INT8 quantized) export of the autoencoder.
- Decision rule: a transaction is "anomalous" when its MSE exceeds a threshold calibrated as the **98th percentile of validation-set reconstruction errors on normal rows only** (so by construction the false-positive rate on normal data ≈ 2%).
- Around the model there is a full streaming pipeline: **Redis Streams** ingestion (producer → consumer group → DLQ), a **FastAPI** inference service, **Prometheus** metrics, an in-memory telemetry buffer feeding a **React (Vite)** dashboard, a **statistical drift monitor** (Kolmogorov–Smirnov + Wasserstein), and an **automated retrain pipeline** with a model hot-reload.

It is a portfolio/learning project, not a deployed production system. Some README claims (latency benchmarks, "production") are aspirational rather than measured; treat the code, not the README superlatives, as ground truth.

---

## 2. Critical context (do not re-derive, just know)

- **One dataset only**: everything is built around the Kaggle **creditcard** schema: 30 features = `["Time", "V1", "V2", ..., "V28", "Amount"]` where V1..V28 are anonymized PCA components. A **PaySim dataset** (`data/paysim.csv`, 11-feature schema) exists in `data/` and there was a working dual-dataset training feature at one point, but it was **reverted by the repo owner — all current code is creditcard-only** and `config.DATASET_PATH` points to `data/creditcard.csv`.
- **`data/creditcard.csv` (150 MB) and `data/paysim.csv` (493 MB) are gitignored**; the repo cannot train without them.
- Repo layout: `src/` package with `main` at several module entrypoints, run via `python -m`.
- **Git**: all currently `.md` files are in `.gitignore` (a bare `.md` pattern) — new markdown files won't be committed unless force-added. README.md and PROJECT_CONTEXT.md are tracked (force-added earlier).
- **Repository history is small** (7 commits) and written in "Phase 1..5" style. The current HEAD is `a216769` and the working tree is clean.

---

## 3. Tech stack & environment

| Component | Choice | Notes |
|---|---|---|
| Python | 3.14 (repo targets 3.11+) | Runs on Windows; repo lives under OneDrive |
| Model framework | PyTorch (`torch>=2.2`) | `.venv` has `2.14.1+cu130`, CUDA on RTX 4060 available |
| Inference runtime | ONNX Runtime (CPU) | Quantized dynamic INT8 export is preferred at load time |
| API | FastAPI + Uvicorn | App factory pattern, lifespan hooks |
| Streaming | Redis Streams + consumer groups | `redis>=5` async client |
| Schemas/settings | Pydantic v2 + pydantic-settings | `.env` supported, `case_sensitive=False` |
| Drift stats | scipy KS test + Wasserstein (`stats.ks_2samp`, `stats.wasserstein_distance`) | Evidently is listed in requirements but **not actually used in code** |
| Tracking | MLflow (optional) | try/except import; training degrades gracefully if server/package missing |
| Metrics | prometheus-client | `/metrics` endpoint |
| LLM-of-note | none | — |
| Frontend | React 19 + Vite (`frontend/`) | node_modules present locally |
| Tests | pytest + pytest-asyncio + httpx TestClient | 2 test files (see §9) |

`requirements.txt` (runtime) includes: pydantic(-settings), redis, fastapi, uvicorn[standard], torch, onnx, onnxruntime, scikit-learn, numpy<2, pandas, evidently, mlflow, prometheus-client, locust, httpx, pytest, pytest-asyncio. There is **no** `pyproject.toml`, no lint/typecheck config, no lockfile, no pinned exact versions.

---

## 4. End-to-end data flow

```
                     ┌────────────────────────────┐
 data/creditcard.csv │  TransactionProducer       │  batch ~50 tx  (XADD, MAXLEN ~100k)
 (or synthetic with  │  + scheduled drift         │  every 50ms, +30% mean-shift every 20 batches
  injected drift) ─▶ │  injection                 │
                     └─────────────┬──────────────┘
                                   ▼
                     Redis Stream "transactions:stream"
                     (msgpack-ish: every field is str(value))
                                   ▼
                     ┌────────────────────────────┐
                     │  TransactionConsumer (group│  XREADGROUP count=100, block=2000ms
                     │  sentinel_consumers)       │  Pydantic validate → DLQ on failure → XACK
                     └─────────────┬──────────────┘
                                   ▼  (only via HTTP calls / frontend / locust)
                     ┌────────────────────────────┐
                     │ FastAPI  POST /predict     │
                     │  payload → 30-dim vector → │
                     │  StandardScaler → ONNX AE →│
                     │  MSE → threshold 2.2521…   │
                     └─────────────┬──────────────┘
                                   ▼
                     Prometheus /metrics + in-memory telemetry
                     (React dashboard polls /api/telemetry/live)

 Offline: train (normal rows) → threshold p98 → ONNX export → INT8 quantize →
 drift monitor → retrain pipeline (drift alert → retrain → quantize → reload)
```

Important: **the consumer does not call the model**. The consumer only validates/acknowledges/counts. Scoring happens through the FastAPI service (called by `/predict` APIs, the locust benchmark, or the frontend). The drift monitor and retrain pipeline are **standalone classes not wired into the consumer** in the current commit (see gotcha list).

---

## 5. Directory map (file by file)

```
mlops-anomaly/
├── src/
│   ├── __init__.py
│   ├── config.py                  # All settings (see §8)
│   ├── api/
│   │   ├── main.py                # FastAPI factory, CORS, /metrics, lifespan
│   │   ├── telemetry.py           # Prometheus metric objects + TelemetryStore
│   │   └── routes/
│   │       ├── health.py          # /healthz /ready /api/telemetry/live
│   │       └── predict.py         # /predict /predict/batch
│   ├── ingestion/
│   │   ├── producer.py            # Redis stream producer + drift injection
│   │   └── consumer.py            # Consumer group worker + DLQ + latency stats
│   ├── mlops/
│   │   ├── drift_monitor.py       # KS/Wasserstein drift scoring + alert callback
│   │   └── retrain_pipeline.py    # drift alert → background retrain → hot reload
│   ├── model/
│   │   ├── autoencoder.py         # PyTorch AE definition
│   │   ├── train.py               # CLI training entrypoint (`python -m src.model.train`)
│   │   ├── quantize.py            # CLI ONNX export + INT8 (…`src.model.quantize`)
│   │   └── predictor.py           # ONNX Runtime scoring engine
│   └── schemas/
│       ├── transaction.py         # TransactionPayload + AnomalyPredictionResult
│       └── telemetry.py           # LatencyTelemetry, DriftAlertPayload, SystemHealthResponse
├── benchmarks/
│   └── locustfile.py              # Locust load test against POST /predict
├── tests/
│   ├── test_phase1.py             # settings + payload schema + redis roundtrip tests
│   └── test_phase2_3.py           # AE shapes, predictor scoring, API endpoints, drift smoke
├── docker/
│   ├── docker-compose.yml
│   ├── Dockerfile.api
│   ├── prometheus.yml
│   └── grafana/provisioning/…     # auto-provisioned dashboard + datasource
├── frontend/
│   └── src/{App.jsx, main.jsx, index.css}   # Vite React dashboard
├── data/                          # gitignored: creditcard.csv, paysim.csv,
│                                  # baseline_distribution.csv
├── models/                        # mostly gitignored artifacts (see §7)
├── .github/workflows/ci.yml       # GitHub Actions pipeline
├── generate_screenshot_log.py     # standalone rich-console fake telemetry (never imported)
├── PROJECT_CONTEXT.md             # earlier hand-written context doc
└── README.md                      # marketing-heavy; read with skepticism
```

---

## 6. Subsystem details (the exact behavior, gotchas included)

### 6.1 Feature contract
- Feature order is hard-coded in **multiple** places: `train.py` (`expected_cols`), `producer.py` (synthetic gen + CSV replay), `TransactionPayload.to_feature_vector()` (`[time_step, v1..v28, amount]`), drift monitor's `feature_names`. These must stay consistent — the very first training bug in this repo came from a dataset whose columns didn't match this list (PaySim), causing `KeyError: None of [Index(['Time','V1',…])] are in the [columns]`.
- `Amount` is **not** log-transformed anywhere in the current code — it is fed raw (raw values up to ~25k, scaled by the StandardScaler). This is why validation MSE has a very heavy tail (max seen ≈ 568) and the threshold histogram bucketing in the README is coarse.

### 6.2 Training (`src/model/train.py`, CLI: `python -m src.model.train --epochs N --batch-size B --lr X`)
Defaults: 15 epochs, batch 256, lr 1e-3. Also `settings.FEATURE_DIM=30, HIDDEN_DIM=16, LATENT_DIM=8`.
1. If `data/creditcard.csv` exists → load it; else **silently generate 25k synthetic rows** and save to `data/baseline_distribution.parquet` (CSV fallback) — check logs for the "Dataset not found" warning, because synthetic results look plausible but mean nothing.
2. Keep only `Class == 0` rows (284,315 of 284,807 in the real dataset). No `Class` column → uses all rows (this is the mode that burned the repo owner with PaySim).
3. `train_test_split(X, test_size=0.20, random_state=42)` → 227,452 train / 56,863 val.
4. `StandardScaler` fit on train only, **both** splits transformed; scaler persisted to `models/scaler.joblib` (joblib).
5. Model: encoder `Linear(30,16) → LayerNorm → LeakyReLU(0.1) → Linear(16,8) → LeakyReLU`, decoder `Linear(8,16) → LayerNorm → LeakyReLU → Linear(16,30)`. MSE loss, Adam(lr, weight_decay=1e-5), `ReduceLROnPlateau(factor=0.5, patience=3)`.
6. Device: `cuda if torch.cuda.is_available() else cpu`. (5 epochs on the RTX 4060 with batch 256: ~12 s; 15 epochs on 6.35M PaySim rows: ~7 min.)
7. Threshold = `numpy.percentile(val_errors, 98)`; metadata JSON saved to `models/model_metadata.json` with keys: `model_type, input_dim, hidden_dim, latent_dim, anomaly_threshold, percentile, val_mse_mean, trained_at, train_samples`. Logged threshold seen with the current tag of the checkpoint: **2.2521579837799135**.
8. MLflow is optional: wraps `mlflow.set_tracking_uri("http://localhost:5000")` + experiment "Sentinel-Anomaly-Detection" in try/except — without a server it just logs a warning and continues. Metrics/params logged: epochs, batch/lr, dims, `train_mse`, `val_mse` per step, artifacts themselves.

### 6.3 Export & quantization (`src/model/quantize.py`, CLI: `python -m src.model.quantize`)
- Loads `models/autoencoder.pt`, exports `models/autoencoder.onnx` (opset 14, dynamic batch axis, constant folding; `dynamo=False`) with dummy input `randn(1, input_dim)`.
- Runs `onnx.checker.check_model`.
- Dynamic INT8 quantization (`onnxruntime.quantization.quantize_dynamic`, `QuantType.QInt8`) → `models/autoencoder_int8.onnx`.
- Benchmarks both sessions with batch-1 random inputs (500 iters); reports FP32/INT8 per-sample latency (observed ~0.075 ms FP32, ~0.033 ms INT8 — numbers only meaningful for this tiny graph).
- Dims come from `settings.FEATURE_DIM/HIDDEN_DIM/LATENT_DIM`, **not** from the checkpoint metadata — they must match training.

### 6.4 Inference (`src/model/predictor.py`)
`AnomalyPredictor(settings)` on init:
1. Loads scaler from `models/scaler.joblib` (warns + uses identity if missing).
2. Loads threshold from `models/model_metadata.json` (`anomaly_threshold`, fallback `settings.ANOMALY_THRESHOLD_FALLBACK = 0.045`).
3. Prefers `models/autoencoder_int8.onnx`, else `models/autoencoder.onnx`; `ort.InferenceSession` with CPU provider, 2 intra-op threads, sequential execution.
4. If **no ONNX file exists**, `session=None`, `model_loaded=False` — and here is the single most important gotcha:
   **`predict_one` returns a fake `reconstruction_mse = 0.01` (always "normal") when the model is absent.** `predict_batch` returns all-zero MSEs. Tests skipped in that mode. This makes the service appear healthy-ish while returning garbage — check `model_loaded` before trusting results.

- Scoring path: `payload.to_feature_vector()` (30 raw floats) → `scaler.transform` → `session.run` → `MSE = mean((scaled − reconstructed)²)` → `is_anomaly = mse > threshold`.
- `predict_batch` is a single ONNX call over the matrix; per-item latency = total/len.
- Global cached instance via `get_predictor()` (`lru_cache`); API routes import it lazily per request.

### 6.5 Streaming ingestion
**Producer** (`src/ingestion/producer.py`, CLI: `python -m src.ingestion.producer [--csv path] [--batch-size n] [--interval s] [--drift-interval n] [--batches n]`):
- Two modes: replay `--csv` (DictReader over creditcard.csv rows: V1..V28, Time, Amount; missing columns default to `row.get(..., 0.0)` so other CSVs fail with cryptic errors — the owner hit exactly that with PaySim) or synthetic mode.
- Synthetic transactions: lognormal amounts, geo jittered around (40.71, −74.01), V1..V28 ~ N(0,1).
- **Drift injection**: every `PRODUCER_DRIFT_INJECTION_INTERVAL=20` batches, all V-features get mean +0.35 (= `PRODUCER_DRIFT_MAGNITUDE 0.30 * 1.0` scale factor… effectively +0.35 shift) and amount ×1.30. This exists so the consumer-side drift monitor has something to catch.
- Publishes via pipeline `XADD` with `MAXLEN ~ 100,000` approximate; log line every 10 batches.
- Batch size 50, sleep 50 ms (≈ 1000 tx/s ceiling).

**Consumer** (`src/ingestion/consumer.py`, CLI: `python -m src.ingestion.consumer [--name …] [--batch-size n] [--max-messages n]`):
- Connects, `XGROUP CREATE … mkstream=True` idempotently ("BUSYGROUP" tolerated).
- Loop: `XREADGROUP > count=100 block=2000ms`; each message → `TransactionPayload.from_redis_dict` (bytes→str keys, numeric casts) → on `ValidationError` (or any exception): to `transactions:dlq` with `original_id, failed_at, error_reason, raw_payload`, message **still XACK'd** (errors are acknowledged after routing).
- Latency bookkeeping: rolling 1,000-sample dequeues; p50/p95/p99 logged every ~5 s; budget breach counting against `LATENCY_BUDGET_MS=10.0`.
- Callback hook `on_payload_callback(async)` exists but nothing passes one by default; consumer standalone `main()` runs validation-only.

### 6.6 Drift monitor (`src/mlops/drift_monitor.py`) — standalone, **not wired in**
- `DataDriftMonitor()`: rolling `deque(maxlen=1000)` of 30-dim vectors (`record_inference_features(vec)` only accepts length == `settings.FEATURE_DIM` — **nothing calls this in the current commit**).
- Baseline: `data/baseline_distribution.parquet` → `.csv` fallback → the training CSV. (A 25k-row synthetic baseline CSV exists, generated during the walled-garden runs.)
- `compute_drift()` needs ≥100 samples: 2-sample KS statistic + p-value, Wasserstein distance normalized by baseline std; feature is "drifted" when **`p_value < 0.01 AND w_dist > 0.20`** (with ~1000 samples the p-value alone fires on everything, hence the AND, though it's still fragile).
- Score = fraction of the 30 features drifted; `data_drift_score` gauge updated; if score `> settings.DRIFT_SCORE_THRESHOLD=0.15` → emits `DriftAlertPayload(drift_score, threshold=0.15, drifted_features, alert_level="CRITICAL", retrigger_training=True)` and calls `on_drift_alert` callback (only if provided).

### 6.7 Retrain pipeline (`src/mlops/retrain_pipeline.py`) — also standalone, **not wired in**
- `retrain_pipeline.handle_drift_alert(alert) → bool`: guarded by a lock, cooldown 300 s, single concurrent retrain.
- On trigger, spawns a daemon thread doing: `train_autoencoder(epochs=10)` → `run_export_and_quantization()` → `get_predictor()._initialize_runtime()` (hot swap of the ONNX session, so the endpoint never serves a half-loaded model). No exception propagates; failures just log.
- **In the current commit nothing calls `handle_drift_alert`** — wiring it into the consumer/API is the known next step (a version with the wiring existed and was reverted).

### 6.8 API (`src/api/main.py` + routes)
- `create_app()`: FastAPI "Sentinel MLOps Anomaly Detection Engine", CORS `allow_origins=["*"]` **with `allow_credentials=True`** (spec-violating combo; browsers will reject - flag it if it matters).
- Routes: `POST /predict` (1 payload), `POST /predict/batch` (list), `GET /healthz`, `GET /ready`, `GET /metrics`, `GET /api/telemetry/live`, `GET /`. CORS via CORSMiddleware.
- `POST /predict` body = `TransactionPayload` (see §6.9); returns `AnomalyPredictionResult{transaction_id, is_anomaly, reconstruction_mse, threshold, inference_latency_ms}`.
- `/predict` increments `HTTP_REQUESTS_TOTAL`, `ANOMALY_COUNT_TOTAL{decision}`, `INFERENCE_LATENCY_SECONDS` histogram, `RECONSTRUCTION_MSE_HISTOGRAM`, and pushes to the in-memory `telemetry_store` (rolling buffer of 100 predictions + latency deque; aggregates for `/api/telemetry/live` which the React dashboard polls).
- `/healthz` checks Redis by opening a **new** `redis.asyncio` client per call with 1 s timeout (fine for liveness; wasteful per call), returns `SystemHealthResponse{status="healthy"|"degraded", version, redis_connected, model_loaded, uptime_seconds}` — **always HTTP 200 even when degraded** (`/ready` returns 503 when the model isn't loaded).

### 6.9 Schemas (`src/schemas/transaction.py`)
`TransactionPayload` (Pydantic v2, frozen, `extra="forbid"`):
- `transaction_id` (auto `tx_{uuid4()}`), `timestamp` (auto now, `gt=0`), `user_id` (3..64, required), `amount` (0 < a ≤ 1,000,000, required), `currency` (3 chars, uppercased by validator), `latitude` (±90, required), `longitude` (±180, required), `time_step` (≥0, default 0), `v1..v28` floats each bounded `[-150, 150]`, default 0.
- `to_feature_vector()` → `[time_step, v1, …, v28, amount]` (30 floats, exact model order).
- `to_redis_dict()` / `from_redis_dict()` — every value stringified for `XADD`; `from_redis_dict` decodes bytes keys/values and numeric-casts (amount, lat, lon, timestamp, time_step, and any `v*` key; `v1` used for drift but not a real Number type — beware float("nan") text).
- Note: extra keys are forbidden → any unmodeled field in Redis messages causes a DLQ entry.

### 6.10 Observability (src/api/telemetry.py)
- Prometheus: `http_requests_total{method,endpoint,status}`, `inference_latency_seconds` (buckets .001…0.1), `anomaly_count_total{decision}`, `reconstruction_mse` (buckets tuned for the **old threshold** scale 0.005–0.5; current threshold 2.25 means most normal traffic clamps into the top bucket), `stream_throughput_tx_per_sec` gauge (referenced but only set externally), `data_drift_score` gauge.
- `TelemetryStore`: 100-item `deque` of recent predictions + latencies; `get_live_metrics()` is a plain dict for the dashboard.

---

## 7. Artifacts currently in `models/` (mix of generations — read carefully)

| File | Tracked in git? | Notes |
|---|---|---|
| `autoencoder.pt` | No (gitignored) | PyTorch state_dict, 30/16/8 architecture |
| `scaler.joblib` | No (gitignored) | fit on train split |
| `autoencoder.onnx` + `autoencoder.onnx.data` | `.data` tracked; `.onnx` ignored | external-data split export |
| `autoencoder_int8.onnx` | No (gitignored) | dynamic INT8 |
| `model_metadata.json` | **Yes (tracked)** | threshold `2.2521579837799135`, `input_dim` 30, `train_samples` 227452 |
| `evaluation.json`, `feature_schema.json` | Not present at this commit | |

**Caveat worth flagging:** because artifacts are generated locally while `.pt/.onnx/.joblib` are gitignored but `model_metadata.json` is tracked, it is possible for the committed metadata to describe an older training run than the local checkpoints. Always regenerate metadata, checkpoint and scaler **together** and verify `metadata["anomaly_threshold"]` against a fresh `models/model_metadata.json` after any retrain. Owner has retrained multiple times (latest recorded run: creditcard 5 epochs on CUDA, batch 256 — val loss ≈ 0.516, threshold ≈ 2.259) so do not be surprised by slightly-different numbers in logs vs these notes.

---

## 8. Settings reference (`src/config.py`, env-var overridable, case-insensitive, `.env` supported)

| Setting | Default | Used for |
|---|---|---|
| `APP_NAME`, `APP_VERSION`, `ENVIRONMENT`, `DEBUG` | Sentinel-…, 1.0.0, development, False | misc |
| `REDIS_HOST/PORT/DB/PASSWORD` | localhost / 6379 / 0 / None | redis URLs everywhere |
| `REDIS_STREAM_KEY` | transactions:stream | producer + consumer |
| `REDIS_DLQ_KEY` | transactions:dlq | consumer |
| `REDIS_CONSUMER_GROUP` | sentinel_consumers | consumer |
| `REDIS_CONSUMER_NAME` | consumer_worker_1 | consumer |
| `REDIS_STREAM_MAXLEN` | 100,000 | XADD maxlen (approximate) |
| `PRODUCER_BATCH_SIZE` | 50 | producer emit batch |
| `PRODUCER_EMIT_INTERVAL_SEC` | 0.05 | producer sleep |
| `PRODUCER_DRIFT_INJECTION_INTERVAL` | 20 | drift batch frequency |
| `PRODUCER_DRIFT_MAGNITUDE` | 0.30 | +30% mean shift |
| `CONSUMER_BATCH_SIZE` | 100 | XREADGROUP count |
| `CONSUMER_BLOCK_MS` | 2000 | XREADGROUP block |
| `LATENCY_BUDGET_MS` | 10.0 | breach counter |
| `DATASET_PATH` | data/creditcard.csv | training + drift fallback |
| `BASELINE_DATA_PATH` | data/baseline_distribution.parquet | drift baseline |
| `SCALER_PATH` | models/scaler.joblib | train + predictor |
| `MODEL_CHECKPOINT_PATH` | models/autoencoder.pt | train, quantize, predictor metadata |
| `ONNX_MODEL_PATH` / `ONNX_QUANTIZED_PATH` | models/autoencoder.onnx / autoencoder_int8.onnx | inference |
| `FEATURE_DIM` | 30 | model input width (30) |
| `HIDDEN_DIM` / `LATENT_DIM` | 16 / 8 | encoder/decoder widths |
| `BATCH_SIZE`, `EPOCHS`, `LEARNING_RATE` | 256 / 20 / 1e-3 | train defaults (CLI flags override) |
| `ANOMALY_PERCENTILE_THRESHOLD` | 98.0 | threshold calibration |
| `ANOMALY_THRESHOLD_FALLBACK` | 0.045 | predictor metadata-missing fallback |
| `MLFLOW_TRACKING_URI`, `MLFLOW_EXPERIMENT_NAME` | http://localhost:5000, Sentinel-Anomaly-Detection | optional tracking |
| `DRIFT_SCORE_THRESHOLD` | 0.15 | retrain trigger |
| `DRIFT_WINDOW_SIZE` | 1000 | rolling buffer max |
| `API_HOST/PORT/WORKERS` | 0.0.0.0 / 8000 / 4 | uvicorn |

---

## 9. Tests & CI

**Tests** (run `pytest tests/ -v`):
- `tests/test_phase1.py` — settings defaults, payload valid/bounds/roundtrip (to_redis_dict → bytes → from_redis_dict → feature vector equality). These are pure, no artifacts needed.
- `tests/test_phase2_3.py` — AE shapes & non-negative MSE; `predict_one` scoring + asserts `inference_latency_ms < 15.0`; FastAPI `TestClient` for `/healthz` (asserts 200 + keys), `/predict` (200 + fields, latency < 15 ms), `/metrics` (contains `http_requests_total`); drift monitor with 150 random N(0,1) records — asserts score in [0,1].
- **Important**: the inference/API tests silently rely on the `predictor`'s no-model fallback path (or on artifacts having been trained) — if you make failure explicit (e.g. raise instead of returning fake MSE), those tests must skip when artifacts are missing.
- Pytest config comes from nowhere formal (no `pytest.ini`/pyproject); asyncio tests work via `pytest-asyncio` default mode. Compiled `.venv` Python 3.14 works; CI pins 3.11.

**GitHub Actions** (`.github/workflows/ci.yml`, ubuntu, Python 3.11, Redis service container):
1. install `requirements.txt` plus onnxscript + pyarrow,
2. `pytest tests/ -v` (yes, all tests run *before* training — they pass only because of the fake-MSE fallback),
3. `python -m src.model.train --epochs 5 --batch-size 256`,
4. `python -m src.model.quantize`,
5. run `tests/test_phase2_3.py` again (now with real artifacts → the latency/scoring assertions run the real ONNX path).

---

## 10. docker & frontend

- `docker/docker-compose.yml`: api + redis + prometheus + grafana; `Dockerfile.api` builds the FastAPI image on python:3.11-slim; Grafana auto-provisions datasource (prometheus) + `sentinel_dashboard.json`.
- `frontend/`: Vite + React 19 dashboard. It consumes `/api/telemetry/live` and shows rolling telemetry (throughput, fraud count, recent transactions). Convention: `npm install && npm run dev`, CORS already permissive in the API.
- `benchmarks/locustfile.py`: Locust user hitting `/predict` with random payloads (500+ RPS target); `locust -f benchmarks/locustfile.py --host http://localhost:8000`.
- `generate_screenshot_log.py`: standalone `rich`-based script that *fakes* colored telemetry output for screenshots; it's never imported by anything — delete-proof but not part of the pipeline.

---

## 11. Known gotchas / next steps (what the code itself leaves undone)

1. **Fake-MSE fallback in `predictor.predict_one` / `predict_batch`** when no ONNX is loaded — silent wrong results. Change to raise / explicit degraded flag.
2. **Drift monitor + retrain pipeline are dead code** until wired: nothing calls `record_inference_features` or `handle_drift_alert`. The cooling-off also exists in retrain but is only reached if wired.
3. **README vs behavior**: latency table numbers, "0.064 ms" and PR stats are aspirational; there is **no labeled evaluation script** committed — a `benchmarks/evaluate.py` (PR-AUC, ROC-AUC, precision/recall at percentile sweep, writing `models/evaluation.json`) existed in a work session and was reverted; the repo owner has a prior run showing AP ≈ 0.218, ROC AUC ≈ 0.9365, recall ≈ 0.817 @ 98pct on creditcard.
4. **CORS** `allow_origins=["*"]` + `allow_credentials=True` is invalid per spec.
5. **`/healthz` per-call Redis connection** and always-200 status.
6. **`Amount` is untransformed raw** → heavy-tailed MSE, weak fraud-vs-normal separation at the tails; `log1p` on money features is the known quality lever (a version with it existed and was reverted).
7. **Producer can’t replay PaySim** (hard-coded V1..V28 columns); a damage-control version with `verify_columns` existed and was reverted.
8. **`RECONSTRUCTION_MSE_HISTOGRAM` buckets** (0.005–0.5) don't match the 2.25 threshold scale — most of the mass lands in the last bucket.
9. **No lint/format/type check config**, no ruff/mypy; tests are thin (2 files, mostly happy-path).
10. **`generate_screenshot_log.py`** contains fabricated "production" numbers — it's cosmetic.
11. Repo owner **repeatedly resets the working tree to HEAD** after experiments (this has already happened twice after two large improvements landed). When contributing: expect these resets, and after any retrain always re-check that the persisted artifact set (checkpoint, scaler, ONNX, metadata) was produced by the *same* training run.
12. Questionable p-value gating in drift detection is fragile at ~1000 samples (KS p<0.01 fires nearly always given a small distributional shift); the AND with Wasserstein>0.2 is the effective rule.

---

## 12. How to run everything (quick reference)

```powershell
# 1. deps
python -m venv .venv; .venv\Scripts\pip install -r requirements.txt
# transformers of ONNX need: pip install onnx onnxscript pyarrow

# 2. train + export
python -m src.model.train --epochs 5 --batch-size 256
python -m src.model.quantize
python -m benchmarks.evaluate            # if present

# 3. infra
docker compose -f docker/docker-compose.yml up -d   # api + redis + prometheus + grafana

# 4. streaming simulation
python -m src.ingestion.producer --batch-size 50 --interval 0.05
python -m src.ingestion.consumer

# 5. scoring
uvicorn src.api.main:app --port 8000     # or python src/api/main.py? -> python -m src.api.main?
# (main.py runs uvicorn directly when executed as a script: python src/api/main.py)
# POST http://localhost:8000/predict with a TransactionPayload JSON

# 6. load test
locust -f benchmarks/locustfile.py --host http://localhost:8000

# 7. dashboard (separate shell)
cd frontend; npm run dev
```

`uvicorn src.api.main:app` works because `create_app()` is additionally exposed as the module-level `app`.

---

End of summary. Anything not listed here either lives in git history (PaySim dual-dataset training, feature-schema persistence file `models/feature_schema.json`, evaluation suite, ruff config, refactored logging) or is a Python cache.
