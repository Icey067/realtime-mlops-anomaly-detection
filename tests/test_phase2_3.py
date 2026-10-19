"""Integration and unit tests for Model Inference, FastAPI Microservice, and Drift Monitor."""

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from src.api.main import app
from src.model.autoencoder import Autoencoder
from src.model.predictor import AnomalyPredictor
from src.mlops.drift_monitor import DataDriftMonitor
from src.schemas.transaction import TransactionPayload


@pytest.fixture
def test_client() -> TestClient:
    """FastAPI TestClient fixture."""
    return TestClient(app)


def test_autoencoder_forward_shape() -> None:
    """Verify PyTorch Autoencoder preserves input dimension across bottleneck."""
    model = Autoencoder(input_dim=30, hidden_dim=16, latent_dim=8)
    dummy_x = torch.randn(10, 30)

    # Encode & Decode
    z = model.encode(dummy_x)
    assert z.shape == (10, 8)

    reconstructed = model.decode(z)
    assert reconstructed.shape == (10, 30)

    errors = model.compute_reconstruction_error(dummy_x)
    assert errors.shape == (10,)
    assert (errors >= 0.0).all()


def test_predictor_scoring() -> None:
    """Verify AnomalyPredictor scores valid transaction and enforces sub-15ms budget."""
    predictor = AnomalyPredictor()

    payload = TransactionPayload(
        user_id="usr_test_score",
        amount=145.20,
        currency="USD",
        latitude=40.7128,
        longitude=-74.0060,
        time_step=100.0,
        v1=0.25,
        v2=-0.12,
    )

    result = predictor.predict_one(payload)
    assert result.transaction_id == payload.transaction_id
    assert isinstance(result.is_anomaly, bool)
    assert result.reconstruction_mse >= 0.0
    # Strict SLA check: single CPU inference must be well under 15ms
    assert result.inference_latency_ms < 15.0


def test_fastapi_healthz_endpoint(test_client: TestClient) -> None:
    """Verify GET /healthz returns 200 with status."""
    response = test_client.get("/healthz")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "version" in data


def test_fastapi_predict_endpoint(test_client: TestClient) -> None:
    """Verify POST /predict correctly scores payload."""
    payload = {
        "transaction_id": "tx_api_test_100",
        "timestamp": 1729384000.0,
        "user_id": "usr_api_999",
        "amount": 250.00,
        "currency": "USD",
        "latitude": 40.7128,
        "longitude": -74.0060,
        "time_step": 500.0,
        "v1": -0.15,
        "v2": 0.42,
    }
    response = test_client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["transaction_id"] == "tx_api_test_100"
    assert "is_anomaly" in data
    assert "reconstruction_mse" in data
    assert data["inference_latency_ms"] < 15.0


def test_fastapi_metrics_endpoint(test_client: TestClient) -> None:
    """Verify GET /metrics exposes Prometheus formatted metrics."""
    response = test_client.get("/metrics")
    assert response.status_code == 200
    assert "http_requests_total" in response.text


def test_data_drift_monitor_calculation() -> None:
    """Verify DataDriftMonitor evaluates 2-sample KS test and Wasserstein distance."""
    monitor = DataDriftMonitor()

    # Feed 150 normal samples
    for _ in range(150):
        vec = np.random.normal(0.0, 1.0, size=30).tolist()
        monitor.record_inference_features(vec)

    score, drifted_features, metrics = monitor.compute_drift()
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0
    assert isinstance(drifted_features, list)
