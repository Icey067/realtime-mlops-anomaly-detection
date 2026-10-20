"""Tests for the FastAPI application."""

import pytest
from fastapi.testclient import TestClient
from src.api.main import app
from src.api.routes import health as health_routes


class _FakeRedis:
    async def ping(self) -> bool:
        return True

    async def get(self, key: str) -> str:
        return "0.25"


@pytest.fixture
def test_client() -> TestClient:
    with TestClient(app) as client:
        yield client


def test_healthz_endpoint(test_client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(health_routes, "get_redis_client", lambda: _FakeRedis())

    response = test_client.get("/healthz")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["version"]
    assert data["redis_connected"] is True
    assert data["model_loaded"] is True


def test_ready_endpoint(test_client: TestClient) -> None:
    response = test_client.get("/ready")
    assert response.status_code == 200
    assert response.json()["ready"] is True


def test_predict_endpoint(test_client: TestClient) -> None:
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
    assert data["status"] == "OK"
    assert isinstance(data["is_anomaly"], bool)
    assert data["reconstruction_mse"] >= 0.0
    assert data["inference_latency_ms"] >= 0.0


def test_predict_batch_rejects_oversized_batch(test_client: TestClient) -> None:
    payload = {
        "user_id": "usr_api_999",
        "amount": 10.0,
        "latitude": 40.7128,
        "longitude": -74.0060,
    }
    response = test_client.post("/predict/batch", json=[payload] * 1001)
    assert response.status_code == 422


def test_metrics_endpoint(test_client: TestClient) -> None:
    response = test_client.get("/metrics")
    assert response.status_code == 200
    assert "http_requests_total" in response.text


def test_live_telemetry_endpoint(
    test_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(health_routes, "get_redis_client", lambda: _FakeRedis())

    response = test_client.get("/api/telemetry/live")
    assert response.status_code == 200
    data = response.json()
    assert data["drift_score"] == 0.25
    assert data["drift_threshold"] == 0.15
    assert "model_threshold" in data
