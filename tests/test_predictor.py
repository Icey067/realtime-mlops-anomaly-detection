"""Tests for the autoencoder, ONNX predictor, and drift monitor."""

import numpy as np
import pandas as pd
import torch
from src.config import Settings
from src.mlops.drift_monitor import DataDriftMonitor
from src.model.autoencoder import Autoencoder
from src.model.predictor import AnomalyPredictor
from src.schemas.transaction import TransactionPayload


def _payload() -> TransactionPayload:
    return TransactionPayload(
        user_id="usr_test_score",
        amount=145.20,
        currency="USD",
        latitude=40.7128,
        longitude=-74.0060,
        time_step=100.0,
        v1=0.25,
        v2=-0.12,
    )


def test_autoencoder_forward_shape() -> None:
    model = Autoencoder(input_dim=30, hidden_dim=16, latent_dim=8)
    dummy_x = torch.randn(10, 30)

    z = model.encode(dummy_x)
    assert z.shape == (10, 8)

    reconstructed = model.decode(z)
    assert reconstructed.shape == (10, 30)

    errors = model.compute_reconstruction_error(dummy_x)
    assert errors.shape == (10,)
    assert (errors >= 0.0).all()


def test_predictor_scoring() -> None:
    predictor = AnomalyPredictor()
    assert predictor.is_ready

    payload = _payload()
    result = predictor.predict_one(payload)
    assert result.transaction_id == payload.transaction_id
    assert result.status == "OK"
    assert isinstance(result.is_anomaly, bool)
    assert result.reconstruction_mse is not None
    assert result.reconstruction_mse >= 0.0
    assert result.inference_latency_ms >= 0.0


def test_predictor_returns_degraded_when_artifacts_missing(tmp_path) -> None:
    settings = Settings(
        SCALER_PATH=tmp_path / "scaler.joblib",
        MODEL_CHECKPOINT_PATH=tmp_path / "autoencoder.pt",
        ONNX_MODEL_PATH=tmp_path / "autoencoder.onnx",
        ONNX_QUANTIZED_PATH=tmp_path / "autoencoder_int8.onnx",
        DATASET_PATH=tmp_path / "creditcard.csv",
        BASELINE_DATA_PATH=tmp_path / "baseline_distribution.csv",
    )
    predictor = AnomalyPredictor(settings)
    assert predictor.is_ready is False

    result = predictor.predict_one(_payload())
    assert result.status == "DEGRADED"
    assert result.is_anomaly is None
    assert result.reconstruction_mse is None

    batch = predictor.predict_batch([_payload()])
    assert len(batch) == 1
    assert batch[0].status == "DEGRADED"
    assert batch[0].is_anomaly is None


def test_data_drift_monitor_detects_shift(tmp_path) -> None:
    rng = np.random.default_rng(0)
    feature_names = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
    baseline = pd.DataFrame(rng.normal(0.0, 1.0, size=(300, 30)), columns=feature_names)
    baseline_path = tmp_path / "baseline_distribution.csv"
    baseline.to_csv(baseline_path, index=False)

    settings = Settings(
        BASELINE_DATA_PATH=baseline_path,
        DATASET_PATH=tmp_path / "creditcard.csv",
    )
    monitor = DataDriftMonitor(settings=settings)

    for row in rng.normal(2.0, 1.0, size=(150, 30)):
        monitor.record_inference_features(row.tolist())

    score, drifted_features, metrics = monitor.compute_drift()
    assert 0.0 < score <= 1.0
    assert drifted_features
    assert set(drifted_features) <= set(feature_names)
    assert all(metrics[name]["drifted"] for name in drifted_features)
