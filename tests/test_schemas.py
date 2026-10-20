"""Tests for configuration and transaction schemas."""

import pytest
from pydantic import ValidationError
from src.config import get_settings
from src.schemas.transaction import TransactionPayload


def test_settings_loaded() -> None:
    settings = get_settings()
    assert settings.APP_NAME == "Sentinel-MLOps-Anomaly-Engine"
    assert settings.REDIS_STREAM_KEY == "transactions:stream"
    assert settings.FEATURE_DIM == 30
    assert settings.PRODUCER_DRIFT_MAGNITUDE == 0.30
    assert settings.LATENCY_BUDGET_MS == 10.0


def test_settings_paths_match_files_on_disk() -> None:
    settings = get_settings()
    assert settings.BASELINE_DATA_PATH.suffix == ".csv"
    assert settings.BASELINE_DATA_PATH.name == "baseline_distribution.csv"


def test_transaction_payload_valid() -> None:
    tx = TransactionPayload(
        user_id="usr_12345",
        amount=250.75,
        currency="usd",
        latitude=40.7128,
        longitude=-74.0060,
        time_step=12.5,
        v1=0.5,
        v2=-0.3,
    )
    assert tx.currency == "USD"
    assert tx.amount == 250.75
    assert len(tx.to_feature_vector()) == 30
    assert tx.to_feature_vector()[0] == 12.5
    assert tx.to_feature_vector()[-1] == 250.75


def test_transaction_payload_strict_bounds() -> None:
    with pytest.raises(ValidationError):
        TransactionPayload(user_id="usr_123", amount=-10.0, latitude=0.0, longitude=0.0)

    with pytest.raises(ValidationError):
        TransactionPayload(user_id="usr_123", amount=1_000_001.0, latitude=0.0, longitude=0.0)

    with pytest.raises(ValidationError):
        TransactionPayload(user_id="usr_123", amount=100.0, latitude=95.0, longitude=0.0)

    with pytest.raises(ValidationError):
        TransactionPayload(user_id="usr_123", amount=100.0, latitude=0.0, longitude=-185.0)


def test_redis_serialization_roundtrip() -> None:
    original = TransactionPayload(
        transaction_id="tx_test_999",
        user_id="usr_abc",
        amount=99.99,
        currency="USD",
        latitude=34.0522,
        longitude=-118.2437,
        time_step=100.0,
        v1=1.234,
        v28=-2.345,
    )

    redis_dict = original.to_redis_dict()
    assert isinstance(redis_dict["amount"], str)
    assert redis_dict["amount"] == "99.99"

    redis_raw_bytes = {k.encode("utf-8"): v.encode("utf-8") for k, v in redis_dict.items()}
    restored = TransactionPayload.from_redis_dict(redis_raw_bytes)

    assert restored.transaction_id == original.transaction_id
    assert restored.user_id == original.user_id
    assert pytest.approx(restored.amount) == original.amount
    assert pytest.approx(restored.latitude) == original.latitude
    assert pytest.approx(restored.longitude) == original.longitude
    assert pytest.approx(restored.v1) == original.v1
    assert pytest.approx(restored.v28) == original.v28
    assert restored.to_feature_vector() == original.to_feature_vector()
