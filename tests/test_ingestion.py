"""Tests for stream producer drift injection."""

from src.config import Settings
from src.ingestion.producer import TransactionProducer
from src.schemas.transaction import TransactionPayload


def test_apply_drift_scales_amount_and_features() -> None:
    producer = TransactionProducer(Settings())
    payload = TransactionPayload(
        user_id="usr_drift_test",
        amount=100.0,
        latitude=40.0,
        longitude=-74.0,
        time_step=50.0,
        v1=2.0,
        v28=-1.0,
    )

    drifted = producer._apply_drift(payload)

    assert drifted.amount == 130.0
    assert drifted.v1 == 2.6
    assert drifted.v28 == -1.3
    assert drifted.time_step == payload.time_step
    assert drifted.transaction_id == payload.transaction_id
