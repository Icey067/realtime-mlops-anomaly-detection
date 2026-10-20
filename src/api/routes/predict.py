"""Inference routes."""

from __future__ import annotations

import logging

from fastapi import APIRouter, BackgroundTasks, HTTPException, status
from src.api.telemetry import (
    ANOMALY_COUNT_TOTAL,
    HTTP_REQUESTS_TOTAL,
    INFERENCE_LATENCY_SECONDS,
    RECONSTRUCTION_MSE_HISTOGRAM,
    telemetry_store,
)
from src.model.predictor import get_predictor
from src.schemas.transaction import AnomalyPredictionResult, TransactionPayload

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Inference"])

MAX_BATCH_SIZE = 1000


def record_telemetry_background(
    payload: TransactionPayload,
    result: AnomalyPredictionResult,
) -> None:
    """Update Prometheus metrics and the dashboard buffer after a response is sent."""
    if result.is_anomaly is None:
        decision_label = "degraded"
    else:
        decision_label = "anomaly" if result.is_anomaly else "normal"

    ANOMALY_COUNT_TOTAL.labels(decision=decision_label).inc()
    INFERENCE_LATENCY_SECONDS.observe(result.inference_latency_ms / 1000.0)
    if result.reconstruction_mse is not None:
        RECONSTRUCTION_MSE_HISTOGRAM.observe(result.reconstruction_mse)

    telemetry_store.record_prediction(
        {
            "transaction_id": result.transaction_id,
            "user_id": payload.user_id,
            "amount": payload.amount,
            "status": result.status,
            "is_anomaly": result.is_anomaly,
            "reconstruction_mse": result.reconstruction_mse,
            "threshold": result.threshold,
            "inference_latency_ms": result.inference_latency_ms,
            "timestamp": result.timestamp,
        }
    )


@router.post(
    "/predict",
    response_model=AnomalyPredictionResult,
    status_code=status.HTTP_200_OK,
    summary="Score one transaction",
)
async def predict_transaction(
    payload: TransactionPayload,
    background_tasks: BackgroundTasks,
) -> AnomalyPredictionResult:
    """Score a transaction and return the reconstruction error and decision."""
    predictor = get_predictor()

    try:
        result = predictor.predict_one(payload)
    except Exception as exc:
        HTTP_REQUESTS_TOTAL.labels(method="POST", endpoint="/predict", status="500").inc()
        logger.exception("Prediction failed for transaction %s", payload.transaction_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Inference failure",
        ) from exc

    HTTP_REQUESTS_TOTAL.labels(method="POST", endpoint="/predict", status="200").inc()
    background_tasks.add_task(record_telemetry_background, payload, result)
    return result


@router.post(
    "/predict/batch",
    response_model=list[AnomalyPredictionResult],
    status_code=status.HTTP_200_OK,
    summary="Score up to 1000 transactions",
)
async def predict_batch_transactions(
    payloads: list[TransactionPayload],
    background_tasks: BackgroundTasks,
) -> list[AnomalyPredictionResult]:
    """Score a batch in one ONNX call."""
    if not payloads:
        return []
    if len(payloads) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=422,
            detail=f"Batch size {len(payloads)} exceeds the limit of {MAX_BATCH_SIZE}",
        )

    predictor = get_predictor()
    try:
        results = predictor.predict_batch(payloads)
    except Exception as exc:
        HTTP_REQUESTS_TOTAL.labels(method="POST", endpoint="/predict/batch", status="500").inc()
        logger.exception("Batch prediction failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Inference failure",
        ) from exc

    HTTP_REQUESTS_TOTAL.labels(method="POST", endpoint="/predict/batch", status="200").inc()
    for payload, result in zip(payloads, results, strict=False):
        background_tasks.add_task(record_telemetry_background, payload, result)
    return results
