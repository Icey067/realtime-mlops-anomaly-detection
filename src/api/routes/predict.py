"""Inference API routes for real-time anomaly detection."""

from __future__ import annotations

import logging
import time
from typing import List
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

logger = logging.getLogger("sentinel.api.predict")
router = APIRouter(prefix="", tags=["Inference"])


def record_telemetry_background(
    payload: TransactionPayload,
    result: AnomalyPredictionResult,
) -> None:
    """Async background task recording Prometheus metrics and updating the live dashboard buffer."""
    # Prometheus metric increments
    decision_label = "anomaly" if result.is_anomaly else "normal"
    ANOMALY_COUNT_TOTAL.labels(decision=decision_label).inc()
    INFERENCE_LATENCY_SECONDS.observe(result.inference_latency_ms / 1000.0)
    RECONSTRUCTION_MSE_HISTOGRAM.observe(result.reconstruction_mse)

    # Telemetry store update
    telemetry_item = {
        "transaction_id": result.transaction_id,
        "user_id": payload.user_id,
        "amount": payload.amount,
        "is_anomaly": result.is_anomaly,
        "reconstruction_mse": result.reconstruction_mse,
        "threshold": result.threshold,
        "inference_latency_ms": result.inference_latency_ms,
        "timestamp": result.timestamp,
    }
    telemetry_store.record_prediction(telemetry_item)


@router.post(
    "/predict",
    response_model=AnomalyPredictionResult,
    status_code=status.HTTP_200_OK,
    summary="Real-time transaction anomaly scoring",
    description="Ingests a validated financial transaction, runs ONNX quantized inference, and flags anomalies exceeding calibrated MSE.",
)
async def predict_transaction(
    payload: TransactionPayload,
    background_tasks: BackgroundTasks,
) -> AnomalyPredictionResult:
    """Score a single incoming transaction."""
    start_time = time.perf_counter()
    predictor = get_predictor()

    try:
        result = predictor.predict_one(payload)
        HTTP_REQUESTS_TOTAL.labels(method="POST", endpoint="/predict", status="200").inc()

        # Append background task for non-blocking telemetry and metrics logging
        background_tasks.add_task(record_telemetry_background, payload, result)

        return result

    except Exception as exc:
        HTTP_REQUESTS_TOTAL.labels(method="POST", endpoint="/predict", status="500").inc()
        logger.error("Error during prediction for tx %s: %s", payload.transaction_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failure: {str(exc)}",
        )


@router.post(
    "/predict/batch",
    response_model=List[AnomalyPredictionResult],
    status_code=status.HTTP_200_OK,
    summary="Vectorized batch transaction scoring",
)
async def predict_batch_transactions(
    payloads: List[TransactionPayload],
    background_tasks: BackgroundTasks,
) -> List[AnomalyPredictionResult]:
    """Score a batch of transactions in a single vectorized ONNX call."""
    if not payloads:
        return []

    predictor = get_predictor()
    try:
        results = predictor.predict_batch(payloads)
        HTTP_REQUESTS_TOTAL.labels(method="POST", endpoint="/predict/batch", status="200").inc()

        # Record metrics in background
        for p, r in zip(payloads, results):
            background_tasks.add_task(record_telemetry_background, p, r)

        return results
    except Exception as exc:
        HTTP_REQUESTS_TOTAL.labels(method="POST", endpoint="/predict/batch", status="500").inc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))
