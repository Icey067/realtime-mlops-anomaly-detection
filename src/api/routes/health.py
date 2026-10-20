"""Health, readiness, and live telemetry routes."""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, Response, status
from src.api.dependencies import get_redis_client
from src.api.telemetry import DATA_DRIFT_SCORE_GAUGE, HTTP_REQUESTS_TOTAL, telemetry_store
from src.config import get_settings
from src.model.predictor import get_predictor
from src.schemas.telemetry import SystemHealthResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Health"])
START_TIME = time.time()


@router.get(
    "/healthz",
    response_model=SystemHealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Liveness probe",
)
async def health_check() -> SystemHealthResponse:
    """Report whether the model and Redis are reachable."""
    settings = get_settings()
    predictor = get_predictor()

    try:
        redis_ok = bool(await get_redis_client().ping())
    except Exception:
        redis_ok = False

    HTTP_REQUESTS_TOTAL.labels(method="GET", endpoint="/healthz", status="200").inc()

    return SystemHealthResponse(
        status="healthy" if (predictor.is_ready and redis_ok) else "degraded",
        version=settings.APP_VERSION,
        redis_connected=redis_ok,
        model_loaded=predictor.is_ready,
        uptime_seconds=round(time.time() - START_TIME, 1),
    )


@router.get("/ready", status_code=status.HTTP_200_OK, summary="Readiness probe")
async def readiness_probe(response: Response) -> dict:
    """Return 503 until the scaler and model are loaded."""
    if not get_predictor().is_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"ready": False, "reason": "Model artifacts not loaded"}
    return {"ready": True}


@router.get("/api/telemetry/live", summary="Live dashboard telemetry")
async def get_live_telemetry() -> dict:
    """Return rolling throughput, latency, anomaly, and drift statistics."""
    settings = get_settings()
    metrics = telemetry_store.get_live_metrics()
    metrics["drift_threshold"] = settings.DRIFT_SCORE_THRESHOLD
    metrics["drift_score"] = None

    try:
        raw_score = await get_redis_client().get(settings.DRIFT_SCORE_KEY)
        if raw_score is not None:
            metrics["drift_score"] = round(float(raw_score), 4)
            DATA_DRIFT_SCORE_GAUGE.set(metrics["drift_score"])
    except Exception:
        logger.debug("Could not read drift score from Redis", exc_info=True)

    return metrics
