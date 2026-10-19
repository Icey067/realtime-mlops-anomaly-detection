"""Health, readiness, and liveness probe routes."""

from __future__ import annotations

import time
from fastapi import APIRouter, Response, status
import redis.asyncio as aioredis

from src.api.telemetry import HTTP_REQUESTS_TOTAL, telemetry_store
from src.config import get_settings
from src.model.predictor import get_predictor
from src.schemas.telemetry import SystemHealthResponse

router = APIRouter(prefix="", tags=["Health"])
START_TIME = time.time()


@router.get(
    "/healthz",
    response_model=SystemHealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Liveness probe",
)
async def health_check() -> SystemHealthResponse:
    """Return system liveness status, model load state, and Redis connection state."""
    settings = get_settings()
    predictor = get_predictor()

    redis_ok = True
    try:
        r = aioredis.from_url(settings.redis_url, socket_connect_timeout=1.0)
        await r.ping()
        await r.close()
    except Exception:
        redis_ok = False

    HTTP_REQUESTS_TOTAL.labels(method="GET", endpoint="/healthz", status="200").inc()

    return SystemHealthResponse(
        status="healthy" if (predictor.model_loaded and redis_ok) else "degraded",
        version=settings.APP_VERSION,
        redis_connected=redis_ok,
        model_loaded=predictor.model_loaded,
        uptime_seconds=round(time.time() - START_TIME, 1),
    )


@router.get("/ready", status_code=status.HTTP_200_OK, summary="Kubernetes Readiness Probe")
async def readiness_probe(response: Response) -> dict:
    """Readiness probe checking critical dependencies."""
    predictor = get_predictor()
    if not predictor.model_loaded:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"ready": False, "reason": "Model artifacts not loaded"}
    return {"ready": True}


@router.get("/api/telemetry/live", summary="Live dashboard telemetry")
async def get_live_telemetry() -> dict:
    """Provide real-time statistics, throughput, fraud rates, and recent transactions for React 19."""
    return telemetry_store.get_live_metrics()
