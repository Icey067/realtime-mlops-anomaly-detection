"""FastAPI application for transaction anomaly scoring."""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from src.api.dependencies import close_redis_client, get_redis_client
from src.api.routes.health import router as health_router
from src.api.routes.predict import router as predict_router
from src.config import get_settings
from src.logging_config import configure_logging
from src.model.predictor import get_predictor

configure_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Warm the model and Redis client on startup; release the pool on shutdown."""
    settings = get_settings()
    logger.info("Starting %s v%s", settings.APP_NAME, settings.APP_VERSION)

    get_redis_client()
    predictor = get_predictor()
    if predictor.is_ready:
        logger.info("Model and scaler loaded")
    else:
        logger.error("Predictor started in degraded mode; /predict will return status=DEGRADED")

    yield

    await close_redis_client()
    logger.info("Shutdown complete")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="Sentinel Anomaly Detection API",
        description="Scores financial transactions with a quantized autoencoder.",
        version=settings.APP_VERSION,
        lifespan=lifespan,
    )

    allow_credentials = settings.CORS_ALLOW_CREDENTIALS and "*" not in settings.CORS_ALLOW_ORIGINS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ALLOW_ORIGINS,
        allow_credentials=allow_credentials,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "Authorization"],
    )

    app.include_router(predict_router)
    app.include_router(health_router)

    @app.get("/metrics", tags=["Telemetry"], summary="Prometheus metrics")
    def metrics_endpoint() -> Response:
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/", tags=["Root"])
    def root() -> dict:
        return {
            "name": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "status": "operational",
            "docs_url": "/docs",
            "metrics_url": "/metrics",
            "health_url": "/healthz",
        }

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "src.api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
    )
