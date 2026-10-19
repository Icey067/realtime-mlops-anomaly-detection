"""Production FastAPI Microservice for Sentinel MLOps Real-Time Anomaly Detection Engine.

Exposes sub-15ms prediction endpoints, custom Prometheus telemetry metrics (/metrics),
system liveness probes (/healthz), and real-time dashboard telemetry streams.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from src.api.routes.health import router as health_router
from src.api.routes.predict import router as predict_router
from src.config import get_settings
from src.model.predictor import get_predictor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sentinel.api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan context manager for startup and shutdown routines."""
    settings = get_settings()
    logger.info("Starting up %s v%s...", settings.APP_NAME, settings.APP_VERSION)

    # Pre-warm ONNX inference session and load models
    predictor = get_predictor()
    if predictor.model_loaded:
        logger.info("ONNX Runtime Predictor successfully pre-warmed.")
    else:
        logger.warning("Predictor started in degraded mode (no compiled ONNX model yet).")

    yield

    logger.info("Shutting down %s...", settings.APP_NAME)


def create_app() -> FastAPI:
    """Factory creating and configuring the FastAPI instance."""
    settings = get_settings()

    app = FastAPI(
        title="Sentinel MLOps Anomaly Detection Engine",
        description="High-frequency financial transaction anomaly detection powered by Quantized ONNX Runtime.",
        version=settings.APP_VERSION,
        lifespan=lifespan,
    )

    # Enable CORS for React 19 Frontend
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount Route Modules
    app.include_router(predict_router)
    app.include_router(health_router)

    # Prometheus Scraping Endpoint
    @app.get("/metrics", tags=["Telemetry"], summary="Prometheus metric endpoint")
    def metrics_endpoint() -> Response:
        """Expose Prometheus metrics for Prometheus server scraping."""
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
        workers=1,
        reload=settings.DEBUG,
    )
