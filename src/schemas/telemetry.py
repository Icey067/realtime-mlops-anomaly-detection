"""Telemetry and drift payload schemas."""

import time

from pydantic import BaseModel, Field


class LatencyTelemetry(BaseModel):
    """Latency breakdown for ingestion and inference."""

    ingestion_latency_ms: float = Field(..., description="Producer emission to consumer receipt latency")
    validation_latency_ms: float = Field(..., description="Pydantic validation time in ms")
    inference_latency_ms: float | None = Field(None, description="ONNX Runtime inference time in ms")
    total_pipeline_latency_ms: float = Field(..., description="End-to-end processing latency")


class SystemHealthResponse(BaseModel):
    """Health check response schema."""

    status: str = "healthy"
    version: str = "1.0.0"
    redis_connected: bool = True
    model_loaded: bool = False
    timestamp: float = Field(default_factory=time.time)
    uptime_seconds: float = 0.0


class DriftAlertPayload(BaseModel):
    """Emitted when the drift score exceeds the configured threshold."""

    timestamp: float = Field(default_factory=time.time)
    drift_score: float = Field(..., description="Fraction of features flagged as drifted")
    threshold: float = Field(..., description="Configured drift cutoff")
    drifted_features: list[str] = Field(default_factory=list)
    alert_level: str = "CRITICAL"
    retrigger_training: bool = True
