"""Pydantic v2 schemas for real-time telemetry, Prometheus metrics, and drift monitoring."""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field
import time


class LatencyTelemetry(BaseModel):
    """Latency metrics breakdown for the ingestion and inference pipeline."""

    ingestion_latency_ms: float = Field(..., description="Producer emission to consumer receipt latency")
    validation_latency_ms: float = Field(..., description="Pydantic validation time in ms")
    inference_latency_ms: Optional[float] = Field(None, description="ONNX Runtime inference time in ms")
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
    """Payload emitted when Evidently AI detects statistical drift beyond threshold."""

    timestamp: float = Field(default_factory=time.time)
    drift_score: float = Field(..., description="Computed drift metric (e.g. KS statistic or Wasserstein distance)")
    threshold: float = Field(..., description="Configured drift cutoff threshold")
    drifted_features: List[str] = Field(default_factory=list, description="Names of individual features displaying drift")
    alert_level: str = "CRITICAL"
    retrigger_training: bool = True
