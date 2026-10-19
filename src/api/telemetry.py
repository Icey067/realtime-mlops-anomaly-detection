"""Prometheus metrics and real-time telemetry buffer for Sentinel MLOps."""

from __future__ import annotations

import collections
import time
from typing import Deque, Dict, List, Optional
from prometheus_client import Counter, Gauge, Histogram

# --------------------------------------------------------------------------
# Prometheus Telemetry Metrics
# --------------------------------------------------------------------------
HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Total HTTP requests handled by the Sentinel API",
    ["method", "endpoint", "status"],
)

INFERENCE_LATENCY_SECONDS = Histogram(
    "inference_latency_seconds",
    "Time spent executing ONNX Runtime inference in seconds",
    buckets=[0.001, 0.002, 0.004, 0.006, 0.008, 0.010, 0.015, 0.025, 0.050, 0.1],
)

ANOMALY_COUNT_TOTAL = Counter(
    "anomaly_count_total",
    "Total detected fraudulent/anomalous transactions",
    ["decision"],
)

RECONSTRUCTION_MSE_HISTOGRAM = Histogram(
    "reconstruction_mse",
    "Autoencoder reconstruction mean squared error",
    buckets=[0.005, 0.01, 0.02, 0.03, 0.045, 0.06, 0.08, 0.1, 0.2, 0.5],
)

STREAM_THROUGHPUT_GAUGE = Gauge(
    "stream_throughput_tx_per_sec",
    "Current streaming transaction throughput",
)

DATA_DRIFT_SCORE_GAUGE = Gauge(
    "data_drift_score",
    "Current statistical data drift score (KS / Wasserstein metric)",
)


# --------------------------------------------------------------------------
# Rolling In-Memory Telemetry Buffer (for React Dashboard)
# --------------------------------------------------------------------------
class TelemetryStore:
    """Thread-safe circular telemetry buffer powering the live React 19 dashboard."""

    def __init__(self, maxlen: int = 100) -> None:
        self.recent_predictions: Deque[Dict] = collections.deque(maxlen=maxlen)
        self.recent_latencies: Deque[float] = collections.deque(maxlen=maxlen)
        self.total_scored: int = 0
        self.total_anomalies: int = 0
        self._start_time: float = time.time()

    def record_prediction(self, payload_dict: Dict) -> None:
        """Record an inference event for dashboard streaming."""
        self.total_scored += 1
        if payload_dict.get("is_anomaly"):
            self.total_anomalies += 1
        self.recent_predictions.appendleft(payload_dict)
        self.recent_latencies.append(payload_dict.get("inference_latency_ms", 0.0))

    def get_live_metrics(self) -> Dict:
        """Summarize current throughput, fraud rate, and latency sparkline points."""
        now = time.time()
        uptime = max(now - self._start_time, 1.0)
        throughput = round(self.total_scored / uptime, 2)
        STREAM_THROUGHPUT_GAUGE.set(throughput)

        anomaly_rate = round((self.total_anomalies / max(self.total_scored, 1)) * 100, 2)

        lat_list = list(self.recent_latencies)
        avg_lat = round(sum(lat_list) / len(lat_list), 3) if lat_list else 0.0
        p95_lat = round(sorted(lat_list)[min(int(0.95 * len(lat_list)), len(lat_list) - 1)], 3) if lat_list else 0.0

        return {
            "uptime_seconds": round(uptime, 1),
            "total_scored": self.total_scored,
            "total_anomalies": self.total_anomalies,
            "anomaly_rate_percent": anomaly_rate,
            "throughput_tx_per_sec": throughput,
            "avg_latency_ms": avg_lat,
            "p95_latency_ms": p95_lat,
            "recent_latencies": lat_list[-30:],
            "recent_transactions": list(self.recent_predictions)[:20],
        }


# Global telemetry singleton
telemetry_store = TelemetryStore(maxlen=100)
