"""High-performance CPU inference engine using quantized ONNX Runtime.

Executes sub-millisecond anomaly detection scoring against incoming transaction payloads.
Computes reconstruction MSE and flags transactions exceeding the calibrated 98th percentile cutoff.
"""

from __future__ import annotations

import json
import logging
import os
import time
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple, Union

import joblib
import numpy as np
import onnxruntime as ort

from src.config import Settings, get_settings
from src.schemas.transaction import AnomalyPredictionResult, TransactionPayload

logger = logging.getLogger("sentinel.predictor")


class AnomalyPredictor:
    """Production ONNX Runtime inference engine for sub-millisecond anomaly detection."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self.session: Optional[ort.InferenceSession] = None
        self.input_name: str = "input"
        self.scaler = None
        self.threshold: float = self.settings.ANOMALY_THRESHOLD_FALLBACK
        self.model_loaded: bool = False

        self._initialize_runtime()

    def _initialize_runtime(self) -> None:
        """Load quantized ONNX model, feature scaler, and anomaly threshold."""
        # 1. Load fitted scaler
        if self.settings.SCALER_PATH.exists():
            try:
                self.scaler = joblib.load(self.settings.SCALER_PATH)
                logger.info("Loaded feature scaler from: %s", self.settings.SCALER_PATH)
            except Exception as e:
                logger.warning("Failed to load scaler from %s (%s). Using identity scaler.", self.settings.SCALER_PATH, e)

        # 2. Load threshold metadata
        metadata_path = self.settings.MODEL_CHECKPOINT_PATH.parent / "model_metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    self.threshold = float(meta.get("anomaly_threshold", self.settings.ANOMALY_THRESHOLD_FALLBACK))
                logger.info("Loaded calibrated anomaly threshold: %.6f", self.threshold)
            except Exception as e:
                logger.warning("Failed loading metadata (%s). Using fallback: %.6f", e, self.threshold)

        # 3. Choose best available model (Quantized INT8 -> Standard FP32 ONNX)
        model_path = None
        if self.settings.ONNX_QUANTIZED_PATH.exists():
            model_path = self.settings.ONNX_QUANTIZED_PATH
            logger.info("Using Quantized INT8 ONNX model: %s", model_path)
        elif self.settings.ONNX_MODEL_PATH.exists():
            model_path = self.settings.ONNX_MODEL_PATH
            logger.info("Using Standard FP32 ONNX model: %s", model_path)

        if model_path:
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            opts.intra_op_num_threads = 2
            opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

            self.session = ort.InferenceSession(
                str(model_path),
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
            self.input_name = self.session.get_inputs()[0].name
            self.model_loaded = True
            logger.info("ONNX Runtime CPU InferenceSession initialized successfully.")
        else:
            logger.warning("No ONNX model found at %s or %s. Predictor awaiting model artifacts.", self.settings.ONNX_QUANTIZED_PATH, self.settings.ONNX_MODEL_PATH)

    def predict_one(self, payload: TransactionPayload) -> AnomalyPredictionResult:
        """Score a single transaction payload for anomaly probability."""
        start_time = time.perf_counter()

        # Extract 30-dim raw features: [Time, V1..V28, Amount]
        raw_vec = np.array([payload.to_feature_vector()], dtype=np.float32)

        # Scale features
        scaled_vec = self.scaler.transform(raw_vec).astype(np.float32) if self.scaler is not None else raw_vec

        if self.session is not None:
            # ONNX inference
            reconstructed = self.session.run(None, {self.input_name: scaled_vec})[0]
            mse = float(np.mean((scaled_vec - reconstructed) ** 2))
        else:
            # Fallback heuristic if model not yet compiled
            mse = 0.01

        inference_time_ms = (time.perf_counter() - start_time) * 1000
        is_anomaly = bool(mse > self.threshold)

        return AnomalyPredictionResult(
            transaction_id=payload.transaction_id,
            is_anomaly=is_anomaly,
            reconstruction_mse=round(mse, 6),
            threshold=round(self.threshold, 6),
            inference_latency_ms=round(inference_time_ms, 3),
        )

    def predict_batch(self, payloads: List[TransactionPayload]) -> List[AnomalyPredictionResult]:
        """Vectorized batch scoring for streaming consumers."""
        if not payloads:
            return []

        start_time = time.perf_counter()

        # Vectorize batch
        raw_batch = np.array([p.to_feature_vector() for p in payloads], dtype=np.float32)
        scaled_batch = self.scaler.transform(raw_batch).astype(np.float32) if self.scaler is not None else raw_batch

        if self.session is not None:
            reconstructed_batch = self.session.run(None, {self.input_name: scaled_batch})[0]
            mses = np.mean((scaled_batch - reconstructed_batch) ** 2, axis=1)
        else:
            mses = np.zeros(len(payloads), dtype=np.float32)

        total_latency_ms = (time.perf_counter() - start_time) * 1000
        per_item_latency_ms = total_latency_ms / len(payloads)

        results: List[AnomalyPredictionResult] = []
        for p, mse in zip(payloads, mses):
            mse_val = float(mse)
            results.append(
                AnomalyPredictionResult(
                    transaction_id=p.transaction_id,
                    is_anomaly=bool(mse_val > self.threshold),
                    reconstruction_mse=round(mse_val, 6),
                    threshold=round(self.threshold, 6),
                    inference_latency_ms=round(per_item_latency_ms, 3),
                )
            )

        return results


@lru_cache(maxsize=1)
def get_predictor() -> AnomalyPredictor:
    """Cached singleton instance of AnomalyPredictor."""
    return AnomalyPredictor()
