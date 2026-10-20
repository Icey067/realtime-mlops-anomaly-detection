"""ONNX Runtime inference wrapper for transaction reconstruction scoring."""

from __future__ import annotations

import json
import logging
import time
from functools import lru_cache

import joblib
import numpy as np
import onnxruntime as ort

from src.config import Settings, get_settings
from src.schemas.transaction import AnomalyPredictionResult, TransactionPayload

logger = logging.getLogger(__name__)


class AnomalyPredictor:
    """Scores transactions with the exported autoencoder and calibrated MSE threshold."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.session: ort.InferenceSession | None = None
        self.input_name: str = "input"
        self.scaler = None
        self.threshold: float = self.settings.ANOMALY_THRESHOLD_FALLBACK
        self.threshold_loaded: bool = False
        self.model_loaded: bool = False

        self._initialize_runtime()

    @property
    def is_ready(self) -> bool:
        return (
            self.model_loaded
            and self.threshold_loaded
            and self.session is not None
            and self.scaler is not None
        )

    def _initialize_runtime(self) -> None:
        """Load the scaler, threshold metadata, and the best available ONNX graph."""
        self.threshold_loaded = False
        if self.settings.SCALER_PATH.exists():
            try:
                self.scaler = joblib.load(self.settings.SCALER_PATH)
                logger.info("Loaded feature scaler from %s", self.settings.SCALER_PATH)
            except Exception:
                logger.exception("Could not load scaler from %s", self.settings.SCALER_PATH)
                self.scaler = None
        else:
            logger.warning("Scaler not found at %s", self.settings.SCALER_PATH)

        metadata_path = self.settings.MODEL_CHECKPOINT_PATH.parent / "model_metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path, encoding="utf-8") as handle:
                    metadata = json.load(handle)
                self.threshold = float(metadata["anomaly_threshold"])
                self.threshold_loaded = True
                logger.info("Loaded anomaly threshold %.6f", self.threshold)
            except Exception:
                logger.exception("Could not read a calibrated threshold from %s", metadata_path)
        else:
            logger.warning("Model metadata not found at %s", metadata_path)

        model_path = None
        if self.settings.ONNX_QUANTIZED_PATH.exists():
            model_path = self.settings.ONNX_QUANTIZED_PATH
        elif self.settings.ONNX_MODEL_PATH.exists():
            model_path = self.settings.ONNX_MODEL_PATH

        if model_path is None:
            self.session = None
            self.model_loaded = False
            logger.error(
                "No ONNX model found at %s or %s; predictions are disabled",
                self.settings.ONNX_QUANTIZED_PATH,
                self.settings.ONNX_MODEL_PATH,
            )
            return

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = 2
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        try:
            session = ort.InferenceSession(
                str(model_path),
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )
        except Exception:
            self.session = None
            self.model_loaded = False
            logger.exception("Failed to create an ONNX session for %s", model_path)
            return

        self.session = session
        self.input_name = session.get_inputs()[0].name
        self.model_loaded = self.scaler is not None
        logger.info("Loaded ONNX model from %s", model_path)
        if self.scaler is None:
            logger.error("Predictions disabled because the feature scaler is unavailable")

    def _degraded_result(self, transaction_id: str, latency_ms: float) -> AnomalyPredictionResult:
        return AnomalyPredictionResult(
            transaction_id=transaction_id,
            status="DEGRADED",
            is_anomaly=None,
            reconstruction_mse=None,
            threshold=round(self.threshold, 6) if self.threshold_loaded else None,
            inference_latency_ms=round(latency_ms, 3),
        )

    def score_features(self, features: np.ndarray) -> np.ndarray:
        """Return per-row reconstruction MSE for a 2-D raw feature array."""
        if not self.is_ready or self.session is None or self.scaler is None:
            raise RuntimeError("Model artifacts are not loaded; scoring is unavailable")

        scaled = self.scaler.transform(features).astype(np.float32)
        reconstructed = self.session.run(None, {self.input_name: scaled})[0]
        return np.mean((scaled - reconstructed) ** 2, axis=1)

    def predict_one(self, payload: TransactionPayload) -> AnomalyPredictionResult:
        """Score one transaction, or return a DEGRADED result when artifacts are missing."""
        start = time.perf_counter()
        if not self.is_ready:
            return self._degraded_result(payload.transaction_id, (time.perf_counter() - start) * 1000)

        features = np.array([payload.to_feature_vector()], dtype=np.float32)
        mse = float(self.score_features(features)[0])
        elapsed_ms = (time.perf_counter() - start) * 1000

        return AnomalyPredictionResult(
            transaction_id=payload.transaction_id,
            status="OK",
            is_anomaly=mse > self.threshold,
            reconstruction_mse=round(mse, 6),
            threshold=round(self.threshold, 6),
            inference_latency_ms=round(elapsed_ms, 3),
        )

    def predict_batch(self, payloads: list[TransactionPayload]) -> list[AnomalyPredictionResult]:
        """Score a batch with one ONNX call, or mark every row DEGRADED."""
        if not payloads:
            return []

        start = time.perf_counter()
        if not self.is_ready:
            return [
                self._degraded_result(payload.transaction_id, (time.perf_counter() - start) * 1000)
                for payload in payloads
            ]

        features = np.array([payload.to_feature_vector() for payload in payloads], dtype=np.float32)
        scores = self.score_features(features)
        per_item_ms = ((time.perf_counter() - start) * 1000) / len(payloads)

        return [
            AnomalyPredictionResult(
                transaction_id=payload.transaction_id,
                status="OK",
                is_anomaly=float(score) > self.threshold,
                reconstruction_mse=round(float(score), 6),
                threshold=round(self.threshold, 6),
                inference_latency_ms=round(per_item_ms, 3),
            )
            for payload, score in zip(payloads, scores, strict=False)
        ]

    def hot_reload(self) -> bool:
        """Reload scaler, metadata, and ONNX session from disk. Returns readiness."""
        self._initialize_runtime()
        logger.info("Predictor reload complete; ready=%s", self.is_ready)
        return self.is_ready


@lru_cache(maxsize=1)
def get_predictor() -> AnomalyPredictor:
    """Return the process-wide predictor instance."""
    return AnomalyPredictor()
