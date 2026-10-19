"""Automated retraining pipeline triggered on statistical data drift alerts.

Retrains the PyTorch Autoencoder on fresh baseline distributions, exports and quantizes
new INT8 ONNX weights, and hot-reloads the active inference session with zero service downtime.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

from src.config import Settings, get_settings
from src.model.predictor import get_predictor
from src.model.quantize import run_export_and_quantization
from src.model.train import train_autoencoder
from src.schemas.telemetry import DriftAlertPayload

logger = logging.getLogger("sentinel.retrain")


class RetrainPipeline:
    """Manages automated model retraining and hot-reloading with cooldown safeguards."""

    def __init__(self, settings: Optional[Settings] = None, cooldown_seconds: float = 300.0) -> None:
        self.settings = settings or get_settings()
        self.cooldown_seconds = cooldown_seconds
        self._last_retrain_time: float = 0.0
        self._lock = threading.Lock()
        self._is_retraining = False

    def handle_drift_alert(self, alert: DriftAlertPayload) -> bool:
        """Invoked when the Drift Monitor identifies critical distribution divergence."""
        now = time.time()
        with self._lock:
            if self._is_retraining:
                logger.warning("Retrain trigger ignored: Retraining is already in progress.")
                return False

            elapsed = now - self._last_retrain_time
            if elapsed < self.cooldown_seconds:
                logger.info(
                    "Retrain trigger ignored: In cooldown period (%.1fs remaining / %.1fs)",
                    self.cooldown_seconds - elapsed,
                    self.cooldown_seconds,
                )
                return False

            self._is_retraining = True

        # Spawn retraining in a background thread to prevent blocking event loops
        thread = threading.Thread(
            target=self._execute_retrain_job,
            args=(alert,),
            name="sentinel-retrain-worker",
            daemon=True,
        )
        thread.start()
        return True

    def _execute_retrain_job(self, alert: DriftAlertPayload) -> None:
        """Worker executing complete retrain, quantization, and predictor reload cycle."""
        logger.info(
            "[AUTOMATED RETRAIN START] Triggered by drift score: %.3f | Drifted features: %s",
            alert.drift_score,
            alert.drifted_features,
        )
        start_time = time.perf_counter()

        try:
            # 1. Train updated PyTorch Autoencoder
            logger.info("Step 1/3: Retraining Autoencoder on updated distribution...")
            train_autoencoder(settings=self.settings, epochs=10)

            # 2. Export and Quantize to INT8 ONNX
            logger.info("Step 2/3: Exporting and quantizing new INT8 ONNX graph...")
            run_export_and_quantization(settings=self.settings)

            # 3. Hot-reload active Predictor instance
            logger.info("Step 3/3: Hot-reloading active predictor runtime...")
            predictor = get_predictor()
            predictor._initialize_runtime()

            duration = time.perf_counter() - start_time
            self._last_retrain_time = time.time()
            logger.info("[AUTOMATED RETRAIN SUCCESS] Pipeline completed in %.2f seconds! Zero downtime.", duration)

        except Exception as exc:
            logger.error("[AUTOMATED RETRAIN FAILED] Execution error: %s", exc, exc_info=True)
        finally:
            with self._lock:
                self._is_retraining = False


# Global retrain pipeline singleton
retrain_pipeline = RetrainPipeline()
