"""Retraining orchestration triggered by drift alerts."""

from __future__ import annotations

import logging
import threading
import time

from src.config import Settings, get_settings
from src.model.predictor import get_predictor
from src.model.quantize import run_export_and_quantization
from src.model.train import train_autoencoder
from src.schemas.telemetry import DriftAlertPayload

logger = logging.getLogger(__name__)


class RetrainPipeline:
    """Runs retraining in a background thread with a cooldown and single-flight guard."""

    def __init__(self, settings: Settings | None = None, cooldown_seconds: float = 300.0) -> None:
        self.settings = settings or get_settings()
        self.cooldown_seconds = cooldown_seconds
        self._last_retrain_time: float = 0.0
        self._lock = threading.Lock()
        self._is_retraining = False

    def handle_drift_alert(self, alert: DriftAlertPayload) -> bool:
        """Start a retrain job if none is running and the cooldown has elapsed."""
        now = time.time()
        with self._lock:
            if self._is_retraining:
                logger.warning("Ignoring retrain trigger: a retrain is already running")
                return False

            elapsed = now - self._last_retrain_time
            if elapsed < self.cooldown_seconds:
                logger.info(
                    "Ignoring retrain trigger: %.1fs of cooldown remaining",
                    self.cooldown_seconds - elapsed,
                )
                return False

            self._is_retraining = True

        thread = threading.Thread(
            target=self._execute_retrain_job,
            args=(alert,),
            name="sentinel-retrain-worker",
            daemon=True,
        )
        thread.start()
        return True

    def _execute_retrain_job(self, alert: DriftAlertPayload) -> None:
        logger.info(
            "Starting retrain for drift score %.3f; drifted features: %s",
            alert.drift_score,
            alert.drifted_features,
        )
        start_time = time.perf_counter()

        try:
            logger.info("Step 1/3: retraining")
            train_autoencoder(settings=self.settings, epochs=10)

            logger.info("Step 2/3: exporting and quantizing")
            run_export_and_quantization(settings=self.settings)

            logger.info("Step 3/3: reloading predictor")
            get_predictor().hot_reload()

            self._last_retrain_time = time.time()
            logger.info("Retrain finished in %.2f seconds", time.perf_counter() - start_time)
        except Exception:
            logger.exception("Retrain failed")
        finally:
            with self._lock:
                self._is_retraining = False


retrain_pipeline = RetrainPipeline()
