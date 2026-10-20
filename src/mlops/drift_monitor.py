"""Distribution drift detection using two-sample KS tests and Wasserstein distance."""

from __future__ import annotations

import collections
import logging
from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy import stats

from src.api.telemetry import DATA_DRIFT_SCORE_GAUGE
from src.config import Settings, get_settings
from src.schemas.telemetry import DriftAlertPayload

logger = logging.getLogger(__name__)

FEATURE_NAMES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
MIN_SAMPLES = 100
MAX_BASELINE_SAMPLES = 2000


class DataDriftMonitor:
    """Compares recent inference features against a baseline dataset."""

    def __init__(
        self,
        settings: Settings | None = None,
        on_drift_alert: Callable[[DriftAlertPayload], None] | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.on_drift_alert = on_drift_alert
        self.window_size = self.settings.DRIFT_WINDOW_SIZE
        self.drift_threshold = self.settings.DRIFT_SCORE_THRESHOLD

        self.feature_buffer: collections.deque[list[float]] = collections.deque(maxlen=self.window_size)
        self.baseline_df: pd.DataFrame | None = None
        self.feature_names: list[str] = FEATURE_NAMES

        self._load_baseline()

    def _load_baseline(self) -> None:
        """Load the baseline distribution, falling back to the labeled dataset."""
        for path in (self.settings.BASELINE_DATA_PATH, self.settings.DATASET_PATH):
            if not path.exists():
                continue
            try:
                frame = pd.read_csv(path)
                missing = [column for column in self.feature_names if column not in frame.columns]
                if missing:
                    logger.warning("Baseline %s is missing columns %s", path, missing)
                    continue
                self.baseline_df = frame
                logger.info("Loaded drift baseline from %s (%d rows)", path, len(frame))
                return
            except Exception:
                logger.exception("Could not read drift baseline %s", path)

        logger.warning("No usable drift baseline found; drift checks will be skipped")

    def record_inference_features(self, feature_vec: list[float]) -> None:
        """Append a raw 30-feature vector to the rolling buffer."""
        if len(feature_vec) == self.settings.FEATURE_DIM:
            self.feature_buffer.append(list(feature_vec))

    def compute_drift(self) -> tuple[float, list[str], dict[str, dict[str, float]]]:
        """Return the drift score, drifted feature names, and per-feature statistics.

        The score is the fraction of features whose KS p-value is below 0.01 and
        whose normalized Wasserstein distance exceeds 0.20.
        """
        if len(self.feature_buffer) < MIN_SAMPLES:
            logger.debug("Not enough samples for drift analysis (%d/%d)", len(self.feature_buffer), MIN_SAMPLES)
            return 0.0, [], {}

        if self.baseline_df is None:
            return 0.0, [], {}

        current = np.asarray(self.feature_buffer, dtype=np.float32)
        baseline = self.baseline_df[self.feature_names].sample(
            n=min(len(self.baseline_df), MAX_BASELINE_SAMPLES),
            random_state=42,
        )

        drifted_features: list[str] = []
        feature_metrics: dict[str, dict[str, float]] = {}

        for index, column in enumerate(self.feature_names):
            baseline_values = baseline[column].to_numpy()
            current_values = current[:, index]

            ks_stat, p_value = stats.ks_2samp(baseline_values, current_values)
            baseline_std = float(np.std(baseline_values))
            wasserstein = stats.wasserstein_distance(baseline_values, current_values) / (
                baseline_std if baseline_std > 1e-6 else 1.0
            )

            drifted = bool(p_value < 0.01 and wasserstein > 0.20)
            if drifted:
                drifted_features.append(column)

            feature_metrics[column] = {
                "ks_stat": round(float(ks_stat), 4),
                "p_value": round(float(p_value), 5),
                "wasserstein_distance": round(float(wasserstein), 4),
                "drifted": drifted,
            }

        overall_score = len(drifted_features) / len(self.feature_names)
        DATA_DRIFT_SCORE_GAUGE.set(overall_score)
        logger.info(
            "Drift score %.3f (threshold %.2f); drifted features: %s",
            overall_score,
            self.drift_threshold,
            drifted_features,
        )

        if overall_score > self.drift_threshold:
            alert = DriftAlertPayload(
                drift_score=round(overall_score, 4),
                threshold=self.drift_threshold,
                drifted_features=drifted_features,
                alert_level="CRITICAL",
                retrigger_training=True,
            )
            logger.critical(
                "Drift threshold exceeded (%.3f > %.2f)",
                overall_score,
                self.drift_threshold,
            )
            if self.on_drift_alert:
                self.on_drift_alert(alert)

        return overall_score, drifted_features, feature_metrics
