"""Statistical data drift engine powered by Kolmogorov-Smirnov and Wasserstein distance metrics.

Collects rolling production prediction payloads and evaluates statistical distribution divergence
against the baseline training dataset. Emits alerts when drift score exceeds 0.15.
"""

from __future__ import annotations

import collections
import json
import logging
import os
import time
from pathlib import Path
from typing import Callable, Deque, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from src.api.telemetry import DATA_DRIFT_SCORE_GAUGE
from src.config import Settings, get_settings
from src.schemas.telemetry import DriftAlertPayload

logger = logging.getLogger("sentinel.drift")


class DataDriftMonitor:
    """Rolling data drift engine comparing inference stream features with baseline distributions."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        on_drift_alert: Optional[Callable[[DriftAlertPayload], None]] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.on_drift_alert = on_drift_alert
        self.window_size = self.settings.DRIFT_WINDOW_SIZE
        self.drift_threshold = self.settings.DRIFT_SCORE_THRESHOLD

        # Feature buffer for rolling inference samples
        self.feature_buffer: Deque[List[float]] = collections.deque(maxlen=self.window_size)
        self.baseline_df: Optional[pd.DataFrame] = None
        self.feature_names: List[str] = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

        self._load_baseline()

    def _load_baseline(self) -> None:
        """Load baseline distribution from parquet or CSV."""
        if self.settings.BASELINE_DATA_PATH.exists():
            try:
                self.baseline_df = pd.read_parquet(self.settings.BASELINE_DATA_PATH)
                logger.info("Loaded baseline data from parquet: %s", self.settings.BASELINE_DATA_PATH)
            except Exception as e:
                logger.warning("Failed loading baseline parquet: %s", e)

        if self.baseline_df is None:
            csv_baseline = self.settings.BASELINE_DATA_PATH.with_suffix(".csv")
            if csv_baseline.exists():
                try:
                    self.baseline_df = pd.read_csv(csv_baseline)
                    logger.info("Loaded baseline data from CSV: %s", csv_baseline)
                except Exception as e:
                    logger.warning("Failed loading baseline CSV: %s", e)

        if self.baseline_df is None and self.settings.DATASET_PATH.exists():
            try:
                self.baseline_df = pd.read_csv(self.settings.DATASET_PATH)
                logger.info("Loaded baseline data from dataset CSV: %s", self.settings.DATASET_PATH)
            except Exception as e:
                logger.warning("Failed loading baseline CSV: %s", e)

    def record_inference_features(self, feature_vec: List[float]) -> None:
        """Append a 30-dimensional feature vector to the rolling production buffer."""
        if len(feature_vec) == self.settings.FEATURE_DIM:
            self.feature_buffer.append(feature_vec)

    def compute_drift(self) -> Tuple[float, List[str], Dict[str, Dict[str, float]]]:
        """Compute 2-sample Kolmogorov-Smirnov test and Wasserstein distance per feature.

        Returns:
            Tuple of:
              - overall_drift_score (fraction of features drifted, 0.0 to 1.0)
              - list of drifted feature names
              - per-feature metric dictionary
        """
        if len(self.feature_buffer) < 100:
            logger.info("Insufficient production samples for drift analysis (%d / 100 minimum)", len(self.feature_buffer))
            return 0.0, [], {}

        if self.baseline_df is None:
            logger.warning("No baseline dataset available for drift calculation.")
            return 0.0, [], {}

        current_arr = np.array(self.feature_buffer, dtype=np.float32)
        drifted_features: List[str] = []
        feature_metrics: Dict[str, Dict[str, float]] = {}

        # Subsample baseline if large
        baseline_sample_size = min(len(self.baseline_df), 2000)
        baseline_sub = self.baseline_df[self.feature_names].sample(n=baseline_sample_size, random_state=42)

        for idx, col in enumerate(self.feature_names):
            base_col = baseline_sub[col].values
            curr_col = current_arr[:, idx]

            # 1. Kolmogorov-Smirnov 2-sample test
            ks_stat, p_value = stats.ks_2samp(base_col, curr_col)

            # 2. Wasserstein Distance (normalized by baseline std)
            base_std = np.std(base_col)
            denom = base_std if base_std > 1e-6 else 1.0
            w_dist = stats.wasserstein_distance(base_col, curr_col) / denom

            is_feature_drifted = (p_value < 0.01) and (w_dist > 0.20)
            if is_feature_drifted:
                drifted_features.append(col)

            feature_metrics[col] = {
                "ks_stat": round(float(ks_stat), 4),
                "p_value": round(float(p_value), 5),
                "wasserstein_distance": round(float(w_dist), 4),
                "drifted": is_feature_drifted,
            }

        # Overall drift score: proportion of features exhibiting drift
        overall_score = len(drifted_features) / len(self.feature_names)
        DATA_DRIFT_SCORE_GAUGE.set(overall_score)

        logger.info(
            "Drift Evaluation: Overall Score: %.3f (Threshold: %.2f) | Drifted Features: %s",
            overall_score,
            self.drift_threshold,
            drifted_features,
        )

        # Trigger alert if drift score breaches threshold
        if overall_score > self.drift_threshold:
            alert = DriftAlertPayload(
                drift_score=round(overall_score, 4),
                threshold=self.drift_threshold,
                drifted_features=drifted_features,
                alert_level="CRITICAL",
                retrigger_training=True,
            )
            logger.critical(
                "[DATA DRIFT ALERT] Production stream drift score (%.3f) exceeds threshold (%.2f)! Features: %s",
                overall_score,
                self.drift_threshold,
                drifted_features,
            )
            if self.on_drift_alert:
                self.on_drift_alert(alert)

        return overall_score, drifted_features, feature_metrics
