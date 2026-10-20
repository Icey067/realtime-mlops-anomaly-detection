"""Evaluate the trained ONNX autoencoder against labeled fraud data.

Run with: python -m src.model.eval --dataset data/creditcard.csv

The script scores all labeled fraud records plus the held-out normal validation
split, reports PR-AUC/ROC-AUC and threshold sensitivity, and writes the results
to models/model_metadata.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.config import get_settings
from src.logging_config import configure_logging
from src.model.predictor import AnomalyPredictor

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
PERCENTILES = (90.0, 95.0, 98.0, 99.0)
SCORE_CHUNK_SIZE = 4096


def load_evaluation_data(
    dataset_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Return normal validation features, fraud features, labels, and class counts."""
    frame = pd.read_csv(dataset_path)

    missing = [column for column in FEATURE_COLUMNS + ["Class"] if column not in frame.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    normal = frame[frame["Class"] == 0]
    fraud = frame[frame["Class"] == 1]
    if fraud.empty:
        raise ValueError("Dataset contains no labeled fraud rows (Class == 1)")

    normal_features = normal[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    _, normal_validation = train_test_split(normal_features, test_size=0.20, random_state=42)
    fraud_features = fraud[FEATURE_COLUMNS].to_numpy(dtype=np.float32)

    labels = np.concatenate(
        [np.zeros(len(normal_validation), dtype=np.int8), np.ones(len(fraud_features), dtype=np.int8)]
    )
    return normal_validation, fraud_features, labels, len(normal_validation), len(fraud_features)


def _score_in_chunks(predictor: AnomalyPredictor, features: np.ndarray) -> np.ndarray:
    """Score features in fixed-size chunks to bound memory use."""
    scores = np.empty(len(features), dtype=np.float64)
    for start in range(0, len(features), SCORE_CHUNK_SIZE):
        stop = start + SCORE_CHUNK_SIZE
        scores[start:stop] = predictor.score_features(features[start:stop])
    return scores


def _threshold_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    predictions = scores > threshold
    true_positives = int(np.sum(predictions & (labels == 1)))
    false_positives = int(np.sum(predictions & (labels == 0)))
    false_negatives = int(np.sum(~predictions & (labels == 1)))
    true_negatives = int(np.sum(~predictions & (labels == 0)))
    return {
        "threshold": round(float(threshold), 6),
        "precision": round(float(precision_score(labels, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(labels, predictions, zero_division=0)), 4),
        "f1": round(float(f1_score(labels, predictions, zero_division=0)), 4),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "true_negatives": true_negatives,
        "false_negatives": false_negatives,
    }


def evaluate(dataset_path: Path | None = None) -> dict:
    """Evaluate the loaded model and persist metrics to model_metadata.json."""
    settings = get_settings()
    dataset = dataset_path or settings.DATASET_PATH
    if not dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset}")

    predictor = AnomalyPredictor(settings)
    if not predictor.is_ready:
        raise RuntimeError("Model artifacts are not loaded; cannot evaluate")

    normal_validation, fraud_features, labels, n_normal, n_fraud = load_evaluation_data(dataset)
    scores = _score_in_chunks(predictor, np.vstack([normal_validation, fraud_features]))
    normal_scores = scores[:n_normal]
    fraud_scores = scores[n_normal:]

    calibrated = _threshold_metrics(labels, scores, predictor.threshold)

    sensitivity: list[dict[str, float]] = []
    for percentile in PERCENTILES:
        cutoff = float(np.percentile(normal_scores, percentile))
        entry = _threshold_metrics(labels, scores, cutoff)
        entry["percentile"] = percentile
        sensitivity.append(entry)

    evaluation = {
        "dataset": dataset.name,
        "evaluated_at": time.time(),
        "normal_validation_samples": n_normal,
        "fraud_samples": n_fraud,
        "pr_auc": round(float(average_precision_score(labels, scores)), 4),
        "roc_auc": round(float(roc_auc_score(labels, scores)), 4),
        "normal_mse_mean": round(float(np.mean(normal_scores)), 6),
        "fraud_mse_mean": round(float(np.mean(fraud_scores)), 6),
        "at_calibrated_threshold": calibrated,
        "sensitivity": sensitivity,
    }

    metadata_path = settings.MODEL_CHECKPOINT_PATH.parent / "model_metadata.json"
    metadata: dict = {}
    if metadata_path.exists():
        with open(metadata_path, encoding="utf-8") as handle:
            metadata = json.load(handle)
    metadata["evaluation"] = evaluation
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    logger.info("Normal validation samples: %d | fraud samples: %d", n_normal, n_fraud)
    logger.info(
        "PR-AUC: %.4f | ROC-AUC: %.4f | normal MSE mean: %.6f | fraud MSE mean: %.6f",
        evaluation["pr_auc"],
        evaluation["roc_auc"],
        evaluation["normal_mse_mean"],
        evaluation["fraud_mse_mean"],
    )
    logger.info("At calibrated threshold %.6f:", predictor.threshold)
    logger.info(
        "  precision %.4f | recall %.4f | f1 %.4f | TP %d | FP %d | FN %d",
        calibrated["precision"],
        calibrated["recall"],
        calibrated["f1"],
        calibrated["true_positives"],
        calibrated["false_positives"],
        calibrated["false_negatives"],
    )
    logger.info("Sensitivity by percentile:")
    for entry in sensitivity:
        logger.info(
            "  p%.0f -> threshold %.6f | precision %.4f | recall %.4f | f1 %.4f",
            entry["percentile"],
            entry["threshold"],
            entry["precision"],
            entry["recall"],
            entry["f1"],
        )
    logger.info("Wrote evaluation metrics to %s", metadata_path)

    return evaluation


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Evaluate the ONNX model on labeled fraud data")
    parser.add_argument("--dataset", type=Path, default=None, help="Path to creditcard.csv")
    args = parser.parse_args()
    evaluate(args.dataset)


if __name__ == "__main__":
    main()
