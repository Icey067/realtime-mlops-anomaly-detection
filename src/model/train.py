"""Training pipeline for Sentinel PyTorch Autoencoder.

Trains exclusively on normal financial records (Class == 0) from Kaggle creditcard.csv,
fits and persists StandardScaler, computes 98th percentile reconstruction threshold,
and logs metrics, hyperparams, and artifacts to MLflow.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.config import Settings, get_settings
from src.model.autoencoder import Autoencoder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sentinel.train")


def generate_synthetic_creditcard_data(num_samples: int = 15000) -> pd.DataFrame:
    """Generate high-fidelity synthetic Kaggle creditcard dataset if CSV is absent."""
    logger.info("Generating %d synthetic baseline credit card records...", num_samples)
    np.random.seed(42)

    # Time: continuous increasing seconds
    times = np.sort(np.random.uniform(0, 172800, size=num_samples))

    # 28 PCA features V1..V28 (multivariate normal with slight correlations)
    v_data = np.random.normal(loc=0.0, scale=1.0, size=(num_samples, 28))
    cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]

    # Amount: skewed log-normal distribution
    amounts = np.random.lognormal(mean=3.0, sigma=1.2, size=num_samples)

    # Class: 99.8% normal (0), 0.2% fraud (1)
    classes = np.zeros(num_samples, dtype=int)
    fraud_indices = np.random.choice(num_samples, size=int(num_samples * 0.002), replace=False)
    classes[fraud_indices] = 1

    df_dict = {"Time": times}
    for i in range(1, 29):
        df_dict[f"V{i}"] = v_data[:, i - 1]
    df_dict["Amount"] = amounts
    df_dict["Class"] = classes

    return pd.DataFrame(df_dict)


def load_and_preprocess_data(
    data_path: Path,
    baseline_path: Path,
    scaler_path: Path,
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Load dataset, filter normal transactions, fit scaler, and split train/val sets."""
    if data_path.exists():
        logger.info("Loading Kaggle dataset from: %s", data_path)
        df = pd.read_csv(data_path)
    else:
        logger.warning("Dataset not found at '%s'. Generating realistic baseline dataset.", data_path)
        df = generate_synthetic_creditcard_data(num_samples=25000)
        try:
            df.to_parquet(baseline_path, index=False)
            logger.info("Persisted baseline distribution to: %s", baseline_path)
        except Exception as e:
            csv_path = baseline_path.with_suffix(".csv")
            df.to_csv(csv_path, index=False)
            logger.info("Persisted baseline distribution to CSV fallback: %s (parquet unavailable: %s)", csv_path, e)

    # Verify required columns
    expected_cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
    if "Class" in df.columns:
        normal_mask = df["Class"] == 0
        df_normal = df[normal_mask]
        logger.info(
            "Filtered dataset for normal transactions (Class == 0): %d / %d records retained",
            len(df_normal),
            len(df),
        )
    else:
        df_normal = df
        logger.info("No 'Class' column detected. Using all %d records.", len(df_normal))

    # Extract 30 features: Time, V1..V28, Amount
    X = df_normal[expected_cols].values.astype(np.float32)

    # Train / Validation Split (80% / 20%)
    X_train, X_val = train_test_split(X, test_size=0.20, random_state=42)
    logger.info("Training set size: %d | Validation set size: %d", len(X_train), len(X_val))

    # Fit StandardScaler strictly on training set
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_val_scaled = scaler.transform(X_val).astype(np.float32)

    # Persist fitted scaler
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_path)
    logger.info("Persisted fitted StandardScaler to: %s", scaler_path)

    return X_train_scaled, X_val_scaled, scaler


def train_autoencoder(
    settings: Optional[Settings] = None,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    learning_rate: Optional[float] = None,
) -> Tuple[Autoencoder, float]:
    """Train unsupervised Autoencoder and calibrate anomaly threshold."""
    settings = settings or get_settings()
    num_epochs = epochs or settings.EPOCHS
    b_size = batch_size or settings.BATCH_SIZE
    lr = learning_rate or settings.LEARNING_RATE

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training Autoencoder on computation device: %s", device)

    # Load and scale training / validation data
    X_train, X_val, scaler = load_and_preprocess_data(
        data_path=settings.DATASET_PATH,
        baseline_path=settings.BASELINE_DATA_PATH,
        scaler_path=settings.SCALER_PATH,
    )

    train_dataset = TensorDataset(torch.from_numpy(X_train))
    val_dataset = TensorDataset(torch.from_numpy(X_val))

    train_loader = DataLoader(train_dataset, batch_size=b_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=b_size, shuffle=False)

    # Initialize Autoencoder (30 -> 16 -> 8 -> 16 -> 30)
    model = Autoencoder(
        input_dim=settings.FEATURE_DIM,
        hidden_dim=settings.HIDDEN_DIM,
        latent_dim=settings.LATENT_DIM,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    # Optional MLflow tracking
    mlflow_active = False
    try:
        import mlflow
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)
        mlflow.start_run(run_name=f"sentinel_train_{int(time.time())}")
        mlflow.log_params({
            "input_dim": settings.FEATURE_DIM,
            "hidden_dim": settings.HIDDEN_DIM,
            "latent_dim": settings.LATENT_DIM,
            "epochs": num_epochs,
            "batch_size": b_size,
            "learning_rate": lr,
            "device": str(device),
        })
        mlflow_active = True
        logger.info("MLflow tracking active (URI: %s)", settings.MLFLOW_TRACKING_URI)
    except Exception as mlflow_err:
        logger.warning("MLflow server unavailable (%s). Continuing training with local logging.", mlflow_err)

    best_val_loss = float("inf")

    logger.info("Beginning %d training epochs...", num_epochs)
    start_train_time = time.perf_counter()

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss_acc = 0.0

        for (batch_x,) in train_loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            reconstructed = model(batch_x)
            loss = criterion(reconstructed, batch_x)
            loss.backward()
            optimizer.step()
            train_loss_acc += loss.item() * len(batch_x)

        train_loss = train_loss_acc / len(X_train)

        # Validation evaluation
        model.eval()
        val_loss_acc = 0.0
        with torch.no_grad():
            for (val_batch_x,) in val_loader:
                val_batch_x = val_batch_x.to(device)
                val_reconstructed = model(val_batch_x)
                v_loss = criterion(val_reconstructed, val_batch_x)
                val_loss_acc += v_loss.item() * len(val_batch_x)

        val_loss = val_loss_acc / len(X_val)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        if epoch % 5 == 0 or epoch == num_epochs:
            logger.info("Epoch [%2d/%2d] | Train Loss: %.6f | Val Loss: %.6f", epoch, num_epochs, train_loss, val_loss)

        if mlflow_active:
            import mlflow
            mlflow.log_metrics({"train_mse": train_loss, "val_mse": val_loss}, step=epoch)

    train_duration = time.perf_counter() - start_train_time
    logger.info("Training completed in %.2f seconds.", train_duration)

    # Calculate 98th percentile MSE reconstruction threshold on validation set
    model.eval()
    val_errors: list[float] = []
    with torch.no_grad():
        for (v_batch,) in val_loader:
            v_batch = v_batch.to(device)
            reconstructed = model(v_batch)
            mse_per_sample = torch.mean((v_batch - reconstructed) ** 2, dim=1).cpu().numpy()
            val_errors.extend(mse_per_sample.tolist())

    val_errors_arr = np.array(val_errors)
    anomaly_threshold = float(np.percentile(val_errors_arr, settings.ANOMALY_PERCENTILE_THRESHOLD))
    logger.info(
        "Calibrated %.1fth percentile anomaly threshold: %.6f (Validation MSE Mean: %.6f, Max: %.6f)",
        settings.ANOMALY_PERCENTILE_THRESHOLD,
        anomaly_threshold,
        float(val_errors_arr.mean()),
        float(val_errors_arr.max()),
    )

    # Save PyTorch model checkpoint
    settings.MODEL_CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), settings.MODEL_CHECKPOINT_PATH)
    logger.info("Saved model weights to: %s", settings.MODEL_CHECKPOINT_PATH)

    # Save model metadata
    metadata = {
        "model_type": "Autoencoder",
        "input_dim": settings.FEATURE_DIM,
        "hidden_dim": settings.HIDDEN_DIM,
        "latent_dim": settings.LATENT_DIM,
        "anomaly_threshold": anomaly_threshold,
        "percentile": settings.ANOMALY_PERCENTILE_THRESHOLD,
        "val_mse_mean": float(val_errors_arr.mean()),
        "trained_at": time.time(),
        "train_samples": len(X_train),
    }
    metadata_path = settings.MODEL_CHECKPOINT_PATH.parent / "model_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved model metadata to: %s", metadata_path)

    if mlflow_active:
        import mlflow
        mlflow.log_metrics({
            "anomaly_threshold_p98": anomaly_threshold,
            "val_mse_mean": float(val_errors_arr.mean()),
            "best_val_loss": best_val_loss,
        })
        mlflow.log_artifact(str(settings.MODEL_CHECKPOINT_PATH))
        mlflow.log_artifact(str(metadata_path))
        mlflow.end_run()

    return model, anomaly_threshold


def main() -> None:
    """CLI entrypoint for running model training."""
    parser = argparse.ArgumentParser(description="Train Sentinel Autoencoder")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()

    train_autoencoder(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr)


if __name__ == "__main__":
    main()
