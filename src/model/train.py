"""Train the autoencoder on normal transactions and calibrate the anomaly threshold."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.config import Settings, get_settings
from src.logging_config import configure_logging
from src.model.autoencoder import Autoencoder

logger = logging.getLogger(__name__)


def generate_synthetic_creditcard_data(num_samples: int = 15000) -> pd.DataFrame:
    """Generate a synthetic dataset with the same schema as creditcard.csv."""
    logger.info("Generating %d synthetic credit card records", num_samples)
    np.random.seed(42)

    times = np.sort(np.random.uniform(0, 172800, size=num_samples))
    v_data = np.random.normal(loc=0.0, scale=1.0, size=(num_samples, 28))
    amounts = np.random.lognormal(mean=3.0, sigma=1.2, size=num_samples)

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
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Load the dataset, keep normal transactions, fit the scaler, and split train/val."""
    if data_path.exists():
        logger.info("Loading dataset from %s", data_path)
        df = pd.read_csv(data_path)
    else:
        logger.warning("Dataset not found at %s; generating a synthetic fallback", data_path)
        df = generate_synthetic_creditcard_data(num_samples=25000)

    expected_cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
    if "Class" in df.columns:
        df_normal = df[df["Class"] == 0]
        logger.info("Kept %d of %d normal records", len(df_normal), len(df))
    else:
        df_normal = df
        logger.info("No 'Class' column found; using all %d records", len(df_normal))

    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_sample = df_normal.sample(n=min(len(df_normal), 25000), random_state=42)
    baseline_sample.to_csv(baseline_path, index=False)
    logger.info("Wrote drift baseline with %d normal rows to %s", len(baseline_sample), baseline_path)

    X = df_normal[expected_cols].values.astype(np.float32)
    X_train, X_val = train_test_split(X, test_size=0.20, random_state=42)
    logger.info("Training set: %d rows; validation set: %d rows", len(X_train), len(X_val))

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_val_scaled = scaler.transform(X_val).astype(np.float32)

    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_path)
    logger.info("Persisted fitted StandardScaler to: %s", scaler_path)

    return X_train_scaled, X_val_scaled, scaler


def train_autoencoder(
    settings: Settings | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
) -> tuple[Autoencoder, float]:
    """Train the autoencoder and calibrate the anomaly threshold from validation MSE."""
    settings = settings or get_settings()
    num_epochs = epochs or settings.EPOCHS
    b_size = batch_size or settings.BATCH_SIZE
    lr = learning_rate or settings.LEARNING_RATE

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on %s", device)

    X_train, X_val, _ = load_and_preprocess_data(
        data_path=settings.DATASET_PATH,
        baseline_path=settings.BASELINE_DATA_PATH,
        scaler_path=settings.SCALER_PATH,
    )

    train_dataset = TensorDataset(torch.from_numpy(X_train))
    val_dataset = TensorDataset(torch.from_numpy(X_val))

    train_loader = DataLoader(train_dataset, batch_size=b_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=b_size, shuffle=False)

    model = Autoencoder(
        input_dim=settings.FEATURE_DIM,
        hidden_dim=settings.HIDDEN_DIM,
        latent_dim=settings.LATENT_DIM,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

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
        logger.info("MLflow tracking at %s", settings.MLFLOW_TRACKING_URI)
    except Exception as mlflow_err:
        logger.warning("MLflow unavailable (%s); continuing without tracking", mlflow_err)

    best_val_loss = float("inf")

    logger.info("Training for %d epochs", num_epochs)
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
    logger.info("Training finished in %.2f seconds", train_duration)

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
        "Threshold (p%.1f): %.6f | validation MSE mean: %.6f | max: %.6f",
        settings.ANOMALY_PERCENTILE_THRESHOLD,
        anomaly_threshold,
        float(val_errors_arr.mean()),
        float(val_errors_arr.max()),
    )

    settings.MODEL_CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), settings.MODEL_CHECKPOINT_PATH)
    logger.info("Saved weights to %s", settings.MODEL_CHECKPOINT_PATH)

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
    logger.info("Saved metadata to %s", metadata_path)

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
    configure_logging()
    parser = argparse.ArgumentParser(description="Train the Sentinel autoencoder")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()

    train_autoencoder(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr)


if __name__ == "__main__":
    main()
