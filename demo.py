"""Score one normal and one anomalous transaction with the loaded ONNX model."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model.predictor import get_predictor
from src.schemas.transaction import TransactionPayload


def main() -> None:
    predictor = get_predictor()

    threshold = f"{predictor.threshold:.6f}" if predictor.threshold_loaded else "unavailable"
    print("Sentinel anomaly detection demo")
    print(f"Model loaded: {predictor.is_ready}")
    print(f"Anomaly threshold: {threshold}")

    if not predictor.is_ready:
        print("Model artifacts are missing; run training and quantization first.")
        return

    normal_tx = TransactionPayload(
        user_id="usr_normal_001",
        amount=45.50,
        currency="USD",
        latitude=40.7128,
        longitude=-74.0060,
        time_step=120.0,
        v1=0.15,
        v2=-0.22,
        v3=0.08,
    )
    fraud_tx = TransactionPayload(
        user_id="usr_suspect_999",
        amount=95000.00,
        currency="USD",
        latitude=40.7128,
        longitude=-74.0060,
        time_step=125.0,
        v1=18.5,
        v2=-22.0,
        v3=30.2,
        v4=-14.8,
    )

    for label, tx in (("normal", normal_tx), ("anomalous", fraud_tx)):
        result = predictor.predict_one(tx)
        decision = "anomaly" if result.is_anomaly else "normal"
        print(
            f"{label}: amount=${tx.amount:,.2f} | mse={result.reconstruction_mse:.6f} | "
            f"decision={decision} | latency={result.inference_latency_ms:.3f}ms"
        )


if __name__ == "__main__":
    main()
