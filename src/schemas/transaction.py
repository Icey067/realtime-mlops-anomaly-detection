"""Transaction and prediction schemas used by the API and stream workers."""

from __future__ import annotations

import time
import uuid
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TransactionPayload(BaseModel):
    """Validated transaction payload with Kaggle-compatible feature fields."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "transaction_id": "tx_4a9e52e4-c361-4c7b-b5d1-ec4be672e391",
                "timestamp": 1729384000.123,
                "user_id": "usr_882914",
                "amount": 142.50,
                "currency": "USD",
                "latitude": 40.7128,
                "longitude": -74.0060,
                "time_step": 842.0,
                "v1": -0.852,
                "v2": 0.412,
                "v3": 1.250,
                "v4": -0.120,
                "v5": 0.054,
                "v6": -0.321,
                "v7": 0.890,
                "v8": -0.110,
                "v9": 0.450,
                "v10": -0.210,
                "v11": 0.330,
                "v12": -0.420,
                "v13": 0.150,
                "v14": -0.610,
                "v15": 0.220,
                "v16": 0.090,
                "v17": -0.180,
                "v18": 0.240,
                "v19": -0.050,
                "v20": 0.120,
                "v21": -0.080,
                "v22": 0.310,
                "v23": -0.140,
                "v24": 0.060,
                "v25": 0.190,
                "v26": -0.030,
                "v27": 0.040,
                "v28": 0.010,
            }
        },
    )

    transaction_id: str = Field(
        default_factory=lambda: f"tx_{uuid.uuid4()}",
        min_length=5,
        max_length=64,
        description="Unique identifier for the financial transaction",
    )
    timestamp: float = Field(
        default_factory=time.time,
        gt=0.0,
        description="High-resolution POSIX timestamp of transaction origin",
    )
    user_id: str = Field(
        ...,
        min_length=3,
        max_length=64,
        description="Encrypted or anonymized user account identifier",
    )

    amount: float = Field(
        ...,
        gt=0.0,
        le=1_000_000.0,
        description="Transaction amount in currency units (strictly bounded: (0, 1,000,000])",
    )
    currency: str = Field(
        default="USD",
        min_length=3,
        max_length=3,
        description="ISO 4217 3-letter currency code",
    )

    latitude: float = Field(
        ...,
        ge=-90.0,
        le=90.0,
        description="Latitude of transaction origin (-90 to +90 degrees)",
    )
    longitude: float = Field(
        ...,
        ge=-180.0,
        le=180.0,
        description="Longitude of transaction origin (-180 to +180 degrees)",
    )

    time_step: float = Field(
        default=0.0,
        ge=0.0,
        description="Elapsed seconds since stream or reference epoch (maps to Kaggle 'Time')",
    )

    v1: float = Field(default=0.0, ge=-150.0, le=150.0)
    v2: float = Field(default=0.0, ge=-150.0, le=150.0)
    v3: float = Field(default=0.0, ge=-150.0, le=150.0)
    v4: float = Field(default=0.0, ge=-150.0, le=150.0)
    v5: float = Field(default=0.0, ge=-150.0, le=150.0)
    v6: float = Field(default=0.0, ge=-150.0, le=150.0)
    v7: float = Field(default=0.0, ge=-150.0, le=150.0)
    v8: float = Field(default=0.0, ge=-150.0, le=150.0)
    v9: float = Field(default=0.0, ge=-150.0, le=150.0)
    v10: float = Field(default=0.0, ge=-150.0, le=150.0)
    v11: float = Field(default=0.0, ge=-150.0, le=150.0)
    v12: float = Field(default=0.0, ge=-150.0, le=150.0)
    v13: float = Field(default=0.0, ge=-150.0, le=150.0)
    v14: float = Field(default=0.0, ge=-150.0, le=150.0)
    v15: float = Field(default=0.0, ge=-150.0, le=150.0)
    v16: float = Field(default=0.0, ge=-150.0, le=150.0)
    v17: float = Field(default=0.0, ge=-150.0, le=150.0)
    v18: float = Field(default=0.0, ge=-150.0, le=150.0)
    v19: float = Field(default=0.0, ge=-150.0, le=150.0)
    v20: float = Field(default=0.0, ge=-150.0, le=150.0)
    v21: float = Field(default=0.0, ge=-150.0, le=150.0)
    v22: float = Field(default=0.0, ge=-150.0, le=150.0)
    v23: float = Field(default=0.0, ge=-150.0, le=150.0)
    v24: float = Field(default=0.0, ge=-150.0, le=150.0)
    v25: float = Field(default=0.0, ge=-150.0, le=150.0)
    v26: float = Field(default=0.0, ge=-150.0, le=150.0)
    v27: float = Field(default=0.0, ge=-150.0, le=150.0)
    v28: float = Field(default=0.0, ge=-150.0, le=150.0)

    @field_validator("currency")
    @classmethod
    def validate_currency_uppercase(cls, val: str) -> str:
        return val.upper()

    def to_feature_vector(self) -> list[float]:
        """Return [time_step, v1..v28, amount] as a 30-element list."""
        return [
            self.time_step,
            self.v1,
            self.v2,
            self.v3,
            self.v4,
            self.v5,
            self.v6,
            self.v7,
            self.v8,
            self.v9,
            self.v10,
            self.v11,
            self.v12,
            self.v13,
            self.v14,
            self.v15,
            self.v16,
            self.v17,
            self.v18,
            self.v19,
            self.v20,
            self.v21,
            self.v22,
            self.v23,
            self.v24,
            self.v25,
            self.v26,
            self.v27,
            self.v28,
            self.amount,
        ]

    def to_redis_dict(self) -> dict[str, str]:
        """Return a string-valued mapping suitable for Redis XADD."""
        return {k: str(v) for k, v in self.model_dump().items()}

    @classmethod
    def from_redis_dict(cls, raw: Mapping[str | bytes, str | bytes]) -> TransactionPayload:
        """Rebuild a payload from a Redis stream message, accepting bytes or str values."""
        decoded: dict[str, Any] = {}
        for k, v in raw.items():
            key = k.decode("utf-8") if isinstance(k, bytes) else str(k)
            val = v.decode("utf-8") if isinstance(v, bytes) else str(v)

            if key in {"amount", "latitude", "longitude", "timestamp", "time_step"} or key.startswith("v"):
                try:
                    decoded[key] = float(val)
                except ValueError:
                    decoded[key] = val
            else:
                decoded[key] = val

        return cls.model_validate(decoded)


class AnomalyPredictionResult(BaseModel):
    """Result of scoring one transaction.

    ``status`` is ``"OK"`` for a real prediction and ``"DEGRADED"`` when the model
    or scaler is unavailable. Degraded results have no anomaly decision.
    """

    transaction_id: str
    status: str = "OK"
    is_anomaly: bool | None = None
    reconstruction_mse: float | None = None
    threshold: float | None = None
    inference_latency_ms: float
    timestamp: float = Field(default_factory=time.time)
    drift_detected: bool = False
