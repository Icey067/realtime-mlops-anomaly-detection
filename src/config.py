"""Application settings loaded from environment variables and optional .env file."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Base project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"


class Settings(BaseSettings):
    """Central configuration class backed by environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    APP_NAME: str = "Sentinel-MLOps-Anomaly-Engine"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    REDIS_HOST: str = Field(default="localhost", description="Redis server hostname")
    REDIS_PORT: int = Field(default=6379, description="Redis server port")
    REDIS_DB: int = Field(default=0, description="Redis database index")
    REDIS_PASSWORD: str | None = Field(default=None, description="Redis authentication password")
    REDIS_STREAM_KEY: str = Field(default="transactions:stream", description="Redis stream key")
    REDIS_DLQ_KEY: str = Field(default="transactions:dlq", description="Dead Letter Queue key")
    REDIS_CONSUMER_GROUP: str = Field(
        default="sentinel_consumers",
        description="Consumer group name",
    )
    REDIS_CONSUMER_NAME: str = Field(default="consumer_worker_1", description="Consumer identifier")
    REDIS_STREAM_MAXLEN: int = Field(
        default=100_000,
        description="Approximate stream length cap (XADD MAXLEN ~)",
    )

    @property
    def redis_url(self) -> str:
        """Construct a standard Redis connection URL."""
        if self.REDIS_PASSWORD:
            return f"redis://:{self.REDIS_PASSWORD}@{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

    PRODUCER_BATCH_SIZE: int = Field(default=50, description="Transaction count per batch")
    PRODUCER_EMIT_INTERVAL_SEC: float = Field(
        default=0.05,
        description="Seconds between producer batches",
    )
    PRODUCER_DRIFT_INJECTION_INTERVAL: int = Field(
        default=20,
        description="Every N batches, inject controlled feature drift into the synthetic stream",
    )
    PRODUCER_DRIFT_MAGNITUDE: float = Field(
        default=0.30,
        description="Magnitude of statistical drift (+30% mean shift)",
    )

    CONSUMER_BATCH_SIZE: int = Field(default=100, description="Max messages per XREADGROUP call")
    CONSUMER_BLOCK_MS: int = Field(default=2000, description="XREADGROUP block timeout in ms")
    LATENCY_BUDGET_MS: float = Field(
        default=10.0,
        description="Ingestion latency budget in milliseconds",
    )

    DATASET_PATH: Path = DATA_DIR / "creditcard.csv"
    BASELINE_DATA_PATH: Path = DATA_DIR / "baseline_distribution.csv"
    SCALER_PATH: Path = MODELS_DIR / "scaler.joblib"
    MODEL_CHECKPOINT_PATH: Path = MODELS_DIR / "autoencoder.pt"
    ONNX_MODEL_PATH: Path = MODELS_DIR / "autoencoder.onnx"
    ONNX_QUANTIZED_PATH: Path = MODELS_DIR / "autoencoder_int8.onnx"

    FEATURE_DIM: int = Field(default=30, description="Input feature dimension (Time, V1..V28, Amount)")
    HIDDEN_DIM: int = Field(default=16, description="First hidden layer dimension for Autoencoder")
    LATENT_DIM: int = Field(default=8, description="Bottleneck latent dimension for Autoencoder")
    BATCH_SIZE: int = Field(default=256, description="Training batch size")
    EPOCHS: int = Field(default=20, description="Training epochs")
    LEARNING_RATE: float = Field(default=1e-3, description="Optimizer learning rate")
    ANOMALY_PERCENTILE_THRESHOLD: float = Field(
        default=98.0,
        description="Percentile of validation reconstruction error used as the anomaly cutoff",
    )
    ANOMALY_THRESHOLD_FALLBACK: float = Field(
        default=0.045,
        description="Fallback reconstruction error threshold if not calibrated",
    )

    MLFLOW_TRACKING_URI: str = Field(default="http://localhost:5000", description="MLflow tracking server URI")
    MLFLOW_EXPERIMENT_NAME: str = Field(default="Sentinel-Anomaly-Detection", description="MLflow experiment name")
    DRIFT_SCORE_THRESHOLD: float = Field(
        default=0.15,
        description="Fraction of drifted features that triggers automated retraining",
    )
    DRIFT_WINDOW_SIZE: int = Field(
        default=1000,
        description="Rolling window size of production transaction inferences to test for drift",
    )
    DRIFT_SCORE_KEY: str = Field(
        default="sentinel:drift_score",
        description="Redis key where the consumer publishes the latest drift score",
    )

    API_HOST: str = Field(default="0.0.0.0", description="Host interface for FastAPI service")
    API_PORT: int = Field(default=8000, description="Port for FastAPI service")
    CORS_ALLOW_ORIGINS: list[str] = Field(
        default_factory=lambda: ["http://localhost:3000"],
        description="Origins allowed by the CORS middleware",
    )
    CORS_ALLOW_CREDENTIALS: bool = Field(
        default=False,
        description="Whether CORS requests may include credentials",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached singleton instance of application settings."""
    settings = Settings()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return settings
