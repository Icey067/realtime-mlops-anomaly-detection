"""High-throughput async Redis Stream producer with controlled drift injection.

Streams financial transactions into Redis Stream ('transactions:stream') using
either real Kaggle credit card records or high-fidelity synthetic transactions.
Periodically shifts statistical distributions by +30% to simulate real-world data drift.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import logging
import math
import os
import random
import signal
import sys
import time
from pathlib import Path
from typing import AsyncGenerator, Dict, List, Optional

import redis.asyncio as aioredis
from redis.asyncio.client import Redis

from src.config import Settings, get_settings
from src.schemas.transaction import TransactionPayload

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sentinel.producer")


class TransactionProducer:
    """Async Redis Stream producer capable of high-throughput streaming and drift simulation."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self.redis_client: Optional[Redis] = None
        self._running = False
        self._batch_counter = 0
        self._total_emitted = 0

    async def connect(self) -> None:
        """Establish async connection pool to Redis."""
        logger.info("Connecting to Redis at %s:%d (Stream: %s)", self.settings.REDIS_HOST, self.settings.REDIS_PORT, self.settings.REDIS_STREAM_KEY)
        self.redis_client = aioredis.from_url(
            self.settings.redis_url,
            encoding="utf-8",
            decode_responses=False,
            max_connections=20,
        )
        await self.redis_client.ping()
        logger.info("Redis connection established successfully.")

    async def close(self) -> None:
        """Close Redis connection and clean up resources."""
        if self.redis_client:
            await self.redis_client.close()
            logger.info("Redis producer connection closed.")

    def _generate_synthetic_transaction(
        self,
        time_offset: float,
        apply_drift: bool = False,
    ) -> TransactionPayload:
        """Generate a single realistic financial transaction payload.

        If apply_drift is True, the feature distribution mean is shifted by +30%.
        """
        drift_factor = (1.0 + self.settings.PRODUCER_DRIFT_MAGNITUDE) if apply_drift else 1.0

        # Amount follows a skewed log-normal distribution, shifted if drifting
        base_amount = random.lognormvariate(mu=3.5, sigma=1.2)
        amount = round(min(max(base_amount * drift_factor, 0.5), 999_999.0), 2)

        # Standard geo coordinates centered on major commerce hubs with noise
        lat = round(40.7128 + random.gauss(0, 0.05), 4)
        lon = round(-74.0060 + random.gauss(0, 0.05), 4)

        # 28 PCA features (V1-V28) normally distributed, shifted by drift factor
        v_features: Dict[str, float] = {}
        for i in range(1, 29):
            # Normal distribution with mean 0 (or shifted) and std dev 1.0
            mean_shift = (0.35 * drift_factor) if apply_drift else 0.0
            v_val = random.gauss(mean_shift, 1.0)
            v_features[f"v{i}"] = round(v_val, 5)

        return TransactionPayload(
            transaction_id=f"tx_{int(time.time() * 1000)}_{random.randint(1000, 9999)}",
            timestamp=time.time(),
            user_id=f"usr_{random.randint(100000, 999999)}",
            amount=amount,
            currency="USD",
            latitude=lat,
            longitude=lon,
            time_step=time_offset,
            **v_features,
        )

    async def _read_kaggle_csv(self, file_path: Path) -> AsyncGenerator[TransactionPayload, None]:
        """Iterate over real Kaggle creditcard.csv records asynchronously."""
        logger.info("Streaming transactions from CSV dataset: %s", file_path)
        with open(file_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            time_offset = 0.0
            for row in reader:
                if not self._running:
                    break
                v_feats = {f"v{i}": float(row.get(f"V{i}", 0.0)) for i in range(1, 29)}
                amount = max(float(row.get("Amount", 1.0)), 0.01)
                time_step = float(row.get("Time", time_offset))

                payload = TransactionPayload(
                    transaction_id=f"tx_csv_{int(time.time() * 1000)}_{random.randint(1000, 9999)}",
                    timestamp=time.time(),
                    user_id=f"usr_{random.randint(100000, 999999)}",
                    amount=amount,
                    currency="USD",
                    latitude=40.7128,
                    longitude=-74.0060,
                    time_step=time_step,
                    **v_feats,
                )
                yield payload

    async def publish_batch(self, payloads: List[TransactionPayload]) -> int:
        """Publish a batch of transactions to Redis Stream using a pipeline."""
        if not self.redis_client:
            raise RuntimeError("Redis client is not connected.")

        async with self.redis_client.pipeline(transaction=False) as pipe:
            for payload in payloads:
                pipe.xadd(
                    name=self.settings.REDIS_STREAM_KEY,
                    fields=payload.to_redis_dict(),
                    maxlen=self.settings.REDIS_STREAM_MAXLEN,
                    approximate=True,
                )
            results = await pipe.execute()
        return len(results)

    async def start_streaming(
        self,
        csv_file: Optional[Path] = None,
        max_batches: Optional[int] = None,
    ) -> None:
        """Run the main event loop generating and publishing transactions."""
        self._running = True
        logger.info(
            "Starting streaming engine (Batch Size: %d, Emit Interval: %.3fs, Drift Every %d batches)",
            self.settings.PRODUCER_BATCH_SIZE,
            self.settings.PRODUCER_EMIT_INTERVAL_SEC,
            self.settings.PRODUCER_DRIFT_INJECTION_INTERVAL,
        )

        time_offset = 0.0
        csv_generator = self._read_kaggle_csv(csv_file) if (csv_file and csv_file.exists()) else None

        try:
            while self._running:
                self._batch_counter += 1

                # Check if current batch should have controlled drift injected (+30% mean shift)
                is_drift_batch = (
                    self._batch_counter % self.settings.PRODUCER_DRIFT_INJECTION_INTERVAL == 0
                )
                if is_drift_batch:
                    logger.warning(
                        "[DRIFT INJECTION TRIGGERED] Batch %d: Applying +%.0f%% feature mean shift",
                        self._batch_counter,
                        self.settings.PRODUCER_DRIFT_MAGNITUDE * 100,
                    )

                batch_payloads: List[TransactionPayload] = []

                if csv_generator:
                    try:
                        for _ in range(self.settings.PRODUCER_BATCH_SIZE):
                            payload = await anext(csv_generator)
                            batch_payloads.append(payload)
                    except StopAsyncIteration:
                        logger.info("Reached end of CSV dataset, restarting generator...")
                        csv_generator = self._read_kaggle_csv(csv_file)
                else:
                    for _ in range(self.settings.PRODUCER_BATCH_SIZE):
                        time_offset += 0.5
                        payload = self._generate_synthetic_transaction(
                            time_offset=time_offset,
                            apply_drift=is_drift_batch,
                        )
                        batch_payloads.append(payload)

                start_pub = time.perf_counter()
                published_count = await self.publish_batch(batch_payloads)
                pub_latency_ms = (time.perf_counter() - start_pub) * 1000

                self._total_emitted += published_count

                if self._batch_counter % 10 == 0:
                    throughput = published_count / max(self.settings.PRODUCER_EMIT_INTERVAL_SEC, 0.001)
                    logger.info(
                        "Batch %d emitted | Size: %d tx | Pub Latency: %.2fms | Throughput: ~%.0f tx/sec | Total: %d tx",
                        self._batch_counter,
                        published_count,
                        pub_latency_ms,
                        throughput,
                        self._total_emitted,
                    )

                if max_batches and self._batch_counter >= max_batches:
                    logger.info("Reached target batch limit (%d). Halting producer.", max_batches)
                    break

                await asyncio.sleep(self.settings.PRODUCER_EMIT_INTERVAL_SEC)

        except asyncio.CancelledError:
            logger.info("Streaming loop received cancellation signal.")
        finally:
            self._running = False

    def stop(self) -> None:
        """Signal the streaming engine to stop gracefully."""
        self._running = False


async def main() -> None:
    """CLI entrypoint for running the Redis Stream producer standalone."""
    parser = argparse.ArgumentParser(description="Sentinel Redis Stream Transaction Producer")
    parser.add_argument("--csv", type=str, default=None, help="Path to Kaggle creditcard.csv dataset")
    parser.add_argument("--batches", type=int, default=None, help="Number of batches to emit before stopping")
    parser.add_argument("--batch-size", type=int, default=None, help="Transaction count per batch")
    parser.add_argument("--interval", type=float, default=None, help="Interval between batches in seconds")
    parser.add_argument("--drift-interval", type=int, default=None, help="Drift injection batch frequency")
    args = parser.parse_args()

    settings = get_settings()
    if args.batch_size:
        settings.PRODUCER_BATCH_SIZE = args.batch_size
    if args.interval:
        settings.PRODUCER_EMIT_INTERVAL_SEC = args.interval
    if args.drift_interval:
        settings.PRODUCER_DRIFT_INJECTION_INTERVAL = args.drift_interval

    producer = TransactionProducer(settings=settings)

    # Setup signal handlers for graceful shutdown
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, producer.stop)
        except NotImplementedError:
            # Signal handlers not fully supported on some Windows event loops
            pass

    csv_path = Path(args.csv) if args.csv else (settings.DATASET_PATH if settings.DATASET_PATH.exists() else None)

    try:
        await producer.connect()
        await producer.start_streaming(csv_file=csv_path, max_batches=args.batches)
    finally:
        await producer.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Producer terminated by user.")
        sys.exit(0)
