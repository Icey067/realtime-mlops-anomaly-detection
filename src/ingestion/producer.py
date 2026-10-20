"""Redis Stream producer for replaying transactions and injecting feature drift."""

from __future__ import annotations

import argparse
import asyncio
import csv
import logging
import random
import signal
import sys
import time
from collections.abc import AsyncGenerator
from pathlib import Path

import redis.asyncio as aioredis
from redis.asyncio.client import Redis

from src.config import Settings, get_settings
from src.logging_config import configure_logging
from src.schemas.transaction import TransactionPayload

logger = logging.getLogger(__name__)


class TransactionProducer:
    """Publishes transaction payloads to a Redis stream."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.redis_client: Redis | None = None
        self._running = False
        self._batch_counter = 0
        self._total_emitted = 0

    async def connect(self) -> None:
        """Open the Redis connection pool."""
        logger.info("Connecting to Redis at %s:%d", self.settings.REDIS_HOST, self.settings.REDIS_PORT)
        self.redis_client = aioredis.from_url(
            self.settings.redis_url,
            encoding="utf-8",
            decode_responses=False,
            max_connections=20,
        )
        await self.redis_client.ping()

    async def close(self) -> None:
        """Close the Redis connection pool."""
        if self.redis_client:
            await self.redis_client.aclose()

    def _generate_synthetic_transaction(self, time_offset: float) -> TransactionPayload:
        """Generate one synthetic payload."""
        base_amount = random.lognormvariate(mu=3.5, sigma=1.2)
        amount = round(min(max(base_amount, 0.5), 999_999.0), 2)

        lat = round(40.7128 + random.gauss(0, 0.05), 4)
        lon = round(-74.0060 + random.gauss(0, 0.05), 4)

        v_features: dict[str, float] = {}
        for i in range(1, 29):
            v_features[f"v{i}"] = round(random.gauss(0.0, 1.0), 5)

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

    def _apply_drift(self, payload: TransactionPayload) -> TransactionPayload:
        """Scale amount and V features by the configured drift factor."""
        factor = 1.0 + self.settings.PRODUCER_DRIFT_MAGNITUDE
        updates: dict[str, float] = {
            "amount": round(min(max(payload.amount * factor, 0.01), 999_999.0), 2)
        }
        updates.update({f"v{i}": round(getattr(payload, f"v{i}") * factor, 5) for i in range(1, 29)})
        return payload.model_copy(update=updates)

    async def _read_kaggle_csv(self, file_path: Path) -> AsyncGenerator[TransactionPayload, None]:
        """Replay records from a creditcard.csv file."""
        logger.info("Replaying transactions from %s", file_path)
        with open(file_path, encoding="utf-8") as f:
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

    async def publish_batch(self, payloads: list[TransactionPayload]) -> int:
        """Publish a batch through a non-transactional pipeline."""
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
        csv_file: Path | None = None,
        max_batches: int | None = None,
    ) -> None:
        """Generate and publish batches until stopped or the batch limit is reached."""
        self._running = True
        logger.info(
            "Streaming with batch size %d every %.3fs; drift every %d batches",
            self.settings.PRODUCER_BATCH_SIZE,
            self.settings.PRODUCER_EMIT_INTERVAL_SEC,
            self.settings.PRODUCER_DRIFT_INJECTION_INTERVAL,
        )

        time_offset = 0.0
        csv_generator = self._read_kaggle_csv(csv_file) if (csv_file and csv_file.exists()) else None

        try:
            while self._running:
                self._batch_counter += 1

                is_drift_batch = (
                    self._batch_counter % self.settings.PRODUCER_DRIFT_INJECTION_INTERVAL == 0
                )
                if is_drift_batch:
                    logger.warning(
                        "Batch %d: applying a +%.0f%% feature mean shift",
                        self._batch_counter,
                        self.settings.PRODUCER_DRIFT_MAGNITUDE * 100,
                    )

                batch_payloads: list[TransactionPayload] = []

                if csv_generator:
                    try:
                        for _ in range(self.settings.PRODUCER_BATCH_SIZE):
                            payload = await anext(csv_generator)
                            batch_payloads.append(payload)
                    except StopAsyncIteration:
                        logger.info("End of CSV reached; restarting from the first row")
                        csv_generator = self._read_kaggle_csv(csv_file)
                else:
                    for _ in range(self.settings.PRODUCER_BATCH_SIZE):
                        time_offset += 0.5
                        batch_payloads.append(self._generate_synthetic_transaction(time_offset))

                if is_drift_batch:
                    batch_payloads = [self._apply_drift(payload) for payload in batch_payloads]

                start_pub = time.perf_counter()
                published_count = await self.publish_batch(batch_payloads)
                pub_latency_ms = (time.perf_counter() - start_pub) * 1000

                self._total_emitted += published_count

                if self._batch_counter % 10 == 0:
                    throughput = published_count / max(self.settings.PRODUCER_EMIT_INTERVAL_SEC, 0.001)
                    logger.info(
                        "Batch %d | Size: %d tx | Publish: %.2fms | "
                        "Throughput: ~%.0f tx/sec | Total: %d tx",
                        self._batch_counter,
                        published_count,
                        pub_latency_ms,
                        throughput,
                        self._total_emitted,
                    )

                if max_batches and self._batch_counter >= max_batches:
                    logger.info("Reached batch limit (%d); stopping", max_batches)
                    break

                await asyncio.sleep(self.settings.PRODUCER_EMIT_INTERVAL_SEC)

        except asyncio.CancelledError:
            logger.info("Stream loop cancelled")
        finally:
            self._running = False

    def stop(self) -> None:
        """Request a graceful stop."""
        self._running = False


async def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Publish transactions to a Redis stream")
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

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, producer.stop)
        except NotImplementedError:
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
        logger.info("Producer interrupted")
        sys.exit(0)
