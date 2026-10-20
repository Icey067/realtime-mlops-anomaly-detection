"""Redis Streams consumer group worker with validation, DLQ routing, and drift sampling."""

from __future__ import annotations

import argparse
import asyncio
import collections
import logging
import signal
import sys
import time
from collections.abc import Callable, Coroutine
from typing import Any

import redis.asyncio as aioredis
from redis.asyncio.client import Redis
from redis.exceptions import ResponseError

from src.config import Settings, get_settings
from src.logging_config import configure_logging
from src.mlops.drift_monitor import DataDriftMonitor
from src.mlops.retrain_pipeline import retrain_pipeline
from src.schemas.telemetry import LatencyTelemetry
from src.schemas.transaction import TransactionPayload

logger = logging.getLogger(__name__)


class TransactionConsumer:
    """Reads transaction messages from a Redis consumer group."""

    def __init__(
        self,
        settings: Settings | None = None,
        consumer_name: str | None = None,
        on_payload_callback: Callable[[TransactionPayload], Coroutine[Any, Any, None]] | None = None,
        drift_monitor: DataDriftMonitor | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.consumer_name = consumer_name or self.settings.REDIS_CONSUMER_NAME
        self.redis_client: Redis | None = None
        self._running = False
        self._on_payload_callback = on_payload_callback
        self._drift_monitor = drift_monitor

        self.latency_window: collections.deque[float] = collections.deque(maxlen=1000)
        self.validation_window: collections.deque[float] = collections.deque(maxlen=1000)
        self.total_processed = 0
        self.total_validation_errors = 0
        self.total_latency_breaches = 0

    def _get_drift_monitor(self) -> DataDriftMonitor:
        if self._drift_monitor is None:
            self._drift_monitor = DataDriftMonitor(
                settings=self.settings,
                on_drift_alert=retrain_pipeline.handle_drift_alert,
            )
        return self._drift_monitor

    async def connect(self) -> None:
        """Open the connection pool and create the consumer group if needed."""
        logger.info(
            "Starting consumer %s in group %s on stream %s",
            self.consumer_name,
            self.settings.REDIS_CONSUMER_GROUP,
            self.settings.REDIS_STREAM_KEY,
        )
        self.redis_client = aioredis.from_url(
            self.settings.redis_url,
            encoding="utf-8",
            decode_responses=False,
            max_connections=20,
        )
        await self.redis_client.ping()

        try:
            await self.redis_client.xgroup_create(
                name=self.settings.REDIS_STREAM_KEY,
                groupname=self.settings.REDIS_CONSUMER_GROUP,
                id="0",
                mkstream=True,
            )
            logger.info("Created consumer group %s", self.settings.REDIS_CONSUMER_GROUP)
        except ResponseError as exc:
            if "BUSYGROUP" not in str(exc):
                raise
            logger.info("Consumer group %s already exists", self.settings.REDIS_CONSUMER_GROUP)

    async def close(self) -> None:
        if self.redis_client:
            await self.redis_client.aclose()

    async def _send_to_dlq(self, message_id: bytes, raw_data: dict[Any, Any], error_reason: str) -> None:
        """Store an unparseable message for later inspection."""
        if not self.redis_client:
            return
        await self.redis_client.xadd(
            self.settings.REDIS_DLQ_KEY,
            {
                "original_id": message_id.decode("utf-8") if isinstance(message_id, bytes) else str(message_id),
                "failed_at": str(time.time()),
                "error_reason": error_reason,
                "raw_payload": str(raw_data),
            },
            maxlen=10000,
        )
        logger.warning("Sent message %s to DLQ: %s", message_id, error_reason)

    def _calculate_percentiles(self, window: collections.deque[float]) -> tuple[float, float, float]:
        if not window:
            return 0.0, 0.0, 0.0
        ordered = sorted(window)
        last = len(ordered) - 1
        return (
            ordered[int(0.50 * last)],
            ordered[int(0.95 * last)],
            ordered[int(0.99 * last)],
        )

    async def process_message(
        self,
        message_id: bytes,
        raw_fields: dict[Any, Any],
    ) -> tuple[TransactionPayload, LatencyTelemetry] | None:
        """Validate one stream message and record its latency."""
        now = time.time()
        validation_start = time.perf_counter()

        try:
            payload = TransactionPayload.from_redis_dict(raw_fields)
            validation_ms = (time.perf_counter() - validation_start) * 1000
            ingestion_ms = max((now - payload.timestamp) * 1000, 0.0)
            total_ms = ingestion_ms + validation_ms

            self.latency_window.append(total_ms)
            self.validation_window.append(validation_ms)
            if total_ms > self.settings.LATENCY_BUDGET_MS:
                self.total_latency_breaches += 1

            try:
                self._get_drift_monitor().record_inference_features(payload.to_feature_vector())
            except Exception:
                logger.exception("Could not record features for drift analysis")

            return payload, LatencyTelemetry(
                ingestion_latency_ms=round(ingestion_ms, 3),
                validation_latency_ms=round(validation_ms, 3),
                total_pipeline_latency_ms=round(total_ms, 3),
            )
        except Exception as exc:
            self.total_validation_errors += 1
            await self._send_to_dlq(message_id, raw_fields, str(exc))
            return None

    async def start_consuming(self, max_messages: int | None = None) -> None:
        """Poll the consumer group until stopped or until max_messages is reached."""
        if not self.redis_client:
            raise RuntimeError("Consumer is not connected; call connect() first")

        self._running = True
        logger.info(
            "Consuming batches of %d (block %dms, latency budget %.1fms)",
            self.settings.CONSUMER_BATCH_SIZE,
            self.settings.CONSUMER_BLOCK_MS,
            self.settings.LATENCY_BUDGET_MS,
        )
        last_log_time = time.time()

        try:
            while self._running:
                response = await self.redis_client.xreadgroup(
                    groupname=self.settings.REDIS_CONSUMER_GROUP,
                    consumername=self.consumer_name,
                    streams={self.settings.REDIS_STREAM_KEY: ">"},
                    count=self.settings.CONSUMER_BATCH_SIZE,
                    block=self.settings.CONSUMER_BLOCK_MS,
                )
                if not response:
                    continue

                for _, messages in response:
                    ack_ids: list[bytes] = []
                    for message_id, fields in messages:
                        result = await self.process_message(message_id, fields)
                        ack_ids.append(message_id)

                        if result:
                            payload, _ = result
                            self.total_processed += 1
                            if self._on_payload_callback:
                                await self._on_payload_callback(payload)

                        if max_messages and self.total_processed >= max_messages:
                            self._running = False
                            break

                    if ack_ids:
                        await self.redis_client.xack(
                            self.settings.REDIS_STREAM_KEY,
                            self.settings.REDIS_CONSUMER_GROUP,
                            *ack_ids,
                        )

                now = time.time()
                if now - last_log_time >= 5.0 and self.latency_window:
                    p50, p95, p99 = self._calculate_percentiles(self.latency_window)
                    _, val_p95, _ = self._calculate_percentiles(self.validation_window)
                    logger.info(
                        "Processed %d tx | p50 %.2fms | p95 %.2fms | p99 %.2fms | "
                        "validation p95 %.2fms | errors %d | breaches %d",
                        self.total_processed,
                        p50,
                        p95,
                        p99,
                        val_p95,
                        self.total_validation_errors,
                        self.total_latency_breaches,
                    )
                    last_log_time = now
                    try:
                        score, _, feature_metrics = self._get_drift_monitor().compute_drift()
                        if feature_metrics:
                            await self.redis_client.set(
                                self.settings.DRIFT_SCORE_KEY,
                                str(score),
                                ex=300,
                            )
                    except Exception:
                        logger.exception("Drift computation failed")

        except asyncio.CancelledError:
            logger.info("Consumer cancelled")
        finally:
            self._running = False

    def stop(self) -> None:
        """Request a graceful stop."""
        self._running = False


async def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Consume transactions from a Redis stream")
    parser.add_argument("--name", type=str, default=None, help="Consumer instance name")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size for XREADGROUP")
    parser.add_argument("--max-messages", type=int, default=None, help="Stop after N messages")
    args = parser.parse_args()

    settings = get_settings()
    if args.batch_size:
        settings.CONSUMER_BATCH_SIZE = args.batch_size

    consumer = TransactionConsumer(
        settings=settings,
        consumer_name=args.name or settings.REDIS_CONSUMER_NAME,
    )

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, consumer.stop)
        except NotImplementedError:
            pass

    try:
        await consumer.connect()
        await consumer.start_consuming(max_messages=args.max_messages)
    finally:
        await consumer.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Consumer interrupted")
        sys.exit(0)
