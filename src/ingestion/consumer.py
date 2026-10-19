"""High-performance Redis Consumer Group worker for streaming transactions.

Parses stream events, performs strict Pydantic v2 payload validation,
acknowledges messages via XACK, routes corrupt events to a Dead Letter Queue (DLQ),
and tracks end-to-end sub-10ms pipeline latency metrics.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import logging
import signal
import sys
import time
from typing import Any, Callable, Coroutine, Deque, Dict, List, Optional, Tuple

import redis.asyncio as aioredis
from pydantic import ValidationError
from redis.asyncio.client import Redis
from redis.exceptions import ResponseError

from src.config import Settings, get_settings
from src.schemas.telemetry import LatencyTelemetry
from src.schemas.transaction import TransactionPayload

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sentinel.consumer")


class TransactionConsumer:
    """Async Redis Consumer Group processor with latency budget tracking."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        consumer_name: Optional[str] = None,
        on_payload_callback: Optional[
            Callable[[TransactionPayload], Coroutine[Any, Any, None]]
        ] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.consumer_name = consumer_name or self.settings.REDIS_CONSUMER_NAME
        self.redis_client: Optional[Redis] = None
        self._running = False
        self._on_payload_callback = on_payload_callback

        # Latency & throughput tracking metrics (rolling window of 1,000 transactions)
        self.latency_window: Deque[float] = collections.deque(maxlen=1000)
        self.validation_window: Deque[float] = collections.deque(maxlen=1000)
        self.total_processed = 0
        self.total_validation_errors = 0
        self.total_latency_breaches = 0

    async def connect(self) -> None:
        """Establish Redis connection and idempotently create the Consumer Group."""
        logger.info(
            "Initializing consumer '%s' for group '%s' on stream '%s'...",
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

        # Create consumer group if not already present
        try:
            await self.redis_client.xgroup_create(
                name=self.settings.REDIS_STREAM_KEY,
                groupname=self.settings.REDIS_CONSUMER_GROUP,
                id="0",
                mkstream=True,
            )
            logger.info("Created Redis consumer group: %s", self.settings.REDIS_CONSUMER_GROUP)
        except ResponseError as e:
            if "BUSYGROUP" in str(e):
                logger.info("Consumer group '%s' already exists. Reusing.", self.settings.REDIS_CONSUMER_GROUP)
            else:
                logger.error("Failed to create consumer group: %s", e)
                raise

    async def close(self) -> None:
        """Close Redis connection and clean up."""
        if self.redis_client:
            await self.redis_client.close()
            logger.info("Consumer connection closed.")

    async def _send_to_dlq(self, message_id: bytes, raw_data: Dict[Any, Any], error_reason: str) -> None:
        """Route malformed payloads to Dead Letter Queue for auditing."""
        if not self.redis_client:
            return
        dlq_entry = {
            "original_id": message_id.decode("utf-8") if isinstance(message_id, bytes) else str(message_id),
            "failed_at": str(time.time()),
            "error_reason": error_reason,
            "raw_payload": str(raw_data),
        }
        await self.redis_client.xadd(self.settings.REDIS_DLQ_KEY, dlq_entry, maxlen=10000)
        logger.warning("Routed corrupted message %s to DLQ: %s", message_id, error_reason)

    def _calculate_percentiles(self, window: Deque[float]) -> Tuple[float, float, float]:
        """Compute p50, p95, and p99 from a rolling latency window."""
        if not window:
            return 0.0, 0.0, 0.0
        sorted_lat = sorted(window)
        n = len(sorted_lat)
        p50 = sorted_lat[int(0.50 * n)]
        p95 = sorted_lat[min(int(0.95 * n), n - 1)]
        p99 = sorted_lat[min(int(0.99 * n), n - 1)]
        return p50, p95, p99

    async def process_message(
        self,
        message_id: bytes,
        raw_fields: Dict[Any, Any],
    ) -> Optional[Tuple[TransactionPayload, LatencyTelemetry]]:
        """Parse raw stream event, validate schema, record latencies, and return validated payload."""
        now = time.time()
        start_val = time.perf_counter()

        try:
            # High-speed Pydantic v2 validation
            payload = TransactionPayload.from_redis_dict(raw_fields)
            val_latency_ms = (time.perf_counter() - start_val) * 1000

            # End-to-end ingestion latency: (consumer receive time - producer timestamp)
            ingest_latency_ms = max((now - payload.timestamp) * 1000, 0.0)
            total_latency_ms = ingest_latency_ms + val_latency_ms

            # Track rolling latency stats
            self.latency_window.append(total_latency_ms)
            self.validation_window.append(val_latency_ms)

            # Check strict latency budget (e.g. 10ms)
            if total_latency_ms > self.settings.LATENCY_BUDGET_MS:
                self.total_latency_breaches += 1

            telemetry = LatencyTelemetry(
                ingestion_latency_ms=round(ingest_latency_ms, 3),
                validation_latency_ms=round(val_latency_ms, 3),
                total_pipeline_latency_ms=round(total_latency_ms, 3),
            )
            return payload, telemetry

        except (ValidationError, Exception) as err:
            self.total_validation_errors += 1
            await self._send_to_dlq(message_id, raw_fields, str(err))
            return None

    async def start_consuming(self, max_messages: Optional[int] = None) -> None:
        """Main consumer loop reading from Redis Consumer Group."""
        if not self.redis_client:
            raise RuntimeError("Consumer is not connected. Call connect() first.")

        self._running = True
        logger.info(
            "Consumer '%s' running. Fetching batches of %d (Block: %dms, Budget: %.1fms)...",
            self.consumer_name,
            self.settings.CONSUMER_BATCH_SIZE,
            self.settings.CONSUMER_BLOCK_MS,
            self.settings.LATENCY_BUDGET_MS,
        )

        last_log_time = time.time()

        try:
            while self._running:
                # XREADGROUP fetches messages assigned to this consumer
                streams_response = await self.redis_client.xreadgroup(
                    groupname=self.settings.REDIS_CONSUMER_GROUP,
                    consumername=self.consumer_name,
                    streams={self.settings.REDIS_STREAM_KEY: ">"},
                    count=self.settings.CONSUMER_BATCH_SIZE,
                    block=self.settings.CONSUMER_BLOCK_MS,
                )

                if not streams_response:
                    continue

                for stream_name, messages in streams_response:
                    ack_ids: List[bytes] = []

                    for message_id, fields in messages:
                        result = await self.process_message(message_id, fields)
                        ack_ids.append(message_id)

                        if result:
                            payload, telemetry = result
                            self.total_processed += 1

                            if self._on_payload_callback:
                                await self._on_payload_callback(payload)

                        if max_messages and self.total_processed >= max_messages:
                            self._running = False
                            break

                    # Acknowledge processed messages in batch
                    if ack_ids:
                        await self.redis_client.xack(
                            self.settings.REDIS_STREAM_KEY,
                            self.settings.REDIS_CONSUMER_GROUP,
                            *ack_ids,
                        )

                # Periodic performance summary telemetry
                now = time.time()
                if now - last_log_time >= 5.0 and self.latency_window:
                    p50, p95, p99 = self._calculate_percentiles(self.latency_window)
                    _, val_p95, _ = self._calculate_percentiles(self.validation_window)
                    logger.info(
                        "Processed: %d tx | Latency p50: %.2fms | p95: %.2fms | p99: %.2fms | Val p95: %.2fms | Errors: %d | Budget Breaches: %d",
                        self.total_processed,
                        p50,
                        p95,
                        p99,
                        val_p95,
                        self.total_validation_errors,
                        self.total_latency_breaches,
                    )
                    last_log_time = now

        except asyncio.CancelledError:
            logger.info("Consumer received shutdown signal.")
        finally:
            self._running = False

    def stop(self) -> None:
        """Signal consumer loop to cease polling."""
        self._running = False


async def main() -> None:
    """CLI entrypoint for running the consumer standalone."""
    parser = argparse.ArgumentParser(description="Sentinel Redis Stream Consumer")
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
        logger.info("Consumer stopped by user.")
        sys.exit(0)
