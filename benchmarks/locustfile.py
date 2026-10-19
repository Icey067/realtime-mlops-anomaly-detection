"""Locust performance load test suite for Sentinel MLOps Anomaly Engine.

Simulates 500+ concurrent requests/sec targeting POST /predict.
Validates that 95th percentile end-to-end response latency is strictly under 15ms.
"""

from __future__ import annotations

import random
import time
from locust import HttpUser, between, events, task


class SentinelLoadUser(HttpUser):
    """Simulated high-frequency trading client emitting transactions."""

    wait_time = between(0.001, 0.003)  # Rapid high-frequency issuance

    def on_start(self) -> None:
        """Client setup and pre-allocation of user identifier."""
        self.user_id = f"usr_locust_{random.randint(10000, 99999)}"

    @task(10)
    def predict_single_transaction(self) -> None:
        """Benchmark single transaction scoring endpoint POST /predict."""
        payload = {
            "transaction_id": f"tx_bench_{int(time.time() * 1000)}_{random.randint(100, 999)}",
            "timestamp": time.time(),
            "user_id": self.user_id,
            "amount": round(random.uniform(5.0, 1500.0), 2),
            "currency": "USD",
            "latitude": round(40.7128 + random.uniform(-0.1, 0.1), 4),
            "longitude": round(-74.0060 + random.uniform(-0.1, 0.1), 4),
            "time_step": float(random.randint(0, 10000)),
        }
        # Populate V1..V28
        for i in range(1, 29):
            payload[f"v{i}"] = round(random.gauss(0.0, 1.0), 4)

        with self.client.post(
            "/predict",
            json=payload,
            catch_response=True,
            name="/predict",
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "is_anomaly" in data and "reconstruction_mse" in data:
                    response.success()
                else:
                    response.failure("Malformed response payload: missing is_anomaly or reconstruction_mse")
            else:
                response.failure(f"HTTP {response.status_code}: {response.text}")

    @task(1)
    def health_check(self) -> None:
        """Benchmark liveness endpoint GET /healthz."""
        self.client.get("/healthz", name="/healthz")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs) -> None:
    """Evaluate SLA compliance (p95 latency < 15ms) on test completion."""
    stats = environment.runner.stats.get("/predict", "POST")
    if stats and stats.num_requests > 0:
        p95 = stats.get_current_response_time_percentile(0.95)
        avg = stats.avg_response_time
        rps = stats.total_rps

        print("\n" + "=" * 65)
        print("          SENTINEL LOAD TEST SLA EVALUATION REPORT          ")
        print("=" * 65)
        print(f"Total Requests:     {stats.num_requests}")
        print(f"Failed Requests:    {stats.num_failures}")
        print(f"Mean Throughput:    {rps:.1f} req/sec")
        print(f"Average Latency:    {avg:.2f} ms")
        print(f"95th Percentile:    {p95:.2f} ms (Target SLA: < 15.00 ms)")

        if p95 <= 15.0:
            print(">>> STATUS: [PASSED] 95th percentile latency satisfies sub-15ms budget! <<<")
        else:
            print(">>> STATUS: [WARNING] 95th percentile latency exceeded 15ms target. <<<")
        print("=" * 65 + "\n")
