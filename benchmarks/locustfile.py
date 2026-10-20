"""Locust load test for POST /predict and GET /healthz."""

from __future__ import annotations

import random
import time

from locust import HttpUser, between, events, task


class SentinelLoadUser(HttpUser):
    """Sends synthetic transactions to the prediction endpoint."""

    wait_time = between(0.001, 0.003)

    def on_start(self) -> None:
        self.user_id = f"usr_locust_{random.randint(10000, 99999)}"

    @task(10)
    def predict_single_transaction(self) -> None:
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
                if data.get("status") != "OK":
                    response.failure("API returned a degraded prediction; load the model artifacts first")
                elif "is_anomaly" not in data or "reconstruction_mse" not in data:
                    response.failure("Malformed response payload: missing is_anomaly or reconstruction_mse")
                else:
                    response.success()
            else:
                response.failure(f"HTTP {response.status_code}: {response.text}")

    @task(1)
    def health_check(self) -> None:
        self.client.get("/healthz", name="/healthz")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs) -> None:
    """Print a latency summary for the completed run."""
    stats = environment.runner.stats.get("/predict", "POST")
    if not stats or stats.num_requests == 0:
        return

    p95 = stats.get_current_response_time_percentile(0.95)
    print("\n" + "=" * 60)
    print("Load test summary (/predict)")
    print("=" * 60)
    print(f"Total requests:  {stats.num_requests}")
    print(f"Failures:        {stats.num_failures}")
    print(f"Throughput:      {stats.total_rps:.1f} req/s")
    print(f"Average latency: {stats.avg_response_time:.2f} ms")
    print(f"p95 latency:     {p95:.2f} ms")
    print("=" * 60 + "\n")
