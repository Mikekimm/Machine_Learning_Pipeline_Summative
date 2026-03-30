"""
locust_tests/locustfile.py
--------------------------
Locust load test for Intel Image Classifier API.

Usage:
  locust -f locustfile.py --host=http://localhost:8000

Then open http://localhost:8089 for the Locust web UI.

Or headless (CLI only):
  locust -f locustfile.py --host=http://localhost:8000 \
         --users 100 --spawn-rate 10 --run-time 60s --headless

Results table shows:
  - Requests per second (RPS)
  - Average / 50th / 95th / 99th percentile latency
  - Failure rate
"""

import os
import io
import random
import json
from pathlib import Path

import numpy as np
from PIL import Image
from locust import HttpUser, task, between, events


# ── Helpers ─────────────────────────────────────────────────────────────────────
def make_dummy_image(size=(150, 150)) -> bytes:
    """Generate a random RGB JPEG in memory (no disk I/O)."""
    arr = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
    img = Image.fromarray(arr, 'RGB')
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=85)
    return buf.getvalue()

# Pre-generate images so Locust workers don't recreate each request
DUMMY_IMAGES = [make_dummy_image() for _ in range(20)]


# ── User behaviors ───────────────────────────────────────────────────────────────
class ClassifierUser(HttpUser):
    """
    Simulates a typical end-user of the image classifier API.
    Mix of health checks and predictions.
    """
    wait_time = between(0.1, 0.5)   # think time between requests

    @task(1)
    def health_check(self):
        with self.client.get("/health", catch_response=True) as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure(f"Health check failed: {resp.status_code}")

    @task(8)
    def predict_single(self):
        img_bytes = random.choice(DUMMY_IMAGES)
        with self.client.post(
            "/predict",
            files={"file": ("test.jpg", img_bytes, "image/jpeg")},
            catch_response=True
        ) as resp:
            if resp.status_code == 200:
                data = resp.json()
                if 'predicted_class' in data:
                    resp.success()
                else:
                    resp.failure("Missing 'predicted_class' in response")
            else:
                resp.failure(f"Predict failed: {resp.status_code} — {resp.text[:200]}")

    @task(2)
    def get_model_info(self):
        with self.client.get("/model/info", catch_response=True) as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure(f"Model info failed: {resp.status_code}")

    @task(1)
    def get_metrics(self):
        self.client.get("/metrics")


class HeavyUser(HttpUser):
    """
    Simulates a power user sending batch predictions.
    """
    wait_time = between(1, 3)

    @task
    def predict_batch(self):
        batch_size = random.randint(3, 10)
        files = [
            ("files", (f"img_{i}.jpg", random.choice(DUMMY_IMAGES), "image/jpeg"))
            for i in range(batch_size)
        ]
        with self.client.post(
            "/predict/batch",
            files=files,
            catch_response=True
        ) as resp:
            if resp.status_code == 200:
                data = resp.json()
                if data.get('count', 0) == len(files):
                    resp.success()
                else:
                    resp.failure(f"Batch count mismatch: expected {len(files)}, got {data.get('count')}")
            else:
                resp.failure(f"Batch predict failed: {resp.status_code}")


# ── Event hooks (for logging/reporting) ─────────────────────────────────────────
@events.request.add_listener
def on_request(request_type, name, response_time, response_length, **kwargs):
    pass  # Hook for custom metrics / InfluxDB / Prometheus push

@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    stats = environment.stats
    print("\n" + "="*70)
    print("LOCUST LOAD TEST SUMMARY")
    print("="*70)
    print(f"{'Endpoint':<30} {'Reqs':>8} {'Fails':>6} {'Avg(ms)':>9} {'P95(ms)':>9} {'RPS':>8}")
    print("-"*70)
    for name, entry in stats.entries.items():
        print(
            f"{str(name[1]):<30}"
            f"{entry.num_requests:>8}"
            f"{entry.num_failures:>6}"
            f"{entry.avg_response_time:>9.1f}"
            f"{entry.get_response_time_percentile(0.95):>9.1f}"
            f"{entry.current_rps:>8.2f}"
        )
    print("="*70)
