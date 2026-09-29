"""
Measures end-to-end latency of the running API stack.

Run this against the live docker compose stack (docker compose up -d),
not in isolation, so the numbers reflect real deployment conditions
(HTTP overhead, container networking included).

Usage:
    python measure_latency.py --image path/to/image.png --category leather
"""

import argparse
import statistics
import time

import requests

API_BASE = "http://localhost:8000"
WARMUP_RUNS = 3
MEASURED_RUNS = 50


def call_predict(image_path: str, category: str) -> dict:
    with open(image_path, "rb") as f:
        response = requests.post(
            f"{API_BASE}/predict/?category={category}",
            files={"file": f},
        )
    response.raise_for_status()
    return response.json()


def call_explain(prediction: dict) -> dict:
    response = requests.post(f"{API_BASE}/explain/", json=prediction)
    response.raise_for_status()
    return response.json()


def time_calls(fn, n, *args, **kwargs):
    durations = []
    for _ in range(n):
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        durations.append(time.perf_counter() - start)
    return durations, result


def summarize(label: str, durations_seconds: list[float]) -> None:
    durations_ms = sorted(d * 1000 for d in durations_seconds)
    mean_ms = statistics.mean(durations_ms)
    median_ms = statistics.median(durations_ms)
    p95_index = int(len(durations_ms) * 0.95) - 1
    p95_ms = durations_ms[max(p95_index, 0)]

    print(f"\n{label} (n={len(durations_ms)})")
    print(f"  mean:   {mean_ms:.1f} ms")
    print(f"  median: {median_ms:.1f} ms")
    print(f"  p95:    {p95_ms:.1f} ms")
    print(f"  min:    {durations_ms[0]:.1f} ms")
    print(f"  max:    {durations_ms[-1]:.1f} ms")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to a test image")
    parser.add_argument(
        "--category", required=True, choices=["leather", "wood"]
    )
    args = parser.parse_args()

    print(f"Warming up ({WARMUP_RUNS} runs)...")
    for _ in range(WARMUP_RUNS):
        call_predict(args.image, args.category)

    print(f"Measuring /predict/ ({MEASURED_RUNS} runs)...")
    predict_durations, _ = time_calls(
    call_predict, MEASURED_RUNS, args.image, args.category
    )
    summarize("/predict/ only", predict_durations)

    print(f"\nMeasuring /predict/ + /explain/ ({MEASURED_RUNS} runs)...")
    combined_durations = []
    for _ in range(MEASURED_RUNS):
        start = time.perf_counter()
        prediction = call_predict(args.image, args.category)
        call_explain(prediction)
        combined_durations.append(time.perf_counter() - start)
    summarize("/predict/ + /explain/ combined", combined_durations)


if __name__ == "__main__":
    main()