"""Measure an actual replay through HTTP; never infer latency from virtual time."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import statistics
import time

import httpx


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    return values[min(len(values) - 1, int((len(values) - 1) * quantile))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--seconds", type=int, default=60)
    parser.add_argument("--speed", type=int, choices=[1, 5, 20], default=20)
    parser.add_argument("--mode", choices=["dispatcher", "evaluation"], default="dispatcher")
    parser.add_argument("--output", type=Path, default=Path("artifacts/benchmark.json"))
    args = parser.parse_args()
    durations, samples = [], []
    with httpx.Client(base_url=args.url, timeout=60) as client:
        started = time.perf_counter()
        response = client.post("/api/v1/replay/start", json={"mode": args.mode, "speed": args.speed})
        response.raise_for_status()
        startup_s = time.perf_counter() - started
        deadline = time.perf_counter() + args.seconds
        while time.perf_counter() < deadline:
            tick = time.perf_counter()
            snapshot = client.get("/api/v1/snapshot")
            snapshot.raise_for_status()
            durations.append((time.perf_counter() - tick) * 1000)
            body = snapshot.json()
            samples.append({"wall_elapsed_s": time.perf_counter() - started,
                            "run": body.get("run"), "system": body.get("system"),
                            "vehicles": len(body.get("vehicles", [])),
                            "with_prediction": sum(v.get("prediction") is not None for v in body.get("vehicles", []))})
            time.sleep(min(1, max(0, deadline - time.perf_counter())))
        client.post("/api/v1/replay/pause").raise_for_status()
        metrics = client.get("/metrics")
        metrics.raise_for_status()
    report = {"measured_at": datetime.now(timezone.utc).isoformat(),
              "client_platform": platform.platform(), "url": args.url,
              "mode": args.mode, "speed": args.speed, "duration_s": args.seconds,
              "replay_start_request_s": startup_s,
              "snapshot_http_ms": {"p50": statistics.median(durations), "p95": percentile(durations, 0.95),
                                   "max": max(durations), "samples": len(durations)},
              "samples": samples,
              "limitations": ["HTTP snapshot timings are not event-to-prediction latency.",
                              "Server processing metrics and virtual-time progress are recorded separately.",
                              "This is a short replay benchmark on the supplied fleet, not a production capacity guarantee."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    args.output.with_suffix(".prom").write_text(metrics.text, encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in {"samples", "limitations"}}, indent=2))


if __name__ == "__main__":
    main()
