"""Restart and ML-outage checks against this project's Docker Compose stack."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import time

import httpx


def compose(*args: str) -> None:
    subprocess.run(["docker", "compose", *args], check=True, capture_output=True, text=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8010")
    args = parser.parse_args()
    checks, stopped = [], False
    with httpx.Client(base_url=args.url, timeout=30) as client:
        client.post("/api/v1/replay/start", json={"mode": "dispatcher", "speed": 20}).raise_for_status()
        before = client.post("/api/v1/replay/pause").json()
        began = time.perf_counter()
        compose("restart", "backend")
        for _ in range(60):
            try:
                if client.get("/health/ready").is_success:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(0.5)
        client.get("/health/ready").raise_for_status()
        restart_s = time.perf_counter() - began
        after = client.get("/api/v1/snapshot").json()
        assert after["run"]["id"] == before["run"]["id"]
        assert after["run"]["virtual_time"] == before["run"]["virtual_time"]
        assert {v["id"] for v in after["vehicles"]} == {v["id"] for v in before["vehicles"]}
        assert len(after["incidents"]) == len(before["incidents"])
        checks.append("PostgreSQL checkpoint restores run, clock, vehicles and incidents")
        with Path("dataset/labels/labels_test.csv").open() as source:
            point = next(csv.DictReader(source))
        params = {"mode": "evaluation", "speed": 1, "start_time": point["T"]}
        try:
            compose("stop", "ml")
            stopped = True
            degraded = client.post("/api/v1/replay/start", json=params)
            degraded.raise_for_status()
            state = degraded.json()
            value = next(v["prediction"] for v in state["vehicles"] if v["id"] == point["tr_id"])
            assert value["source"] == "baseline" and value["p_late"] is None
            assert abs(value["prediction_delay_s"] - float(point["cur_dev_s"])) < 1e-8
            client.get("/health/ready").raise_for_status()
            checks.append("ML outage uses explicit supplied-deviation baseline without probability")
            compose("start", "ml")
            stopped = False
            for _ in range(30):
                state = client.post("/api/v1/replay/start", json=params).json()
                value = next(v["prediction"] for v in state["vehicles"] if v["id"] == point["tr_id"])
                if value and value["source"] == "model":
                    break
                time.sleep(0.5)
            assert value and value["source"] == "model"
            checks.append("ML predictions resume after service recovery")
        finally:
            if stopped:
                compose("start", "ml")
            client.post("/api/v1/replay/start", json={"mode": "dispatcher", "speed": 20}).raise_for_status()
            client.post("/api/v1/replay/pause").raise_for_status()
    report = {"passed": len(checks), "checks": checks, "backend_restart_to_ready_s": restart_s,
              "limitation": "Restart uses existing images and database; excludes first image download/build time."}
    Path("artifacts/resilience_check.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
