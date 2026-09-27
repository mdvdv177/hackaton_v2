"""End-to-end HTTP checks against a running stack; starts isolated replay runs."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import time

import httpx


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8010")
    parser.add_argument("--output", type=Path, default=Path("artifacts/smoke.json"))
    args = parser.parse_args()
    checks = []
    with httpx.Client(base_url=args.url, timeout=60) as client:
        for endpoint in ("/health/live", "/health/ready", "/docs", "/openapi.json", "/metrics"):
            client.get(endpoint).raise_for_status()
            checks.append(endpoint)
        invalid = client.post("/api/v1/replay/start", json={"speed": 1000})
        assert invalid.status_code == 422
        checks.append("invalid speed rejected")
        response = client.post("/api/v1/replay/start", json={"mode": "dispatcher", "speed": 20})
        response.raise_for_status()
        snap = response.json()
        predictions = [v["prediction"] for v in snap["vehicles"] if v["prediction"]]
        assert predictions, "No dispatcher predictions produced"
        assert any(p["source"] == "model" for p in predictions), "ML did not participate"
        assert all(600 < p["horizon_s"] <= 900 for p in predictions)
        checks.append("dispatcher telemetry → model → predictions, horizons valid")
        selected = next(v for v in snap["vehicles"] if v["prediction"])
        detail = client.get(f"/api/v1/vehicles/{selected['id']}")
        detail.raise_for_status()
        assert {"telemetry", "history", "planned_visits"} <= set(detail.json())
        assert all("time_fact_begin" not in row for row in detail.json()["planned_visits"])
        checks.append("vehicle detail without schedule facts")
        if snap["incidents"]:
            incident = snap["incidents"][0]
            ack = client.post(f"/api/v1/incidents/{incident['id']}/ack")
            ack.raise_for_status()
            assert ack.json()["acknowledged"] is True
            checks.append("incident acknowledgement")
        paused = client.post("/api/v1/replay/pause")
        paused.raise_for_status()
        frozen = paused.json()["run"]["virtual_time"]
        time.sleep(0.5)
        assert client.get("/api/v1/snapshot").json()["run"]["virtual_time"] == frozen
        client.post("/api/v1/replay/resume").raise_for_status()
        time.sleep(0.75)
        assert client.get("/api/v1/snapshot").json()["run"]["virtual_time"] > frozen
        client.post("/api/v1/replay/pause").raise_for_status()
        checks.append("pause/resume virtual clock")
        with client.stream("GET", "/api/v1/events", headers={"Last-Event-ID": "999999999"}) as stream:
            stream.raise_for_status()
            lines = []
            for line in stream.iter_lines():
                lines.append(line)
                if line == "" and lines:
                    break
            assert "event: reset" in lines
        checks.append("SSE reconnect reset for unavailable cursor")
        # Exact official T ensures the online shared extractor reproduces the batch prediction.
        with Path("dataset/labels/labels_test.csv").open() as source:
            point = next(csv.DictReader(source))
        with Path("artifacts/test_predictions.csv").open() as source:
            expected = next(row for row in csv.DictReader(source) if row["sample_id"] == point["sample_id"])
        official = client.post("/api/v1/replay/start", json={"mode": "evaluation", "speed": 1, "start_time": point["T"]})
        official.raise_for_status()
        run = official.json()
        assert run["run"]["id"] != snap["run"]["id"]
        prediction = next(v["prediction"] for v in run["vehicles"] if v["id"] == point["tr_id"])
        assert abs(prediction["prediction_delay_s"] - float(expected["prediction_delay_s"])) < 1e-6
        assert abs(prediction["p_late"] - float(expected["p_late"])) < 1e-6
        checks.append("official point HTTP prediction equals offline artifact")
        client.post("/api/v1/replay/pause").raise_for_status()
    result = {"passed": len(checks), "checks": checks, "url": args.url, "finished_at": datetime.now().isoformat()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
