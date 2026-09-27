"""Isolated supplied-emulator test: NDTP -> stream ML -> SSE -> browser.

The imported scenario is explicitly synthetic and dated today. Zero forecasts,
baseline fallback, missing schedule matching, or missing UI delivery fail.
No existing stack or emulator configuration is modified.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

import httpx
from playwright.async_api import async_playwright

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.check_cold_start import assert_browser_prediction, browser_probe, json_rows
from scripts.ndtp_sender import load_scenario, position
from scripts.stack import compose, doctor, isolated_env


async def run(output: Path) -> dict:
    environment = doctor()
    project = f"transport-ndtp-check-{uuid.uuid4().hex[:12]}"
    env = isolated_env()
    result = {"passed": False, "project": project, "created_at": datetime.now(timezone.utc).isoformat(),
              "environment": environment, "ports": env, "source": "supplied ndtp-telemetry-emulator:1.0",
              "scenario_kind": "explicit synthetic current-date plan, one stationary known vehicle"}
    created = False
    backend = f"http://127.0.0.1:{env['BACKEND_PORT']}"
    emulator = f"http://127.0.0.1:{env['EMULATOR_PORT']}"
    try:
        existing = await asyncio.to_thread(subprocess.run,
            ["docker", "ps", "-aq", "--filter", f"label=com.docker.compose.project={project}"],
            check=True, capture_output=True, text=True)
        volumes = await asyncio.to_thread(subprocess.run,
            ["docker", "volume", "ls", "-q", "--filter", f"label=com.docker.compose.project={project}"],
            check=True, capture_output=True, text=True)
        if existing.stdout.strip() or volumes.stdout.strip():
            raise RuntimeError("Refusing to modify an existing project")
        created = True
        print(f"Starting isolated official emulator test: {project}", flush=True)
        await asyncio.to_thread(compose, "--profile", "ndtp", "up", "-d", "--no-build", "--pull", "never",
                                "--wait", "--wait-timeout", "90", project=project, env=env, capture=True)
        images = await asyncio.to_thread(compose, "--profile", "ndtp", "images", "--format", "json", project=project, env=env, capture=True)
        result["images"] = json_rows(images.stdout)
        async with httpx.AsyncClient(timeout=10, limits=httpx.Limits(keepalive_expiry=1)) as client:
            emulator_polls = 0
            for _ in range(60):
                emulator_polls += 1
                try:
                    response = await client.get(f"{emulator}/api/cells")
                    if response.is_success:
                        break
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(.5)
            else:
                raise AssertionError("Supplied emulator API did not become ready")
            anchor = datetime.now(timezone.utc)
            imported = await client.post(f"{backend}/api/v1/scenarios/import",
                                         json=load_scenario(1, 600, anchor))
            imported.raise_for_status()
            scenario_id = imported.json()["scenario"]["id"]
            response = await client.post(f"{backend}/api/v1/live/start",
                                         json={"scenario_id": scenario_id, "schedule_mode": "as_is"})
            response.raise_for_status()
            run_id = response.json()["run"]["id"]
            async with async_playwright() as runtime:
                browser = await runtime.chromium.launch(channel=os.getenv("PLAYWRIGHT_CHANNEL", "chrome"), headless=True)
                page, errors = await browser_probe(browser)
                try:
                    await page.goto(f"http://127.0.0.1:{env['DASHBOARD_PORT']}", wait_until="domcontentloaded")
                    lat, lon, _ = position(0, 0)
                    config = {"targetHost": "backend", "targetPort": 9201, "units": [{
                        "unitId": 900000, "intervalMs": 1000, "autoGenerate": False,
                        "cells": [{"type": "G6CellNav00", "fields": {
                            "longitude": round(abs(lon) * 1e7), "latitude": round(abs(lat) * 1e7),
                            "speedAvg": 0, "course": 0, "extraDopBit5": lat >= 0,
                            "extraDopBit6": lon >= 0, "extraDopBit7": True}}]}]}
                    began = time.perf_counter()
                    response = await client.post(f"{emulator}/api/config", json=config)
                    response.raise_for_status()
                    configured = await client.get(f"{emulator}/api/config")
                    configured.raise_for_status()
                    fields = configured.json()["units"][0]["cells"][0]["fields"]
                    assert fields["latitude"] == config["units"][0]["cells"][0]["fields"]["latitude"]
                    assert fields["speedAvg"] == 0 and fields["extraDopBit7"] is True
                    snapshot, vehicle, prediction, polls = {}, None, None, 0
                    deadline = time.perf_counter() + 50
                    while time.perf_counter() < deadline:
                        response = await client.get(f"{backend}/api/v1/snapshot")
                        response.raise_for_status()
                        snapshot = response.json()
                        polls += 1
                        vehicle = next((row for row in snapshot["vehicles"] if row["id"] == "load-000"), None)
                        prediction = vehicle.get("prediction") if vehicle else None
                        if (prediction and prediction.get("source") == "model"
                                and prediction.get("cur_dev_source") == "estimated"
                                and snapshot["system"]["telemetry_count"] >= 3):
                            break
                        await asyncio.sleep(.25)
                    assert prediction and prediction.get("source") == "model", snapshot
                    assert prediction.get("model_profile") == "stream", prediction
                    assert prediction.get("cur_dev_source") == "estimated", prediction
                    assert vehicle["cur_dev_source"] == "estimated"
                    assert 60 <= vehicle["cur_dev_s"] <= 180, vehicle
                    assert abs(vehicle["lat"] - lat) < 1e-6 and abs(vehicle["lon"] - lon) < 1e-6
                    assert math.isfinite(prediction["prediction_delay_s"])
                    assert prediction["p_late"] is not None and 0 <= prediction["p_late"] <= 1
                    assert snapshot["system"]["predictions_count"] > 0
                    horizon = prediction.get("publication_horizon_s", prediction["horizon_s"])
                    assert 600 < horizon <= 900
                    rendered = await assert_browser_prediction(page, prediction)
                    assert not errors, errors
                    result.update(passed=True, run_id=run_id, scenario_id=scenario_id,
                        elapsed_to_prediction_ui_s=time.perf_counter() - began,
                        emulator_readiness_polls=emulator_polls, snapshot_polls=polls,
                        model_version=prediction["model_version"], model_profile=prediction["model_profile"],
                        prediction=prediction, vehicle=vehicle, system=snapshot["system"],
                        configured_fields=fields, rendered_event_id=rendered["event_id"], browser_errors=errors,
                        checks=["official emulator nested fields", "NDTP CRC/handshake and navigation",
                            "current plan matching and estimated deviation", "real stream model forecast",
                            "publication in 10–15 minute planned horizon", "SSE delivery", "same forecast rendered in browser"],
                        limitations=["Synthetic scenario verifies integration; it is not a measurement of model accuracy."])
                finally:
                    await page.close()
                    await browser.close()
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        if isinstance(error, subprocess.CalledProcessError):
            result["command_stdout"], result["command_stderr"] = error.stdout, error.stderr
        if created:
            with contextlib.suppress(Exception):
                logs = await asyncio.to_thread(compose, "logs", "--tail", "60", "backend", "ml", "emulator",
                                                project=project, env=env, capture=True)
                result["failure_logs"] = logs.stdout[-20000:]
    finally:
        if created:
            try:
                await asyncio.to_thread(compose, "--profile", "ndtp", "down", "--volumes", "--remove-orphans",
                                        project=project, env=env, capture=True)
                result["owned_resources_removed"] = True
            except Exception as error:
                result.update(passed=False, cleanup_error=str(error))
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            previous = output.with_name(f"{output.stem}-previous-{uuid.uuid4().hex[:10]}.json")
            previous.write_bytes(output.read_bytes())
        serialized = json.dumps(result, ensure_ascii=False, indent=2)
        output.with_name(f"{output.stem}-{project.rsplit('-', 1)[-1]}.json").write_text(serialized, encoding="utf-8")
        output.write_text(serialized, encoding="utf-8")
    print(json.dumps({key: result.get(key) for key in ("passed", "project", "elapsed_to_prediction_ui_s", "model_version", "error")}, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/acceptance/ndtp_official.json"))
    args = parser.parse_args()
    if not asyncio.run(run(args.output))["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
