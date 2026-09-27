"""Measure three fresh-volume starts and NDTP -> stream ML -> SSE -> browser.

Uses only already-built local images. Every attempt creates its own random
Compose project and deletes only that project's newly created resources.
Run: python -m scripts.check_cold_start
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
import platform
import statistics
import subprocess
import sys
import time
import uuid

import httpx
from playwright.async_api import async_playwright

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.ndtp_sender import FleetSender, load_scenario
from scripts.stack import compose, doctor, isolated_env


def docker_json(*arguments: str):
    result = subprocess.run(["docker", *arguments], check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def json_rows(value: str) -> list[dict]:
    value = value.strip()
    if not value:
        return []
    if value.startswith("["):
        return json.loads(value)
    return [json.loads(line) for line in value.splitlines() if line.strip()]


async def browser_probe(browser):
    page = await browser.new_page(viewport={"width": 1440, "height": 1000})
    errors: list[str] = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    await page.add_init_script("""(() => {
      window.__coldRendered = []; window.__coldSSE = [];
      window.addEventListener('dispatcher:rendered', e => {
        window.__coldRendered.push(e.detail); window.__coldRendered.splice(0, Math.max(0, window.__coldRendered.length - 100));
      });
      const Native = window.EventSource;
      window.EventSource = class extends Native {
        constructor(url, options) { super(url, options); this.addEventListener('snapshot', e => {
          try { window.__coldSSE.push(JSON.parse(e.data)); window.__coldSSE.splice(0, Math.max(0, window.__coldSSE.length - 100)); } catch (_) {}
        }); }
      };
    })();""")
    # Tile network availability is deliberately outside the pipeline assertion.
    await page.route("**/*.tile.openstreetmap.org/**", lambda route: route.abort())
    return page, errors


async def assert_browser_prediction(page, prediction: dict) -> dict:
    await page.wait_for_function("""id => window.__coldSSE.some(s => s.vehicles.some(v => v.prediction?.id === id))
        && window.__coldRendered.some(e => e.snapshot.vehicles.some(v => v.prediction?.id === id))""",
        arg=prediction["id"], timeout=10000)
    await page.locator(f'[data-vehicle-id="{prediction["tr_id"]}"]').click()
    await page.locator(f'[data-testid="incident-card"][data-target-id="{prediction["target_visit_id"]}"]').wait_for(timeout=10000)
    # Technical model metadata is intentionally inside the expandable detail.
    details = page.locator('[data-testid="incident-card"] details.detail-extra')
    if await details.count() and await details.get_attribute("open") is None:
        await details.locator("summary").click()
    await page.get_by_text(prediction["model_version"], exact=False).last.wait_for(timeout=10000)
    return await page.evaluate("""id => window.__coldRendered.find(e => e.snapshot.vehicles.some(v => v.prediction?.id === id))""", prediction["id"])


async def exercise_pipeline(browser, env: dict) -> dict:
    backend = f"http://127.0.0.1:{env['BACKEND_PORT']}"
    dashboard = f"http://127.0.0.1:{env['DASHBOARD_PORT']}"
    page, errors = await browser_probe(browser)
    sender, sender_task = None, None
    async with httpx.AsyncClient(base_url=backend, timeout=10, limits=httpx.Limits(keepalive_expiry=1)) as client:
        try:
            anchor = datetime.now(timezone.utc)
            scenario = load_scenario(vehicles=1, duration_s=600, anchor=anchor)
            imported = await client.post("/api/v1/scenarios/import", json=scenario)
            imported.raise_for_status()
            scenario_id = imported.json()["scenario"]["id"]
            started = await client.post("/api/v1/live/start", json={"scenario_id": scenario_id, "schedule_mode": "as_is"})
            started.raise_for_status()
            run_id = started.json()["run"]["id"]
            await page.goto(dashboard, wait_until="domcontentloaded")
            sender = FleetSender("127.0.0.1", int(env["NDTP_PORT"]), 1, anchor.timestamp())
            began = time.perf_counter()

            async def emit():
                while True:
                    await sender.tick(time.time() - anchor.timestamp())
                    await asyncio.sleep(1)

            sender_task = asyncio.create_task(emit())
            prediction, snapshot, polls = None, {}, 0
            deadline = began + 45
            while time.perf_counter() < deadline:
                response = await client.get("/api/v1/snapshot")
                response.raise_for_status()
                snapshot = response.json()
                polls += 1
                prediction = next((vehicle["prediction"] for vehicle in snapshot["vehicles"]
                    if vehicle.get("prediction") and vehicle["prediction"].get("source") == "model"), None)
                if prediction:
                    break
                await asyncio.sleep(.25)
            if not prediction:
                raise AssertionError(f"No model prediction after real NDTP: {snapshot.get('system')}")
            assert prediction.get("model_profile") == "stream", prediction
            assert math.isfinite(prediction["prediction_delay_s"])
            horizon = prediction.get("publication_horizon_s", prediction["horizon_s"])
            assert 600 < horizon <= 900, prediction
            assert prediction["run_id"] == run_id
            model_prediction_s = time.perf_counter() - began
            rendered = await assert_browser_prediction(page, prediction)
            assert not errors, errors
            return {"passed": True, "run_id": run_id, "scenario_id": scenario_id,
                "model_profile": prediction["model_profile"], "model_version": prediction["model_version"],
                "prediction_id": prediction["id"], "publication_horizon_s": horizon,
                "first_model_prediction_s": model_prediction_s, "first_prediction_ui_s": time.perf_counter() - began,
                "rendered_event_id": rendered["event_id"], "tcp_packets_sent": sender.sent,
                "sender_connections": sender.connections, "sender_failures": sender.failures,
                "snapshot_polls": polls, "browser_errors": errors,
                "system": snapshot["system"], "checks": ["import current-date plan", "real binary NDTP TCP",
                    "stream model result", "publication horizon", "SSE delivery", "React rendered same prediction", "vehicle detail model version"]}
        finally:
            sender_error = None
            if sender_task:
                sender_task.cancel()
                for error in await asyncio.gather(sender_task, return_exceptions=True):
                    if isinstance(error, BaseException) and not isinstance(error, asyncio.CancelledError):
                        sender_error = error
            if sender:
                await sender.disconnect()
            await page.close()
            if sender_error:
                raise sender_error


async def one_attempt(browser, number: int, limit_s: float) -> dict:
    project = f"transport-cold-{uuid.uuid4().hex[:12]}-{number}"
    env = isolated_env()
    record = {"attempt": number, "project": project, "ports": env, "fresh_volume": True,
              "images_prebuilt": True, "started_at": datetime.now(timezone.utc).isoformat(), "passed": False}
    created = False
    try:
        # Never reuse a project, even if an improbable random-name collision occurs.
        existing = await asyncio.to_thread(subprocess.run, ["docker", "ps", "-aq", "--filter", f"label=com.docker.compose.project={project}"],
                                           check=True, capture_output=True, text=True)
        volume = await asyncio.to_thread(subprocess.run, ["docker", "volume", "ls", "-q", "--filter", f"label=com.docker.compose.project={project}"],
                                         check=True, capture_output=True, text=True)
        if existing.stdout.strip() or volume.stdout.strip():
            raise RuntimeError("Refusing to use pre-existing project resources")
        created = True
        began = time.perf_counter()
        print(f"Cold start {number}: {project}; local images, empty database", flush=True)
        await asyncio.to_thread(compose, "up", "-d", "--no-build", "--pull", "never", "--wait", "--wait-timeout", "90",
                                project=project, env=env, capture=True)
        async with httpx.AsyncClient(timeout=5, limits=httpx.Limits(keepalive_expiry=1)) as client:
            api = await client.get(f"http://127.0.0.1:{env['BACKEND_PORT']}/health/ready")
            api.raise_for_status()
            document = await client.get(f"http://127.0.0.1:{env['DASHBOARD_PORT']}/")
            document.raise_for_status()
            assert "<html" in document.text.lower()
            swagger = await client.get(f"http://127.0.0.1:{env['BACKEND_PORT']}/docs")
            swagger.raise_for_status()
            assert "swagger-ui" in swagger.text.lower()
            schema = await client.get(f"http://127.0.0.1:{env['BACKEND_PORT']}/openapi.json")
            schema.raise_for_status()
            assert schema.json().get("openapi") and schema.json().get("paths")
            record["http_checks"] = {"backend_ready": api.status_code, "dashboard": document.status_code,
                                     "swagger": swagger.status_code, "openapi": schema.status_code}
        record["all_services_ready_s"] = time.perf_counter() - began
        record["ready_within_limit"] = record["all_services_ready_s"] <= limit_s
        state = await asyncio.to_thread(compose, "ps", "--format", "json", project=project, env=env, capture=True)
        record["services"] = json_rows(state.stdout)
        record["pipeline"] = await exercise_pipeline(browser, env)
        record["first_prediction_ui_total_s"] = time.perf_counter() - began
        record["first_prediction_ui_within_limit"] = record["first_prediction_ui_total_s"] <= limit_s
        record["within_limit"] = record["ready_within_limit"] and record["first_prediction_ui_within_limit"]
        images = await asyncio.to_thread(compose, "images", "--format", "json", project=project, env=env, capture=True)
        record["images"] = json_rows(images.stdout)
        containers = await asyncio.to_thread(compose, "ps", "-q", project=project, env=env, capture=True)
        stats = await asyncio.to_thread(subprocess.run, ["docker", "stats", "--no-stream", "--format", "{{json .}}", *containers.stdout.split()],
                                       check=True, capture_output=True, text=True)
        record["memory_snapshot"] = json_rows(stats.stdout)
        record["passed"] = record["within_limit"] and record["pipeline"]["passed"]
    except Exception as error:
        record["error"] = f"{type(error).__name__}: {error}"
        if isinstance(error, subprocess.CalledProcessError):
            record["command_stdout"] = error.stdout
            record["command_stderr"] = error.stderr
        if created:
            with contextlib.suppress(Exception):
                logs = await asyncio.to_thread(compose, "logs", "--tail", "60", "backend", "ml", "frontend", project=project, env=env, capture=True)
                record["failure_logs"] = logs.stdout[-20000:]
    finally:
        if created:
            # The UUID namespace was proven empty above; these are only our data.
            try:
                await asyncio.to_thread(compose, "down", "--volumes", "--remove-orphans", project=project, env=env, capture=True)
                record["owned_resources_removed"] = True
            except Exception as error:
                record.update(passed=False, cleanup_error=str(error))
        print(f"Cold start {number}: passed={record['passed']}; ready={record.get('all_services_ready_s')}; "
              f"first_ui_total={record.get('first_prediction_ui_total_s')}; {record.get('error', '')}", flush=True)
    return record


async def run(args) -> dict:
    environment = doctor()
    environment.update(platform=platform.platform(), docker_info=docker_json("info", "--format",
        '{"cpus":{{.NCPU}},"memory_bytes":{{.MemTotal}},"server_version":"{{.ServerVersion}}","architecture":"{{.Architecture}}"}'))
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "environment": environment,
              "ready_limit_s": args.limit_seconds, "requested_runs": args.runs, "attempts": [],
              "protocol": "Prebuilt local images, new PostgreSQL volume for every isolated project; no existing stack changes.",
              "limitations": ["Image build/download time is excluded and must be reported separately.",
                              "Three timings yield median/max, not a meaningful p95 estimate.",
                              "Memory entries are point samples, not peak usage."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    archive = args.output.with_name(f"{args.output.stem}-{uuid.uuid4().hex[:10]}.json")
    if args.output.exists():
        previous = args.output.with_name(f"{args.output.stem}-previous-{uuid.uuid4().hex[:10]}.json")
        previous.write_bytes(args.output.read_bytes())
    async with async_playwright() as runtime:
        browser = await runtime.chromium.launch(channel=os.getenv("PLAYWRIGHT_CHANNEL", "chrome"), headless=True)
        try:
            for number in range(1, args.runs + 1):
                report["attempts"].append(await one_attempt(browser, number, args.limit_seconds))
                serialized = json.dumps(report, ensure_ascii=False, indent=2)
                archive.write_text(serialized, encoding="utf-8")
                args.output.write_text(serialized, encoding="utf-8")
        finally:
            await browser.close()
    times = [row["all_services_ready_s"] for row in report["attempts"] if "all_services_ready_s" in row]
    ui_times = [row["first_prediction_ui_total_s"] for row in report["attempts"] if "first_prediction_ui_total_s" in row]
    report.update(passed=all(row["passed"] for row in report["attempts"]),
                  successful_runs=sum(row["passed"] for row in report["attempts"]),
                  three_runs_confirmed=len(report["attempts"]) >= 3,
                  ready_median_s=statistics.median(times) if times else None,
                  ready_max_s=max(times) if times else None,
                  first_prediction_ui_median_s=statistics.median(ui_times) if ui_times else None,
                  first_prediction_ui_max_s=max(ui_times) if ui_times else None)
    serialized = json.dumps(report, ensure_ascii=False, indent=2)
    archive.write_text(serialized, encoding="utf-8")
    args.output.write_text(serialized, encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--limit-seconds", type=float, default=60)
    parser.add_argument("--output", type=Path, default=Path("artifacts/acceptance/cold_start.json"))
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be positive")
    report = asyncio.run(run(args))
    print(json.dumps({key: report[key] for key in ("passed", "successful_runs", "three_runs_confirmed", "ready_median_s",
        "ready_max_s", "first_prediction_ui_median_s", "first_prediction_ui_max_s")}, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
