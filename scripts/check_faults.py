"""Isolated live NDTP failures, durable restart and browser reconnect acceptance."""
from __future__ import annotations

import argparse
import asyncio
import contextlib
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import uuid

import httpx
from playwright.async_api import async_playwright

from scripts.check_cold_start import browser_probe, assert_browser_prediction
from scripts.ndtp_sender import FleetSender, load_scenario
from scripts.stack import compose, isolated_env


async def until(action, condition, timeout=45):
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        try:
            last = await action()
            if condition(last):
                return last
        except (httpx.HTTPError, OSError):
            pass
        await asyncio.sleep(.25)
    raise AssertionError(f"Condition was not reached in {timeout}s: {last}")


async def run(output: Path):
    project = f"transport-fault-{uuid.uuid4().hex[:10]}"
    env = isolated_env()
    report = {"project": project, "environment": env, "started_at": datetime.now(timezone.utc).isoformat(), "checks": []}
    output.parent.mkdir(parents=True, exist_ok=True)
    sender, emit_task = None, None
    backend, dashboard = f"http://127.0.0.1:{env['BACKEND_PORT']}", f"http://127.0.0.1:{env['DASHBOARD_PORT']}"
    try:
        await asyncio.to_thread(compose, "up", "-d", "--no-build", "--pull", "never", "--wait", project=project, env=env, capture=True)
        report["images"] = (await asyncio.to_thread(compose, "images", "--format", "json", project=project, env=env, capture=True)).stdout
        async with httpx.AsyncClient(base_url=backend, timeout=3, limits=httpx.Limits(keepalive_expiry=1)) as client:
            async def snapshot():
                response = await client.get("/api/v1/snapshot")
                response.raise_for_status()
                return response.json()
            async def ready():
                return (await client.get("/health/ready")).status_code
            def prediction(state):
                return next((v.get("prediction") for v in state.get("vehicles", []) if v.get("prediction")), None)
            anchor = datetime.now(timezone.utc)
            imported = await client.post("/api/v1/scenarios/import", json=load_scenario(1, 1800, anchor))
            imported.raise_for_status()
            started = await client.post("/api/v1/live/start", json={"scenario_id": imported.json()["scenario"]["id"]})
            started.raise_for_status()
            run_id = started.json()["run"]["id"]
            sender = FleetSender("127.0.0.1", int(env["NDTP_PORT"]), 1, anchor.timestamp())
            async def emit():
                while True:
                    await sender.tick(time.time() - anchor.timestamp())
                    await asyncio.sleep(1)
            emit_task = asyncio.create_task(emit())
            async with async_playwright() as runtime:
                browser = await runtime.chromium.launch(channel="chrome", headless=True)
                page, page_errors = await browser_probe(browser)
                await page.goto(dashboard, wait_until="domcontentloaded")
                initial = await until(snapshot, lambda s: prediction(s) and prediction(s)["source"] == "model"
                                      and s["vehicles"][0].get("cur_dev_source") == "estimated")
                await assert_browser_prediction(page, prediction(initial))
                report["checks"].append({"name": "initial live stream forecast rendered", "passed": True})

                # Stop only sender sockets: Backend, ML and browser keep running.
                sender.enabled = False
                await sender.disconnect()
                last_position = (initial["vehicles"][0]["lat"], initial["vehicles"][0]["lon"])
                stale = await until(snapshot, lambda s: s["vehicles"][0]["stale"], timeout=75)
                vehicle = stale["vehicles"][0]
                assert (vehicle["lat"], vehicle["lon"]) == last_position
                assert not vehicle.get("prediction") or vehicle["prediction"]["p_late"] is None
                assert (await client.get("/health/live")).status_code == 200
                await page.wait_for_function("() => window.__coldRendered.some(e => e.snapshot.vehicles.some(v => v.stale))")
                restored_at = datetime.now(timezone.utc).isoformat()
                sender.enabled = True
                await until(snapshot, lambda s: not s["vehicles"][0]["stale"], timeout=10)
                restored = await until(snapshot, lambda s: prediction(s) and prediction(s)["source"] == "model" and prediction(s)["generated_at"] > restored_at)
                await assert_browser_prediction(page, prediction(restored))
                report["checks"].append({"name": "NDTP disconnect → last position/stale UI → reconnect/model UI", "passed": True})

                # ML outage must not block ingestion or turn baseline into probability.
                before_count = restored["system"]["telemetry_count"]
                await asyncio.to_thread(compose, "stop", "ml", project=project, env=env, capture=True)
                down_at = time.monotonic()
                fallback = await until(snapshot, lambda s: prediction(s) and prediction(s)["source"] == "baseline", timeout=40)
                assert prediction(fallback)["p_late"] is None
                assert fallback["system"]["telemetry_count"] > before_count
                await asyncio.sleep(max(0, 60 - (time.monotonic() - down_at)))
                assert (await client.get("/health/live")).status_code == 200
                await asyncio.to_thread(compose, "start", "ml", project=project, env=env, capture=True)
                restored = await until(snapshot, lambda s: prediction(s) and prediction(s)["source"] == "model", timeout=50)
                await assert_browser_prediction(page, prediction(restored))
                report["checks"].append({"name": "60s ML outage → explicit baseline/no probability → model restored", "passed": True})

                # Pending raw data survives transient DB loss; old state remains readable.
                await asyncio.to_thread(compose, "stop", "postgres", project=project, env=env, capture=True)
                await until(ready, lambda status: status == 503, timeout=15)
                assert (await client.get("/health/live")).status_code == 200
                assert (await snapshot())["vehicles"]
                await asyncio.sleep(30)
                admitted_barrier = (await snapshot())["system"]["ingress_admitted"]
                began = time.monotonic()
                await asyncio.to_thread(compose, "start", "postgres", project=project, env=env, capture=True)
                await until(ready, lambda status: status == 200, timeout=max(.1, 10 - (time.monotonic() - began)))
                readiness_recovery_s = time.monotonic() - began
                recovered = await until(snapshot, lambda s: s["system"].get("apply_queue_depth") == 0 and s["system"].get("ingress_queue_depth") == 0
                                        and s["system"]["telemetry_count"] >= admitted_barrier,
                                        timeout=max(.1, 10 - readiness_recovery_s))
                recovery_s = time.monotonic() - began
                assert recovered["system"].get("ingress_rejected", 0) == 0
                report["checks"].append({"name": "30s PostgreSQL outage → bounded queue → automatic recovery", "passed": recovery_s <= 10,
                                        "admitted_barrier": admitted_barrier, "readiness_recovery_s": readiness_recovery_s,
                                        "queue_drained_recovery_s": recovery_s, "recovery_s": recovery_s})

                # Abrupt death followed by a recreated container exercises stored state
                # and Nginx's Docker-DNS resolution; no graceful final checkpoint.
                before = await snapshot()
                ack_id = next((i["id"] for i in before["incidents"] if not i["acknowledged"]), None)
                if ack_id:
                    (await client.post(f"/api/v1/incidents/{ack_id}/ack")).raise_for_status()
                await asyncio.to_thread(compose, "kill", "-s", "KILL", "backend", project=project, env=env, capture=True)
                await asyncio.sleep(3)
                await asyncio.to_thread(compose, "up", "-d", "--no-deps", "--force-recreate", "--wait", "backend", project=project, env=env, capture=True)
                after = await until(snapshot, lambda s: s.get("run", {}).get("id") == run_id and bool(s["vehicles"]), timeout=30)
                assert after["vehicles"][0]["id"] == before["vehicles"][0]["id"]
                if ack_id:
                    incident = await client.get(f"/api/v1/incidents/{ack_id}")
                    incident.raise_for_status()
                    assert incident.json()["acknowledged"]
                # Reconnect TCP deterministically even when OS has not yet noticed peer death.
                await sender.disconnect()
                after_at = datetime.now(timezone.utc).isoformat()
                final = await until(snapshot, lambda s: prediction(s) and prediction(s)["source"] == "model" and prediction(s)["generated_at"] > after_at, timeout=45)
                await assert_browser_prediction(page, prediction(final))
                report["checks"].append({"name": "abrupt backend recreation → same run → TCP/SSE/UI recover", "passed": True, "ack_verified": bool(ack_id)})
                report["final_images"] = (await asyncio.to_thread(compose, "images", "--format", "json", project=project, env=env, capture=True)).stdout
                initial_images = {row["ContainerName"]: row["ID"] for row in json.loads(report["images"])}
                final_images = {row["ContainerName"]: row["ID"] for row in json.loads(report["final_images"])}
                report["images_unchanged"] = bool(initial_images) and initial_images == final_images
                report["final_system"] = final["system"]
                report["browser_errors"] = page_errors
                await page.screenshot(path=str(output.with_suffix(".png")), full_page=True)
                await browser.close()
                assert not page_errors, page_errors
                report["passed"] = report["images_unchanged"] and all(check["passed"] for check in report["checks"])
    except Exception as error:
        report.update(passed=False, error=f"{type(error).__name__}: {error}")
        with contextlib.suppress(Exception):
            report["logs"] = compose("logs", "--tail", "80", project=project, env=env, capture=True).stdout
        raise
    finally:
        if emit_task:
            emit_task.cancel()
            for error in await asyncio.gather(emit_task, return_exceptions=True):
                if isinstance(error, BaseException) and not isinstance(error, asyncio.CancelledError):
                    report.update(passed=False, sender_task_error=f"{type(error).__name__}: {error}")
        if sender:
            await sender.disconnect()
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        serialized = json.dumps(report, ensure_ascii=False, indent=2)
        output.with_name(f"{output.stem}-{project.rsplit('-', 1)[-1]}.json").write_text(serialized)
        output.write_text(serialized)
        await asyncio.to_thread(compose, "down", "--volumes", project=project, env=env, capture=True)
    print(json.dumps({k: v for k, v in report.items() if k != "logs"}, ensure_ascii=False, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--postgres-only", action="store_true", help="30s outage with direct read-only SQL durability proof")
    args = parser.parse_args()
    if args.postgres_only:
        from scripts.check_pg_recovery import run as runner
    else:
        runner = run
    output = args.output or Path("artifacts/acceptance/PG.json" if args.postgres_only else "artifacts/acceptance/faults.json")
    report = asyncio.run(runner(output))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
