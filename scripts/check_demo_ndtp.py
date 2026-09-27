"""Isolated original-CSV NDTP demo -> stream ML -> browser acceptance check."""
from __future__ import annotations

import argparse
import asyncio
import contextlib
from datetime import datetime, timezone
import json
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

from scripts.check_cold_start import assert_browser_prediction, browser_probe
from scripts.stack import ROOT, compose, doctor, isolated_env


async def check(args) -> dict:
    project = f"transport-demo-check-{uuid.uuid4().hex[:12]}"
    env = isolated_env()
    report = {"project": project, "ports": env, "passed": False,
              "created_at": datetime.now(timezone.utc).isoformat(), "duration_s": args.duration,
              "protocol": "Original test traffic over TCP with demo manifest shift; default zero warmup; 1x clock"}
    created = False
    process = None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        args.output.with_name(f"{args.output.stem}-previous-{uuid.uuid4().hex[:10]}.json").write_bytes(args.output.read_bytes())
    try:
        doctor()
        for kind in ("ps", "volume"):
            command = (["docker", "ps", "-aq"] if kind == "ps" else ["docker", "volume", "ls", "-q"])
            check_result = await asyncio.to_thread(subprocess.run, command + ["--filter", f"label=com.docker.compose.project={project}"],
                                                   capture_output=True, text=True, check=True)
            if check_result.stdout.strip():
                raise RuntimeError("Refusing to reuse pre-existing resources")
        created = True
        await asyncio.to_thread(compose, "up", "-d", "--no-build", "--pull", "never", "--wait", "--wait-timeout", "90",
                                project=project, env=env, capture=True)
        report["images"] = json.loads((await asyncio.to_thread(compose, "images", "--format", "json", project=project, env=env, capture=True)).stdout)
        backend = f"http://127.0.0.1:{env['BACKEND_PORT']}"
        dashboard = f"http://127.0.0.1:{env['DASHBOARD_PORT']}"
        async with async_playwright() as runtime:
            browser = await runtime.chromium.launch(channel=os.getenv("PLAYWRIGHT_CHANNEL", "chrome"), headless=True)
            page, errors = await browser_probe(browser)
            try:
                await page.goto(dashboard, wait_until="domcontentloaded")
                began = time.perf_counter()
                process = await asyncio.create_subprocess_exec(sys.executable, "-m", "scripts.demo_ndtp",
                    "--backend-url", backend, "--dashboard-url", dashboard, "--ndtp-port", env["NDTP_PORT"],
                    "--duration", str(args.duration), cwd=ROOT, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
                async with httpx.AsyncClient(base_url=backend, timeout=10, limits=httpx.Limits(keepalive_expiry=1)) as client:
                    snapshot, prediction = {}, None
                    while time.perf_counter() - began < args.duration + 10:
                        response = await client.get("/api/v1/snapshot")
                        response.raise_for_status()
                        snapshot = response.json()
                        prediction = next((v["prediction"] for v in snapshot["vehicles"]
                            if v.get("prediction") and v["prediction"].get("source") == "model"
                            and v["prediction"].get("model_profile") == "stream"), None)
                        if prediction:
                            report["first_model_prediction_s"] = time.perf_counter() - began
                            rendered = await assert_browser_prediction(page, prediction)
                            report["first_prediction_ui_s"] = time.perf_counter() - began
                            report["rendered_event_id"] = rendered["event_id"]
                            break
                        if process.returncode is not None:
                            break
                        await asyncio.sleep(.5)
                    stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=args.duration + 15)
                    logs = [json.loads(line) for line in stdout.decode().splitlines() if line.strip()]
                    report.update(runner_exit_code=process.returncode, runner_logs=logs, runner_stderr=stderr.decode(),
                                  browser_errors=errors, prediction=prediction, system=snapshot.get("system"))
                    assert process.returncode == 0, report["runner_stderr"] or logs
                    assert prediction is not None, "No real stream model prediction from original NDTP traffic"
                    started = next(row for row in logs if row["event"] == "started")
                    stopped = next(row for row in logs if row["event"] == "stopped")
                    assert started["warmup_packets"] == 0 and stopped["sent"] > 0
                    assert stopped["reason"] == "duration_reached"
                    after_response = await client.get("/api/v1/snapshot")
                    after_response.raise_for_status()
                    after = after_response.json()
                    assert after["run"]["id"] == started["run_id"] and after["run"]["status"] == "running"
                    report["sender_stop_leaves_run_active"] = True
                    report["prediction_run_matches"] = prediction["run_id"] == started["run_id"]
                    assert report["prediction_run_matches"] and not errors
                    screenshot = args.output.with_suffix(".png")
                    await page.screenshot(path=str(screenshot), full_page=True)
                    report["screenshot"] = str(screenshot)
                    report["final_images"] = json.loads((await asyncio.to_thread(compose, "images", "--format", "json", project=project, env=env, capture=True)).stdout)
                    report["images_unchanged"] = ({row["ContainerName"]: row["ID"] for row in report["images"]}
                                                  == {row["ContainerName"]: row["ID"] for row in report["final_images"]})
                    report["passed"] = bool(report["images"]) and report["images_unchanged"]
            finally:
                await browser.close()
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        if isinstance(error, subprocess.CalledProcessError):
            report.update(command_stdout=error.stdout, command_stderr=error.stderr)
        if created:
            with contextlib.suppress(Exception):
                logs = await asyncio.to_thread(compose, "logs", "--tail", "50", "backend", "ml", project=project, env=env, capture=True)
                report["failure_logs"] = logs.stdout[-20000:]
    finally:
        if process and process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=10)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
        if created:
            try:
                await asyncio.to_thread(compose, "down", "--volumes", "--remove-orphans", project=project, env=env, capture=True)
                report["owned_resources_removed"] = True
            except Exception as error:
                report.update(passed=False, cleanup_error=str(error))
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        serialized = json.dumps(report, ensure_ascii=False, indent=2)
        args.output.with_name(f"{args.output.stem}-{project.rsplit('-', 1)[-1]}.json").write_text(serialized, encoding="utf-8")
        args.output.write_text(serialized, encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=60)
    parser.add_argument("--output", type=Path, default=Path("artifacts/acceptance/demo_ndtp.json"))
    args = parser.parse_args()
    if args.duration < 45:
        parser.error("--duration must be at least 45 seconds to observe a 30s prediction cadence")
    report = asyncio.run(check(args))
    print(json.dumps({key: report.get(key) for key in ("passed", "error", "first_model_prediction_s", "first_prediction_ui_s")}, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
