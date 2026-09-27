"""Isolated full-history diagnostic: 180s NDTP catch-up, then 120s live measurement.

This deliberately accelerated startup is never a full-hour acceptance result.
Only prebuilt images and a new, owned Compose project are used.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time
import uuid

import httpx
from playwright.async_api import async_playwright

from scripts.acceptance_load import BROWSER_PROBE, browser_quality, continuity_checks, distribution, image_identity
from scripts.ndtp_sender import FleetSender, handshake, load_scenario, navigation, position
from scripts.stack import compose, isolated_env
from scripts.watch_durable_load import query_text

VEHICLES = 100
HISTORY_POINTS = 900
WARM_RATE_HZ = 5
WARM_SECONDS = HISTORY_POINTS / WARM_RATE_HZ


def warm_event_offset(index: int) -> float:
    """Event seconds relative to warm start; every event is in the available past."""
    return -(HISTORY_POINTS - WARM_SECONDS) + index


class HistorySender(FleetSender):
    async def tick_at(self, source_elapsed_s: float, event_at: float) -> None:
        if event_at > time.time():
            raise ValueError("Refusing to send future telemetry during catch-up")
        async def send(index):
            peer = 900000 + index
            try:
                if index not in self.writers:
                    _, writer = await asyncio.wait_for(asyncio.open_connection(self.host, self.port), timeout=3)
                    self.writers[index] = writer
                    writer.write(handshake(peer))
                    self.connections += 1
                writer = self.writers[index]
                self.requests[index] += 1
                writer.write(navigation(peer, self.requests[index], *position(index, source_elapsed_s), at=event_at))
                await asyncio.wait_for(writer.drain(), timeout=3)
                self.sent += 1
            except (OSError, asyncio.TimeoutError, ConnectionError):
                self.failures += 1
                writer = self.writers.pop(index, None)
                if writer:
                    writer.close()
        await asyncio.gather(*(send(index) for index in range(self.vehicles)))


WARM_PROBE = """() => {
  window.__warm = {began:Infinity, ends:Infinity, seen:{}, predictions:[]};
  window.addEventListener('dispatcher:rendered', event => {
    const w=window.__warm, now=Date.now();
    for (const v of event.detail.snapshot.vehicles || []) {
      const p=v.prediction;
      if (!p || p.source!=='model' || p.timing_status!=='verified' || !p.id || w.seen[p.id]) continue;
      const triggered=Date.parse(p.triggered_at);
      if (!(triggered >= w.began && triggered < w.ends)) continue;
      w.seen[p.id]=true;
      w.predictions.push({id:p.id, tr_id:v.id, run_id:p.run_id, triggered_at:p.triggered_at,
        rendered_at:now, latency_ms:now-triggered, history_points:p.data_quality?.history_points,
        horizon_s:p.publication_horizon_s, model_version:p.model_version});
    }
  });
}"""


def history_checks(rows: list[dict], run_id: str, vehicles: int = VEHICLES) -> tuple[dict, dict]:
    per_vehicle = {f"load-{index:03}": [] for index in range(vehicles)}
    seen = set()
    for row in rows:
        if row.get("id") and row["id"] not in seen and row.get("tr_id") in per_vehicle:
            per_vehicle[row["tr_id"]].append(row)
            seen.add(row["id"])
    counts = [row.get("history_points") for row in rows]
    sufficient = lambda value: isinstance(value, int) and value >= 890
    checks = {"each_vehicle_has_three_postwarm_model_renders": all(len(value) >= 3 for value in per_vehicle.values()),
              "all_postwarm_models_have_900s_history": bool(rows) and all(sufficient(value) for value in counts),
              "all_model_run_ids_match": bool(rows) and all(row.get("run_id") == run_id for row in rows),
              "all_publication_horizons_valid": bool(rows) and all(isinstance(row.get("horizon_s"), (int, float))
                    and 600 < row["horizon_s"] <= 900 for row in rows)}
    history_count = lambda row: row["history_points"] if isinstance(row.get("history_points"), int) else 0
    evidence = {identifier: {"model_renders": len(values),
                            "min_history_points": min((history_count(r) for r in values), default=0),
                            "max_history_points": max((history_count(r) for r in values), default=0)}
                for identifier, values in per_vehicle.items()}
    return checks, evidence


async def wait_until(deadline: float) -> None:
    while time.monotonic() < deadline:
        await asyncio.sleep(min(.05, max(0, deadline - time.monotonic())))


async def live_tick(sender: HistorySender, anchor: float, due: float) -> float:
    await wait_until(due)
    lag = max(0, time.monotonic() - due)
    # Monotonic pacing must not manufacture future event times when the
    # system wall clock is adjusted during the run.
    event_at = time.time()
    await sender.tick_at(event_at - anchor, event_at)
    return lag


async def cancel_sender_task(task: asyncio.Task) -> None:
    task.cancel()
    # An already failed task must not abort owned-container cleanup.
    await asyncio.gather(task, return_exceptions=True)


async def run(args) -> dict:
    project = f"transport-warm-{uuid.uuid4().hex[:10]}"
    env = isolated_env()
    backend, dashboard = f"http://127.0.0.1:{env['BACKEND_PORT']}", f"http://127.0.0.1:{env['DASHBOARD_PORT']}"
    report = {"passed": False, "full_acceptance": False, "kind": "accelerated_history_diagnostic",
              "project": project, "environment": env, "started_at": datetime.now(timezone.utc).isoformat(),
              "vehicles": VEHICLES, "duration_s": args.seconds, "samples": [], "warm_samples": [],
              "protocol": {"warm_history_points_per_vehicle": HISTORY_POINTS, "warm_events_per_second": VEHICLES * WARM_RATE_HZ,
                           "warm_duration_s": WARM_SECONDS, "initial_event_age_s": HISTORY_POINTS - WARM_SECONDS,
                           "measurement_events_per_second": VEHICLES, "measurement_duration_s": args.seconds,
                           "latency": "postwarm triggered_at to React commit; coalesced latest telemetry received_at to React",
                           "history_proof": "each of 100 vehicles >=3 postwarm models, every reported history_points >=890",
                           "limitation": "Accelerated past-event receipt is a diagnostic, not an unmodified one-hour real-time protocol"}}
    report["code_sha256"] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                             for folder in ("backend", "predictor", "ml") for path in sorted(Path(folder).glob("*.py"))}
    sender = None
    created = False
    active_task = None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        args.output.with_name(f"{args.output.stem}-previous-{uuid.uuid4().hex[:8]}.json").write_bytes(args.output.read_bytes())
    def save():
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    try:
        for kind in ("containers", "volumes"):
            command = ["docker", "ps", "-aq"] if kind == "containers" else ["docker", "volume", "ls", "-q"]
            existing = await asyncio.to_thread(subprocess.run, [*command, "--filter", f"label=com.docker.compose.project={project}"],
                                               capture_output=True, text=True, check=True)
            if existing.stdout.strip():
                raise RuntimeError("Refusing to reuse existing project resources")
        created = True
        await asyncio.to_thread(compose, "up", "-d", "--no-build", "--pull", "never", "--wait", project=project, env=env, capture=True)
        report["images"] = (await asyncio.to_thread(compose, "images", "--format", "json", project=project, env=env, capture=True)).stdout
        command = "import hashlib,json; from pathlib import Path; print(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for f in ('backend','predictor','ml') for p in sorted(Path(f).glob('*.py'))}))"
        report["runtime_code_sha256"] = json.loads((await asyncio.to_thread(compose, "exec", "-T", "backend", "python", "-c", command,
                                                                            project=project, env=env, capture=True)).stdout)
        async with httpx.AsyncClient(base_url=backend, timeout=30, limits=httpx.Limits(keepalive_expiry=1)) as client:
            async with async_playwright() as runtime:
                browser = await runtime.chromium.launch(channel="chrome", headless=True)
                page = await browser.new_page(viewport={"width": 1440, "height": 900})
                errors = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                await page.add_init_script(f"({BROWSER_PROBE})(); ({WARM_PROBE})()")
                await page.route("**/*.tile.openstreetmap.org/**", lambda route: route.abort())
                await page.goto(dashboard, wait_until="domcontentloaded")
                warm_wall = time.time() + 3
                warm_mono = time.monotonic() + 3
                anchor = warm_wall - (HISTORY_POINTS - WARM_SECONDS)
                package = load_scenario(VEHICLES, HISTORY_POINTS + args.seconds, datetime.fromtimestamp(anchor, timezone.utc))
                imported = await client.post("/api/v1/scenarios/import", json=package)
                imported.raise_for_status()
                started = await client.post("/api/v1/live/start", json={"scenario_id": imported.json()["scenario"]["id"]})
                started.raise_for_status()
                run_id = started.json()["run"]["id"]
                report.update(run_id=run_id, scenario_id=imported.json()["scenario"]["id"], warm_started_at=datetime.fromtimestamp(warm_wall, timezone.utc).isoformat())
                sender = HistorySender("127.0.0.1", int(env["NDTP_PORT"]), VEHICLES, anchor)

                async def sample(phase: str, origin: float):
                    response = await client.get("/api/v1/system/status")
                    response.raise_for_status()
                    probe = await page.evaluate("() => ({now_ms:Date.now(), vehicles:window.__acceptance.vehicles, last:window.__acceptance.last, lastSseAt:window.__acceptance.lastSseAt})")
                    row = {"elapsed_s": time.monotonic() - origin, "sent": sender.sent, "system": response.json(),
                           "browser_quality": browser_quality(probe, VEHICLES, run_id)}
                    report["samples" if phase == "measurement" else "warm_samples"].append(row)
                    report["phase"] = phase
                    save()

                async def send_warm():
                    max_lag = 0.0
                    for index in range(HISTORY_POINTS):
                        due = warm_mono + index / WARM_RATE_HZ
                        await wait_until(due)
                        max_lag = max(max_lag, time.monotonic() - due)
                        await sender.tick_at(index, warm_wall + warm_event_offset(index))
                    await wait_until(warm_mono + WARM_SECONDS)
                    report["warmup"] = {"sent": sender.sent, "expected": VEHICLES * HISTORY_POINTS,
                                        "actual_s": time.monotonic() - warm_mono, "max_sender_lag_s": max_lag}

                print(f"Warm history: {project}; 100×900 points over 180s, then {args.seconds}s live", flush=True)
                active_task = asyncio.create_task(send_warm())
                next_sample = warm_mono
                while not active_task.done():
                    await wait_until(next_sample)
                    await sample("warmup", warm_mono)
                    next_sample += 5
                await active_task
                active_task = None

                measurement_mono = time.monotonic()
                measurement_wall = time.time()
                report["measurement_started_at"] = datetime.fromtimestamp(measurement_wall, timezone.utc).isoformat()
                await page.evaluate("value => { window.__acceptance.telemetry=[]; window.__acceptance.predictions=[]; window.__acceptance.horizons=[]; window.__warm.began=value.began; window.__warm.ends=value.ends; }",
                                    {"began": measurement_wall * 1000, "ends": (measurement_wall + args.seconds) * 1000})
                warm_sent = sender.sent
                async def send_live():
                    max_lag = 0.0
                    for index in range(args.seconds):
                        due = measurement_mono + index
                        max_lag = max(max_lag, await live_tick(sender, anchor, due))
                    await wait_until(measurement_mono + args.seconds)
                    report["actual_emission_s"] = time.monotonic() - measurement_mono
                    report["measurement_sender"] = {"sent": sender.sent - warm_sent, "expected": VEHICLES * args.seconds,
                                                     "max_sender_lag_s": max_lag}
                active_task = asyncio.create_task(send_live())
                next_sample = measurement_mono
                while not active_task.done():
                    await wait_until(next_sample)
                    await sample("measurement", measurement_mono)
                    next_sample += 5
                await active_task
                active_task = None
                for _ in range(100):
                    result = (await client.get("/api/v1/system/status")).json()
                    if result.get("telemetry_count") == sender.sent and result.get("durable_queue_depth") == 0 and result.get("ingress_queue_depth") == 0:
                        break
                    await asyncio.sleep(.1)
                await asyncio.sleep(2)
                await sample("measurement", measurement_mono)
                probe = await page.evaluate("() => ({telemetry:window.__acceptance.telemetry, predictions:window.__warm.predictions, sseSnapshots:window.__acceptance.sseSnapshots})")
                await page.screenshot(path=str(args.output.with_suffix(".png")), full_page=True)
                await browser.close()

                report["postwarm_predictions"] = probe["predictions"]
                report["browser"] = {"errors": errors, "sse_snapshots": probe["sseSnapshots"],
                                      "telemetry_received_to_rendered": distribution(probe["telemetry"]),
                                      "prediction_triggered_to_rendered": distribution([p["latency_ms"] for p in probe["predictions"]])}
                checks, history = history_checks(probe["predictions"], run_id)
                report["history_by_vehicle"] = history
                sql = await asyncio.to_thread(compose, "exec", "-T", "postgres", "psql", "-U", "predictor", "-d", "predictor",
                    "-X", "-qAt", "-v", "ON_ERROR_STOP=1", "-c", query_text(run_id), project=project, env=env, capture=True)
                report["durable_counts"] = json.loads(sql.stdout.strip())
                report["final_images"] = (await asyncio.to_thread(compose, "images", "--format", "json", project=project, env=env, capture=True)).stdout
                final = report["samples"][-1]["system"]
                for name, metric in report["browser"].items():
                    if name.endswith("to_rendered"):
                        checks[f"{name}_p95_le_2s"] = metric["p95_ms"] is not None and metric["p95_ms"] <= 2000 and metric["invalid_samples"] == 0
                checks.update({"warmup_complete": report["warmup"]["sent"] == VEHICLES * HISTORY_POINTS,
                    "measurement_complete": report["measurement_sender"]["sent"] == VEHICLES * args.seconds and report["actual_emission_s"] >= args.seconds,
                    "sender_kept_cadence": max(report["warmup"]["max_sender_lag_s"], report["measurement_sender"]["max_sender_lag_s"]) <= 1,
                    "no_sender_or_ingress_failures": sender.failures == 0 and final.get("ingress_rejected", 0) == 0,
                    "same_container_images": image_identity(report["images"]) == image_identity(report["final_images"]),
                    "runtime_code_matches_workspace": report["runtime_code_sha256"] == report["code_sha256"],
                    "durable_all_sent_applied": report["durable_counts"]["inbox_pending"] == 0 and all(report["durable_counts"][key] == sender.sent for key in
                        ("telemetry_events", "inbox_total", "inbox_applied", "checkpoint_telemetry_count")),
                    "no_js_errors": not errors})
                checks.update(continuity_checks(report))
                report.update(checks=checks, passed=all(checks.values()), sender={"sent": sender.sent, "failures": sender.failures, "connections": sender.connections})
    except Exception as error:
        report.update(passed=False, error=f"{type(error).__name__}: {error}")
        if created:
            with contextlib.suppress(Exception):
                report["logs"] = (await asyncio.to_thread(compose, "logs", "--tail", "60", project=project, env=env, capture=True)).stdout
    finally:
        if active_task:
            await cancel_sender_task(active_task)
        if sender:
            await sender.disconnect()
        if created:
            try:
                await asyncio.to_thread(compose, "down", "--volumes", project=project, env=env, capture=True)
                report["owned_resources_removed"] = True
            except Exception as error:
                report.update(passed=False, cleanup_error=str(error))
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        save()
    print(json.dumps({key: report.get(key) for key in ("passed", "full_acceptance", "project", "warmup", "measurement_sender", "browser", "checks", "error", "owned_resources_removed")}, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=int, default=120, help="Postwarm live measurement seconds (minimum 120)")
    parser.add_argument("--output", type=Path, default=Path("artifacts/acceptance/warm-history.json"))
    args = parser.parse_args()
    if args.seconds < 120:
        parser.error("At least 120s are required to observe multiple model cycles per vehicle")
    raise SystemExit(0 if asyncio.run(run(args))["passed"] else 1)


if __name__ == "__main__":
    main()
