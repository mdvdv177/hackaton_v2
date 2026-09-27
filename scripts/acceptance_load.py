"""Isolated real-time NDTP → model → rendered UI capacity acceptance.

Default is the agreed one-hour 100-device run. Short runs are diagnostics only.
Creates and cleans ONLY its own Compose project and newly-created volumes.
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import signal
import statistics
import subprocess
import time
import uuid

import httpx
from playwright.async_api import async_playwright

from scripts.ndtp_sender import FleetSender, load_scenario
from scripts.stack import compose, isolated_env


BROWSER_PROBE = """() => {
  window.__acceptance = {telemetry: [], predictions: [], renders: 0, modelIds: {},
    packets: {}, horizons: [], last: null, vehicles: {}, lastSseAt: null, sseSnapshots: 0};
  const NativeEventSource = window.EventSource;
  window.EventSource = class extends NativeEventSource {
    constructor(...args) {
      super(...args);
      this.addEventListener('snapshot', () => {
        window.__acceptance.lastSseAt = Date.now();
        window.__acceptance.sseSnapshots++;
      });
    }
  };
  window.addEventListener('dispatcher:rendered', event => {
    const a=window.__acceptance, s=event.detail.snapshot, now=Date.now();
    if (!s || !s.vehicles) return;
    a.renders++; a.last={at:now, event_id:s.event_id, vehicles:s.vehicles.length, run_id:s.run?.id};
    for (const v of s.vehicles) {
      const seen = a.vehicles[v.id] ||= {packetAt:null, modelAt:null};
      if (v.packet_id && a.packets[v.id] !== v.packet_id && v.received_at) {
        a.packets[v.id]=v.packet_id;
        a.telemetry.push(now-Date.parse(v.received_at));
        seen.packetAt=now;
      }
      const p=v.prediction;
      seen.currentModel=Boolean(p && p.source==='model' && p.current_prediction && p.timing_status==='verified');
      if (p && p.id && p.source==='model' && p.timing_status==='verified' && a.modelIds[v.id]!==p.id) {
        a.modelIds[v.id]=p.id;
        seen.modelAt=now;
        if (p.triggered_at) a.predictions.push(now-Date.parse(p.triggered_at));
        a.horizons.push(p.publication_horizon_s);
      }
    }
  });
}"""


def distribution(values):
    valid = sorted(v for v in values if isinstance(v, (float, int)) and not isinstance(v, bool)
                   and math.isfinite(v) and v >= 0)
    invalid = len(values) - len(valid)
    if not valid:
        return {"samples": 0, "invalid_samples": invalid, "p50_ms": None, "p95_ms": None, "max_ms": None}
    return {"samples": len(valid), "invalid_samples": invalid, "p50_ms": statistics.median(valid),
            "p95_ms": valid[max(0, math.ceil(len(valid) * .95) - 1)], "max_ms": max(valid)}


def browser_quality(probe: dict, vehicles: int, run_id: str) -> dict:
    """Freshness is measured on the browser's own clock, not the Backend clock."""
    now = probe["now_ms"]
    seen = probe.get("vehicles", {})
    expected = [f"load-{index:03}" for index in range(vehicles)]
    def recent(at, limit):
        return isinstance(at, (int, float)) and 0 <= now - at <= limit
    stale_packets = [identifier for identifier in expected if not recent(seen.get(identifier, {}).get("packetAt"), 10000)]
    stale_models = [identifier for identifier in expected if not recent(seen.get(identifier, {}).get("modelAt"), 60000)]
    last = probe.get("last") or {}
    return {"telemetry_coverage": (vehicles - len(stale_packets)) / vehicles,
            "model_coverage": (vehicles - len(stale_models)) / vehicles,
            "stale_telemetry_ids": stale_packets, "stale_model_ids": stale_models,
            "render_fresh": recent(last.get("at"), 10000),
            "sse_fresh": recent(probe.get("lastSseAt"), 10000),
            "run_matches": last.get("run_id") == run_id,
            "vehicle_count_matches": last.get("vehicles") == vehicles}


def continuity_checks(report: dict) -> dict:
    """A low latency on the first few renders cannot certify the rest of an hour."""
    samples = report["samples"]
    eligible = [sample for sample in samples if sample["elapsed_s"] >= 60]
    final = samples[-1]["browser_quality"]
    healthy = lambda value: (value["render_fresh"] and value["sse_fresh"] and value["run_matches"]
                            and value["vehicle_count_matches"] and value["telemetry_coverage"] == 1
                            and value["model_coverage"] == 1)
    gaps = [{"elapsed_s": sample["elapsed_s"], **sample["browser_quality"]}
            for sample in eligible if not healthy(sample["browser_quality"])]
    windows = []
    for minute in range(1, math.ceil(report["actual_emission_s"] / 60)):
        rows = [sample for sample in eligible if minute * 60 <= sample["elapsed_s"] < (minute + 1) * 60]
        windows.append({"minute": minute, "samples": len(rows),
                        "min_telemetry_coverage": min((s["browser_quality"]["telemetry_coverage"] for s in rows), default=0),
                        "min_model_coverage": min((s["browser_quality"]["model_coverage"] for s in rows), default=0)})
    observed = [0.0, *[s["elapsed_s"] for s in samples if s["elapsed_s"] <= report["actual_emission_s"]], report["actual_emission_s"]]
    max_gap = max((right - left for left, right in zip(observed, observed[1:])), default=float("inf"))
    report["ui_continuity"] = {"warmup_s": 60, "telemetry_max_age_s": 10, "model_max_age_s": 60,
                               "sampling_max_gap_s": max_gap, "samples_after_warmup": len(eligible),
                               "gaps": gaps, "minute_windows": windows, "final": final}
    return {"final_ui_fresh_all_vehicles": healthy(final),
            "continuous_ui_all_vehicles": (bool(eligible) if report["duration_s"] >= 60 else True) and not gaps,
            "ui_minute_coverage_ge_99pct": all(w["samples"] > 0 and w["min_telemetry_coverage"] >= .99
                                                and w["min_model_coverage"] >= .99 for w in windows),
            "measurement_sampling_gap_le_10s": max_gap <= 10}


def emission_checks(report: dict) -> dict:
    duration, vehicles = report["duration_s"], report["vehicles"]
    burst_seconds = [max(0, min(duration, end) - start) for start, end in ((1200, 1260), (2400, 2460))]
    expected = {"base": (duration - sum(burst_seconds)) * vehicles,
                "burst_1": burst_seconds[0] * vehicles * 5, "burst_2": burst_seconds[1] * vehicles * 5}
    actual = {key: sum(tick["sent"] for tick in report["emission_ticks"] if tick["phase"] == key) for key in expected}
    report["emission_counts"] = {"expected": expected, "actual": actual, "boundary_tolerance_events": vehicles}
    # At most one fleet tick may straddle a wall-clock phase boundary.
    return {"emitted_expected_workload": all(abs(actual[key] - count) <= vehicles for key, count in expected.items()),
            "sender_sustained_rate": report.get("sender_overruns", 0) == 0}


def image_identity(raw: str) -> list[str]:
    """Compose versions emit either an array or newline-delimited objects."""
    try:
        values = json.loads(raw)
    except json.JSONDecodeError:
        values = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if isinstance(values, dict):
        values = [values]
    return sorted(json.dumps(value, sort_keys=True) for value in values)


def stats(project, env):
    ids = compose("ps", "-q", project=project, env=env, capture=True).stdout.split()
    if not ids:
        return []
    result = subprocess.run(["docker", "stats", "--no-stream", "--format", "{{json .}}", *ids],
                            check=True, capture_output=True, text=True)
    return [json.loads(line) for line in result.stdout.splitlines() if line.strip()]


def memory_mib(value):
    match = re.match(r"([\d.]+)([A-Za-z]+)", value.split("/")[0].strip())
    if not match:
        raise ValueError(f"Unrecognized Docker memory measurement: {value}")
    amount, unit = match.groups()
    return float(amount) * {"B": 1 / 1048576, "KiB": 1 / 1024, "MiB": 1,
                            "GiB": 1024, "kB": 1000 / 1048576,
                            "MB": 1000000 / 1048576, "GB": 1000000000 / 1048576}[unit]


def capacity_checks(report):
    """Explicit hour-run gates; short diagnostics cannot certify sustained capacity."""
    final = report["samples"][-1]["system"]
    checks = {"model_opportunity_coverage_ge_99pct":
              final.get("model_opportunity_coverage") is not None and final["model_opportunity_coverage"] >= .99}
    memory = [{"elapsed_s": sample["elapsed_s"], "mib": memory_mib(row["MemUsage"])}
              for sample in report["resources"] for row in sample["containers"]
              if "backend" in row["Name"]]
    report["backend_memory"] = {"samples": len(memory), "max_mib": max((v["mib"] for v in memory), default=None)}
    checks["backend_memory_le_1gib"] = bool(memory) and report["backend_memory"]["max_mib"] <= 1024
    if report["full_acceptance"]:
        for end in (1260, 2460):
            checks[f"burst_{end}_recovered_within_30s"] = any(
                end <= sample["elapsed_s"] <= end + 30
                and sample["system"]["apply_lag_s"] <= 2
                and sample["system"]["apply_queue_depth"] <= report["vehicles"]
                and sample["system"]["ingress_queue_depth"] <= report["vehicles"]
                for sample in report["samples"])
        early = [v["mib"] for v in memory if 1800 <= v["elapsed_s"] < 2700]
        late = [v["mib"] for v in memory if v["elapsed_s"] >= 2700]
        # A bounded 16-minute window should plateau after warming. Allow 128 MiB
        # allocator headroom; retain every sample to make the limit reviewable.
        checks["memory_plateau"] = len(early) >= 10 and len(late) >= 10 and statistics.median(late) <= statistics.median(early) + 128
        measurements = [0.0, *[v["elapsed_s"] for v in memory if v["elapsed_s"] <= report["actual_emission_s"]], report["actual_emission_s"]]
        gaps = [right - left for left, right in zip(measurements, measurements[1:])]
        report["backend_memory"]["sampling_max_gap_s"] = max(gaps, default=float("inf"))
        checks["resource_sampling_complete"] = not report.get("resource_errors") and max(gaps, default=float("inf")) <= 90
        lags = [s["system"]["apply_lag_s"] for s in report["samples"] if s["elapsed_s"] >= 3000]
        checks["no_sustained_queue_lag"] = bool(lags) and statistics.median(lags) <= 1 and max(lags) <= 5
    return checks


async def run(args):
    project = f"transport-load-{uuid.uuid4().hex[:8]}"
    env = isolated_env()
    backend, dashboard = f"http://127.0.0.1:{env['BACKEND_PORT']}", f"http://127.0.0.1:{env['DASHBOARD_PORT']}"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    state_path = args.output.with_suffix(".running.json")
    state_path.write_text(json.dumps({"project": project, "env": env, "backend": backend, "dashboard": dashboard}))
    report = {"started_at": datetime.now(timezone.utc).isoformat(), "project": project, "environment": env,
              "duration_s": args.seconds, "vehicles": args.vehicles, "platform": platform.platform(),
              "full_acceptance": args.seconds >= 3600 and args.vehicles == 100,
              "requested_full_acceptance": args.seconds >= 3600 and args.vehicles == 100,
              "samples": [], "resources": [], "emission_ticks": []}
    report["code_sha256"] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                             for folder in ("backend", "predictor", "ml")
                             for path in sorted(Path(folder).glob("*.py"))}
    report["harness_sha256"] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in (Path(__file__), Path("scripts/ndtp_sender.py"))}
    report["frontend_source_sha256"] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                       for path in sorted(Path("frontend/src").rglob("*")) if path.is_file()}
    report["protocol"] = {"base_events_per_second": args.vehicles, "burst_events_per_second": args.vehicles * 5,
                          "burst_intervals_s": [[1200, 1260], [2400, 2460]],
                          "latency_endpoint": "React useLayoutEffect after applying snapshot; SSE freshness checked independently",
                          "latency_clock": "Backend UTC to browser Date.now; negative/nonfinite samples fail",
                          "ui_continuity": "5s samples after 60s warmup: every vehicle new packet <=10s, new verified model <=60s; live SSE <=10s",
                          "coverage": "first eligible targets with at least 30s fresh eligibility exposure",
                          "memory_limit": "Backend <= 1GiB; last15min median <= preceding15min +128MiB"}
    sender = None
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signum, stop.set)
    started = time.monotonic()
    try:
        await asyncio.to_thread(compose, "up", "-d", "--wait", project=project, env=env)
        report["startup_s"] = time.monotonic() - started
        report["images"] = (await asyncio.to_thread(compose, "images", "--format", "json", project=project, env=env, capture=True)).stdout
        runtime_hash_command = "import hashlib,json; from pathlib import Path; print(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for f in ('backend','predictor','ml') for p in sorted(Path(f).glob('*.py'))}))"
        report["runtime_code_sha256"] = json.loads((await asyncio.to_thread(compose, "exec", "-T", "backend", "python", "-c",
                                                                           runtime_hash_command, project=project, env=env, capture=True)).stdout)
        # Polling every 5s otherwise races Uvicorn's 5s keep-alive close.
        async with httpx.AsyncClient(base_url=backend, timeout=30, limits=httpx.Limits(keepalive_expiry=1)) as client:
            anchor = datetime.now(timezone.utc)
            scenario = load_scenario(args.vehicles, args.seconds, anchor)
            imported = await client.post("/api/v1/scenarios/import", json=scenario)
            imported.raise_for_status()
            scenario_id = imported.json()["scenario"]["id"]
            response = await client.post("/api/v1/live/start", json={"scenario_id": scenario_id})
            response.raise_for_status()
            report["run_id"] = response.json()["run"]["id"]
            report["scenario_id"] = scenario_id
            sender = FleetSender("127.0.0.1", int(env["NDTP_PORT"]), args.vehicles, anchor.timestamp())
            async with async_playwright() as browser_runtime:
                browser = await browser_runtime.chromium.launch(channel="chrome", headless=True)
                page = await browser.new_page(viewport={"width": 1440, "height": 900})
                errors = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                await page.add_init_script(f"({BROWSER_PROBE})()")
                await page.goto(dashboard, wait_until="domcontentloaded")
                report["frontend_assets"] = {}
                for asset in await page.locator('script[src], link[rel="stylesheet"]').evaluate_all("nodes => nodes.map(n => n.src || n.href)"):
                    resource = await page.request.get(asset)
                    report["frontend_assets"][asset.rsplit("/", 1)[-1]] = hashlib.sha256(await resource.body()).hexdigest()
                began = time.monotonic()
                next_tick = began
                next_sample = began
                next_stats = began + 15
                async def sample(elapsed):
                    result = await client.get("/api/v1/system/status")
                    result.raise_for_status()
                    probe = await page.evaluate("() => ({now_ms:Date.now(), telemetry:window.__acceptance.telemetry.slice(-1000), predictions:window.__acceptance.predictions.slice(-1000), renders:window.__acceptance.renders, vehicles:window.__acceptance.vehicles, last:window.__acceptance.last, lastSseAt:window.__acceptance.lastSseAt})")
                    report["samples"].append({"elapsed_s": time.monotonic() - began, "requested_elapsed_s": elapsed,
                                              "sent": sender.sent, "system": result.json(),
                                              "browser_quality": browser_quality(probe, args.vehicles, report["run_id"])})
                    report["recent_browser"] = {"telemetry": distribution(probe["telemetry"]),
                                                "predictions": distribution(probe["predictions"]), "renders": probe["renders"]}
                pending_stats = None
                while time.monotonic() - began < args.seconds and not stop.is_set():
                    now = time.monotonic()
                    elapsed = now - began
                    if now >= next_tick:
                        sent_before = sender.sent
                        await sender.tick(elapsed)
                        burst = (1200 <= elapsed < 1260) or (2400 <= elapsed < 2460)
                        report["emission_ticks"].append({"elapsed_s": elapsed, "sent": sender.sent - sent_before,
                                                        "phase": "burst_1" if 1200 <= elapsed < 1260 else "burst_2" if 2400 <= elapsed < 2460 else "base"})
                        next_tick += .2 if burst else 1
                        if next_tick < time.monotonic() - 1:
                            report.setdefault("sender_overruns", 0)
                            report["sender_overruns"] += 1
                            next_tick = time.monotonic()
                    if now >= next_sample:
                        await sample(elapsed)
                        next_sample += 5
                        state_path.write_text(json.dumps({"project": project, "env": env, "elapsed_s": elapsed,
                            "sent": sender.sent, "latest": report["samples"][-1]["system"],
                            "recent_browser": report["recent_browser"]}, ensure_ascii=False))
                    if pending_stats and pending_stats.done():
                        try:
                            report["resources"].append({"elapsed_s": elapsed, "containers": pending_stats.result()})
                        except Exception as error:
                            report.setdefault("resource_errors", []).append(str(error))
                        pending_stats = None
                    if now >= next_stats and not pending_stats:
                        pending_stats = asyncio.create_task(asyncio.to_thread(stats, project, env))
                        next_stats += 60
                    await asyncio.sleep(.02)
                report["actual_emission_s"] = time.monotonic() - began
                report["interrupted"] = stop.is_set()
                # Drain events already sent; do not count stale timeout as lost delivery.
                for _ in range(100):
                    await asyncio.sleep(.1)
                    latest = (await client.get("/api/v1/system/status")).json()
                    if latest.get("ingress_committed", 0) >= sender.sent and latest.get("apply_queue_depth", 1) == 0:
                        break
                await asyncio.sleep(2)
                probe = await page.evaluate("window.__acceptance")
                await page.screenshot(path=str(args.output.with_suffix(".png")), full_page=True)
                await sample(time.monotonic() - began)
                if pending_stats:
                    report["resources"].append({"elapsed_s": time.monotonic()-began, "containers": await pending_stats})
                await browser.close()
                report["browser"] = {"errors": errors, "renders": probe["renders"], "last": probe["last"],
                    "sse_snapshots": probe["sseSnapshots"],
                    "telemetry_received_to_rendered": distribution(probe["telemetry"]),
                    "prediction_triggered_to_rendered": distribution(probe["predictions"]),
                    "observed_model_predictions": len(probe["horizons"]),
                    "invalid_publication_horizons": sum(not isinstance(v, (int,float)) or not 600 < v <= 900 for v in probe["horizons"])}
                final = report["samples"][-1]["system"]
                report["sender"] = {"sent": sender.sent, "connection_failures": sender.failures, "connections": sender.connections}
                telemetry_ms = report["browser"]["telemetry_received_to_rendered"]
                predictions_ms = report["browser"]["prediction_triggered_to_rendered"]
                final_snapshot = (await client.get("/api/v1/snapshot")).json()
                report["final_run_id"] = (final_snapshot.get("run") or {}).get("id")
                report["final_images"] = (await asyncio.to_thread(compose, "images", "--format", "json", project=project, env=env, capture=True)).stdout
                report["checks"] = {
                    "requested_duration_completed": report["actual_emission_s"] >= args.seconds and not stop.is_set(),
                    "all_sent_committed": final.get("ingress_committed") == sender.sent,
                    "all_committed_applied": final.get("telemetry_count") == sender.sent,
                    "no_rejections": final.get("ingress_rejected", 0) == 0 and sender.failures == 0,
                    "drained": final.get("apply_queue_depth") == 0 and final.get("ingress_queue_depth") == 0 and final.get("durable_queue_depth") == 0,
                    "telemetry_p95_le_2s": telemetry_ms["p95_ms"] is not None and telemetry_ms["p95_ms"] <= 2000 and telemetry_ms["invalid_samples"] == 0,
                    "prediction_p95_le_2s": predictions_ms["p95_ms"] is not None and predictions_ms["p95_ms"] <= 2000 and predictions_ms["invalid_samples"] == 0,
                    "model_predictions_rendered": report["browser"]["observed_model_predictions"] > 0,
                    "all_horizons_valid": report["browser"]["invalid_publication_horizons"] == 0,
                    "no_js_errors": not errors,
                    "same_run_through_completion": report["final_run_id"] == report["run_id"],
                    "same_container_images": image_identity(report["images"]) == image_identity(report["final_images"]),
                    "runtime_code_matches_workspace": report["runtime_code_sha256"] == report["code_sha256"],
                }
                report["checks"].update(capacity_checks(report))
                report["checks"].update(continuity_checks(report))
                report["checks"].update(emission_checks(report))
                report["full_acceptance"] = report["full_acceptance"] and report["checks"]["requested_duration_completed"]
                report["passed"] = all(report["checks"].values())
    except Exception as error:
        report["passed"] = False
        report["error"] = f"{type(error).__name__}: {error}"
        try:
            report["logs"] = compose("logs", "--tail", "100", project=project, env=env, capture=True).stdout
        except Exception:
            pass
        raise
    finally:
        if sender:
            await sender.disconnect()
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        if not args.keep_stack:
            # Project ID is generated by this process; its volume has no user data.
            await asyncio.to_thread(compose, "down", "--volumes", project=project, env=env)
        for signum in (signal.SIGINT, signal.SIGTERM):
            loop.remove_signal_handler(signum)
    console = {key: value for key, value in report.items() if key not in {
        "samples", "resources", "logs", "emission_ticks", "code_sha256", "runtime_code_sha256",
        "frontend_source_sha256", "harness_sha256"}}
    if "ui_continuity" in console:
        console["ui_continuity"] = {key: value for key, value in report["ui_continuity"].items()
                                    if key not in {"minute_windows", "gaps"}}
        console["ui_continuity"]["gap_count"] = len(report["ui_continuity"].get("gaps", []))
    print(json.dumps(console, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=int, default=3600)
    parser.add_argument("--vehicles", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("artifacts/acceptance/load.json"))
    parser.add_argument("--keep-stack", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.vehicles <= 1000 or args.seconds < 1:
        parser.error("vehicles must be 1..1000 and seconds positive")
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
