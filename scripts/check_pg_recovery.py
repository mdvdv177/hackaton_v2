"""Isolated 30-second PostgreSQL outage, measured through committed application."""
from __future__ import annotations

import asyncio
import contextlib
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import uuid

import httpx

from scripts.ndtp_sender import FleetSender, load_scenario
from scripts.stack import compose, isolated_env
from scripts.watch_durable_load import query_text


async def run(output: Path) -> dict:
    from scripts.check_faults import until

    project, env = f"transport-pg-{uuid.uuid4().hex[:10]}", isolated_env()
    report = {"passed": False, "project": project, "environment": env,
              "started_at": datetime.now(timezone.utc).isoformat(), "sql_samples": [],
              "protocol": "1 vehicle/second; 30s PostgreSQL outage; continued NDTP; recovery from compose start through committed inbox and run checkpoint, <=10s."}
    sender, task = None, None
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        await asyncio.to_thread(compose, "up", "-d", "--no-build", "--pull", "never", "--wait", project=project, env=env, capture=True)
        report["images"] = json.loads((await asyncio.to_thread(compose, "images", "--format", "json", project=project, env=env, capture=True)).stdout)
        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{env['BACKEND_PORT']}", timeout=3,
                                     limits=httpx.Limits(keepalive_expiry=1)) as client:
            async def snapshot():
                response = await client.get("/api/v1/snapshot")
                response.raise_for_status()
                return response.json()

            async def ready():
                return (await client.get("/health/ready")).status_code

            anchor = datetime.now(timezone.utc)
            imported = await client.post("/api/v1/scenarios/import", json=load_scenario(1, 600, anchor))
            imported.raise_for_status()
            started = await client.post("/api/v1/live/start", json={"scenario_id": imported.json()["scenario"]["id"]})
            started.raise_for_status()
            run_id = started.json()["run"]["id"]
            report["run_id"] = run_id
            sender = FleetSender("127.0.0.1", int(env["NDTP_PORT"]), 1, anchor.timestamp())

            async def emit():
                while True:
                    await sender.tick(time.time() - anchor.timestamp())
                    await asyncio.sleep(1)

            task = asyncio.create_task(emit())
            before = await until(snapshot, lambda state: state["system"]["telemetry_count"] >= 3, timeout=15)
            await asyncio.to_thread(compose, "stop", "postgres", project=project, env=env, capture=True)
            await until(ready, lambda status: status == 503, timeout=10)
            await asyncio.sleep(30)
            queued = await snapshot()
            target = queued["system"]["ingress_admitted"]
            report.update(admitted_barrier=target, admitted_during_outage=target - before["system"]["ingress_admitted"],
                          outage_snapshot=queued["system"])
            assert report["admitted_during_outage"] >= 25
            assert (await client.get("/health/live")).status_code == 200
            began = time.monotonic()
            await asyncio.to_thread(compose, "start", "postgres", project=project, env=env, capture=True)
            await until(ready, lambda status: status == 200, timeout=max(.1, 10 - (time.monotonic() - began)))
            report["readiness_recovery_s"] = time.monotonic() - began
            deadline = began + 10
            while time.monotonic() < deadline:
                sql = await asyncio.to_thread(compose, "exec", "-T", "postgres", "psql", "-U", "predictor", "-d", "predictor",
                    "-X", "-qAt", "-v", "ON_ERROR_STOP=1", "-c", query_text(run_id), project=project, env=env, capture=True)
                sample = json.loads(sql.stdout.strip())
                sample["recovery_elapsed_s"] = time.monotonic() - began
                report["sql_samples"].append(sample)
                current = (await snapshot())["system"]
                if (sample["transaction_read_only"] == "on" and sample["inbox_pending"] == 0
                        and all(sample[key] >= target for key in ("telemetry_events", "inbox_total", "inbox_applied", "checkpoint_telemetry_count"))
                        and current["ingress_queue_depth"] == 0 and current["telemetry_count"] >= target):
                    report.update(durable_recovery_s=time.monotonic() - began, final_system=current)
                    break
                await asyncio.sleep(.1)
            assert report.get("durable_recovery_s", float("inf")) <= 10, report
            assert current["ingress_rejected"] == 0 and sender.failures == 0
            report.update(passed=True, sender_packets=sender.sent, sender_failures=sender.failures)
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        with contextlib.suppress(Exception):
            report["logs"] = (await asyncio.to_thread(compose, "logs", "--tail", "40", project=project, env=env, capture=True)).stdout
    finally:
        if task:
            task.cancel()
            for error in await asyncio.gather(task, return_exceptions=True):
                if isinstance(error, BaseException) and not isinstance(error, asyncio.CancelledError):
                    report.update(passed=False, sender_task_error=f"{type(error).__name__}: {error}")
        if sender:
            await sender.disconnect()
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        serialized = json.dumps(report, ensure_ascii=False, indent=2)
        output.with_name(f"{output.stem}-{project.rsplit('-', 1)[-1]}.json").write_text(serialized)
        output.write_text(serialized)
        await asyncio.to_thread(compose, "down", "--volumes", project=project, env=env, capture=True)
    print(json.dumps({key: report.get(key) for key in ("passed", "project", "admitted_barrier", "admitted_during_outage",
        "readiness_recovery_s", "durable_recovery_s", "error")}, indent=2), flush=True)
    return report
