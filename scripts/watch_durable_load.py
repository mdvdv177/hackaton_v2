"""Observe an existing load project's durable SQL state without changing services."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import signal
import time
import uuid

from scripts.stack import compose

PORT_KEYS = {"BACKEND_PORT", "DASHBOARD_PORT", "NDTP_PORT", "EMULATOR_PORT"}


def validate_identity(project: str, run_id: str, environment: dict) -> dict[str, str]:
    if not re.fullmatch(r"transport-load-[0-9a-f]{8}", project):
        raise ValueError("Expected an isolated transport-load-<8 hex digits> project")
    if str(uuid.UUID(run_id)) != run_id:
        raise ValueError("Expected a canonical lowercase UUID run_id")
    if set(environment) != PORT_KEYS:
        raise ValueError("Only the four load-test port variables are accepted")
    if any(not str(value).isdigit() or not 1 <= int(value) <= 65535 for value in environment.values()):
        raise ValueError("Invalid load-test port")
    return {key: str(value) for key, value in environment.items()}


def query_text(run_id: str) -> str:
    if str(uuid.UUID(run_id)) != run_id:
        raise ValueError("Expected a canonical run UUID")
    # One MVCC snapshot covers raw rows, inbox state and the committed checkpoint.
    # There are no INSERT/UPDATE/DELETE/DDL statements or application API writes.
    return f"""BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY;
SET LOCAL statement_timeout = '3s';
WITH raw AS (
  SELECT count(*) AS total FROM telemetry_events WHERE run_id = '{run_id}'
), inbox AS (
  SELECT count(*) AS total,
         count(*) FILTER (WHERE applied_at IS NULL) AS pending,
         count(*) FILTER (WHERE applied_at IS NOT NULL) AS applied
  FROM telemetry_inbox WHERE run_id = '{run_id}'
), checkpoint AS (
  SELECT (payload->>'telemetry_count')::bigint AS telemetry_count,
         payload->>'virtual_time' AS virtual_time
  FROM runs WHERE id = '{run_id}'
)
SELECT json_build_object(
  'database_at', clock_timestamp(), 'run_id', '{run_id}',
  'transaction_read_only', current_setting('transaction_read_only'),
  'telemetry_events', raw.total, 'inbox_total', inbox.total,
  'inbox_pending', inbox.pending, 'inbox_applied', inbox.applied,
  'checkpoint_telemetry_count', checkpoint.telemetry_count,
  'checkpoint_virtual_time', checkpoint.virtual_time
) FROM raw CROSS JOIN inbox LEFT JOIN checkpoint ON true;
COMMIT;"""


def reconcile(samples: list[dict], final: dict, project: str, run_id: str) -> dict:
    sender = final.get("sender") or {}
    count = sender.get("sent")
    identity = final.get("project") == project and final.get("run_id") == run_id
    count_valid = isinstance(count, int) and not isinstance(count, bool) and count > 0
    matching = [index for index, sample in enumerate(samples)
                if identity and count_valid and sample.get("run_id") == run_id
                and sample.get("transaction_read_only") == "on"
                and sample.get("inbox_pending") == 0
                and all(sample.get(key) == count for key in
                        ("telemetry_events", "inbox_total", "inbox_applied", "checkpoint_telemetry_count"))]
    return {"passed": bool(matching), "identity_matches": identity, "final_sender_sent": count,
            "matching_sample_indices": matching, "load_passed": final.get("passed"),
            "load_full_acceptance": final.get("full_acceptance"),
            "load_finished_at": final.get("finished_at")}


def read_json(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else None
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def write_report(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def run(args) -> int:
    state = read_json(args.state)
    if not state or state.get("project") != args.expected_project:
        raise ValueError("Running state does not belong to the explicitly selected project")
    run_id = args.run_id or ((state.get("latest") or {}).get("run") or {}).get("id")
    if not isinstance(run_id, str):
        raise ValueError("Running state has no run_id yet; provide --run-id")
    environment = validate_identity(args.expected_project, run_id, state.get("env", {}))
    report = {"passed": False, "status": "RUNNING", "project": args.expected_project, "run_id": run_id,
              "started_at": datetime.now(timezone.utc).isoformat(), "state_file": str(args.state),
              "source_report": str(args.report), "environment": environment, "samples": [], "errors": [],
              "protocol": {"database": "PostgreSQL directly through Compose exec postgres psql",
                           "transaction": "REPEATABLE READ READ ONLY; one SELECT statement",
                           "sparse_interval_s": 30, "dense_from_elapsed_s": 3590, "dense_interval_s": .5,
                           "purpose": "Durable count reconciliation, independently of Backend cached counters",
                           "load_latency_scope": "SQL polling adds sparse reads, then ~2 reads/s during final 10s"}}
    stopped = False
    def stop(_signum, _frame):
        nonlocal stopped
        stopped = True
    previous_handlers = {signum: signal.signal(signum, stop) for signum in (signal.SIGINT, signal.SIGTERM)}
    started = time.monotonic()
    last_elapsed = float(state.get("elapsed_s", 0))
    origin = started - last_elapsed
    unavailable_since = None
    try:
        while not stopped and time.monotonic() - started <= args.max_wait_seconds:
            cycle_started = time.monotonic()
            latest = read_json(args.state)
            if latest and latest.get("project") != args.expected_project:
                raise ValueError("Running-state project changed; refusing to follow another project")
            if latest:
                latest_run = ((latest.get("latest") or {}).get("run") or {}).get("id")
                if latest_run and latest_run != run_id:
                    raise ValueError("Active run changed; refusing to combine different runs")
                value = latest.get("elapsed_s")
                if isinstance(value, (int, float)) and value > last_elapsed:
                    # The state file is updated once every 5s; its mtime supplies
                    # only cadence estimation, never proof of SQL durability.
                    age = max(0, time.time() - args.state.stat().st_mtime)
                    last_elapsed = value
                    origin = cycle_started - value - age
            elapsed = cycle_started - origin
            final = read_json(args.report)
            if final and final.get("project") == args.expected_project:
                result = reconcile(report["samples"], final, args.expected_project, run_id)
                report["reconciliation"] = result
                if result["passed"]:
                    report.update(passed=True, status="PASS")
                    break
            try:
                response = compose("exec", "-T", "postgres", "psql", "-U", "predictor", "-d", "predictor",
                                   "-X", "-qAt", "-v", "ON_ERROR_STOP=1", "-c", query_text(run_id),
                                   project=args.expected_project, env=environment, capture=True)
                sample = json.loads(response.stdout.strip())
                sample.update(observed_at=datetime.now(timezone.utc).isoformat(), estimated_elapsed_s=elapsed,
                              query_elapsed_ms=(time.monotonic() - cycle_started) * 1000)
                report["samples"].append(sample)
                unavailable_since = None
                print(json.dumps({"elapsed_s": round(elapsed, 2), "telemetry_events": sample["telemetry_events"],
                                  "inbox_pending": sample["inbox_pending"], "inbox_applied": sample["inbox_applied"],
                                  "checkpoint": sample["checkpoint_telemetry_count"]}), flush=True)
            except Exception as error:
                unavailable_since = unavailable_since or time.monotonic()
                report["errors"].append({"observed_at": datetime.now(timezone.utc).isoformat(),
                                         "estimated_elapsed_s": elapsed, "error": f"{type(error).__name__}: {error}"})
                if final and final.get("project") == args.expected_project:
                    report.update(status="FAIL", failure_reason="No captured SQL snapshot reconciles final sender count")
                    break
                if time.monotonic() - unavailable_since > 30:
                    report.update(status="FAIL", failure_reason="PostgreSQL unavailable and matching final report absent")
                    break
            write_report(args.output, report)
            interval = .5 if elapsed >= 3590 or final and final.get("project") == args.expected_project else 30
            # Interruptible, short sleeps; no actions that keep or change services.
            deadline = cycle_started + interval
            while not stopped and time.monotonic() < deadline:
                time.sleep(min(.5, deadline - time.monotonic()))
        else:
            report.update(status="INCOMPLETE", failure_reason="Observer interrupted or timed out")
    except Exception as error:
        report.update(status="FAIL", failure_reason=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_report(args.output, report)
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
    print(json.dumps({key: value for key, value in report.items() if key not in {"samples", "errors"}}, ensure_ascii=False, indent=2), flush=True)
    return 0 if report["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, default=Path("artifacts/acceptance/load.running.json"))
    parser.add_argument("--report", type=Path, default=Path("artifacts/acceptance/load.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/acceptance/load-durability.json"))
    parser.add_argument("--expected-project", required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--max-wait-seconds", type=float, default=4200)
    args = parser.parse_args()
    if args.max_wait_seconds <= 0:
        parser.error("max-wait-seconds must be positive")
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
