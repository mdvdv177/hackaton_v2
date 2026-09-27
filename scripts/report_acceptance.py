"""Render saved acceptance measurements; never starts or changes running services."""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path


def release_images(value) -> list[tuple[str, str]]:
    """Compare application/DB images across projects; emulator is a separate input."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = [json.loads(line) for line in value.splitlines() if line.strip()]
    if isinstance(value, dict):
        value = [value]
    repositories = {"transport-predictor-python", "transport-predictor-frontend", "postgres"}
    return sorted((row["Repository"], row["ID"]) for row in value or []
                  if row.get("Repository") in repositories and row.get("ID"))


def plot_load(folder: Path, load: dict) -> str | None:
    """Plot actual samples only; an absent hour must not produce an empty chart."""
    samples = load.get("samples", [])
    if not samples:
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scripts.acceptance_load import memory_mib

    figure, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True, layout="constrained")
    minutes = [row["elapsed_s"] / 60 for row in samples]
    for key, label in (("ingress_queue_depth", "Ingress"), ("apply_queue_depth", "Ready inbox"), ("ml_queue_depth", "ML jobs")):
        axes[0].plot(minutes, [row["system"].get(key, math.nan) for row in samples], label=label, linewidth=1)
    state = "PASS" if load.get("passed") else "FAILED / INCOMPLETE"
    protocol = "full-hour protocol" if load.get("full_acceptance") else "short diagnostic"
    axes[0].set(ylabel="Queued events / jobs", title=f"NDTP {protocol}: {state} · measured queues, lag and memory")
    axes[0].legend(loc="upper left", ncols=3)
    axes[1].plot(minutes, [row["system"].get("apply_lag_s", math.nan) for row in samples], color="#a63c32")
    axes[1].set(ylabel="Ready-event lag (s)")
    for service in ("backend", "ml", "postgres", "frontend"):
        values = [(sample["elapsed_s"] / 60, memory_mib(row["MemUsage"]))
                  for sample in load.get("resources", []) for row in sample["containers"]
                  if f"-{service}-" in row["Name"]]
        if values:
            axes[2].plot(*zip(*values), label=service, linewidth=1.5)
    axes[2].set(xlabel="Elapsed real time (minutes)", ylabel="Container RAM (MiB)")
    if axes[2].lines:
        axes[2].legend(loc="upper left", ncols=4)
    for axis in axes:
        for start, end in ((20, 21), (40, 41)):
            axis.axvspan(start, end, color="#e6aa31", alpha=.15)
        axis.grid(alpha=.2)
        axis.set_xlim(0, max(1, max(minutes, default=60)))
    figure.savefig(folder / "load-metrics.png", dpi=160)
    plt.close(figure)
    return "load-metrics.png"


def generate(folder: Path) -> dict:
    load_path = folder / "load.json"
    load = json.loads(load_path.read_text()) if load_path.exists() else None
    summary = {"generated_at": datetime.now(timezone.utc).isoformat()}
    if load is not None:
        summary["load"] = {key: load.get(key) for key in (
            "passed", "full_acceptance", "duration_s", "actual_emission_s", "interrupted", "vehicles", "sender",
            "checks", "browser", "backend_memory", "error", "started_at", "finished_at", "project", "run_id",
            "images", "code_sha256", "protocol", "ui_continuity")}
        samples = load.get("samples", [])
        summary["load"]["final_system"] = samples[-1]["system"] if samples else None
        summary["load"]["chart"] = plot_load(folder, load)
        summary["load"]["source_report"] = load_path.name
        reference_name, reference = "load", load
    else:
        status_path = folder / "load-status.json"
        status = json.loads(status_path.read_text()) if status_path.exists() else {"status": "NOT_RUN"}
        summary["load"] = {
            **status, "passed": None, "full_acceptance": False,
            "source_report": None, "chart": None, "final_system": None,
        }
        reference_name = "warm-history"
        reference_path = folder / f"{reference_name}.json"
        reference = json.loads(reference_path.read_text()) if reference_path.exists() else {}
    reference_images = release_images(reference.get("images"))
    release_checks = {}
    for name in ("faults", "cold_start", "ndtp_official", "demo_ndtp", "PG", "load-durability", "warm-history"):
        path = folder / f"{name}.json"
        if path.exists():
            result = json.loads(path.read_text())
            summary[name] = {key: result[key] for key in (
                "passed", "checks", "successful_runs", "ready_median_s", "ready_max_s",
                "first_prediction_ui_median_s", "first_prediction_ui_max_s", "first_prediction_ui_s",
                "elapsed_to_prediction_ui_s", "first_model_prediction_s", "images_unchanged",
                "images", "project", "created_at", "started_at", "finished_at", "error",
                "status", "run_id", "protocol", "admitted_barrier", "admitted_during_outage",
                "readiness_recovery_s", "durable_recovery_s", "sender_packets", "sender_failures",
                "reconciliation", "failure_reason", "errors", "full_acceptance", "kind",
                "browser", "browser_errors", "warmup", "measurement_sender", "durable_counts",
                "owned_resources_removed", "duration_s", "sender_stop_leaves_run_active",
                "derived_history_summary", "derived_postwarm_stages") if key in result}
            if name == "PG":
                summary[name]["final_sql_snapshot"] = next(reversed(result.get("sql_samples", [])), None)
            if name == "load-durability":
                matching = result.get("reconciliation", {}).get("matching_sample_indices", [])
                summary[name]["final_sql_snapshot"] = result["samples"][matching[-1]] if matching else None
            elif name == "cold_start":
                attempts = result.get("attempts", [])
                release_checks[name] = bool(attempts) and all(release_images(attempt.get("images")) == reference_images for attempt in attempts)
            else:
                release_checks[name] = release_images(result.get("images")) == reference_images
    required = {"faults", "cold_start", "ndtp_official", "demo_ndtp", "PG", "warm-history"}
    summary["release_consistency"] = {
        "reference_report": f"{reference_name}.json" if reference else None,
        "reference_images": reference_images, "matches_reference_images": release_checks,
        "missing_reports": sorted(required - release_checks.keys()),
        "passed": len(reference_images) == 4 and required <= release_checks.keys() and all(release_checks.values()),
    }
    summary["short_acceptance"] = {
        "reports": sorted(required),
        "passed": all(summary.get(name, {}).get("passed") is True for name in required),
        "confirms_hourly_stability": False,
    }
    summary["external_checks"] = {"human_five_second_usability": "NOT_RUN", "real_carrier_route_geometry": "NOT_SUPPLIED"}
    (folder / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, default=Path("artifacts/acceptance"))
    args = parser.parse_args()
    print(json.dumps(generate(args.folder), ensure_ascii=False, indent=2))
