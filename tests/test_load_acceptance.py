"""Negative controls for acceptance evidence; no Docker, browser or load starts."""
import copy

import pytest

from scripts.acceptance_load import browser_quality, capacity_checks, continuity_checks, distribution, emission_checks, image_identity


def probe(now_ms=120_000, vehicles=100):
    return {"now_ms": now_ms, "lastSseAt": now_ms - 100,
            "last": {"at": now_ms - 100, "vehicles": vehicles, "run_id": "current"},
            "vehicles": {f"load-{index:03}": {"packetAt": now_ms - 1000, "modelAt": now_ms - 30_000,
                                             "currentModel": True} for index in range(vehicles)}}


def healthy_report(duration=180):
    return {"duration_s": duration, "actual_emission_s": duration + .01, "vehicles": 100,
            "samples": [{"elapsed_s": elapsed, "browser_quality": browser_quality(probe(), 100, "current")}
                        for elapsed in range(0, duration + 1, 5)]}


def test_stable_browser_has_complete_minute_evidence():
    report = healthy_report()
    assert all(continuity_checks(report).values())
    assert all(window["min_telemetry_coverage"] == 1 for window in report["ui_continuity"]["minute_windows"])


def test_frozen_browser_cannot_pass_from_good_initial_latencies():
    report = healthy_report()
    initial = probe()
    for sample in report["samples"]:
        if sample["elapsed_s"] >= 60:
            frozen = {**initial, "now_ms": initial["now_ms"] + sample["elapsed_s"] * 1000}
            sample["browser_quality"] = browser_quality(frozen, 100, "current")
    checks = continuity_checks(report)
    assert not checks["continuous_ui_all_vehicles"]
    assert not checks["final_ui_fresh_all_vehicles"]
    assert not checks["ui_minute_coverage_ge_99pct"]
    assert report["ui_continuity"]["gaps"]


@pytest.mark.parametrize("failure", ["one_vehicle_packets", "one_vehicle_model", "sse", "wrong_run", "partial_fleet"])
def test_partial_or_polling_only_updates_are_explicit_failures(failure):
    value = probe()
    if failure == "one_vehicle_packets":
        value["vehicles"]["load-099"]["packetAt"] -= 20_000
    elif failure == "one_vehicle_model":
        value["vehicles"]["load-099"]["modelAt"] -= 60_000
    elif failure == "sse":
        value["lastSseAt"] -= 20_000
    elif failure == "wrong_run":
        value["last"]["run_id"] = "superseded"
    else:
        value["last"]["vehicles"] = 99
    report = healthy_report()
    report["samples"][-1]["browser_quality"] = browser_quality(value, 100, "current")
    assert not continuity_checks(report)["final_ui_fresh_all_vehicles"]
    assert report["ui_continuity"]["gaps"]


def test_target_boundary_does_not_invalidate_recent_model_flow():
    value = probe()
    value["vehicles"]["load-099"]["currentModel"] = False
    assert browser_quality(value, 100, "current")["model_coverage"] == 1


def test_missing_measurement_minute_does_not_certify_ui():
    report = healthy_report()
    report["samples"] = [sample for sample in report["samples"] if not 60 <= sample["elapsed_s"] < 120]
    checks = continuity_checks(report)
    assert not checks["ui_minute_coverage_ge_99pct"]
    assert not checks["measurement_sampling_gap_le_10s"]


def test_hour_resource_claim_requires_measurement_coverage():
    system = {"model_opportunity_coverage": 1, "apply_lag_s": 0, "apply_queue_depth": 0, "ingress_queue_depth": 0}
    report = {"full_acceptance": True, "actual_emission_s": 3600, "vehicles": 100,
              "samples": [{"elapsed_s": at, "system": system} for at in (1261, 2461, 3599)],
              "resources": [{"elapsed_s": at, "containers": [{"Name": "load-backend-1", "MemUsage": "100MiB / 4GiB"}]}
                            for at in (1801, 2701)]}
    checks = capacity_checks(report)
    assert not checks["resource_sampling_complete"]
    assert not checks["memory_plateau"]


def test_emitted_burst_volume_is_checked_separately_from_no_overruns():
    report = {"duration_s": 3600, "vehicles": 100, "sender_overruns": 0,
              "emission_ticks": [{"phase": phase, "sent": count} for phase, count in
                                 (("base", 348_000), ("burst_1", 30_000), ("burst_2", 30_000))]}
    assert all(emission_checks(report).values())
    report["emission_ticks"][-1]["sent"] = 6000
    assert not emission_checks(report)["emitted_expected_workload"]
    assert emission_checks(report)["sender_sustained_rate"]


def test_invalid_latencies_are_not_silently_used_as_fast_samples():
    result = distribution([0, 100, -2, float("nan"), float("inf"), None, True])
    assert result == {"samples": 2, "invalid_samples": 5, "p50_ms": 50, "p95_ms": 100, "max_ms": 100}
    assert distribution(list(range(1, 21)))["p95_ms"] == 19


def test_image_identity_is_order_independent_across_compose_formats():
    assert image_identity('[{"ID":"b"},{"ID":"a"}]') == image_identity('{"ID":"a"}\n{"ID":"b"}')
