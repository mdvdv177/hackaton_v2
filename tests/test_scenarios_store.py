from copy import deepcopy
from datetime import datetime, timezone
import json
import sqlite3

import pytest
from sqlalchemy import select, update

from backend.scenarios import demo_package, scenario_network, validate_package
from backend.storage import Store, SchemaMigration, Telemetry, Prediction, Incident, Inbox


def package():
    return {"source_id": "fixture", "planned_visits": [
        {"tt_action_item_id": "a", "tr_id": "v", "time_begin": "2026-01-06T07:00:00Z", "geom": "POINT (37.6 55.7)", "building_address": "A", "time_fact_begin": "DO NOT COPY"},
        {"tt_action_item_id": "b", "tr_id": "v", "time_begin": "2026-01-06T07:13:00Z", "geom": "POINT (37.61 55.71)", "building_address": "B"}], "device_bindings": {"123": "v"}}


def test_versioned_import_excludes_facts_and_preserves_previous_version():
    store = Store("sqlite:///:memory:")
    first = validate_package(package())
    assert "time_fact_begin" not in first["plan"][0]
    assert store.save_scenario(first) == store.save_scenario(first)
    updated = package()
    updated["planned_visits"][0]["building_address"] = "Updated A"
    second = validate_package(updated)
    store.save_scenario(second)
    assert first["id"] != second["id"]
    assert store.load_scenario(first["id"])["plan"][0]["building_address"] == "A"
    bad = deepcopy(first)
    bad["name"] = "overwrite"
    with pytest.raises(ValueError, match="immutable"):
        store.save_scenario(bad)


@pytest.mark.parametrize("change", [
    lambda p: p["planned_visits"].append(p["planned_visits"][0]),
    lambda p: p["device_bindings"].update({"999": "unknown"}),
    lambda p: p["planned_visits"][0].update(geom="POINT (190 55)"),
    lambda p: p["planned_visits"][0].update(trip_id="unknown"),
])
def test_reject_invalid_package(change):
    value = package()
    change(value)
    with pytest.raises(ValueError):
        validate_package(value)


def test_network_all_vehicles_and_explicit_import_geometry():
    value = package()
    value["network"] = {"routes": [{"id": "r", "name": "Route"}], "segments": [{"id": "road", "route_id": "r", "from_visit_id": "a", "to_visit_id": "b", "geometry": {"type": "LineString", "coordinates": [[37.6, 55.7], [37.605, 55.706], [37.61, 55.71]]}}]}
    result = validate_package(value)
    assert result["network"]["geometry_kind"] == "supplied_route"
    assert result["network"]["segments"][0]["geometry"]["coordinates"][1] == [37.605, 55.706]
    value["network"]["segments"][0]["geometry"]["coordinates"].reverse()
    with pytest.raises(ValueError, match="direction"):
        validate_package(value)
    schematic = validate_package(package())["network"]
    assert schematic["geometry_kind"] == "schedule_schematic"
    assert schematic["segments"][0]["route_id"] is None


def test_ambiguous_and_missing_geometry_do_not_invent_segments():
    value = package()
    value["planned_visits"][1]["time_begin"] = value["planned_visits"][0]["time_begin"]
    result = validate_package(value)
    assert result["network"]["segments"] == []
    assert result["network"]["gaps"][0]["reason"] == "ambiguous_order"


def telemetry(identifier, time=100):
    return {"id": identifier, "run_id": "r", "tr_id": "v", "event_at": time,
            "payload": {"event_time": datetime.fromtimestamp(time, timezone.utc).isoformat(),
                        "received_at": datetime.fromtimestamp(100, timezone.utc).isoformat(), "packet_id": identifier}}


def test_inbox_future_does_not_block_ready_and_commit_is_recoverable(tmp_path):
    url = f"sqlite:///{tmp_path / 'state.db'}"
    store = Store(url)
    assert store.enqueue_telemetry([telemetry("future", 300), telemetry("ready")]) == ["future", "ready"]
    assert store.enqueue_telemetry([telemetry("ready")]) == []
    assert [r["id"] for r in store.pending_telemetry("r", 100)] == ["ready"]
    assert store.recent_applied_telemetry("r", 0) == []
    run = {"id": "r", "created_at": "2026-01-01T00:00:00Z", "last_applied": "ready"}
    store.commit_applied(["ready"], run)
    reopened = Store(url)
    assert reopened.latest_run() == run
    assert reopened.pending_telemetry("r", 100) == []
    assert [r["id"] for r in reopened.pending_telemetry("r", 300)] == ["future"]
    assert len(reopened.recent_applied_telemetry("r", 0)) == 1
    assert reopened.inbox_stats("r", 301) == {"pending_count": 1, "ready_count": 1, "oldest_ready_age_s": 1}


def test_failed_apply_rolls_back_checkpoint_and_pending_flags():
    store = Store("sqlite:///:memory:")
    store.enqueue_telemetry([telemetry("a")])
    run = {"id": "r", "created_at": "2026-01-01T00:00:00Z"}
    with pytest.raises(Exception):
        store.commit_applied(["a"], run, [{"id": "broken", "run_id": None, "payload": {}}])
    assert store.latest_run() is None
    assert len(store.pending_telemetry("r", 100)) == 1


def test_additive_migration_keeps_legacy_data(tmp_path):
    path = tmp_path / "legacy.db"
    url = f"sqlite:///{path}"
    # Build the actual pre-migration schema, without ORM defaults/new columns.
    old = telemetry("old")
    prediction = {"id": "p", "prediction_time": "2026-01-06T07:00:00Z", "timing_status": "legacy_unknown"}
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE runs (id VARCHAR PRIMARY KEY, created_at VARCHAR NOT NULL, payload JSON NOT NULL)")
        db.execute("CREATE TABLE telemetry_events (id VARCHAR PRIMARY KEY, run_id VARCHAR NOT NULL, tr_id VARCHAR NOT NULL, event_at FLOAT NOT NULL, payload JSON NOT NULL)")
        db.execute("CREATE TABLE predictions (id VARCHAR PRIMARY KEY, run_id VARCHAR NOT NULL, tr_id VARCHAR NOT NULL, payload JSON NOT NULL)")
        db.execute("INSERT INTO runs VALUES (?, ?, ?)", ("r", "2026-09-26T07:00:00Z", json.dumps({"id": "r"})))
        db.execute("INSERT INTO telemetry_events VALUES (?, ?, ?, ?, ?)", (old["id"], "r", "v", old["event_at"], json.dumps(old["payload"])))
        db.execute("INSERT INTO predictions VALUES (?, ?, ?, ?)", ("p", "r", "v", json.dumps(prediction)))
    reopened = Store(url)
    assert len(reopened.rows(Telemetry)) == 1
    assert len(reopened.recent_applied_telemetry("r", 0)) == 1
    with reopened.engine.connect() as connection:
        assert len(connection.execute(select(SchemaMigration.id)).all()) == 3
        assert connection.execute(select(Telemetry.stored_at)).scalar() == datetime(2026, 9, 26, 7, tzinfo=timezone.utc).timestamp()
    assert reopened.rows(Prediction) == [prediction]
    assert "published_at" not in reopened.rows(Prediction)[0]
    assert Store(url).rows(Telemetry) == [old["payload"]]


def test_retention_uses_storage_clock_and_protects_pending_and_active_window():
    store = Store("sqlite:///:memory:")
    now = datetime(2026, 9, 26, tzinfo=timezone.utc).timestamp()
    store.enqueue_telemetry([telemetry("pending"), telemetry("applied"), telemetry("active", 500)])
    store.commit_applied(["applied", "active"], {"id": "r", "created_at": "2026-09-26T00:00:00Z"})
    store.insert_telemetry([telemetry("newly_received_historical")])
    with store.engine.begin() as connection:
        connection.execute(update(Telemetry).where(Telemetry.id != "newly_received_historical").values(stored_at=now - 90000))
    store.put_many(Prediction, [{"id": "old_prediction", "run_id": "r", "tr_id": "v", "payload": {}, "stored_at": now - 8 * 86400}])
    store.put_many(Incident, [{"id": name, "run_id": "r", "payload": {"id": name, "status": status}, "stored_at": now - 8 * 86400}
                             for name, status in [("resolved", "resolved"), ("active_incident", "monitoring")]])
    result = store.prune(now, active_run_id="r", active_cutoff=400)
    assert result == {"telemetry_deleted": 1, "results_deleted": 2}
    with store.engine.connect() as connection:
        assert set(connection.scalars(select(Telemetry.id))) == {"pending", "active", "newly_received_historical"}
        assert set(connection.scalars(select(Inbox.event_id))) == {"pending", "active"}
    assert store.rows(Incident) == [{"id": "active_incident", "status": "monitoring"}]
    assert len(store.pending_telemetry("r", 600)) == 1
