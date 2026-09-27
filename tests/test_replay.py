from __future__ import annotations

import asyncio
import csv
from datetime import timedelta

import httpx

from backend.data import iso, normalize_event, timestamp
from backend.service import Dispatcher
from backend.storage import Run, Store
from test_backend import miniature_dataset, service, mock_ml


def test_duplicate_and_late_packet_do_not_move_current_position(service):
    async def scenario():
        await service.start_replay("dispatcher", 1)
        event = dict(service.latest["v1"])
        before = service.metrics["telemetry_count"]
        await service.ingest_batch([event, event])
        assert service.metrics["telemetry_count"] == before
        late = {**event, "packet_id": "late", "event_time": "2026-01-06T06:59:45+00:00", "lat": 0.0}
        await service.ingest_batch([late])
        assert service.positions["v1"]["lat"] == 55.7
        assert service.metrics["late_events"] == 1
        await service.close()
    asyncio.run(scenario())


def test_replay_recovers_checkpoint_predictions_and_ack(service, miniature_dataset, tmp_path):
    database = f"sqlite:///{tmp_path / 'restart.db'}"
    async def scenario():
        first = Dispatcher(Store(database), miniature_dataset, "http://ml", httpx.AsyncClient(transport=httpx.MockTransport(mock_ml)))
        await first.start_replay("dispatcher", 1)
        await first.set_status("paused")
        run_id = first.run["id"]
        identifier = next(iter(first.incidents))
        await first.acknowledge(identifier)
        cursor = first.cursor
        await first.close()
        second = Dispatcher(Store(database), miniature_dataset, "http://ml", httpx.AsyncClient(transport=httpx.MockTransport(mock_ml)))
        await second.recover()
        assert second.run["id"] == run_id
        assert second.cursor == cursor
        assert second.predictions["v1"]["prediction_delay_s"] == 180
        assert second.incidents[identifier]["acknowledged"] is True
        assert second.run["status"] == "paused"
        await second.close()
    asyncio.run(scenario())


def test_official_evaluation_ignores_receive_time_delay(service, miniature_dataset):
    path = miniature_dataset / "test" / "traffic.csv"
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row["receive_time"] = iso(timestamp(row["event_time"]) + timedelta(hours=1))
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    async def scenario():
        await service.start_replay("evaluation", 1)
        assert service.predictions["v1"]["prediction_delay_s"] == 180
        assert service.latest["v1"]["event_time"] == "2026-01-06T07:00:00+00:00"
        await service.close()
    asyncio.run(scenario())


def test_live_requires_explicit_mode_and_binding(service):
    async def scenario():
        event = {"unit_id": "unknown", "event_time": "2026-01-06T07:00:00Z", "location_valid": True, "lat": 55.7, "lon": 37.6}
        await service.ingest_live(event)
        assert service.metrics["ignored_live_events"] == 1
        await service.start_live(schedule_mode="demo_rebased")
        await service.ingest_live(event)
        assert service.metrics["unknown_devices"] == 1
        assert not service.latest
        await service.ingest_live({**event, "unit_id": "1"})
        await asyncio.wait_for(service.ingress.join(), 2)
        await service._apply_inbox()
        assert "v1" in service.latest
        await service.close()
    asyncio.run(scenario())


def test_received_future_event_waits_for_event_clock(service):
    async def scenario():
        await service.start_replay("dispatcher", 1)
        before = service.latest["v1"]["event_time"]
        event = {**service.latest["v1"], "packet_id": "clock-ahead", "event_time": "2026-01-06T07:00:20+00:00", "lat": 55.8}
        await service.ingest_batch([event])
        assert service.latest["v1"]["event_time"] == before
        assert service.future
        service.run["virtual_time"] = "2026-01-06T07:00:20+00:00"
        service._drain_future()
        assert service.latest["v1"]["event_time"] == event["event_time"]
        assert service.positions["v1"]["lat"] == 55.8
        await service.close()
    asyncio.run(scenario())


def test_arrival_requires_exit_before_reentering_same_stop(service):
    async def scenario():
        await service.start_replay("dispatcher", 1)
        assert service.visits["v1"]["target_visit_id"] == "past"
        assert service.visits["v1"]["departed"] is False
        next_visit = {**service.plans["v1"][0], "tt_action_item_id": "repeated", "time_begin": "2026-01-06T07:01:00+00:00"}
        service.plans["v1"].insert(1, next_visit)
        service.plan_ticks["v1"].insert(1, timestamp(next_visit["time_begin"]).timestamp())
        previous_state = service.observer.dump_state()
        service.plan.append(next_visit)
        service._index_plan()
        service.observer.restore_state(previous_state)
        service.visits, service.candidates, service.deviations = service.observer.visits, service.observer.candidates, service.observer.deviations
        service.run["virtual_time"] = "2026-01-06T07:01:20+00:00"
        event = {**service.latest["v1"], "event_time": "2026-01-06T07:00:30+00:00"}
        service._observe(event)
        service._observe({**event, "event_time": "2026-01-06T07:00:45+00:00"})
        assert service.visits["v1"]["target_visit_id"] == "past"
        service._observe({**event, "lat": 55.72, "event_time": "2026-01-06T07:00:50+00:00"})
        assert service.visits["v1"]["departed"] is True
        service._observe({**event, "event_time": "2026-01-06T07:01:00+00:00"})
        service._observe({**event, "event_time": "2026-01-06T07:01:15+00:00"})
        assert service.visits["v1"]["target_visit_id"] == "repeated"
        await service.close()
    asyncio.run(scenario())


def test_ambiguous_visits_and_long_gap_do_not_confirm(service):
    async def scenario():
        await service.start_replay("dispatcher", 1)
        service.visits.clear()
        service.deviations.clear()
        event = dict(service.latest["v1"])
        service._observe(event)
        service._observe({**event, "event_time": "2026-01-06T07:02:00+00:00"})
        assert not service.visits
        duplicate = {**service.plans["v1"][0], "tt_action_item_id": "ambiguous", "time_begin": "2026-01-06T07:01:00+00:00"}
        service.plans["v1"].insert(1, duplicate)
        service.plan_ticks["v1"].insert(1, timestamp(duplicate["time_begin"]).timestamp())
        service._observe({**event, "event_time": "2026-01-06T07:02:15+00:00"})
        service._observe({**event, "event_time": "2026-01-06T07:02:30+00:00"})
        assert not service.visits
        assert not service.deviations
        await service.close()
    asyncio.run(scenario())


def test_invalid_start_preserves_active_source_and_order(service, miniature_dataset):
    import shutil
    import pytest
    shutil.copytree(miniature_dataset / "test", miniature_dataset / "validate")
    (miniature_dataset / "validate" / "schedule.csv").rename(miniature_dataset / "validate" / "schedule_plan.csv")
    shutil.copyfile(miniature_dataset / "labels" / "labels_test.csv", miniature_dataset / "validate" / "points.csv")
    traffic = miniature_dataset / "validate" / "traffic.csv"
    traffic.write_text(traffic.read_text().replace("v1", "v2"))
    async def scenario():
        await service.start_replay("dispatcher", 1)
        identifier, cursor = service.run["id"], service.cursor
        ticks = list(service.event_ticks)
        with pytest.raises(ValueError):
            await service.start_replay("evaluation", 20, "2099-01-01T00:00:00Z", "validate")
        assert service.run["id"] == identifier
        assert service.run["status"] == "running"
        assert service.loaded_source == "test"
        assert service.event_ticks == ticks
        assert service.cursor == cursor
        assert service.traffic[0]["tr_id"] == "v1"
        with pytest.raises(ValueError):
            await service.start_replay("evaluation", 20, "not-a-date", "validate")
        assert service.loaded_source == "test"
        await service.close()
    asyncio.run(scenario())


def test_failed_live_persistence_retains_admission_and_retries(service):
    async def scenario():
        await service.start_live(schedule_mode="demo_rebased")
        original = service.store.enqueue_telemetry
        def failing(rows):
            raise RuntimeError("database offline")
        service.store.enqueue_telemetry = failing
        event = {"unit_id": "1", "packet_id": "retryable", "event_time": iso(service.now), "lat": 55.7, "lon": 37.6, "location_valid": True}
        await service.ingest_live(event)
        await asyncio.sleep(.2)
        assert service.metrics["ingress_admitted"] == 1
        assert service.metrics["ingress_committed"] == 0
        assert service.metrics["telemetry_count"] == 0
        assert not service.latest
        assert not service.seen
        service.store.enqueue_telemetry = original
        await asyncio.wait_for(service.ingress.join(), 3)
        await service._apply_inbox()
        assert service.metrics["telemetry_count"] == 1
        assert service.database_ready is True
        assert service.run["status"] == "running"
        await service.close()
    asyncio.run(scenario())


def test_restart_keeps_last_known_position_beyond_rolling_window(miniature_dataset, tmp_path):
    database = f"sqlite:///{tmp_path / 'stale.db'}"
    async def scenario():
        first = Dispatcher(Store(database), miniature_dataset, "http://ml", httpx.AsyncClient(transport=httpx.MockTransport(mock_ml)))
        await first.start_replay("dispatcher", 1)
        first.run["virtual_time"] = "2026-01-06T07:40:00+00:00"
        first.run["status"] = "paused"
        await first.set_status("paused")
        await first.close()
        second = Dispatcher(Store(database), miniature_dataset, "http://ml", httpx.AsyncClient(transport=httpx.MockTransport(mock_ml)))
        await second.recover()
        vehicle = second.vehicle("v1")
        assert vehicle["lat"] == 55.7
        assert vehicle["stale"] is True
        assert vehicle["position_age_s"] >= 2400
        assert not second.events.get("v1")
        await second.close()
    asyncio.run(scenario())


def test_invalid_replay_preserves_active_ndtp_scenario(service):
    import copy
    import pytest

    async def scenario():
        await service.start_live(schedule_mode="demo_rebased")
        run_id, observer = service.run["id"], service.observer
        original = copy.deepcopy({"source": service.loaded_source, "plan": service.plan,
            "bindings": service.bindings, "network": service.network,
            "observer": observer.dump_state(), "traffic": service.traffic})
        for start in ("2099-01-01T00:00:00Z", "not-a-date"):
            with pytest.raises(ValueError):
                await service.start_replay("dispatcher", 20, start, "test")
            assert service.run["id"] == run_id
            assert service.run["mode"] == "ndtp" and service.run["status"] == "running"
            assert service.observer is observer
            assert {"source": service.loaded_source, "plan": service.plan,
                "bindings": service.bindings, "network": service.network,
                "observer": observer.dump_state(), "traffic": service.traffic} == original
        await service.close()
    asyncio.run(scenario())


def test_control_database_writes_do_not_hold_state_lock(service):
    async def scenario():
        for name in ("save_run", "put_many", "save_scenario", "commit_applied"):
            original = getattr(service.store, name)
            def guarded(*args, _original=original, **kwargs):
                assert not service.lock.locked()
                return _original(*args, **kwargs)
            setattr(service.store, name, guarded)
        await service.start_live(schedule_mode="demo_rebased")
        old_run = service.run["id"]
        old_observer = service.observer.dump_state()
        await service.start_replay("dispatcher", 1)
        stopped = next(row for row in service.store.rows(Run) if row["id"] == old_run)
        assert stopped["status"] == "stopped"
        assert stopped["observer_state"] == old_observer
        await service.set_status("paused")
        await service.set_status(speed=5)
        await service.close()
    asyncio.run(scenario())


def test_failed_pending_checkpoint_prevents_run_switch(service):
    import pytest

    async def scenario():
        await service.start_live(schedule_mode="demo_rebased")
        old_run, old_source, old_observer = service.run["id"], service.loaded_source, service.observer
        pending = ([], service._state_payload(), [])
        service.uncommitted = pending
        original = service.store.commit_applied
        def unavailable(*args):
            raise RuntimeError("durable apply unavailable")
        service.store.commit_applied = unavailable
        with pytest.raises(RuntimeError, match="durable apply"):
            await service.start_replay("dispatcher", 1)
        assert service.run["id"] == old_run and service.run["status"] == "running"
        assert service.loaded_source == old_source and service.observer is old_observer
        assert service.uncommitted is pending
        service.store.commit_applied = original
        await service.start_replay("dispatcher", 1)
        assert service.run["id"] != old_run and service.uncommitted is None
        await service.close()
    asyncio.run(scenario())


def test_controls_serialize_without_blocking_snapshot_on_database(service):
    import threading

    async def scenario():
        await service.start_replay("dispatcher", 1)
        await service.set_status("paused")
        entered, release = threading.Event(), threading.Event()
        original = service.store.save_run
        def slow_save(payload):
            entered.set()
            assert release.wait(2)
            return original(payload)
        service.store.save_run = slow_save
        first = asyncio.create_task(service.set_status(speed=5))
        second = None
        try:
            assert await asyncio.to_thread(entered.wait, 1)
            second = asyncio.create_task(service.set_status(speed=20))
            await asyncio.sleep(.02)
            assert not service.lock.locked()
            assert service.snapshot()["run"]["speed"] == 5
            assert not second.done()
        finally:
            release.set()
            await first
            if second:
                await second
            service.store.save_run = original
        assert service.run["speed"] == 20
        await service.close()
    asyncio.run(scenario())


def test_ack_survives_abrupt_dispatcher_recreation(miniature_dataset, tmp_path):
    import copy
    database = f"sqlite:///{tmp_path / 'abrupt-ack.db'}"

    async def scenario():
        first = Dispatcher(Store(database), miniature_dataset, "http://ml", httpx.AsyncClient(transport=httpx.MockTransport(mock_ml)))
        await first.start_replay("dispatcher", 1)
        await first.set_status("paused")
        identifier = next(iter(first.incidents))
        await first.acknowledge(identifier)
        acknowledged = copy.deepcopy(first.incidents[identifier])
        # Cancel tasks directly: deliberately skip close() and its final checkpoint.
        tasks = [task for task in (first.task, first.ingress_task, first.publisher_task,
            first.prediction_task, first.maintenance_task) if task]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        second = Dispatcher(Store(database), miniature_dataset, "http://ml", httpx.AsyncClient(transport=httpx.MockTransport(mock_ml)))
        try:
            await second.recover()
            recovered = second.incidents[identifier]
            assert recovered["acknowledged"] is True
            assert recovered["acknowledged_at"] == acknowledged["acknowledged_at"]
            assert recovered["prediction_id"] == acknowledged["prediction_id"]
            assert recovered["prediction"] == acknowledged["prediction"]
        finally:
            await second.close()
            await first.client.aclose()
            await second.client.aclose()
    asyncio.run(scenario())
