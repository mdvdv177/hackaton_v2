"""Publication-time guarantees, coverage, retrospective isolation and clocks."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import json

import httpx
import pytest

from backend.data import iso, timestamp
from backend.storage import Prediction
from test_backend import miniature_dataset, mock_ml, service


@pytest.fixture
def deterministic(service):
    elapsed = [0.0]
    wall = datetime(2026, 9, 26, 12, tzinfo=timezone.utc)
    service.monotonic = lambda: elapsed[0]
    service.wall_clock = lambda: wall + timedelta(seconds=elapsed[0])
    service.start_task = lambda: None
    return service, elapsed


def test_replay_clock_advances_during_work_and_freezes_only_on_pause(deterministic):
    dispatcher, elapsed = deterministic
    async def scenario():
        await dispatcher.start_replay("dispatcher", 20)
        start = dispatcher.now
        elapsed[0] += 1
        assert (dispatcher.now - start).total_seconds() == 20
        await dispatcher.set_status("paused")
        paused = dispatcher.now
        elapsed[0] += 50
        assert dispatcher.now == paused
        await dispatcher.set_status(speed=5)
        await dispatcher.set_status("running")
        assert dispatcher.now == paused
        elapsed[0] += 2
        assert (dispatcher.now - paused).total_seconds() == 10
        await dispatcher.close()
    asyncio.run(scenario())


@pytest.mark.parametrize("seconds,reason", [(600, "horizon"), (601, None), (900, None), (901, "horizon")])
def test_publication_boundary_is_left_open_right_closed(deterministic, seconds, reason):
    dispatcher, _ = deterministic
    async def scenario():
        await dispatcher.start_replay("dispatcher", 1)
        prediction = {**dispatcher.predictions["v1"], "target_time_begin": iso(dispatcher.now + timedelta(seconds=seconds))}
        assert dispatcher._publication_rejection(prediction) == reason
        await dispatcher.close()
    asyncio.run(scenario())


def configure_boundary(dispatcher):
    dispatcher.predictions.clear(); dispatcher.incidents.clear(); dispatcher.histories.clear(); dispatcher.opportunities.clear()
    target = next(row for row in dispatcher.plan if row["tt_action_item_id"] == "future")
    target["time_begin"] = iso(dispatcher.now + timedelta(seconds=601))
    dispatcher._index_plan()
    dispatcher.next_prediction = None
    return dispatcher._collect_due()


def test_delayed_inference_suppressed_and_failed_opportunity_visible(deterministic):
    dispatcher, elapsed = deterministic
    async def scenario():
        await dispatcher.start_replay("dispatcher", 1)
        points = configure_boundary(dispatcher)
        def delayed(request):
            elapsed[0] += 2
            return mock_ml(request)
        dispatcher.client = httpx.AsyncClient(transport=httpx.MockTransport(delayed))
        await dispatcher.predict(points)
        assert not dispatcher.predictions
        assert not dispatcher.incidents
        suppressed = dispatcher.histories["v1"][-1]
        assert suppressed["suppression_reason"] == "horizon"
        assert suppressed["published_at"] is None
        assert dispatcher.system()["eligible_opportunity_count"] == 1
        assert dispatcher.system()["opportunity_count"] == 0  # only one second of valid opportunity
        assert dispatcher.system()["opportunity_coverage"] is None
        assert dispatcher.metrics["suppressed_horizon"] == 1
        await dispatcher.close()
    asyncio.run(scenario())


def test_slow_database_commit_rechecks_horizon_before_visibility(deterministic):
    dispatcher, elapsed = deterministic
    async def scenario():
        await dispatcher.start_replay("dispatcher", 1)
        points = configure_boundary(dispatcher)
        original = dispatcher.store.commit_predictions
        first = [True]
        def slow_commit(predictions, incidents):
            original(predictions, incidents)
            if first[0]:
                first[0] = False
                elapsed[0] += 2
        dispatcher.store.commit_predictions = slow_commit
        await dispatcher.predict(points)
        assert not dispatcher.predictions and not dispatcher.incidents
        assert dispatcher.histories["v1"][-1]["suppression_reason"] == "horizon"
        saved = dispatcher.store.rows(Prediction, dispatcher.run["id"])
        assert any(row.get("suppression_reason") == "horizon" for row in saved)
        await dispatcher.close()
    asyncio.run(scenario())


def test_visit_confirmed_during_inference_cannot_create_alert(deterministic):
    dispatcher, elapsed = deterministic
    async def scenario():
        await dispatcher.start_replay("dispatcher", 1)
        dispatcher.predictions.clear(); dispatcher.incidents.clear()
        dispatcher.next_prediction = None
        points = dispatcher._collect_due()
        def arrive(request):
            elapsed[0] += 2
            for second in (1, 2):
                event = {**dispatcher.latest["v1"], "event_time": f"2026-01-06T07:00:0{second}+00:00",
                         "received_at": f"2026-01-06T07:00:0{second}+00:00", "lat": 55.8, "lon": 37.7}
                dispatcher._observe(event)
            return mock_ml(request)
        dispatcher.client = httpx.AsyncClient(transport=httpx.MockTransport(arrive))
        await dispatcher.predict(points)
        assert dispatcher.visits["v1"]["target_visit_id"] == "future"
        assert not dispatcher.predictions and not dispatcher.incidents
        assert dispatcher.histories["v1"][-1]["suppression_reason"] == "observed"
        dispatcher.next_prediction = None
        assert not dispatcher._collect_due()
        await dispatcher.close()
    asyncio.run(scenario())


def test_evaluation_catchup_never_creates_operational_alerts(deterministic):
    dispatcher, _ = deterministic
    async def scenario():
        await dispatcher.start_replay("evaluation", 1)
        dispatcher.run.update(status="paused", virtual_time="2026-01-06T07:20:00+00:00")
        dispatcher.point_cursor = 0
        await dispatcher._predict_due()
        result = dispatcher.predictions["v1"]
        assert result["prediction_time"] == "2026-01-06T07:00:00+00:00"
        assert result["timing_status"] == "retrospective"
        assert result["published_at"] is None
        assert not dispatcher.incidents
        assert dispatcher.vehicle("v1")["prediction"]["risk"] == "gray"
        assert dispatcher.system()["opportunity_count"] == 0
        await dispatcher.close()
    asyncio.run(scenario())


def test_existing_incident_monitored_after_prediction_window(deterministic):
    dispatcher, _ = deterministic
    async def scenario():
        await dispatcher.start_replay("dispatcher", 1)
        identifier = next(iter(dispatcher.incidents))
        first_alert = dispatcher.incidents[identifier]["first_alert_at"]
        dispatcher.run.update(status="paused", virtual_time="2026-01-06T07:04:00+00:00")
        dispatcher.latest["v1"]["event_time"] = iso(dispatcher.now)
        dispatcher._stale_incidents()
        result = dispatcher.incident_detail(identifier)
        assert result["status"] == "monitoring"
        assert result["risk"] == "gray" and result["last_known_risk"] == "red"
        assert result["p_late"] is None and result["last_known_p_late"] == .85
        assert result["first_alert_at"] == first_alert
        assert result["prediction_id"] == result["prediction"]["id"]
        dispatcher.run["virtual_time"] = "2026-01-06T07:14:00+00:00"
        dispatcher.latest["v1"]["event_time"] = iso(dispatcher.now)
        dispatcher._stale_incidents()
        assert dispatcher.incidents[identifier]["status"] == "monitoring"
        await dispatcher.close()
    asyncio.run(scenario())


def test_new_run_drops_old_inflight_result(deterministic):
    dispatcher, _ = deterministic
    async def scenario():
        await dispatcher.start_replay("dispatcher", 1)
        dispatcher.next_prediction = None
        points = dispatcher._collect_due()
        dispatcher.predictions.clear(); dispatcher.incidents.clear()
        def switched(request):
            dispatcher.run["id"] = "different-run"
            return mock_ml(request)
        dispatcher.client = httpx.AsyncClient(transport=httpx.MockTransport(switched))
        await dispatcher.predict(points)
        assert not dispatcher.predictions and not dispatcher.incidents
        assert dispatcher.metrics["suppressed_superseded"] >= 1
        await dispatcher.close()
    asyncio.run(scenario())


def test_ml_batches_are_bounded(deterministic):
    dispatcher, _ = deterministic
    async def scenario():
        await dispatcher.start_replay("evaluation", 1)
        sizes = []
        def counting(request):
            sizes.append(len(json.loads(request.content)["items"]))
            return mock_ml(request)
        dispatcher.client = httpx.AsyncClient(transport=httpx.MockTransport(counting))
        point = dispatcher.points[0]
        await dispatcher.predict([{**point, "sample_id": f"sample-{i}", "cur_dev_source": "provided"} for i in range(257)])
        assert sizes == [256, 1]
        assert dispatcher.metrics["max_ml_batch_size"] == 256
        await dispatcher.close()
    asyncio.run(scenario())


def test_concurrent_apply_checkpoint_does_not_suppress_fresh_prediction(deterministic):
    dispatcher, _ = deterministic
    async def scenario():
        await dispatcher.start_replay("dispatcher", 1)
        dispatcher.predictions.clear(); dispatcher.incidents.clear()
        dispatcher.next_prediction = None
        points = dispatcher._collect_due()
        # Raw telemetry is durable; only its reducer checkpoint is still pending.
        dispatcher.uncommitted = (["other-event"], dispatcher._state_payload(), [])
        await dispatcher.predict(points)
        assert dispatcher.predictions["v1"]["timing_status"] == "verified"
        assert dispatcher.metrics["suppressed_stale"] == 0
        dispatcher.uncommitted = None
        await dispatcher.close()
    asyncio.run(scenario())


def test_model_failure_is_counted_against_mature_first_target_opportunity(deterministic):
    dispatcher, elapsed = deterministic
    async def scenario():
        async def unavailable(request):
            return httpx.Response(503)
        dispatcher.client = httpx.AsyncClient(transport=httpx.MockTransport(unavailable))
        await dispatcher.start_replay("dispatcher", 1)
        # Fresh fallback is visible separately and must not masquerade as ML coverage.
        elapsed[0] += 30
        status = dispatcher.system()
        assert status["opportunity_count"] == 1
        assert status["model_opportunity_coverage"] == 0
        assert status["opportunity_coverage"] == 1
        assert status["ml_failures"] >= 1
        await dispatcher.close()
    asyncio.run(scenario())


def test_durable_apply_failure_replays_once_after_restart(deterministic, miniature_dataset):
    from backend.service import Dispatcher
    from backend.storage import ObservedVisit
    dispatcher, _ = deterministic
    async def scenario():
        await dispatcher.start_live(schedule_mode="demo_rebased")
        run_id = dispatcher.run["id"]
        raw = []
        for index, age in enumerate((60, 45)):
            event = {"tr_id": "v1", "unit_id": "1", "packet_id": f"durable-{index}",
                     "event_time": iso(dispatcher.now - timedelta(seconds=age)), "received_at": iso(dispatcher.now),
                     "lat": 55.7, "lon": 37.6, "speed": 5, "heading": 0, "location_valid": True}
            raw.append({"id": f"{run_id}:v1:{event['packet_id']}", "run_id": run_id, "tr_id": "v1",
                        "event_at": timestamp(event["event_time"]).timestamp(), "payload": event})
        dispatcher.store.enqueue_telemetry(raw)
        original = dispatcher.store.commit_applied
        def fail(*args):
            raise RuntimeError("apply checkpoint unavailable")
        dispatcher.store.commit_applied = fail
        with pytest.raises(RuntimeError):
            await dispatcher._apply_inbox()
        assert dispatcher.uncommitted
        assert dispatcher.store.inbox_stats(run_id, dispatcher.now.timestamp())["pending_count"] == 2
        await dispatcher.close()  # Must not checkpoint ahead of consumed inbox flags.
        dispatcher.store.commit_applied = original
        recovered = Dispatcher(dispatcher.store, miniature_dataset, "http://ml", httpx.AsyncClient(transport=httpx.MockTransport(mock_ml)))
        recovered.wall_clock = dispatcher.wall_clock
        recovered.start_task = lambda: None
        await recovered.recover()
        assert not recovered.latest
        await recovered._apply_inbox()
        assert recovered.metrics["telemetry_count"] == 2
        assert recovered.visits["v1"]["target_visit_id"] == "past"
        assert len(recovered.store.rows(ObservedVisit, run_id)) == 1
        await recovered._apply_inbox()
        assert recovered.metrics["telemetry_count"] == 2
        assert recovered.store.inbox_stats(run_id, recovered.now.timestamp())["pending_count"] == 0
        await recovered.close()
    asyncio.run(scenario())


def test_legacy_prediction_recovers_without_invented_publication_time(deterministic, miniature_dataset):
    from backend.service import Dispatcher
    dispatcher, _ = deterministic
    async def scenario():
        await dispatcher.start_replay("dispatcher", 1)
        await dispatcher.set_status("paused")
        old = dict(dispatcher.predictions["v1"])
        for field in ("timing_status", "feature_cutoff_at", "generated_at", "published_at", "published_scenario_at", "publication_horizon_s"):
            old.pop(field, None)
        dispatcher.store.put_many(Prediction, dispatcher._prediction_rows([old]))
        await dispatcher.close()
        recovered = Dispatcher(dispatcher.store, miniature_dataset, "http://ml", httpx.AsyncClient(transport=httpx.MockTransport(mock_ml)))
        recovered.start_task = lambda: None
        await recovered.recover()
        public = recovered.vehicle("v1")["prediction"]
        assert public["timing_status"] == "unknown_legacy"
        assert public["published_at"] is None
        assert public["feature_cutoff_at"] == old["prediction_time"]
        assert public["risk"] == "gray" and public["p_late"] is None
        await recovered.close()
    asyncio.run(scenario())


def test_acknowledgement_failure_does_not_change_memory(deterministic):
    dispatcher, _ = deterministic
    async def scenario():
        await dispatcher.start_replay("dispatcher", 1)
        identifier = next(iter(dispatcher.incidents))
        original = dispatcher.store.put_many
        def unavailable(*args):
            raise RuntimeError("database unavailable")
        dispatcher.store.put_many = unavailable
        with pytest.raises(RuntimeError):
            await dispatcher.acknowledge(identifier)
        assert not dispatcher.incidents[identifier]["acknowledged"]
        dispatcher.store.put_many = original
        await dispatcher.close()
    asyncio.run(scenario())


def test_queued_incident_snapshot_cannot_revoke_committed_ack(deterministic):
    from copy import deepcopy
    from backend.storage import Incident
    dispatcher, _ = deterministic
    async def scenario():
        await dispatcher.start_replay("dispatcher", 1)
        identifier = next(iter(dispatcher.incidents))
        old = deepcopy(dispatcher.incidents[identifier])
        await dispatcher.acknowledge(identifier)
        await dispatcher._persist_predictions([], [{"id": identifier, "run_id": dispatcher.run["id"], "payload": old}])
        saved = dispatcher.store.rows(Incident, dispatcher.run["id"])[0]
        assert saved["acknowledged"] is True
        assert saved["acknowledged_at"] == dispatcher.incidents[identifier]["acknowledged_at"]
        await dispatcher.close()
    asyncio.run(scenario())


def test_queued_prediction_keeps_segment_features_from_cutoff(deterministic):
    dispatcher, _ = deterministic
    async def scenario():
        await dispatcher.start_replay("dispatcher", 1)
        dispatcher.next_prediction = None
        points = dispatcher._collect_due()
        assert points[0]["_segment_features"]["segment_speed_kmh"] is None
        def future_segment(*args):
            return {"segment_speed_kmh": 199, "speed_coverage": 1, "elapsed_s": 90, "max_gap_s": 1}
        dispatcher.observer.segment = future_segment
        seen = []
        def record(request):
            seen.append(json.loads(request.content)["items"][0]["features"])
            return mock_ml(request)
        dispatcher.client = httpx.AsyncClient(transport=httpx.MockTransport(record))
        await dispatcher.predict(points)
        assert seen[0]["segment_speed_kmh"] is None
        assert seen[0]["segment_speed_missing"] == 1
        await dispatcher.close()
    asyncio.run(scenario())


def test_prepared_history_filters_event_and_receive_cutoffs_with_late_events(deterministic):
    import math
    from backend.data import normalize_event
    from predictor.features import build_features
    from predictor.stream_features import enrich_stream_features

    dispatcher, _ = deterministic
    async def scenario():
        await dispatcher.start_replay("dispatcher", 1)
        await dispatcher.set_status("paused")
        dispatcher.next_prediction = None
        points = dispatcher._collect_due()
        point, cutoff = points[0], timestamp(points[0]["T"])
        dispatcher.run["virtual_time"] = iso(cutoff + timedelta(seconds=3))
        template = dispatcher.latest["v1"]
        for packet, event_delta, received_delta, speed in (("late-ok", -2, -1, 13),
                ("received-after-cutoff", -1, 2, 199), ("event-after-cutoff", 2, -1, 198)):
            dispatcher._accept(normalize_event({**template, "packet_id": packet, "speed": speed,
                "event_time": iso(cutoff + timedelta(seconds=event_delta)),
                "received_at": iso(cutoff + timedelta(seconds=received_delta))}))
        eligible = [row for row in dispatcher.events["v1"] if timestamp(row["event_time"]) <= cutoff and timestamp(row["received_at"]) <= cutoff]
        expected = enrich_stream_features(build_features(eligible, dispatcher.plans["v1"], point), point["_segment_features"])
        expected = {key: float(value) if math.isfinite(float(value)) else None for key, value in expected.items()}
        seen = []
        def record(request):
            seen.append(json.loads(request.content)["items"][0]["features"])
            return mock_ml(request)
        dispatcher.client = httpx.AsyncClient(transport=httpx.MockTransport(record))
        await dispatcher.predict(points)
        assert seen[0] == expected
        ticks = [row.event_ns for row in dispatcher.prepared_events["v1"]]
        assert ticks == sorted(ticks)
        await dispatcher.close()
    asyncio.run(scenario())


def test_feature_worker_keeps_immutable_snapshot_and_loop_responsive(deterministic):
    import threading
    from backend.data import normalize_event

    dispatcher, _ = deterministic
    async def scenario():
        await dispatcher.start_replay("dispatcher", 1)
        await dispatcher.set_status("paused")
        dispatcher.next_prediction = None
        points = dispatcher._collect_due()
        count = len(dispatcher.prepared_events["v1"])
        entered, release = threading.Event(), threading.Event()
        original = dispatcher.feature_plan.with_prepared_histories
        def slow(history, cutoff):
            entered.set()
            assert release.wait(2)
            return original(history, cutoff)
        dispatcher.feature_plan.with_prepared_histories = slow
        prediction = asyncio.create_task(dispatcher.predict(points))
        try:
            assert await asyncio.to_thread(entered.wait, 1)
            assert not prediction.done()
            dispatcher._accept(normalize_event({**dispatcher.latest["v1"], "packet_id": "arrived-during-feature-build", "speed": 199}))
            before = dispatcher.sequence
            await asyncio.wait_for(dispatcher.publish(), .2)
            assert dispatcher.sequence == before + 1
            assert len(dispatcher.prepared_events["v1"]) == count + 1
        finally:
            release.set()
            await prediction
        assert dispatcher.predictions["v1"]["data_quality"]["history_points"] == count
        stages = dispatcher.system()["prediction_stages_ms"]
        assert all(stages[key] >= 0 for key in ("scheduler_wait_ms", "snapshot_ms", "feature_build_ms", "feature_wait_ms",
            "inference_ms", "candidate_commit_ms", "publication_ms", "final_commit_ms", "total_ms"))
        assert dispatcher.system()["prediction_stages_cutoff_at"] == points[0]["T"]
        await dispatcher.close()
    asyncio.run(scenario())


def test_encoded_sse_retains_snapshot_and_reconnect_returns_current_state(deterministic):
    from backend.app import create_app

    dispatcher, _ = deterministic
    class Request:
        async def is_disconnected(self):
            return False

    async def scenario():
        await dispatcher.start_replay("dispatcher", 1)
        original_id, original_wire = dispatcher.stream[-1]
        original = json.loads(original_wire)
        dispatcher.latest["v1"] = {**dispatcher.latest["v1"], "speed": 19}
        await dispatcher.publish()
        assert json.loads(original_wire) == original
        app = create_app(dispatcher, start_listener=False)
        app.state.dispatcher = dispatcher
        endpoint = next(route.endpoint for route in app.routes if route.path == "/api/v1/events")
        backlog = await endpoint(Request(), last_event_id=str(original_id), since=None)
        wire = await anext(backlog.body_iterator)
        data = json.loads(next(line[6:] for line in wire.splitlines() if line.startswith("data: ")))
        assert isinstance(data, dict) and data["vehicles"][0]["speed"] == 19
        assert data["event_id"] == dispatcher.sequence
        await backlog.body_iterator.aclose()
        reconnect = await endpoint(Request(), last_event_id=None, since=None)
        current = await anext(reconnect.body_iterator)
        assert json.loads(next(line[6:] for line in current.splitlines() if line.startswith("data: ")))["vehicles"][0]["speed"] == 19
        await reconnect.body_iterator.aclose()
        reset = await endpoint(Request(), last_event_id=str(dispatcher.sequence + 100), since=None)
        assert "event: reset" in await anext(reset.body_iterator)
        await reset.body_iterator.aclose()
        await dispatcher.close()
    asyncio.run(scenario())


def test_sse_delayed_consumer_does_not_miss_already_published_frame(deterministic):
    from backend.app import create_app

    dispatcher, _ = deterministic
    class Request:
        async def is_disconnected(self):
            return False

    async def scenario():
        await dispatcher.start_replay("dispatcher", 1)
        app = create_app(dispatcher, start_listener=False)
        app.state.dispatcher = dispatcher
        endpoint = next(route.endpoint for route in app.routes if route.path == "/api/v1/events")
        response = await endpoint(Request(), last_event_id=str(dispatcher.sequence - 1), since=None)
        iterator = response.body_iterator
        try:
            first = await anext(iterator)
            first_id = dispatcher.sequence
            assert f"id: {first_id}\n" in first
            # The generator is suspended at yield while the browser consumes it.
            # Notification now precedes its next condition.wait().
            await dispatcher.publish()
            second = await asyncio.wait_for(anext(iterator), timeout=.1)
            assert f"id: {first_id + 1}\n" in second
            data = json.loads(next(line[6:] for line in second.splitlines() if line.startswith("data: ")))
            assert data["event_id"] == first_id + 1
        finally:
            await iterator.aclose()
            await dispatcher.close()
    asyncio.run(scenario())
