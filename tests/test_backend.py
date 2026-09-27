from __future__ import annotations

import asyncio
import csv
from datetime import timedelta
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from backend.app import create_app
from backend.data import iso, safe_plan, timestamp
from backend.service import Dispatcher
from backend.storage import Plan, Prediction, Store


@pytest.fixture
def miniature_dataset(tmp_path: Path) -> Path:
    folder = tmp_path / "dataset"
    (folder / "test").mkdir(parents=True)
    (folder / "labels").mkdir()
    rows = [
        {"tt_action_item_id": "past", "tr_id": "v1", "time_begin": "2026-01-06 06:59:00", "time_fact_begin": "2099-01-01", "geom": "POINT (37.6 55.7)", "building_address": "Past"},
        {"tt_action_item_id": "future", "tr_id": "v1", "time_begin": "2026-01-06 07:13:00", "time_fact_begin": "2099-01-01", "geom": "POINT (37.7 55.8)", "building_address": "Target"},
    ]
    with (folder / "test" / "schedule.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    traffic = []
    for index, at in enumerate(["06:59:00", "06:59:15", "06:59:30", "07:00:00", "07:00:15", "07:20:00"]):
        traffic.append({"packet_id": str(index), "unit_id": "1", "tr_id": "v1", "event_time": f"2026-01-06 {at}",
                        "receive_time": f"2026-01-06 {at}", "lat": "55.7", "lon": "37.6", "speed": "5", "heading": "0", "location_valid": "True"})
    with (folder / "test" / "traffic.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=traffic[0].keys()); writer.writeheader(); writer.writerows(traffic)
    points = [{"sample_id": "sample", "tr_id": "v1", "T": "2026-01-06 07:00:00", "target_stop_id": "future",
               "target_time_begin": "2026-01-06 07:13:00", "cur_dev_s": "100", "target_delay_s": "99999"}]
    with (folder / "labels" / "labels_test.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=points[0].keys()); writer.writeheader(); writer.writerows(points)
    return folder


def mock_ml(request: httpx.Request) -> httpx.Response:
    import json
    payload = json.loads(request.content)
    return httpx.Response(200, json={"model_version": "test-model", "feature_schema_version": "1.0.0", "items": [
        {"request_id": item["request_id"], "prediction_delay_s": 180, "p_late": .85, "factors": [], "status": "ok"}
        for item in payload["items"]]})


@pytest.fixture
def service(miniature_dataset: Path) -> Dispatcher:
    client = httpx.AsyncClient(transport=httpx.MockTransport(mock_ml))
    return Dispatcher(Store("sqlite:///:memory:"), miniature_dataset, "http://ml", client)


def test_api_replay_incident_ack_and_run_isolation(service: Dispatcher):
    with TestClient(create_app(service, start_listener=False)) as client:
        assert client.get("/health/ready").status_code == 200
        assert client.get("/api/v1/snapshot").json()["run"] is None
        response = client.post("/api/v1/replay/start", json={"mode": "dispatcher", "speed": 1})
        assert response.status_code == 200, response.text
        state = response.json()
        run_id = state["run"]["id"]
        prediction = state["vehicles"][0]["prediction"]
        assert prediction["horizon_s"] == pytest.approx(780, abs=.1)
        assert 600 < prediction["publication_horizon_s"] <= prediction["horizon_s"]
        assert prediction["risk"] == "red"
        assert prediction["cur_dev_source"] == "estimated"
        identifier = state["incidents"][0]["id"]
        assert client.post(f"/api/v1/incidents/{identifier}/ack").json()["acknowledged"] is True
        assert client.post("/api/v1/replay/speed", json={"speed": 5}).json()["run"]["speed"] == 5
        assert client.post("/api/v1/replay/speed", json={"speed": 3}).status_code == 422
        assert client.post("/api/v1/replay/pause").json()["run"]["status"] == "paused"
        assert client.get("/api/v1/vehicles/v1").json()["planned_visits"]
        assert client.get("/api/v1/vehicles/missing").status_code == 404
        state2 = client.post("/api/v1/replay/start", json={"mode": "evaluation", "speed": 1}).json()
        assert state2["run"]["id"] != run_id
        assert all(not item["acknowledged"] for item in state2["incidents"])
        assert state2["vehicles"][0]["prediction"]["cur_dev_s"] == 100
        assert state2["vehicles"][0]["prediction"]["prediction_delay_s"] == 180
        assert client.post(f"/api/v1/incidents/{identifier}/ack").status_code == 404
        assert "dispatcher_processing_latency_p95_ms" in client.get("/metrics").text


def test_plan_import_excludes_actual_and_targets(service: Dispatcher):
    service.load_data()
    assert all("time_fact_begin" not in row for row in service.plan)
    assert all("time_fact_begin" not in row for row in service.store.rows(Plan))
    assert "target_delay_s" not in service.points[0]


def test_stale_telemetry_hides_probability_preserves_position(service: Dispatcher):
    async def scenario():
        await service.start_replay("dispatcher", 1)
        service.run["virtual_time"] = "2026-01-06T07:02:00+00:00"
        item = service.vehicle("v1")
        assert item["stale"] is True
        assert item["prediction"]["p_late"] is None
        assert item["prediction"]["risk"] == "gray"
        assert item["lat"] == 55.7
        service._stale_incidents()
        assert next(iter(service.incidents.values()))["status"] == "data_stale"
        await service.close()
    asyncio.run(scenario())


def test_risk_changes_immediately_without_hysteresis(service: Dispatcher):
    async def scenario():
        await service.start_replay("dispatcher", 1)
        initial = service.predictions["v1"]
        low = {**initial, "risk": "green", "p_late": .1}
        service._incident(low)
        assert next(iter(service.incidents.values()))["risk"] == "green"
        assert next(iter(service.incidents.values()))["status"] == "resolved"
        await service.close()
    asyncio.run(scenario())


def test_model_timeout_fallback_does_not_invent_probability(service: Dispatcher):
    async def unavailable(request: httpx.Request):
        raise httpx.ReadTimeout("down", request=request)
    service.client = httpx.AsyncClient(transport=httpx.MockTransport(unavailable))
    async def scenario():
        await service.start_replay("evaluation", 1)
        prediction = service.predictions["v1"]
        assert prediction["prediction_delay_s"] == 100
        assert prediction["p_late"] is None
        assert prediction["risk"] == "gray"
        assert prediction["source"] == "baseline"
        await service.close()
    asyncio.run(scenario())


def test_no_current_deviation_does_not_become_zero(service: Dispatcher):
    async def unavailable(request: httpx.Request):
        return httpx.Response(503)
    service.client = httpx.AsyncClient(transport=httpx.MockTransport(unavailable))
    async def scenario():
        await service.start_replay("dispatcher", 1)
        service.deviations.clear()
        service.predictions.clear()
        await service.predict([{**service.points[0], "cur_dev_s": None, "cur_dev_source": "missing"}])
        assert "v1" not in service.predictions
        await service.close()
    asyncio.run(scenario())


def test_incident_expiry_and_reason_require_evidence(service: Dispatcher):
    async def scenario():
        await service.start_replay("dispatcher", 1)
        prediction = service.predictions["v1"]
        service.latest["v1"]["speed"] = 0
        service._incident(prediction, {"stop_duration_s": 15})
        assert "длительный простой" not in next(iter(service.incidents.values()))["reason"]
        service._incident(prediction, {"stop_duration_s": 150})
        assert "длительный простой" in next(iter(service.incidents.values()))["reason"]
        service.run["virtual_time"] = "2026-01-06T07:14:00+00:00"
        service._stale_incidents()
        assert next(iter(service.incidents.values()))["status"] == "data_stale"
        service.run["status"] = "paused"
        service.run["virtual_time"] = "2026-01-06T07:29:00+00:00"
        service._stale_incidents()
        assert next(iter(service.incidents.values()))["status"] == "expired"
        await service.close()
    asyncio.run(scenario())


def test_openapi_defines_snapshot_and_degraded_readiness(service: Dispatcher):
    with TestClient(create_app(service, start_listener=False)) as client:
        schema = client.get("/openapi.json").json()
        assert "SnapshotOutput" in schema["components"]["schemas"]
        service.database_ready = False
        assert client.get("/health/ready").status_code == 503
        assert client.get("/health/live").status_code == 200


def test_stale_incident_hides_live_probability_preserves_history(service: Dispatcher):
    async def scenario():
        await service.start_replay("dispatcher", 1)
        service.run["virtual_time"] = "2026-01-06T07:02:00+00:00"
        public = service.snapshot()["incidents"][0]
        assert public["p_late"] is None
        assert public["last_known_p_late"] == .85
        assert next(iter(service.incidents.values()))["p_late"] == .85
        await service.close()
    asyncio.run(scenario())
