"""Causal telemetry processing, independent clocks, and auditable publication."""
from __future__ import annotations

import asyncio
import bisect
import contextlib
import copy
import heapq
import json
import logging
import math
import os
import time
import uuid
from collections import OrderedDict, defaultdict, deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx

from backend.data import coordinates, iso, normalize_event, read_traffic, safe_plan, safe_points, timestamp
from backend.storage import Binding, Incident, ModelVersion, ObservedVisit, Plan, Prediction, Store, Vehicle

logger = logging.getLogger(__name__)
RISK_ORDER = {"gray": -1, "green": 0, "yellow": 1, "red": 2}
INGRESS_LIMIT = 5000
INGRESS_BATCH = 500
ML_BATCH = 256


class Dispatcher:
    def __init__(self, store: Store, dataset: Path, ml_url: str, http_client: httpx.AsyncClient | None = None):
        self.store, self.dataset, self.ml_url = store, dataset, ml_url.rstrip("/")
        self.client = http_client or httpx.AsyncClient(timeout=httpx.Timeout(3.0))
        self.owns_client = http_client is None
        self.run: dict | None = None
        self.traffic: list[dict] = []
        self.event_ticks: list[float] = []
        self.plan: list[dict] = []
        self.plans: dict[str, list[dict]] = defaultdict(list)
        self.plan_by_id: dict[str, dict] = {}
        self.plan_ticks: dict[str, list[float]] = {}
        self.points: list[dict] = []
        self.bindings: dict[str, str] = {}
        self.network: dict = {"paths": [], "visits": [], "segments": []}
        self.observer: Any = None
        self.events: dict[str, deque] = defaultdict(deque)
        self.prepared_events: dict[str, deque] = defaultdict(deque)
        self.feature_plan: Any = None
        self.latest: dict[str, dict] = {}
        self.positions: dict[str, dict] = {}
        self.deviations: dict[str, dict] = {}
        self.visits: dict[str, dict] = {}
        self.candidates: dict[str, Any] = {}
        self.predictions: dict[str, dict] = {}
        self.histories: dict[str, deque] = defaultdict(lambda: deque(maxlen=240))
        self.incidents: dict[str, dict] = {}
        self.dirty_incidents: set[str] = set()
        self.segments_by_id: dict[str, dict] = {}
        self.target_segments: dict[tuple[str, str], dict] = {}
        self.seen: set[str] = set()
        self.seen_order: deque[tuple[str, float]] = deque()
        self.next_prune = 0.0
        self.future: list[tuple[float, str, dict]] = []
        self.observed_pending: list[dict] = []
        self.opportunities: dict[str, dict] = {}
        self.vehicle_incident_ids: dict[str, set[str]] = defaultdict(set)
        self.new_vehicles: set[str] = set()
        self.cursor = self.point_cursor = 0
        self.next_prediction: datetime | None = None
        self.task: asyncio.Task | None = None
        self.ingress_task: asyncio.Task | None = None
        self.publisher_task: asyncio.Task | None = None
        self.maintenance_task: asyncio.Task | None = None
        self.prediction_task: asyncio.Task | None = None
        self.jobs: OrderedDict[str, dict] = OrderedDict()
        self.ingress: asyncio.Queue = asyncio.Queue(maxsize=INGRESS_LIMIT)
        self.ingress_writing = 0
        self.inbox = {"pending_count": 0, "ready_count": 0, "oldest_ready_age_s": 0.0}
        self.uncommitted: tuple[list[str], dict, list[dict]] | None = None
        self.lock = asyncio.Lock()
        self.control_lock = asyncio.Lock()
        self.apply_lock = asyncio.Lock()
        self.incident_write_lock = asyncio.Lock()
        self.condition = asyncio.Condition()
        self.sequence = 0
        self.stream: deque[tuple[int, str]] = deque(maxlen=256)
        self.prediction_stages_ms: dict[str, float] = {}
        self.prediction_stages_cutoff_at: str | None = None
        self.latencies: deque[float] = deque(maxlen=1000)
        self.ingress_latencies: deque[float] = deque(maxlen=1000)
        self.started_at = time.monotonic()
        self.metrics = {"telemetry_count": 0, "duplicates": 0, "late_events": 0, "unknown_devices": 0,
                        "ml_failures": 0, "predictions_count": 0, "processing_latency_ms": 0.0,
                        "invalid_events": 0, "ignored_live_events": 0, "dropped_live_events": 0,
                        "ingress_admitted": 0, "ingress_committed": 0, "ingress_rejected": 0,
                        "backpressure_waits": 0,
                        "suppressed_horizon": 0, "suppressed_observed": 0, "suppressed_stale": 0,
                        "suppressed_superseded": 0, "prediction_jobs_replaced": 0,
                        "published_predictions": 0, "retrospective_predictions": 0,
                        "late_alerts": 0, "max_ingress_queue_depth": 0, "max_ml_batch_size": 0}
        self.ml_status = "unavailable"
        self.model_version: str | None = None
        self.database_ready = True
        self.last_error: str | None = None
        self.pending = 0
        self.loaded_source: str | None = None
        self._anchor_scenario: datetime | None = None
        self._anchor_mono = 0.0
        self._clock_checkpoint: str | None = None
        self.wall_clock = lambda: datetime.now(timezone.utc)
        self.monotonic = time.monotonic

    @property
    def now(self) -> datetime:
        if not self.run or self.run["mode"] == "ndtp":
            return self.wall_clock()
        # Explicit virtual_time changes remain supported for deterministic tests/tools.
        if self._clock_checkpoint != self.run["virtual_time"]:
            self._anchor_scenario = timestamp(self.run["virtual_time"])
            self._anchor_mono = self.monotonic()
            self._clock_checkpoint = self.run["virtual_time"]
        if self.run["status"] != "running" or self._anchor_scenario is None:
            return timestamp(self.run["virtual_time"])
        result = self._anchor_scenario + timedelta(seconds=(self.monotonic() - self._anchor_mono) * self.run["speed"])
        return min(result, timestamp(self.event_ticks[-1])) if self.event_ticks else result

    def _anchor(self, at: datetime) -> None:
        self._anchor_scenario, self._anchor_mono = at, self.monotonic()
        if self.run:
            self.run["virtual_time"] = self._clock_checkpoint = iso(at)

    def _reset(self) -> None:
        self.started_at = self.monotonic()
        self.events.clear(); self.prepared_events.clear(); self.latest.clear(); self.positions.clear()
        self.deviations.clear(); self.visits.clear(); self.candidates.clear()
        self.predictions.clear(); self.histories.clear(); self.incidents.clear()
        self.dirty_incidents.clear()
        self.seen.clear(); self.future.clear(); self.opportunities.clear(); self.jobs.clear()
        self.seen_order.clear(); self.next_prune = 0.0
        self.vehicle_incident_ids.clear(); self.new_vehicles.clear()
        self.observed_pending.clear()
        self.cursor = self.point_cursor = 0
        self.next_prediction = None
        self.prediction_stages_ms.clear(); self.prediction_stages_cutoff_at = None
        self.metrics.update({key: 0 for key in self.metrics})
        self._anchor_scenario = None
        self._clock_checkpoint = None
        self.uncommitted = None
        self.inbox = {"pending_count": 0, "ready_count": 0, "oldest_ready_age_s": 0.0}

    def _index_plan(self, network: dict | None = None) -> None:
        from backend.scenarios import scenario_network
        from predictor.observer import CausalObserver
        from predictor.features import FeatureBuilder
        self.plan.sort(key=lambda row: (row["time_begin"], row["tt_action_item_id"]))
        self.plans = defaultdict(list)
        self.plan_by_id = {}
        for row in self.plan:
            self.plans[row["tr_id"]].append(row)
            self.plan_by_id[row["tt_action_item_id"]] = row
        self.plan_ticks = {tr_id: [timestamp(row["time_begin"]).timestamp() for row in rows] for tr_id, rows in self.plans.items()}
        self.network = scenario_network(self.plan, self.loaded_source or "test", network)
        self.segments_by_id = {s["id"]: s for s in self.network.get("segments", [])}
        self.target_segments = {(s["tr_id"], s["to_visit_id"]): s for s in self.network.get("segments", [])}
        self.observer = CausalObserver(self.plan, self.network.get("segments", []))
        self.feature_plan = FeatureBuilder([], self.plan)
        self.deviations, self.visits, self.candidates = self.observer.deviations, self.observer.visits, self.observer.candidates

    def _source_data(self, source: str) -> dict:
        if source not in {"test", "validate"}:
            raise ValueError("source must be test or validate")
        folder = self.dataset / source
        traffic = read_traffic(folder / "traffic.csv")
        plan = safe_plan(folder / ("schedule_plan.csv" if source == "validate" else "schedule.csv"))
        points = safe_points(folder / "points.csv" if source == "validate" else self.dataset / "labels" / "labels_test.csv")
        bindings: dict[str, set[str]] = defaultdict(set)
        for event in traffic:
            if event["unit_id"]:
                bindings[event["unit_id"]].add(event["tr_id"])
        resolved = {key: next(iter(values)) for key, values in bindings.items() if len(values) == 1}
        resolved.update({str(k): str(v) for k, v in json.loads(os.environ.get("DEVICE_BINDINGS_JSON", "{}")).items()})
        return {"source": source, "traffic": traffic, "plan": plan, "points": points, "bindings": resolved}

    def _install_source(self, data: dict) -> None:
        self.loaded_source = data["source"]
        self.traffic, self.plan, self.points, self.bindings = data["traffic"], data["plan"], data["points"], data["bindings"]
        self._index_plan()

    def _persist_source(self, data: dict) -> None:
        source = data["source"]
        self.store.put_many(Plan, [{"id": f"{source}:{row['tt_action_item_id']}", "source": source, "payload": row} for row in data["plan"]])
        self.store.put_many(Vehicle, [{"id": tr_id, "payload": {"id": tr_id}} for tr_id in sorted({r["tr_id"] for r in data["traffic"]})])
        self.store.put_many(Binding, [{"id": key, "payload": {"unit_id": key, "tr_id": value}} for key, value in data["bindings"].items()])

    def load_data(self, source: str = "test") -> None:
        if self.loaded_source == source and self.plan:
            return
        data = self._source_data(source)
        self._persist_source(data)
        self._install_source(data)

    def _activate_package(self, package: dict) -> None:
        self.loaded_source = package["source_id"]
        self.plan = copy.deepcopy(package["plan"])
        self.bindings = dict(package["bindings"])
        self.traffic, self.points, self.event_ticks = [], [], []
        self._index_plan(package.get("network"))

    def _sort_traffic(self, mode: str) -> None:
        def available(row: dict) -> str:
            return row["event_time"] if mode == "evaluation" else max(row["event_time"], row["received_at"])
        self.traffic.sort(key=lambda row: (available(row), row["packet_id"]))
        self.event_ticks = [timestamp(available(row)).timestamp() for row in self.traffic]

    def _state_payload(self) -> dict:
        if not self.run:
            return {}
        payload = {**self.run, "virtual_time": iso(self.now), "cursor": self.cursor,
                   "point_cursor": self.point_cursor, "next_prediction": iso(self.next_prediction) if self.next_prediction else None,
                   "telemetry_count": self.metrics["telemetry_count"], "latest": self.latest, "positions": self.positions,
                   "deviations": self.deviations, "visits": self.visits, "opportunities": self.opportunities,
                   "metrics": self.metrics, "observer_state": self.observer.dump_state() if self.observer else {}}
        return copy.deepcopy(payload)

    async def recover(self) -> None:
        saved = await asyncio.to_thread(self.store.latest_run)
        if not saved or saved.get("status") not in {"running", "paused"}:
            return
        if saved.get("scenario_id"):
            package = await asyncio.to_thread(self.store.load_scenario, saved["scenario_id"])
            if not package:
                self.last_error = "Saved scenario is missing; select a scenario to restart"
                return
            self._activate_package(package)
        else:
            self.load_data(saved.get("source", "test"))
            self._sort_traffic(saved["mode"])
        self.run = saved
        if saved["mode"] != "ndtp":
            self.run["status"] = "paused"
        self._anchor(timestamp(saved["virtual_time"]))
        self.cursor, self.point_cursor = saved.get("cursor", 0), saved.get("point_cursor", 0)
        if saved.get("observer_state"):
            self.observer.restore_state(saved["observer_state"])
            self.deviations, self.visits, self.candidates = self.observer.deviations, self.observer.visits, self.observer.candidates
        else:
            self.deviations.update(saved.get("deviations", {}))
            for tr_id, visit in saved.get("visits", {}).items():
                index = self.observer.indices.get((tr_id, visit["target_visit_id"]))
                if index is not None:
                    self.visits[tr_id] = {**visit, "index": index}
        self.latest, self.positions = saved.get("latest", {}), saved.get("positions", {})
        self.opportunities = saved.get("opportunities", {})
        self.next_prediction = timestamp(saved.get("next_prediction") or self.now)
        reader = self.store.recent_applied_telemetry if saved["mode"] == "ndtp" else self.store.recent_telemetry
        history = await asyncio.to_thread(reader, saved["id"], (self.now - timedelta(minutes=16)).timestamp())
        for event in history:
            self._accept(event, observe=False)
        records = await asyncio.to_thread(self.store.recent_predictions, saved["id"], 240)
        for raw in sorted(records, key=lambda row: row["prediction_time"]):
            prediction = {"feature_cutoff_at": raw["prediction_time"], "timing_status": "unknown_legacy", "published_at": None,
                          "published_scenario_at": None, "publication_horizon_s": None, **raw}
            tr_id = prediction["tr_id"]
            self.histories[tr_id].append(prediction)
            if prediction.get("suppression_reason") or prediction.get("timing_status") == "pending":
                continue
            if tr_id not in self.predictions or prediction["prediction_time"] >= self.predictions[tr_id]["prediction_time"]:
                self.predictions[tr_id] = prediction
        if saved["mode"] != "evaluation":
            self.incidents = {item["id"]: item for item in await asyncio.to_thread(self.store.rows, Incident, saved["id"])}
            for item in self.incidents.values():
                self.vehicle_incident_ids[item["tr_id"]].add(item["id"])
                if not item.get("first_alert_at") and item["status"] not in {"resolved", "expired"}:
                    item["status"], item["timing_status"] = "data_stale", "unknown_legacy"
        self.metrics.update({k: v for k, v in saved.get("metrics", {}).items() if k in self.metrics})
        self.metrics["telemetry_count"] = saved.get("telemetry_count", self.metrics["telemetry_count"])
        self.start_task()

    def start_task(self) -> None:
        for name, factory in (("task", self._loop), ("ingress_task", self._ingress_loop), ("publisher_task", self._publish_loop), ("maintenance_task", self._maintenance_loop)):
            current = getattr(self, name)
            if not current or current.done():
                setattr(self, name, asyncio.create_task(factory(), name=f"dispatcher-{name}"))

    async def close(self) -> None:
        if self.ingress_task and not self.ingress_task.done():
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self.ingress.join(), timeout=5)
        tasks = [t for t in (self.task, self.ingress_task, self.publisher_task, self.prediction_task, self.maintenance_task) if t]
        for task in tasks:
            task.cancel()
        for task in tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task
        if self.run and not self.uncommitted:
            with contextlib.suppress(Exception):
                await asyncio.to_thread(self.store.save_run, self._state_payload())
        if self.owns_client:
            await self.client.aclose()

    def checkpoint(self) -> None:
        if self.run and not self.uncommitted:
            self.store.save_run(self._state_payload())

    async def _finish_current_run(self) -> None:
        """Called under apply_lock: finish durable reducer work before switching."""
        if self.uncommitted:
            commit = self.uncommitted
            await asyncio.to_thread(self.store.commit_applied, *commit)
            if self.uncommitted is commit:
                self.uncommitted = None
        async with self.lock:
            if not self.run:
                return
            frozen = self.now
            payload = self._state_payload()
            payload.update(status="stopped", virtual_time=iso(frozen))
        # A failed write leaves the current run and observer untouched.
        await asyncio.to_thread(self.store.save_run, payload)
        async with self.lock:
            self.run["status"] = "stopped"
            self._anchor(frozen)

    async def _checkpoint_async(self) -> None:
        async with self.lock:
            payload = self._state_payload() if self.run and not self.uncommitted else None
        if payload:
            await asyncio.to_thread(self.store.save_run, payload)

    async def start_replay(self, mode: str, speed: int, start_time: str | None = None, source: str = "test") -> dict:
        async with self.control_lock:
            requested_start = timestamp(start_time) if start_time else None
            # Parse and validate independently: invalid requests cannot replace the
            # active NDTP plan, binding table, observer or network.
            data = await asyncio.to_thread(self._source_data, source)
            traffic, points = data["traffic"], data["points"]
            available = lambda row: row["event_time"] if mode == "evaluation" else max(row["event_time"], row["received_at"])
            traffic.sort(key=lambda row: (available(row), row["packet_id"]))
            ticks = [timestamp(available(row)).timestamp() for row in traffic]
            if not traffic:
                raise ValueError("No telemetry available")
            first, last = timestamp(ticks[0]), timestamp(ticks[-1])
            default = timestamp(points[0]["T"]) if mode == "evaluation" and points else timestamp(min(row["event_time"] for row in traffic)).replace(hour=7, minute=0, second=0, microsecond=0)
            start = requested_start if requested_start else max(first, min(default, last))
            if not first <= start <= last:
                raise ValueError(f"start_time must be between {iso(first)} and {iso(last)}")
            await asyncio.to_thread(self._persist_source, data)
            async with self.apply_lock:
                await self._finish_current_run()
                async with self.lock:
                    self._reset(); self._install_source(data)
                    self.event_ticks = ticks
                    self.run = {"id": str(uuid.uuid4()), "created_at": iso(self.wall_clock()), "mode": mode,
                                "source": source, "status": "paused", "speed": speed, "virtual_time": iso(start), "contract_version": 2}
                    self._anchor(start)
                    lower = bisect.bisect_left(ticks, (start - timedelta(minutes=15)).timestamp())
                    self.cursor = bisect.bisect_right(ticks, start.timestamp())
                    self.point_cursor = bisect.bisect_left([point["T"] for point in points], iso(start))
                await self.ingest_batch(traffic[lower:self.cursor])
                self.run["status"] = "running"; self._anchor(start)
                self.next_prediction = start
            await self._predict_due()
            async with self.apply_lock:
                await self._checkpoint_async()
            await self.publish(); self.start_task()
            return self.snapshot()

    async def start_live(self, scenario_id: str | None = None, schedule_mode: str = "as_is", source_time: str | None = None) -> dict:
        from backend.scenarios import demo_package
        async with self.control_lock:
            if schedule_mode == "demo_rebased":
                package = await asyncio.to_thread(demo_package, self.dataset, source_time or "2026-01-06T07:00:00Z", self.wall_clock())
                await asyncio.to_thread(self.store.save_scenario, package)
            else:
                package = await asyncio.to_thread(self.store.load_scenario, scenario_id) if scenario_id else None
                if not package:
                    raise ValueError("An imported scenario_id is required; select Demo NDTP for a shifted demonstration")
                at = self.wall_clock()
                if not any(at - timedelta(minutes=15) <= timestamp(p["time_begin"]) <= at + timedelta(hours=24) for p in package["plan"]):
                    raise ValueError("The selected scenario has no current schedule; import an up-to-date plan")
            async with self.apply_lock:
                await self._finish_current_run()
                async with self.lock:
                    self._reset(); self._activate_package(package)
                    self.run = {"id": str(uuid.uuid4()), "created_at": iso(self.wall_clock()), "mode": "ndtp",
                                "source": package["source_id"], "scenario_id": package["id"], "schedule_mode": schedule_mode,
                                "scenario_name": package["name"], "manifest": package.get("manifest", {}),
                                "status": "running", "speed": 1, "virtual_time": iso(self.wall_clock()), "contract_version": 2}
                    self.next_prediction = self.now
                await self._checkpoint_async()
            self.start_task()
            await self.publish()
            return self.snapshot()

    async def set_status(self, status: str | None = None, speed: int | None = None) -> dict:
        async with self.control_lock, self.apply_lock:
            async with self.lock:
                if not self.run or self.run["mode"] == "ndtp":
                    raise ValueError("No historical replay is active")
                if self.run["status"] == "completed":
                    raise ValueError("Replay is complete; start a new run")
                frozen = self.now
                if status:
                    self.run["status"] = status
                if speed is not None:
                    self.run["speed"] = speed
                self._anchor(frozen)
            await self._checkpoint_async()
            await self.publish()
            return self.snapshot()

    def _accept(self, event: dict, observe: bool = True) -> bool:
        key = f"{event['tr_id']}:{event['packet_id']}"
        if key in self.seen:
            self.metrics["duplicates"] += 1
            return False
        self.seen.add(key); self.metrics["telemetry_count"] += 1
        self.seen_order.append((key, self.now.timestamp()))
        while len(self.seen_order) > 200000:
            old, _ = self.seen_order.popleft()
            self.seen.discard(old)
        event_dt = timestamp(event["event_time"])
        if event_dt > self.now:
            heapq.heappush(self.future, (event_dt.timestamp(), key, event))
            return True
        self._apply_available(event, observe)
        return True

    def _drain_future(self) -> None:
        while self.future and self.future[0][0] <= self.now.timestamp():
            _, _, event = heapq.heappop(self.future)
            self._apply_available(event)

    def _apply_available(self, event: dict, observe: bool = True) -> None:
        tr_id = event["tr_id"]
        previous = self.latest.get(tr_id)
        late = previous is not None and event["event_time"] < previous["event_time"]
        if late:
            self.metrics["late_events"] += 1
        else:
            self.latest[tr_id] = event
            if previous is None:
                self.new_vehicles.add(tr_id)
            if event["location_valid"]:
                self.positions[tr_id] = event
        if timestamp(event["event_time"]) >= self.now - timedelta(minutes=16):
            from predictor.features import prepare_event
            self.events[tr_id].append(event)
            self.prepared_events[tr_id].append(prepare_event(event))
            if late:
                self.events[tr_id] = deque(sorted(self.events[tr_id], key=lambda row: row["event_time"]))
                self.prepared_events[tr_id] = deque(sorted(self.prepared_events[tr_id], key=lambda row: row.event_ns))
        if observe and not late and self.run and self.run["mode"] != "evaluation":
            self._observe(event)

    def _observe(self, event: dict) -> None:
        if not self.observer:
            return
        observed = self.observer.observe(event, at=self.now)
        if observed:
            observed = {**observed, "id": f"{self.run['id']}:{observed['target_visit_id']}", "run_id": self.run["id"]}
            self.observed_pending.append({"id": observed["id"], "run_id": self.run["id"], "payload": observed})
            key = f"{self.run['id']}:{event['tr_id']}:{observed['target_visit_id']}"
            if key in self.incidents:
                self.incidents[key].update(status="resolved", resolution_reason="arrival_observed", resolved_at=observed["observed_at"])
                self.dirty_incidents.add(key)
            for opportunity in self.opportunities.values():
                if opportunity["tr_id"] == event["tr_id"] and self._already_visited(event["tr_id"], opportunity["target_visit_id"]):
                    opportunity["eligible_until"] = min(opportunity.get("eligible_until", iso(self.now)), iso(self.now))

    def _prune(self) -> None:
        if self.monotonic() < self.next_prune:
            return
        self.next_prune = self.monotonic() + 5
        cutoff = iso(self.now - timedelta(minutes=16))
        for history in self.events.values():
            while history and history[0]["event_time"] < cutoff:
                history.popleft()
        cutoff_ns = int(timestamp(cutoff).timestamp() * 1_000_000_000)
        for history in self.prepared_events.values():
            while history and history[0].event_ns < cutoff_ns:
                history.popleft()
        cutoff_tick = (self.now - timedelta(minutes=20)).timestamp()
        while self.seen_order and self.seen_order[0][1] < cutoff_tick:
            old, _ = self.seen_order.popleft()
            self.seen.discard(old)

    async def ingest_batch(self, events: list[dict]) -> None:
        if not self.run:
            return
        run_id = self.run["id"]
        for offset in range(0, len(events), INGRESS_BATCH):
            accepted, keys = [], set()
            for row in events[offset:offset + INGRESS_BATCH]:
                key = f"{row['tr_id']}:{row['packet_id']}"
                if key in self.seen or key in keys:
                    self.metrics["duplicates"] += 1
                    continue
                keys.add(key); accepted.append(row)
            rows = [{"id": f"{run_id}:{r['tr_id']}:{r['packet_id']}", "run_id": run_id,
                     "tr_id": r["tr_id"], "event_at": timestamp(r["event_time"]).timestamp(), "payload": r} for r in accepted]
            await asyncio.to_thread(self.store.insert_telemetry, rows)
            if not self.run or self.run["id"] != run_id:
                return
            for row in accepted:
                self._accept(row)
            if self.observed_pending:
                pending, self.observed_pending = self.observed_pending, []
                await asyncio.to_thread(self.store.put_many, ObservedVisit, pending)
        self._prune()

    async def ingest_live(self, row: dict) -> None:
        if not self.run or self.run["mode"] != "ndtp":
            self.metrics["ignored_live_events"] += 1
            return
        tr_id = self.bindings.get(str(row.get("unit_id")))
        if tr_id is None:
            self.metrics["unknown_devices"] += 1
            return
        try:
            event = normalize_event({**row, "tr_id": tr_id, "received_at": iso(self.wall_clock())})
        except (ValueError, TypeError, KeyError):
            self.metrics["invalid_events"] += 1
            return
        run_id = self.run["id"]
        async def admit() -> None:
            waited = False
            while self.ingress.qsize() + self.ingress_writing + self.inbox["pending_count"] >= INGRESS_LIMIT:
                if not waited:
                    self.metrics["backpressure_waits"] += 1
                    waited = True
                await asyncio.sleep(.05)
            if not self.run or self.run["id"] != run_id:
                raise ConnectionError("Run changed while waiting for telemetry admission")
            self.ingress.put_nowait((run_id, event, self.monotonic()))
        try:
            await asyncio.wait_for(admit(), timeout=10)
        except asyncio.TimeoutError as error:
            self.metrics["ingress_rejected"] += 1
            raise ConnectionError("Telemetry admission timed out after 10 seconds; reconnect and retry") from error
        self.metrics["ingress_admitted"] += 1
        self.metrics["max_ingress_queue_depth"] = max(self.metrics["max_ingress_queue_depth"], self.ingress.qsize())

    async def _ingress_loop(self) -> None:
        batch: list = []
        while True:
            if not batch:
                batch.append(await self.ingress.get())
                await asyncio.sleep(.1)
                while len(batch) < INGRESS_BATCH and not self.ingress.empty():
                    batch.append(self.ingress.get_nowait())
            self.ingress_writing = len(batch)
            rows = [{"id": f"{run_id}:{event['tr_id']}:{event['packet_id']}", "run_id": run_id,
                     "tr_id": event["tr_id"], "event_at": timestamp(event["event_time"]).timestamp(), "payload": event}
                    for run_id, event, _ in batch]
            try:
                added = await asyncio.to_thread(self.store.enqueue_telemetry, rows)
                self.metrics["ingress_committed"] += len(added)
                self.metrics["duplicates"] += len(rows) - len(added)
                self.ingress_latencies.extend((self.monotonic() - started) * 1000 for _, _, started in batch)
                for _ in batch:
                    self.ingress.task_done()
                batch = []; self.ingress_writing = 0
            except asyncio.CancelledError:
                raise
            except Exception:
                self.database_ready = False
                self.last_error = "Database unavailable; admitted telemetry retained for retry"
                await asyncio.sleep(1)

    async def _apply_inbox(self) -> None:
        async with self.apply_lock:
            await self._apply_inbox_locked()

    async def _apply_inbox_locked(self) -> None:
        if not self.run:
            return
        run_id = self.run["id"]
        previous_commit = self.uncommitted
        if previous_commit:
            await asyncio.to_thread(self.store.commit_applied, *previous_commit)
            async with self.lock:
                if self.uncommitted is previous_commit:
                    self.uncommitted = None
        pending = await asyncio.to_thread(self.store.pending_telemetry, run_id, self.now.timestamp(), INGRESS_BATCH)
        async with self.lock:
            if not self.run or self.run["id"] != run_id:
                return
            commit = None
            if pending:
                for row in pending:
                    self._accept(row["payload"])
                self._drain_future(); self._prune()
                observed, self.observed_pending = self.observed_pending, []
                commit = ([row["id"] for row in pending], self._state_payload(), observed)
                self.uncommitted = commit
        if commit:
            await asyncio.to_thread(self.store.commit_applied, *commit)
            async with self.lock:
                if self.uncommitted is commit:
                    self.uncommitted = None
        inbox = await asyncio.to_thread(self.store.inbox_stats, run_id, self.now.timestamp())
        if self.run and self.run["id"] == run_id:
            self.inbox = inbox

    async def _loop(self) -> None:
        while True:
            await asyncio.sleep(.1)
            if not self.run or self.run["status"] != "running":
                continue
            try:
                if self.run["mode"] == "ndtp":
                    await self._apply_inbox()
                else:
                    async with self.apply_lock:
                        if not self.run or self.run["mode"] == "ndtp" or self.run["status"] != "running":
                            continue
                        end = min(bisect.bisect_right(self.event_ticks, self.now.timestamp()), self.cursor + INGRESS_BATCH)
                        if end > self.cursor:
                            await self.ingest_batch(self.traffic[self.cursor:end]); self.cursor = end
                        self._drain_future()
                self.database_ready = True; self.last_error = None
                for point in self._collect_due():
                    key = point["sample_id"] if self.run["mode"] == "evaluation" else point["tr_id"]
                    if key in self.jobs:
                        self.metrics["prediction_jobs_replaced"] += 1
                    self.jobs[key] = point
                if self.jobs and (not self.prediction_task or self.prediction_task.done()):
                    self.prediction_task = asyncio.create_task(self._prediction_loop(), name="dispatcher-ml")
                self._stale_incidents()
                if self.run["mode"] != "ndtp" and self.cursor >= len(self.traffic) and not self.jobs and (not self.prediction_task or self.prediction_task.done()):
                    frozen = self.now; self.run["status"] = "completed"; self._anchor(frozen)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                self.database_ready = False; self.last_error = str(error)
                logger.exception("Telemetry processing will retry")
                if self.run and self.run["mode"] != "ndtp":
                    frozen = self.now; self.run["status"] = "paused"; self._anchor(frozen)
                await asyncio.sleep(.5)

    async def _publish_loop(self) -> None:
        while True:
            await asyncio.sleep(1)
            if not self.run:
                continue
            try:
                if not self.uncommitted and self.run["mode"] != "ndtp":
                    await asyncio.to_thread(self.store.save_run, self._state_payload())
                if self.dirty_incidents:
                    dirty, self.dirty_incidents = self.dirty_incidents, set()
                    rows = [{"id": key, "run_id": self.run["id"], "payload": copy.deepcopy(self.incidents[key])} for key in dirty if key in self.incidents]
                    try:
                        await self._persist_predictions([], rows)
                    except Exception:
                        self.dirty_incidents.update(dirty)
                        raise
                await self.publish()
            except asyncio.CancelledError:
                raise
            except Exception:
                self.database_ready = False
                self.last_error = "State checkpoint unavailable"
                await self.publish()

    async def _maintenance_loop(self) -> None:
        while True:
            await asyncio.sleep(30)
            if not self.run:
                continue
            try:
                for _ in range(10):
                    result = await asyncio.to_thread(self.store.prune, self.wall_clock().timestamp(),
                        self.run["id"], (self.now - timedelta(minutes=16)).timestamp(), 500)
                    if result["telemetry_deleted"] < 500 and result["results_deleted"] < 500:
                        break
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("Retention maintenance deferred", exc_info=True)

    async def _prediction_loop(self) -> None:
        while self.jobs:
            points = [self.jobs.popitem(last=False)[1] for _ in range(min(ML_BATCH, len(self.jobs)))]
            try:
                await self.predict(points)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Prediction batch failed")
                self.metrics["ml_failures"] += 1
                self.ml_status = "unavailable"

    def _deviation(self, tr_id: str, at: datetime) -> dict:
        value = self.deviations.get(tr_id)
        if not value or not 0 <= (at - timestamp(value["observed_at"])).total_seconds() <= 900:
            return {"value": None, "source": "missing", "observed_at": None}
        return value

    def _already_visited(self, tr_id: str, visit_id: str) -> bool:
        if self.observer and self.observer.has_visited(tr_id, visit_id):
            return True
        previous, target = self.visits.get(tr_id), self.plan_by_id.get(visit_id)
        return bool(previous and target and target["time_begin"] <= previous["planned_at"])

    def _collect_due(self) -> list[dict]:
        if not self.run:
            return []
        at = self.now
        if self.run["mode"] == "evaluation":
            due = []
            while self.point_cursor < len(self.points) and timestamp(self.points[self.point_cursor]["T"]) <= at:
                point = dict(self.points[self.point_cursor])
                point.update(cur_dev_source="provided" if point["cur_dev_s"] is not None else "missing", cur_dev_observed_at=None)
                due.append(point); self.point_cursor += 1
            return due
        targets: dict[str, dict] = {}
        # Record eligible first-stop opportunities independently of inference,
        # including while the worker is busy or the model is unavailable.
        for tr_id, latest in self.latest.items():
            fresh_until = timestamp(latest["event_time"]) + timedelta(seconds=60)
            if not timestamp(latest["event_time"]) <= at <= fresh_until:
                continue
            ticks = self.plan_ticks.get(tr_id, [])
            left, right = bisect.bisect_right(ticks, at.timestamp() + 600), bisect.bisect_right(ticks, at.timestamp() + 900)
            target = next((row for row in self.plans[tr_id][left:right] if not self._already_visited(tr_id, row["tt_action_item_id"])), None)
            if not target:
                continue
            targets[tr_id] = target
            key = f"{tr_id}:{target['tt_action_item_id']}"
            until = min(fresh_until, timestamp(target["time_begin"]) - timedelta(seconds=600))
            opportunity = self.opportunities.setdefault(key, {"tr_id": tr_id, "target_visit_id": target["tt_action_item_id"],
                "first_eligible_at": iso(at), "episode_start": iso(at), "eligible_until": iso(until),
                "past_exposure_s": 0.0, "covered": False, "model_covered": False})
            previous_until = timestamp(opportunity.get("eligible_until", iso(at)))
            if at > previous_until:
                opportunity["past_exposure_s"] = opportunity.get("past_exposure_s", 0) + max(0, (previous_until - timestamp(opportunity.get("episode_start", opportunity["first_eligible_at"]))).total_seconds())
                opportunity["episode_start"] = iso(at)
            opportunity["eligible_until"] = iso(until)
        scheduled = self.next_prediction is None or at >= self.next_prediction
        if not scheduled and not self.new_vehicles:
            return []
        if scheduled:
            self.next_prediction = at + timedelta(seconds=30)
        vehicles = list(self.latest.items()) if scheduled else [(tr_id, self.latest[tr_id]) for tr_id in self.new_vehicles if tr_id in self.latest]
        self.new_vehicles.clear()
        points = []
        for tr_id, latest in vehicles:
            target = targets.get(tr_id)
            if not target:
                continue
            deviation = self._deviation(tr_id, at)
            points.append({"sample_id": f"{tr_id}_{at.timestamp()}", "tr_id": tr_id, "T": iso(at), "triggered_at": iso(self.wall_clock()),
                           "target_stop_id": target["tt_action_item_id"], "target_time_begin": target["time_begin"],
                           "cur_dev_s": deviation["value"], "cur_dev_source": deviation["source"], "cur_dev_observed_at": deviation["observed_at"],
                           "_segment_features": copy.deepcopy(self.observer.segment(tr_id, at)) if self.observer else {},
                           "_current_segment": copy.deepcopy(self._segment(tr_id))})
        return points

    async def _predict_due(self) -> None:
        points = self._collect_due()
        for offset in range(0, len(points), ML_BATCH):
            await self.predict(points[offset:offset + ML_BATCH])

    def _segment(self, tr_id: str, target_id: str | None = None) -> dict | None:
        if target_id is not None:
            segment = self.target_segments.get((tr_id, target_id))
            return dict(segment) if segment else None
        if self.observer:
            state = self.observer.segment(tr_id, self.now)
            identifier = state.get("segment_id")
            segment = self.segments_by_id.get(identifier)
            return {**segment, **state} if segment else None
        return None

    async def predict(self, points: list[dict]) -> None:
        if not points or not self.run:
            return
        if len(points) > ML_BATCH:
            for offset in range(0, len(points), ML_BATCH):
                await self.predict(points[offset:offset + ML_BATCH])
            return
        from predictor.features import FEATURE_SCHEMA_VERSION, FeatureBuilder, timestamp as feature_timestamp
        from predictor.stream_features import STREAM_SCHEMA_VERSION, enrich_stream_features
        started, run_id, mode = self.monotonic(), self.run["id"], self.run["mode"]
        stage_start = time.perf_counter()
        wall_started = self.wall_clock()
        stages = {"scheduler_wait_ms": max((max(0.0, (wall_started - timestamp(p.get("triggered_at") or wall_started)).total_seconds()) * 1000 for p in points), default=0.0)}
        profile = "official" if mode == "evaluation" else "stream"
        schema = FEATURE_SCHEMA_VERSION if profile == "official" else STREAM_SCHEMA_VERSION
        # Immutable scalar records and a plan index survive pruning/run switches.
        # Snapshot only the deque references here; filtering/indexing runs off-loop.
        plan_builder = self.feature_plan
        inputs = []
        for point in points:
            tr_id, at = point["tr_id"], timestamp(point["T"])
            if mode == "evaluation":
                left = bisect.bisect_left(self.event_ticks, (at - timedelta(minutes=15)).timestamp())
                right = bisect.bisect_right(self.event_ticks, at.timestamp())
                history = [row for row in self.traffic[left:right] if row["tr_id"] == tr_id]
                plan = tuple(self.plans.get(tr_id, ()))
            else:
                history = tuple(self.prepared_events.get(tr_id, ()))
                plan = ()
            segment = point.get("_segment_features", {})
            if "_segment_features" not in point and self.observer and mode != "evaluation" and self.observer.last_event.get(tr_id, 0) <= at.timestamp():
                segment = self.observer.segment(tr_id, at)
            inputs.append((dict(point), history, plan, segment))
        stages["snapshot_ms"] = (time.perf_counter() - stage_start) * 1000

        def extract() -> tuple[dict, int, float]:
            began = time.perf_counter()
            work, suppressed = {}, 0
            groups: dict[str, list] = defaultdict(list)
            for item in inputs:
                groups[item[0]["T"] if profile == "stream" else "official"].append(item)
            for group in groups.values():
                if profile == "stream":
                    histories = {point["tr_id"]: history for point, history, _, _ in group}
                    builder = plan_builder.with_prepared_histories(histories, group[0][0]["T"])
                    cutoff_ns = feature_timestamp(group[0][0]["T"]).value
                else:
                    events = {(row["tr_id"], row.get("packet_id", row["event_time"])): row for _, history, _, _ in group for row in history}
                    visits = {(row["tr_id"], row["tt_action_item_id"]): row for _, _, plan, _ in group for row in plan}
                    builder = FeatureBuilder(list(events.values()), list(visits.values()))
                for point, history, _, segment in group:
                    count = len(history)
                    if profile == "stream":
                        ticks = builder.events.get(point["tr_id"], builder.empty_events).ticks
                        count = len(ticks)
                        if not count or cutoff_ns - int(ticks[-1]) > 60_000_000_000:
                            suppressed += 1
                            continue
                    features = builder.build(point)
                    if profile == "stream":
                        features = enrich_stream_features(features, segment)
                    safe = {key: float(value) if value is not None and math.isfinite(float(value)) else None for key, value in features.items()}
                    work[point["sample_id"]] = (point, safe, count)
            return work, suppressed, (time.perf_counter() - began) * 1000

        feature_started = time.perf_counter()
        work, suppressed, feature_ms = await asyncio.to_thread(extract)
        stages["feature_build_ms"] = feature_ms
        stages["feature_wait_ms"] = max(0.0, (time.perf_counter() - feature_started) * 1000 - feature_ms)
        if not self.run or self.run["id"] != run_id:
            return
        self.metrics["suppressed_stale"] += suppressed
        if not work:
            return
        self.pending = len(work)
        self.metrics["max_ml_batch_size"] = max(self.metrics["max_ml_batch_size"], len(work))
        answers: dict[str, dict] = {}
        inference_started = time.perf_counter()
        try:
            response = await self.client.post(f"{self.ml_url}/v1/predict", json={"profile": profile, "feature_schema_version": schema,
                "items": [{"request_id": key, "features": features} for key, (_, features, _) in work.items()]})
            response.raise_for_status(); payload = response.json()
            answers = {item["request_id"]: item for item in payload["items"]}
            self.ml_status, self.model_version = "ready", payload.get("model_version")
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            self.metrics["ml_failures"] += 1; self.ml_status = "unavailable"
        finally:
            self.pending = 0
            stages["inference_ms"] = (time.perf_counter() - inference_started) * 1000
        generated = iso(self.wall_clock())
        if not self.run or self.run["id"] != run_id:
            self.metrics["suppressed_superseded"] += len(work)
            return
        candidates = []
        for key, (point, features, history_count) in work.items():
            answer = answers.get(key, {})
            source, delay, probability = answer.get("source", "model"), answer.get("prediction_delay_s"), answer.get("p_late")
            if delay is None:
                delay, source, probability = point["cur_dev_s"], "baseline", None
                if delay is None:
                    continue
            delay = float(delay)
            if not math.isfinite(delay):
                continue
            if probability is not None:
                probability = float(probability)
                if not math.isfinite(probability) or not 0 <= probability <= 1:
                    probability = None
            risk = "red" if probability is not None and probability >= .7 else "yellow" if ((probability is not None and probability >= .4) or delay < -60) else "green" if probability is not None else "gray"
            target = self.plan_by_id.get(str(point["target_stop_id"]), {})
            lat, lon = coordinates(target.get("geom", ""))
            prediction = {"id": f"{run_id}:{key}", "run_id": run_id, "tr_id": point["tr_id"],
                          "prediction_time": point["T"], "feature_cutoff_at": point["T"], "generated_at": generated,
                          "triggered_at": point.get("triggered_at", generated),
                          "published_at": None, "published_scenario_at": None, "publication_horizon_s": None,
                          "timing_status": "retrospective" if mode == "evaluation" else "pending",
                          "target_visit_id": str(point["target_stop_id"]), "target_name": target.get("building_address") or str(point["target_stop_id"]),
                          "target_lat": lat, "target_lon": lon, "target_time_begin": point["target_time_begin"],
                          "horizon_s": (timestamp(point["target_time_begin"]) - timestamp(point["T"])).total_seconds(),
                          "prediction_delay_s": delay, "predicted_arrival_at": iso(timestamp(point["target_time_begin"]) + timedelta(seconds=delay)),
                          "p_late": probability, "risk": risk, "source": source, "model_profile": profile,
                          "model_version": answer.get("model_version") or (self.model_version if source == "model" else "cur-dev-baseline"),
                          "feature_schema_version": schema, "factors": answer.get("factors", []),
                          "cur_dev_s": point["cur_dev_s"], "cur_dev_source": point.get("cur_dev_source", "missing"),
                          "current_segment": point.get("_current_segment"), "target_segment": self._segment(point["tr_id"], str(point["target_stop_id"])),
                          "data_quality": {"history_points": history_count, "cur_dev_source": point.get("cur_dev_source", "missing")}}
            candidates.append((prediction, features))
        # Durable candidates precede publication. A slow commit can consume the horizon,
        # so the final gate below runs AFTER this await, immediately before visibility.
        commit_started = time.perf_counter()
        await self._persist_predictions(self._prediction_rows([p for p, _ in candidates]), [])
        stages["candidate_commit_ms"] = (time.perf_counter() - commit_started) * 1000
        if not self.run or self.run["id"] != run_id:
            return
        publication_started = time.perf_counter()
        changed_incidents = []
        for prediction, features in candidates:
            if mode == "evaluation":
                self.metrics["retrospective_predictions"] += 1
            else:
                reason = self._publication_rejection(prediction)
                if reason:
                    prediction.update(timing_status="suppressed", suppression_reason=reason)
                    self.metrics[f"suppressed_{reason}"] += 1
                else:
                    publication = self.now
                    prediction.update(timing_status="verified", published_at=iso(self.wall_clock()), published_scenario_at=iso(publication),
                                      publication_horizon_s=(timestamp(prediction["target_time_begin"]) - publication).total_seconds())
                    incident = self._incident(prediction, features, persist=False)
                    if incident:
                        changed_incidents.append({"id": incident["id"], "run_id": run_id, "payload": incident})
                    opportunity = self.opportunities.get(f"{prediction['tr_id']}:{prediction['target_visit_id']}")
                    if opportunity:
                        opportunity["covered"] = True
                        opportunity["model_covered"] |= prediction["source"] == "model"
                        opportunity.setdefault("first_prediction_at", prediction["published_scenario_at"])
                    self.metrics["published_predictions"] += 1
            self.histories[prediction["tr_id"]].append(copy.deepcopy(prediction))
            if prediction["timing_status"] in {"verified", "retrospective"}:
                self.predictions[prediction["tr_id"]] = prediction
            if mode == "evaluation" and prediction["cur_dev_source"] == "provided":
                self.deviations[prediction["tr_id"]] = {"value": prediction["cur_dev_s"], "source": "provided", "observed_at": prediction["prediction_time"]}
        self.metrics["predictions_count"] += len(candidates)
        # No awaits between final validation, state visibility and SSE enqueue.
        await self.publish()
        stages["publication_ms"] = (time.perf_counter() - publication_started) * 1000
        final_commit_started = time.perf_counter()
        await self._persist_predictions(self._prediction_rows([p for p, _ in candidates]), changed_incidents)
        if self.model_version:
            await asyncio.to_thread(self.store.put_many, ModelVersion, [{"id": self.model_version, "payload": {"version": self.model_version, "feature_schema_version": schema}}])
        self.metrics["processing_latency_ms"] = round((self.monotonic() - started) * 1000, 2)
        stages["final_commit_ms"] = (time.perf_counter() - final_commit_started) * 1000
        stages["total_ms"] = self.metrics["processing_latency_ms"]
        self.prediction_stages_ms = {key: round(value, 3) for key, value in stages.items()}
        self.prediction_stages_cutoff_at = points[0]["T"]
        self.latencies.append(self.metrics["processing_latency_ms"])

    @staticmethod
    def _prediction_rows(predictions: list[dict]) -> list[dict]:
        return [{"id": p["id"], "run_id": p["run_id"], "tr_id": p["tr_id"], "payload": copy.deepcopy(p)} for p in predictions]

    async def _persist_predictions(self, predictions: list[dict], incidents: list[dict]) -> None:
        async with self.incident_write_lock:
            # An older queued snapshot must never revoke an already committed ACK.
            for row in incidents:
                current = self.incidents.get(row["id"])
                if current and current.get("acknowledged"):
                    row["payload"]["acknowledged"] = True
                    row["payload"]["acknowledged_at"] = current.get("acknowledged_at")
            await asyncio.to_thread(self.store.commit_predictions, predictions, incidents)

    def _publication_rejection(self, prediction: dict) -> str | None:
        if not self.run or prediction["run_id"] != self.run["id"]:
            return "superseded"
        current = self.predictions.get(prediction["tr_id"])
        if current and current["prediction_time"] > prediction["prediction_time"]:
            return "superseded"
        at = self.now
        if not 600 < (timestamp(prediction["target_time_begin"]) - at).total_seconds() <= 900:
            return "horizon"
        if self._already_visited(prediction["tr_id"], prediction["target_visit_id"]):
            return "observed"
        latest = self.latest.get(prediction["tr_id"])
        # Applied rows already have a committed raw inbox record. An unrelated
        # checkpoint transaction in flight does not make their telemetry stale.
        if not latest or not 0 <= (at - timestamp(latest["event_time"])).total_seconds() <= 60:
            return "stale"
        return None

    def _incident(self, prediction: dict, features: dict | None = None, persist: bool = True) -> dict | None:
        if not self.run or self.run["mode"] == "evaluation":
            return None
        key = f"{self.run['id']}:{prediction['tr_id']}:{prediction['target_visit_id']}"
        current, risk = self.incidents.get(key), prediction["risk"]
        if not current and risk not in {"red", "yellow"}:
            return None
        features = features or {}
        speed = features.get("last_speed")
        reason = "Низкая текущая скорость" if speed is not None and speed < 10 else "Прогнозируемое отклонение от расписания"
        if (features.get("speed_change_1m") or 0) < -5:
            reason = "Наблюдается снижение скорости"
        if (features.get("stop_duration_s") or 0) >= 120:
            reason = "Возможная причина — длительный простой; требуется проверка диспетчером"
        incident = {**(current or {}), "id": key, "tr_id": prediction["tr_id"], "target_visit_id": prediction["target_visit_id"],
                    "risk": risk, "status": "resolved" if risk == "green" else "data_stale" if risk == "gray" else "active",
                    "acknowledged": current.get("acknowledged", False) if current else False,
                    "prediction_id": prediction["id"], "prediction": copy.deepcopy(prediction),
                    "prediction_delay_s": prediction["prediction_delay_s"], "p_late": prediction["p_late"],
                    "target_name": prediction["target_name"], "target_time_begin": prediction["target_time_begin"],
                    "predicted_arrival_at": prediction["predicted_arrival_at"], "prediction_time": prediction["prediction_time"],
                    "timing_status": prediction.get("timing_status", "unknown_legacy"), "reason": reason,
                    "current_segment": prediction.get("current_segment"), "target_segment": prediction.get("target_segment"),
                    "recommendation": "Уточнить обстановку у водителя и оценить необходимость вмешательства"}
        for name in ("published_at", "published_scenario_at", "publication_horizon_s"):
            incident[name] = prediction.get(name)
        if not current:
            incident.update(first_alert_at=prediction.get("published_at"), first_alert_scenario_at=prediction.get("published_scenario_at"),
                            first_alert_horizon_s=prediction.get("publication_horizon_s"))
        self.incidents[key] = incident
        self.vehicle_incident_ids[prediction["tr_id"]].add(key)
        if persist:
            self.store.put_many(Incident, [{"id": key, "run_id": self.run["id"], "payload": incident}])
        return incident

    def _stale_incidents(self) -> None:
        for item in self.incidents.values():
            previous_status = item["status"]
            if item["status"] in {"resolved", "expired"}:
                continue
            latest = self.latest.get(item["tr_id"])
            stale = not latest or (self.now - timestamp(latest["event_time"])).total_seconds() > 60
            if self._already_visited(item["tr_id"], item["target_visit_id"]):
                item.update(status="resolved", resolution_reason="arrival_observed")
            elif self.now > timestamp(item["target_time_begin"]) + timedelta(minutes=15):
                item.update(status="expired", resolution_reason="observation_timeout")
            elif stale:
                item["status"] = "data_stale"
            elif (timestamp(item["target_time_begin"]) - self.now).total_seconds() <= 600:
                item["status"] = "monitoring"
            elif item["status"] == "data_stale":
                item["status"] = "active"
            if item["status"] != previous_status:
                self.dirty_incidents.add(item["id"])

    async def acknowledge(self, identifier: str) -> dict:
        item = self.incidents.get(identifier)
        if item is None:
            raise KeyError(identifier)
        run_id = self.run["id"]
        async with self.incident_write_lock:
            update = {**copy.deepcopy(self.incidents.get(identifier, item)), "acknowledged": True, "acknowledged_at": iso(self.now)}
            await asyncio.to_thread(self.store.put_many, Incident, [{"id": identifier, "run_id": run_id, "payload": update}])
            if self.run and self.run["id"] == run_id and identifier in self.incidents:
                self.incidents[identifier].update(acknowledged=True, acknowledged_at=update["acknowledged_at"])
        await self.publish()
        return self.public_incident(self.incidents.get(identifier, update))

    def public_run(self) -> dict | None:
        if not self.run:
            return None
        return {**{key: self.run[key] for key in ("id", "mode", "source", "status", "speed")}, "virtual_time": iso(self.now),
                "scenario_id": self.run.get("scenario_id"), "scenario_name": self.run.get("scenario_name"),
                "schedule_mode": self.run.get("schedule_mode"), "manifest": self.run.get("manifest"), "contract_version": 2}

    def public_prediction(self, prediction: dict, stale: bool = False) -> dict:
        result = copy.deepcopy(prediction)
        age = (self.now - timestamp(prediction["prediction_time"])).total_seconds()
        remaining = (timestamp(prediction["target_time_begin"]) - self.now).total_seconds()
        retrospective = prediction.get("timing_status") == "retrospective"
        current = not stale and age <= 90 and 600 < remaining <= 900 and prediction.get("timing_status") == "verified"
        result["current_prediction"] = current
        result["display_state"] = "retrospective" if retrospective else "current" if current else "stale" if stale or prediction.get("timing_status") != "verified" else "monitoring" if remaining <= 600 else "prediction_stale"
        if not current:
            result.update(last_known_risk=result["risk"], last_known_p_late=result["p_late"], risk="gray", p_late=None)
        return result

    def vehicle(self, tr_id: str) -> dict:
        latest, position = self.latest[tr_id], self.positions.get(tr_id, {})
        age = max(0, (self.now - timestamp(latest["event_time"])).total_seconds())
        stale = age > 60
        deviation = self._deviation(tr_id, self.now)
        prediction = self.public_prediction(self.predictions[tr_id], stale) if tr_id in self.predictions else None
        current_incidents = [self.incidents[key] for key in self.vehicle_incident_ids.get(tr_id, ()) if key in self.incidents and self.incidents[key]["status"] not in {"resolved", "expired"}]
        return {"id": tr_id, "lat": position.get("lat"), "lon": position.get("lon"), "speed": latest["speed"],
                "packet_id": latest.get("packet_id"), "received_at": latest.get("received_at"),
                "event_time": latest["event_time"], "position_age_s": max(0, (self.now - timestamp(position["event_time"])).total_seconds()) if position else None,
                "telemetry_age_s": age, "stale": stale, "cur_dev_s": deviation["value"], "cur_dev_source": deviation["source"],
                "prediction": prediction, "current_segment": self._segment(tr_id),
                "target_segment": prediction.get("target_segment") if prediction else None,
                "incident_ids": [item["id"] for item in current_incidents]}

    @staticmethod
    def _p95(values: deque) -> float:
        return sorted(values)[max(0, math.ceil(len(values) * .95) - 1)] if values else 0.0

    def system(self) -> dict:
        stale = sum((self.now - timestamp(row["event_time"])).total_seconds() > 60 for row in self.latest.values())
        ndtp = {}
        with contextlib.suppress(ImportError):
            from backend.ndtp import get_stats
            ndtp = get_stats()
        at = self.now
        mature = [value for value in self.opportunities.values() if value.get("past_exposure_s", 0) + max(0,
            (min(at, timestamp(value.get("eligible_until", value["first_eligible_at"]))) - timestamp(value.get("episode_start", value["first_eligible_at"]))).total_seconds()) >= 30]
        opportunities = len(mature)
        covered = sum(v["covered"] for v in mature)
        model_covered = sum(v["model_covered"] for v in mature)
        return {"backend_status": "ok" if self.database_ready else "degraded", "ml_status": self.ml_status,
                "model_version": self.model_version, "active_vehicles": len(self.latest) - stale, "stale_vehicles": stale,
                "queue_depth": self.ingress.qsize() + self.ingress_writing + self.inbox["pending_count"] + len(self.jobs) + self.pending,
                "ingress_queue_depth": self.ingress.qsize() + self.ingress_writing, "durable_queue_depth": self.inbox["pending_count"],
                "apply_queue_depth": self.inbox["ready_count"], "apply_lag_s": self.inbox["oldest_ready_age_s"],
                "ml_queue_depth": len(self.jobs), "ml_inflight": self.pending, "ingress_queue_capacity": INGRESS_LIMIT,
                "processing_latency_p95_ms": self._p95(self.latencies), "ingress_latency_p95_ms": self._p95(self.ingress_latencies),
                "prediction_stages_ms": dict(self.prediction_stages_ms), "prediction_stages_cutoff_at": self.prediction_stages_cutoff_at,
                "opportunity_count": opportunities, "covered_opportunities": covered, "model_covered_opportunities": model_covered,
                "eligible_opportunity_count": len(self.opportunities), "immature_opportunity_count": len(self.opportunities) - opportunities,
                "opportunity_coverage": covered / opportunities if opportunities else None,
                "model_opportunity_coverage": model_covered / opportunities if opportunities else None,
                "telemetry_per_second": round(self.metrics["telemetry_count"] / max(.001, self.monotonic() - self.started_at), 2),
                "last_error": self.last_error, "ndtp_errors": sum(v for k, v in ndtp.items() if "error" in k and isinstance(v, (int, float))),
                "ndtp": ndtp, **self.metrics}

    def public_incident(self, item: dict) -> dict:
        result = copy.deepcopy(item)
        latest = self.latest.get(item["tr_id"])
        stale = not latest or (self.now - timestamp(latest["event_time"])).total_seconds() > 60
        old = (self.now - timestamp(item["prediction_time"])).total_seconds() > 90
        if stale or old or item["status"] in {"monitoring", "data_stale", "resolved", "expired"} or item.get("timing_status") != "verified":
            result.update(last_known_risk=result["risk"], last_known_p_late=result["p_late"], risk="gray", p_late=None)
        if result.get("prediction"):
            result["prediction"] = self.public_prediction(result["prediction"], stale)
        return result

    def incident_detail(self, identifier: str) -> dict:
        if identifier not in self.incidents:
            raise KeyError(identifier)
        return self.public_incident(self.incidents[identifier])

    def snapshot(self) -> dict:
        def order(item: dict) -> tuple:
            latest = self.latest.get(item["tr_id"])
            fresh = latest and (self.now - timestamp(latest["event_time"])).total_seconds() <= 60 and (self.now - timestamp(item["prediction_time"])).total_seconds() <= 90
            return (item["status"] in {"resolved", "expired"}, -RISK_ORDER[item["risk"]] if fresh else 1, -timestamp(item["prediction_time"]).timestamp())
        incidents = sorted(self.incidents.values(), key=order)
        return {"run": self.public_run(), "vehicles": [self.vehicle(tr_id) for tr_id in sorted(self.latest)],
                "incidents": [self.public_incident(item) for item in incidents[:100]], "incidents_total": len(incidents),
                "system": self.system(), "event_id": self.sequence, "published_at": iso(self.wall_clock()),
                "network_version": self.network.get("network_version"), "segment_risks": self.segment_risks()}

    def detail(self, tr_id: str) -> dict:
        if tr_id not in self.latest:
            raise KeyError(tr_id)
        visits = []
        for item in self.plans.get(tr_id, []):
            if self.now - timedelta(minutes=15) <= timestamp(item["time_begin"]) <= self.now + timedelta(minutes=30):
                lat, lon = coordinates(item["geom"])
                visits.append({"id": item["tt_action_item_id"], "name": item["building_address"], "lat": lat, "lon": lon, "time_begin": item["time_begin"]})
        return {**self.vehicle(tr_id), "telemetry": list(self.events[tr_id]), "history": list(self.histories[tr_id]), "planned_visits": visits}

    def segment_risks(self) -> dict[str, str]:
        risks: dict[str, str] = {}
        at = self.now
        for tr_id, latest in self.latest.items():
            prediction = self.predictions.get(tr_id)
            current = (prediction and prediction.get("timing_status") == "verified"
                       and (at - timestamp(latest["event_time"])).total_seconds() <= 60
                       and (at - timestamp(prediction["prediction_time"])).total_seconds() <= 90
                       and 600 < (timestamp(prediction["target_time_begin"]) - at).total_seconds() <= 900)
            if current and prediction.get("target_segment"):
                segment_id = prediction["target_segment"]["id"]
                if RISK_ORDER[prediction["risk"]] > RISK_ORDER.get(risks.get(segment_id), -1):
                    risks[segment_id] = prediction["risk"]
        return risks

    def network_snapshot(self) -> dict:
        network = copy.deepcopy(self.network)
        risks = self.segment_risks()
        for segment in network.get("segments", []):
            segment["risk"] = risks.get(segment["id"], "gray")
        return {**network, "run_id": self.run["id"] if self.run else None}

    async def publish(self) -> None:
        self.sequence += 1
        # Keep one immutable wire representation per ID, not 256 retained object
        # graphs or one serialization for every connected browser.
        self.stream.append((self.sequence, json.dumps(self.snapshot(), ensure_ascii=False)))
        async with self.condition:
            self.condition.notify_all()
