"""Causal stop matching and observed stop-to-stop segment speeds.

This module deliberately has no database, labels, model, or Backend dependency.
Both streaming training and the dispatcher feed events in availability order.
"""

from __future__ import annotations

import bisect
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import math
import re
from typing import Any

OBSERVER_VERSION = "2.0.0"
_POINT = re.compile(r"POINT\s*\(\s*([-+\d.eE]+)\s+([-+\d.eE]+)\s*\)", re.I)


def epoch(value: Any) -> float:
    if isinstance(value, (float, int)):
        return float(value)
    parsed = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return (parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed).timestamp()


def iso(value: Any) -> str:
    return datetime.fromtimestamp(epoch(value), timezone.utc).isoformat()


def finite(value: Any) -> float | None:
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (ValueError, TypeError):
        return None


def distance_m(lat: float, lon: float, lat2: float, lon2: float) -> float:
    a, b = math.radians(lat), math.radians(lat2)
    h = math.sin((b - a) / 2) ** 2 + math.cos(a) * math.cos(b) * math.sin(math.radians(lon2 - lon) / 2) ** 2
    return 6371000 * 2 * math.asin(math.sqrt(min(1.0, max(0.0, h))))


class CausalObserver:
    """Confirm arrivals with two points; require departure before another visit.

    Matching uses plan times corrected only by a previously observed deviation.
    When two spatially valid visits differ by less than 60 seconds in matching
    score neither is asserted. Late packets remain useful to feature histories,
    but never move the stop observer backwards.
    """

    def __init__(self, plan: list[dict], segments: list[dict] | None = None):
        self.plan: dict[str, list[dict]] = defaultdict(list)
        for row in plan:
            match = _POINT.fullmatch(str(row.get("geom", "")).strip())
            lon, lat = map(float, match.groups()) if match else (None, None)
            item = {"id": str(row["tt_action_item_id"]), "tr_id": str(row["tr_id"]),
                    "time": epoch(row["time_begin"]), "lat": lat, "lon": lon}
            self.plan[item["tr_id"]].append(item)
        for rows in self.plan.values():
            rows.sort(key=lambda row: (row["time"], row["id"]))
            for index, row in enumerate(rows):
                row["index"] = index
        self.ticks = {key: [row["time"] for row in rows] for key, rows in self.plan.items()}
        self.indices = {(key, row["id"]): row["index"] for key, rows in self.plan.items() for row in rows}
        self.segment_ids = {(str(row["from_visit_id"]), str(row["to_visit_id"])): str(row["id"])
                            for row in segments or []}
        self.visits: dict[str, dict] = {}
        self.candidates: dict[str, dict] = {}
        self.deviations: dict[str, dict] = {}
        self.last_event: dict[str, float] = {}
        self.traversals: dict[str, dict] = {}
        self.confirmed_count = 0

    def hints(self, tr_id: str, at: Any) -> dict:
        result = self.deviations.get(str(tr_id))
        if not result or not 0 <= epoch(at) - epoch(result["observed_at"]) <= 900:
            return {"value": None, "source": "missing", "observed_at": None}
        return dict(result)

    def has_visited(self, tr_id: str, visit_id: str) -> bool:
        previous = self.visits.get(str(tr_id))
        index = self.indices.get((str(tr_id), str(visit_id)))
        return bool(previous and index is not None and index <= previous["index"])

    def observe(self, event: dict, at: Any | None = None) -> dict | None:
        tr_id, tick = str(event["tr_id"]), epoch(event["event_time"])
        received = event.get("received_at", event.get("receive_time", event["event_time"]))
        if at is not None and (tick > epoch(at) or epoch(received) > epoch(at)):
            return None
        if tick <= self.last_event.get(tr_id, float("-inf")):
            return None
        self.last_event[tr_id] = tick
        lat, lon = finite(event.get("lat")), finite(event.get("lon"))
        valid = str(event.get("location_valid", False)).lower() in {"true", "1"}
        valid = valid and lat is not None and lon is not None and -90 <= lat <= 90 and -180 <= lon <= 180
        speed = finite(event.get("speed"))
        speed = speed if speed is not None and 0 <= speed <= 200 else None
        previous = self.visits.get(tr_id)
        if previous and not previous["departed"] and valid:
            if distance_m(lat, lon, previous["lat"], previous["lon"]) <= 100:
                self.candidates.pop(tr_id, None)
                return None
            previous["departed"] = True
            next_index = previous["index"] + 1
            if next_index < len(self.plan.get(tr_id, [])):
                to_visit = self.plan[tr_id][next_index]["id"]
                self.traversals[tr_id] = {"from_visit_id": previous["target_visit_id"],
                    "to_visit_id": to_visit, "started_at": tick, "ended_at": None, "samples": []}
        traversal = self.traversals.get(tr_id)
        if traversal and traversal["ended_at"] is None:
            samples = traversal["samples"]
            samples.append([tick, speed, bool(valid)])
            while len(samples) > 1 and (samples[1][0] < tick - 900 or len(samples) > 2048):
                samples.pop(0)
        if not valid:
            self.candidates.pop(tr_id, None)
            return None
        hint = self.hints(tr_id, tick)
        expected = tick - (hint["value"] or 0)
        ticks = self.ticks.get(tr_id, [])
        left, right = bisect.bisect_left(ticks, expected - 900), bisect.bisect_right(ticks, expected + 900)
        eligible = []
        for visit in self.plan.get(tr_id, [])[left:right]:
            if previous and visit["index"] <= previous["index"]:
                continue
            # Corrected matching must never drift through later trips on a loop.
            # The hard +/-15 minute plan tolerance applies independently of hints.
            if abs(tick - visit["time"]) > 900:
                continue
            if visit["lat"] is None or distance_m(lat, lon, visit["lat"], visit["lon"]) > 75:
                continue
            eligible.append((abs(expected - visit["time"]), visit))
        eligible.sort(key=lambda item: item[0])
        if not eligible or (len(eligible) > 1 and eligible[1][0] - eligible[0][0] < 60):
            self.candidates.pop(tr_id, None)
            return None
        visit = eligible[0][1]
        candidate = self.candidates.get(tr_id)
        if not candidate or candidate["visit_id"] != visit["id"] or tick - candidate["first_at"] > 60:
            self.candidates[tr_id] = {"visit_id": visit["id"], "first_at": tick}
            return None
        arrived = candidate["first_at"]
        deviation = arrived - visit["time"]
        result = {"tr_id": tr_id, "target_visit_id": visit["id"], "index": visit["index"],
                  "planned_at": iso(visit["time"]), "observed_at": iso(arrived), "cur_dev_s": deviation,
                  "quality": "two_valid_points_within_75m", "lat": visit["lat"], "lon": visit["lon"], "departed": False}
        self.visits[tr_id] = result
        self.deviations[tr_id] = {"value": deviation, "source": "estimated", "observed_at": iso(arrived)}
        if traversal and traversal["ended_at"] is None:
            # Skipping a planned stop does not turn a multi-stop trip into one segment.
            if traversal["to_visit_id"] == visit["id"]:
                traversal["ended_at"] = arrived
            else:
                self.traversals.pop(tr_id, None)
        self.candidates.pop(tr_id, None)
        self.confirmed_count += 1
        return dict(result)

    def segment(self, tr_id: str, at: Any) -> dict:
        segment = self.traversals.get(str(tr_id))
        empty = {"segment_id": None, "from_visit_id": None, "to_visit_id": None,
                 "segment_speed_kmh": None, "speed_coverage": 0.0, "elapsed_s": 0.0,
                 "max_gap_s": None, "quality": "unobserved"}
        if not segment:
            return empty
        tick = epoch(at)
        end = min(tick, segment["ended_at"]) if segment["ended_at"] is not None else tick
        start = segment["started_at"]
        elapsed = max(0, end - start)
        samples = [sample for sample in segment["samples"] if sample[0] <= end]
        covered, weighted, max_gap = 0.0, 0.0, 0.0
        for previous, current in zip(samples, samples[1:]):
            gap = current[0] - previous[0]
            max_gap = max(max_gap, gap)
            if 0 < gap <= 60 and previous[1] is not None and previous[2] and current[2]:
                covered += gap
                weighted += previous[1] * gap
        if samples:
            max_gap = max(max_gap, samples[0][0] - start, end - samples[-1][0])
        else:
            max_gap = elapsed
        coverage = min(1.0, covered / elapsed) if elapsed else 0.0
        fresh = bool(samples and tick - samples[-1][0] <= 60)
        reliable = elapsed >= 30 and coverage >= .8 and max_gap <= 60 and fresh
        pair = (segment["from_visit_id"], segment["to_visit_id"])
        return {"segment_id": self.segment_ids.get(pair, f"{pair[0]}->{pair[1]}"),
                "from_visit_id": pair[0], "to_visit_id": pair[1],
                "segment_speed_kmh": weighted / covered if reliable and covered else None,
                "speed_coverage": coverage, "elapsed_s": elapsed, "max_gap_s": max_gap,
                "quality": "observed" if reliable else "insufficient_coverage"}

    def dump_state(self) -> dict:
        return deepcopy({"version": OBSERVER_VERSION, "visits": self.visits, "candidates": self.candidates,
                         "deviations": self.deviations, "last_event": self.last_event,
                         "traversals": self.traversals, "confirmed_count": self.confirmed_count})

    def restore_state(self, state: dict) -> None:
        if state and state.get("version") != OBSERVER_VERSION:
            raise ValueError("Incompatible observer state version")
        for field in ("visits", "candidates", "deviations", "last_event", "traversals"):
            setattr(self, field, deepcopy(state.get(field, {})))
        self.confirmed_count = int(state.get("confirmed_count", 0))
