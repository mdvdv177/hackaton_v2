"""Versioned plan-only scenarios and explicitly sourced network geometry."""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from backend.data import coordinates, distance_m, iso, safe_plan, timestamp


def fingerprint(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()[:20]


def _coordinate(value: object) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("Coordinates must be [longitude, latitude]")
    lon, lat = map(float, value)
    if not math.isfinite(lon + lat) or not -180 <= lon <= 180 or not -90 <= lat <= 90:
        raise ValueError("Coordinates must be finite WGS84")
    return lon, lat


def _identifier(value: object, field: str) -> str:
    if value is None or isinstance(value, (dict, list, bool)) or not str(value).strip():
        raise ValueError(f"{field} must be a nonempty identifier")
    result = str(value).strip()
    if len(result) > 200:
        raise ValueError(f"{field} is too long")
    return result


_CATALOG_FIELDS = {
    "stops": {"id", "name", "coordinates"},
    "routes": {"id", "name", "direction_id"},
    "trips": {"id", "route_id", "direction_id"},
    "segments": {"id", "route_id", "from_visit_id", "to_visit_id", "geometry", "provenance"},
}


def _catalog(rows: list, label: str) -> dict:
    if not isinstance(rows, list):
        raise ValueError(f"network.{label} must be an array")
    result = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"Invalid {label} record")
        key = _identifier(row.get("id"), f"{label}.id")
        if key in result:
            raise ValueError(f"Duplicate {label} id: {key}")
        clean = {field: copy.deepcopy(value) for field, value in row.items() if field in _CATALOG_FIELDS[label]}
        clean["id"] = key
        for field in ("route_id", "direction_id", "from_visit_id", "to_visit_id"):
            if field in clean:
                clean[field] = _identifier(clean[field], f"{label}.{field}")
        for field in ("name", "provenance"):
            if field in clean:
                if not isinstance(clean[field], str):
                    raise ValueError(f"{label}.{field} must be text")
                clean[field] = clean[field][:500]
        result[key] = clean
    return result


def validate_package(payload: dict) -> dict:
    """Validate once, normalize to UTC, discard unrecognized fields/labels."""
    if not isinstance(payload, dict) or payload.get("schema_version", "1.0") != "1.0":
        raise ValueError("Expected a scenario with schema_version 1.0")
    source_id = _identifier(payload.get("source_id"), "source_id")
    name = str(payload.get("name") or source_id)[:200]
    zone_name = str(payload.get("timezone", "UTC"))
    try:
        zone = ZoneInfo(zone_name)
    except (KeyError, ValueError) as error:
        raise ValueError("Unknown scenario timezone") from error
    rows = payload.get("planned_visits")
    if not isinstance(rows, list) or not rows or len(rows) > 100000:
        raise ValueError("planned_visits must contain 1..100000 entries")
    plan, seen = [], set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Invalid visit")
        identifier = _identifier(row.get("tt_action_item_id"), "tt_action_item_id")
        if identifier in seen:
            raise ValueError(f"Duplicate visit: {identifier}")
        seen.add(identifier)
        tr_id = _identifier(row.get("tr_id"), "tr_id")
        try:
            at = datetime.fromisoformat(str(row["time_begin"]).replace("Z", "+00:00"))
        except (ValueError, KeyError, TypeError) as error:
            raise ValueError(f"Invalid time_begin for {identifier}") from error
        if at.tzinfo is None:
            at = at.replace(tzinfo=zone)
            if at.replace(fold=0).utcoffset() != at.replace(fold=1).utcoffset():
                raise ValueError("Ambiguous local time requires explicit UTC offset")
        geom = str(row.get("geom") or "")
        if geom:
            lat, lon = coordinates(geom)
            if lat is None:
                raise ValueError(f"Invalid POINT geometry: {identifier}")
            _coordinate([lon, lat])
        item = {"tt_action_item_id": identifier, "tr_id": tr_id, "time_begin": iso(at),
                "geom": geom, "building_address": str(row.get("building_address") or "")[:500]}
        for field in ("route_id", "trip_id", "stop_id", "direction_id"):
            if row.get(field) is not None:
                item[field] = _identifier(row[field], field)
        if row.get("sequence") is not None:
            seq = row["sequence"]
            if isinstance(seq, bool) or not isinstance(seq, int) or seq < 0:
                raise ValueError("sequence must be a nonnegative integer")
            item["sequence"] = seq
        plan.append(item)
    plan.sort(key=lambda row: (row["time_begin"], row["tt_action_item_id"]))
    vehicles = {row["tr_id"] for row in plan}
    raw_bindings = payload.get("device_bindings", {})
    if not isinstance(raw_bindings, dict):
        raise ValueError("device_bindings must map NDTP unit IDs to vehicles")
    bindings = {}
    for unit, vehicle in raw_bindings.items():
        unit, vehicle = _identifier(unit, "unit_id"), _identifier(vehicle, "binding tr_id")
        if not unit.isdecimal() or not 0 <= int(unit) <= 2147483647:
            raise ValueError("NDTP unit_id must be an integer between 0 and 2147483647")
        if vehicle not in vehicles:
            raise ValueError(f"Binding references unknown vehicle {vehicle}")
        normalized = str(int(unit))
        if normalized in bindings:
            raise ValueError("Ambiguous device binding")
        bindings[normalized] = vehicle
    supplied = payload.get("network") or {}
    if not isinstance(supplied, dict):
        raise ValueError("network must be an object")
    stops = _catalog(supplied.get("stops", []), "stops")
    routes = _catalog(supplied.get("routes", []), "routes")
    trips = _catalog(supplied.get("trips", []), "trips")
    for stop in stops.values():
        stop["coordinates"] = list(_coordinate(stop.get("coordinates")))
    for trip in trips.values():
        if str(trip.get("route_id")) not in routes:
            raise ValueError("Trip references an unknown route")
    for row in plan:
        for field, catalog in (("stop_id", stops), ("route_id", routes), ("trip_id", trips)):
            if field in row and row[field] not in catalog:
                raise ValueError(f"Visit references unknown {field}")
        if "trip_id" in row and "route_id" in row and str(trips[row["trip_id"]]["route_id"]) != row["route_id"]:
            raise ValueError("Visit route and trip disagree")
        if "stop_id" in row and row["geom"]:
            lat, lon = coordinates(row["geom"])
            stop_lon, stop_lat = stops[row["stop_id"]]["coordinates"]
            if distance_m(lat, lon, stop_lat, stop_lon) > 150:
                raise ValueError("Visit geometry disagrees with its physical stop")
        if "trip_id" in row and "direction_id" in row:
            direction = trips[row["trip_id"]].get("direction_id")
            if direction is not None and direction != row["direction_id"]:
                raise ValueError("Visit direction and trip disagree")
    by_id = {row["tt_action_item_id"]: row for row in plan}
    segments = _catalog(supplied.get("segments", []), "segments")
    pairs = set()
    for segment in segments.values():
        start_id, end_id = str(segment.get("from_visit_id")), str(segment.get("to_visit_id"))
        if start_id not in by_id or end_id not in by_id or start_id == end_id:
            raise ValueError("Segment endpoints must reference distinct planned visits")
        start, end = by_id[start_id], by_id[end_id]
        if start["tr_id"] != end["tr_id"] or start["time_begin"] >= end["time_begin"]:
            raise ValueError("Segment must follow the vehicle's visit order")
        if (start_id, end_id) in pairs:
            raise ValueError("Ambiguous segment mapping")
        pairs.add((start_id, end_id))
        shape = segment.get("geometry", {})
        if not isinstance(shape, dict) or shape.get("type") != "LineString" or not isinstance(shape.get("coordinates"), list) or len(shape["coordinates"]) < 2:
            raise ValueError("Segment geometry must be a GeoJSON LineString")
        line = [_coordinate(value) for value in shape["coordinates"]]
        for visit, endpoint in ((start, line[0]), (end, line[-1])):
            lat, lon = coordinates(visit["geom"])
            if lat is None or distance_m(lat, lon, endpoint[1], endpoint[0]) > 150:
                raise ValueError("Segment geometry direction/endpoints disagree with visits")
        if str(segment.get("route_id")) not in routes:
            raise ValueError("Supplied segment requires a known route_id")
        for visit in (start, end):
            route = visit.get("route_id")
            if route is None and "trip_id" in visit:
                route = trips[visit["trip_id"]]["route_id"]
            if route is not None and route != segment["route_id"]:
                raise ValueError("Segment route disagrees with its planned visits")
        segment["geometry"] = {"type": "LineString", "coordinates": [list(value) for value in line]}
    network_source = {"stops": list(stops.values()), "routes": list(routes.values()),
                      "trips": list(trips.values()), "segments": list(segments.values())}
    content = {"schema_version": "1.0", "source_id": source_id, "name": name, "timezone": zone_name,
               "plan": plan, "bindings": bindings, "network_source": network_source}
    # Only explicitly selected provenance is retained; facts cannot hide in metadata.
    raw_manifest = payload.get("manifest", {})
    if not isinstance(raw_manifest, dict):
        raise ValueError("manifest must be an object")
    manifest = {}
    for key in ("source_time", "wall_anchor"):
        if key in raw_manifest:
            value = raw_manifest[key]
            if not isinstance(value, str):
                raise ValueError(f"manifest.{key} must be an ISO timestamp")
            manifest[key] = iso(value)
    if "offset_s" in raw_manifest:
        value = raw_manifest["offset_s"]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("manifest.offset_s must be a finite number")
        manifest["offset_s"] = value
    if "source_sha256" in raw_manifest:
        value = raw_manifest["source_sha256"]
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdefABCDEF" for c in value):
            raise ValueError("manifest.source_sha256 must be a SHA256 hex digest")
        manifest["source_sha256"] = value.lower()
    if "demonstration" in raw_manifest:
        if not isinstance(raw_manifest["demonstration"], bool):
            raise ValueError("manifest.demonstration must be boolean")
        manifest["demonstration"] = raw_manifest["demonstration"]
    content["manifest"] = manifest
    content["version"] = fingerprint(content)
    content["id"] = f"{source_id}:{content['version']}"
    content["network"] = scenario_network(plan, content["id"], network_source)
    content["starts_at"], content["ends_at"] = plan[0]["time_begin"], plan[-1]["time_begin"]
    content["geometry_kind"] = content["network"]["geometry_kind"]
    content["coverage"] = content["network"]["coverage"]
    return content


def scenario_network(plan: list[dict], source_id: str, network: dict | None = None) -> dict:
    """Build all plan paths; imported geometry only replaces explicitly bound edges."""
    if network and "network_version" in network:
        return copy.deepcopy(network)
    imported = {(str(row["from_visit_id"]), str(row["to_visit_id"])): row for row in (network or {}).get("segments", [])}
    groups = defaultdict(list)
    for row in plan:
        day = timestamp(row["time_begin"]).date().isoformat()
        groups[(row["tr_id"], row.get("trip_id") or day)].append(row)
    paths, visits, segments, gaps = [], [], [], []
    used = set()
    for (tr_id, journey), rows in sorted(groups.items()):
        rows = sorted(rows, key=lambda row: (row["time_begin"], row.get("sequence", 0), row["tt_action_item_id"]))
        path_id = f"path:{fingerprint([source_id, tr_id, journey])}"
        paths.append({"id": path_id, "tr_id": tr_id, "route_id": rows[0].get("route_id"),
                      "name": f"План ТС {tr_id}", "visit_ids": [row["tt_action_item_id"] for row in rows]})
        for row in rows:
            lat, lon = coordinates(row.get("geom", ""))
            name = row.get("building_address") or f"Остановка без адреса · {row['tt_action_item_id'][-6:]}"
            visits.append({"id": row["tt_action_item_id"], "tr_id": tr_id, "path_id": path_id,
                           "name": name, "lat": lat, "lon": lon, "time_begin": row["time_begin"]})
        for previous, current in zip(rows, rows[1:]):
            first, second = coordinates(previous.get("geom", "")), coordinates(current.get("geom", ""))
            key = (previous["tt_action_item_id"], current["tt_action_item_id"])
            reason = None
            if previous["time_begin"] == current["time_begin"]:
                reason = "ambiguous_order"
            elif first[0] is None or second[0] is None:
                reason = "missing_geometry"
            elif first == second:
                reason = "zero_length"
            if reason:
                gaps.append({"from_visit_id": key[0], "to_visit_id": key[1], "reason": reason})
                continue
            supplied = imported.get(key)
            if supplied:
                used.add(key)
            segments.append({"id": f"segment:{fingerprint([path_id, *key])}", "path_id": path_id, "tr_id": tr_id,
                "from_visit_id": key[0], "to_visit_id": key[1],
                "from_name": previous.get("building_address") or f"Остановка · {key[0][-6:]}",
                "to_name": current.get("building_address") or f"Остановка · {key[1][-6:]}",
                "route_id": supplied.get("route_id") if supplied else None,
                "source_segment_id": supplied["id"] if supplied else None,
                "geometry_kind": "supplied_route" if supplied else "schedule_schematic",
                "geometry": supplied["geometry"] if supplied else {"type": "LineString", "coordinates": [[first[1], first[0]], [second[1], second[0]]]}})
    if imported.keys() - used:
        raise ValueError("Imported segments must link consecutive unambiguous visits in the same journey")
    kinds = {row["geometry_kind"] for row in segments}
    kind = "mixed" if len(kinds) > 1 else next(iter(kinds), "schedule_schematic")
    result = {"geometry_kind": kind, "paths": paths, "visits": visits, "segments": segments, "gaps": gaps,
              "coverage": {"visits": len(visits), "segments": len(segments), "supplied_segments": len(used),
                           "supplied_fraction": len(used) / len(segments) if segments else 0.0}}
    result["network_version"] = fingerprint(result)
    return result


def demo_package(dataset: Path, source_time: str = "2026-01-06T07:00:00Z", wall_anchor: datetime | None = None) -> dict:
    anchor = wall_anchor or datetime.now(timezone.utc)
    offset = anchor - timestamp(source_time)
    path = dataset / "test" / "schedule.csv"
    plan = safe_plan(path)
    for row in plan:
        row["time_begin"] = iso(timestamp(row["time_begin"]) + offset)
    vehicles = {row["tr_id"] for row in plan}
    choices = defaultdict(set)
    with (dataset / "test" / "traffic.csv").open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["tr_id"] in vehicles:
                choices[row["unit_id"]].add(row["tr_id"])
    bindings = {unit: next(iter(ids)) for unit, ids in choices.items() if len(ids) == 1}
    return validate_package({"schema_version": "1.0", "source_id": "demo-test", "name": "NDTP · демонстрационный сценарий",
        "timezone": "UTC", "planned_visits": plan, "device_bindings": bindings,
        "manifest": {"source_time": iso(source_time), "wall_anchor": iso(anchor), "offset_s": offset.total_seconds(),
                     "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "demonstration": True}})
