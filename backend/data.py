"""Safe CSV ingestion. Actual arrivals and targets never enter live state."""
from __future__ import annotations

import csv
import math
import re
from datetime import datetime, timezone
from pathlib import Path

PLAN_COLUMNS = ("tt_action_item_id", "tr_id", "time_begin", "geom", "building_address")
POINT_COLUMNS = ("sample_id", "tr_id", "T", "target_stop_id", "target_time_begin", "cur_dev_s")


def timestamp(value: str | datetime | float) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, (float, int)):
        parsed = datetime.fromtimestamp(value, timezone.utc)
    else:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


def iso(value: str | datetime | float) -> str:
    return timestamp(value).isoformat()


def number(value: object) -> float | None:
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def coordinates(geom: str) -> tuple[float | None, float | None]:
    match = re.fullmatch(r"\s*POINT\s*\(\s*([-+\d.eE]+)\s+([-+\d.eE]+)\s*\)\s*", geom or "")
    return (float(match[2]), float(match[1])) if match else (None, None)


def safe_plan(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as stream:
        result = [{key: row[key] for key in PLAN_COLUMNS} for row in csv.DictReader(stream)]
    for row in result:
        row["time_begin"] = iso(row["time_begin"])
    return sorted(result, key=lambda r: r["time_begin"])


def safe_points(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as stream:
        result = [{key: row[key] for key in POINT_COLUMNS} for row in csv.DictReader(stream)]
    for row in result:
        row["T"] = iso(row["T"])
        row["target_time_begin"] = iso(row["target_time_begin"])
        row["cur_dev_s"] = number(row["cur_dev_s"])
        horizon = (timestamp(row["target_time_begin"]) - timestamp(row["T"])).total_seconds()
        if not 600 < horizon <= 900:
            raise ValueError(f"Invalid prediction horizon: {row['sample_id']}")
    return sorted(result, key=lambda row: row["T"])


def normalize_event(row: dict) -> dict:
    event_time = iso(row["event_time"])
    valid = str(row.get("location_valid", False)).lower() in {"true", "1"}
    lat, lon = number(row.get("lat")), number(row.get("lon"))
    valid = valid and lat is not None and lon is not None and -90 <= lat <= 90 and -180 <= lon <= 180
    speed = number(row.get("speed"))
    speed = speed if speed is not None and 0 <= speed <= 200 else None
    return {
        "tr_id": str(row.get("tr_id", "")), "unit_id": str(row.get("unit_id", "")),
        "event_time": event_time, "received_at": iso(row.get("receive_time") or row.get("received_at") or event_time),
        "packet_id": str(row.get("packet_id") or f"{row.get('unit_id', '')}:{event_time}"),
        "lat": lat if valid else None, "lon": lon if valid else None,
        "speed": speed, "heading": number(row.get("heading")), "location_valid": valid,
    }


def read_traffic(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as stream:
        return [normalize_event(row) for row in csv.DictReader(stream)]


def distance_m(lat: float, lon: float, other_lat: float, other_lon: float) -> float:
    rlat, rlat2 = math.radians(lat), math.radians(other_lat)
    dlat, dlon = rlat2 - rlat, math.radians(other_lon - lon)
    h = math.sin(dlat / 2) ** 2 + math.cos(rlat) * math.cos(rlat2) * math.sin(dlon / 2) ** 2
    return 6371000 * 2 * math.asin(min(1, math.sqrt(h)))
