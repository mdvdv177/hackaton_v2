"""CSV ingestion with an explicit boundary between plans and outcomes."""

from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

PLAN_COLUMNS = ("tt_action_item_id", "tr_id", "time_begin", "geom", "building_address")
POINT_COLUMNS = ("sample_id", "tr_id", "T", "target_stop_id", "target_time_begin", "cur_dev_s")
TRAFFIC_COLUMNS = (
    "packet_id", "tr_id", "unit_id", "event_time", "receive_time", "location_valid",
    "lon", "lat", "speed", "heading",
)
_POINT_RE = re.compile(r"POINT\s*\(\s*([-+\d.eE]+)\s+([-+\d.eE]+)\s*\)", re.I)


def timestamp(value: Any) -> pd.Timestamp:
    """Treat naive source timestamps as UTC, consistently in both execution modes."""
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def timestamp_column(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, utc=True, format="mixed", errors="coerce")


def parse_geometry(value: Any) -> tuple[float, float]:
    match = _POINT_RE.fullmatch(str(value).strip())
    if not match:
        return float("nan"), float("nan")
    lon, lat = map(float, match.groups())
    return (lon, lat) if -180 <= lon <= 180 and -90 <= lat <= 90 else (float("nan"), float("nan"))


def safe_plan(frame: pd.DataFrame) -> pd.DataFrame:
    """Never retain time_fact_begin or another unapproved schedule field."""
    required = {"tt_action_item_id", "tr_id", "time_begin"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing plan columns: {sorted(required - set(frame.columns))}")
    result = frame.loc[:, [name for name in PLAN_COLUMNS if name in frame.columns]].copy()
    result["tr_id"] = result["tr_id"].astype(str)
    result["tt_action_item_id"] = result["tt_action_item_id"].astype(str)
    result["time_begin"] = timestamp_column(result["time_begin"])
    for name in ("geom", "building_address"):
        if name not in result:
            result[name] = ""
    if result["time_begin"].isna().any():
        raise ValueError("Invalid plan timestamp")
    return result.sort_values(["tr_id", "time_begin"], kind="stable").reset_index(drop=True)


def load_plan(path: str | Path) -> pd.DataFrame:
    return safe_plan(pd.read_csv(path, usecols=lambda name: name in PLAN_COLUMNS, dtype={"tr_id": str, "tt_action_item_id": str}))


def load_traffic(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=lambda name: name in TRAFFIC_COLUMNS,
                        dtype={"tr_id": str, "unit_id": str, "packet_id": str})
    frame["event_time"] = timestamp_column(frame["event_time"])
    frame["receive_time"] = timestamp_column(frame["receive_time"])
    if frame["event_time"].isna().any():
        raise ValueError("Invalid telemetry timestamp")
    return frame


def load_points(path: str | Path, *, labels: bool = False) -> pd.DataFrame:
    allowed = set(POINT_COLUMNS) | ({"target_delay_s", "target_class"} if labels else set())
    frame = pd.read_csv(path, usecols=lambda name: name in allowed,
                        dtype={"sample_id": str, "tr_id": str, "target_stop_id": str})
    for name in ("T", "target_time_begin"):
        frame[name] = timestamp_column(frame[name])
    if frame["sample_id"].duplicated().any():
        raise ValueError("Duplicate prediction sample_id")
    horizons = (frame["target_time_begin"] - frame["T"]).dt.total_seconds()
    if not ((horizons > 600) & (horizons <= 900)).all():
        raise ValueError("Every prediction target must be in (T+600s, T+900s]")
    if labels and not np.isfinite(frame["target_delay_s"]).all():
        raise ValueError("Labels must be finite signed seconds")
    return frame
