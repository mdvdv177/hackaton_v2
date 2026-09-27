"""Causal features shared by the HTTP application and offline experiments.

Identifiers only select input records; neither identifiers nor observed target
arrivals become features. The caller controls arrival-time availability in live
mode; event-time truncation is always enforced here as a second boundary.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from operator import attrgetter
from typing import Any

import numpy as np
import pandas as pd

from predictor.data import parse_geometry, safe_plan, timestamp, timestamp_column

FEATURE_SCHEMA_VERSION = "1.0.0"
WINDOWS = (60, 180, 300, 900)
_BASE_NAMES = (
    "cur_dev_s", "cur_dev_missing", "cur_dev_provided", "cur_dev_estimated", "cur_dev_age_s",
    "horizon_s", "hour_sin", "hour_cos", "target_lon", "target_lat", "target_geometry_missing",
    "stops_until_target", "last_event_age_s", "last_position_age_s", "last_speed",
    "distance_to_target_m", "distance_change_5m_m", "speed_change_1m", "stop_duration_s",
)
_WINDOW_NAMES = ("point_count", "speed_mean", "speed_median", "speed_std", "stopped_fraction",
                 "invalid_location_fraction", "missing_speed_fraction", "max_gap_s", "heading_consistency")
FEATURE_NAMES = list(_BASE_NAMES) + [f"{name}_{seconds}s" for seconds in WINDOWS for name in _WINDOW_NAMES]


def _number(value: Any, default: float = float("nan")) -> float:
    try:
        parsed = float(value)
        return parsed if np.isfinite(parsed) else default
    except (TypeError, ValueError):
        return default


def _valid(value: Any) -> bool:
    return value is True or str(value).lower() in {"true", "1"}


def haversine_m(lon: Any, lat: Any, target_lon: float, target_lat: float) -> Any:
    lon, lat, target_lon, target_lat = map(np.radians, (lon, lat, target_lon, target_lat))
    a = np.sin((target_lat - lat) / 2) ** 2 + np.cos(lat) * np.cos(target_lat) * np.sin((target_lon - lon) / 2) ** 2
    return 6371000.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _prepare_events(events: pd.DataFrame | Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    columns = ["tr_id", "event_time", "lon", "lat", "speed", "heading", "location_valid"]
    # Ingress carries additional metadata; avoid materializing/copying it for ML.
    frame = events.loc[:, events.columns.intersection(columns)].copy() if isinstance(events, pd.DataFrame) else pd.DataFrame(events, columns=columns)
    if frame.empty:
        return pd.DataFrame({
            "tr_id": pd.Series(dtype=str), "event_time": pd.Series(dtype="datetime64[ns, UTC]"),
            **{key: pd.Series(dtype=float) for key in ("lon", "lat", "speed", "heading")},
            "location_valid": pd.Series(dtype=bool),
        })
    frame["tr_id"] = frame["tr_id"].astype(str)
    frame["event_time"] = timestamp_column(frame["event_time"])
    for name in ("lon", "lat", "speed", "heading"):
        if name not in frame:
            frame[name] = np.nan
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    frame.loc[~frame["speed"].between(0, 200), "speed"] = np.nan
    if "location_valid" not in frame:
        frame["location_valid"] = False
    validity = frame["location_valid"]
    if pd.api.types.is_bool_dtype(validity.dtype):
        validity = validity.fillna(False)
    else:
        validity = validity.map(_valid)
    frame["location_valid"] = validity & frame["lon"].between(-180, 180) & frame["lat"].between(-90, 90)
    frame.loc[~frame["location_valid"], ["lon", "lat"]] = np.nan
    frame.loc[~frame["heading"].between(0, 360), "heading"] = np.nan
    return frame.dropna(subset=["event_time"]).sort_values("event_time", kind="stable").reset_index(drop=True)


@dataclass(frozen=True, slots=True)
class _EventIndex:
    ticks: np.ndarray
    lon: np.ndarray
    lat: np.ndarray
    speed: np.ndarray
    heading: np.ndarray
    valid: np.ndarray


@dataclass(frozen=True, slots=True)
class PreparedEvent:
    """Immutable runtime sidecar; parsed once, never persisted as raw telemetry."""

    event_ns: int
    received_ns: int
    lon: float
    lat: float
    speed: float
    heading: float
    valid: bool


def prepare_event(event: Mapping[str, Any]) -> PreparedEvent:
    """Prepare an ingested, normalized event without changing its public dict."""
    lon, lat = _number(event.get("lon")), _number(event.get("lat"))
    valid = _valid(event.get("location_valid")) and -180 <= lon <= 180 and -90 <= lat <= 90
    speed, heading = _number(event.get("speed")), _number(event.get("heading"))
    return PreparedEvent(
        event_ns=timestamp(event["event_time"]).value,
        received_ns=timestamp(event.get("received_at") or event.get("receive_time") or event["event_time"]).value,
        lon=lon if valid else np.nan, lat=lat if valid else np.nan,
        speed=speed if 0 <= speed <= 200 else np.nan,
        heading=heading if 0 <= heading <= 360 else np.nan,
        valid=bool(valid),
    )


def _scalar_seconds(nanoseconds: int) -> float:
    # Timestamp subtraction followed by Timedelta.total_seconds() truncates to
    # microseconds. Keep this separate from vector gaps, which retain nanoseconds.
    whole_seconds, microseconds = divmod(int(nanoseconds) // 1000, 1_000_000)
    return whole_seconds + microseconds / 1_000_000


class FeatureBuilder:
    """Index immutable history once; use the same extraction for every point."""

    def __init__(self, events: pd.DataFrame | Iterable[Mapping[str, Any]],
                 plan: pd.DataFrame | Iterable[Mapping[str, Any]]) -> None:
        events = _prepare_events(events)
        ticks = events["event_time"].array.as_unit("ns").asi8
        # Preserve source precision for scalar coordinate/speed operations. The
        # original extractor cast only window aggregates to float64.
        numeric = [events[name].to_numpy() for name in ("lon", "lat", "speed", "heading")]
        validity = events["location_valid"]
        valid = (validity.to_numpy(dtype=float, na_value=np.nan) if validity.isna().any()
                 else validity.to_numpy(dtype=bool))
        # Global stable time sorting preserves the input order of duplicate times.
        self.events = {str(key): _EventIndex(ticks[indices], *(values[indices] for values in numeric), valid[indices])
                       for key, indices in events.groupby("tr_id", sort=False).indices.items()}
        empty = np.empty(0, dtype=float)
        self.empty_events = _EventIndex(np.empty(0, dtype=np.int64), empty, empty, empty, empty, np.empty(0, dtype=bool))
        raw_plan = plan.copy() if isinstance(plan, pd.DataFrame) else pd.DataFrame(plan)
        if raw_plan.empty:
            raw_plan = pd.DataFrame(columns=["tt_action_item_id", "tr_id", "time_begin", "geom", "building_address"])
        plan_frame = safe_plan(raw_plan)
        self.plan = {str(key): group["time_begin"].array.as_unit("ns").asi8
                     for key, group in plan_frame.groupby("tr_id", sort=False)}
        self.visits = {(str(row["tr_id"]), str(row["tt_action_item_id"])): row for row in plan_frame.to_dict("records")}

    def with_prepared_histories(self, histories: Mapping[str, Iterable[PreparedEvent]],
                                cutoff: Any) -> "FeatureBuilder":
        """Reuse the plan and compact immutable runtime rows at one causal cutoff.

        This avoids reparsing timestamps, copying raw dictionaries and constructing
        pandas frames every 30 seconds. Both availability clocks are enforced here;
        build() independently enforces the event-time cutoff for each query.
        """
        at = timestamp(cutoff).value
        lower = at - 900_000_000_000
        indexed = {}
        for tr_id, history in histories.items():
            rows = [row for row in history if lower <= row.event_ns <= at and row.received_ns <= at]
            if not rows:
                continue
            count = len(rows)
            ticks = np.fromiter(map(attrgetter("event_ns"), rows), dtype=np.int64, count=count)
            numeric = [np.fromiter(map(attrgetter(name), rows), dtype=float, count=count)
                       for name in ("lon", "lat", "speed", "heading")]
            valid = np.fromiter(map(attrgetter("valid"), rows), dtype=bool, count=count)
            # Late delivery may reorder event time; stable ties retain arrival order.
            if np.any(ticks[1:] < ticks[:-1]):
                order = np.argsort(ticks, kind="stable")
                ticks, numeric, valid = ticks[order], [values[order] for values in numeric], valid[order]
            for values in (ticks, *numeric, valid):
                values.setflags(write=False)
            indexed[str(tr_id)] = _EventIndex(ticks, *numeric, valid)
        result = object.__new__(FeatureBuilder)
        result.events, result.empty_events = indexed, self.empty_events
        result.plan, result.visits = self.plan, self.visits
        result.prepared_cutoff_ns = at
        return result

    def build(self, point: Mapping[str, Any]) -> dict[str, float]:
        now, target_time = timestamp(point["T"]), timestamp(point["target_time_begin"])
        horizon = (target_time - now).total_seconds()
        if not 600 < horizon <= 900:
            raise ValueError("Target must fall in (T+600s, T+900s]")
        tr_id = str(point["tr_id"])
        events = self.events.get(tr_id, self.empty_events)
        now_ns = now.value
        if getattr(self, "prepared_cutoff_ns", now_ns) != now_ns:
            raise ValueError("Prepared history cutoff must match query T")
        left = events.ticks.searchsorted(now_ns - 900_000_000_000, side="left")
        right = events.ticks.searchsorted(now_ns, side="right")
        times, speed, valid = events.ticks[left:right], events.speed[left:right], events.valid[left:right]
        visit = self.visits.get((tr_id, str(point["target_stop_id"])), {})
        target_lon, target_lat = parse_geometry(visit.get("geom"))
        dev = _number(point.get("cur_dev_s"))
        source = point.get("cur_dev_source", "provided" if np.isfinite(dev) else "missing")
        if source == "missing":
            dev = float("nan")
        observed_at = point.get("cur_dev_observed_at")
        dev_age = max(0, (now - timestamp(observed_at)).total_seconds()) if observed_at is not None and not pd.isna(observed_at) else np.nan
        seconds_of_day = now.hour * 3600 + now.minute * 60 + now.second
        angle = 2 * np.pi * seconds_of_day / 86400
        vehicle_plan = self.plan.get(tr_id)
        stops = 0 if vehicle_plan is None else int(vehicle_plan.searchsorted(target_time.value, side="left")
                                                   - vehicle_plan.searchsorted(now_ns, side="right"))
        f: dict[str, float] = {
            "cur_dev_s": dev, "cur_dev_missing": float(not np.isfinite(dev)),
            "cur_dev_provided": float(source == "provided" and np.isfinite(dev)),
            "cur_dev_estimated": float(source == "estimated" and np.isfinite(dev)), "cur_dev_age_s": dev_age,
            "horizon_s": horizon, "hour_sin": float(np.sin(angle)), "hour_cos": float(np.cos(angle)),
            "target_lon": target_lon, "target_lat": target_lat,
            "target_geometry_missing": float(not np.isfinite(target_lon + target_lat)),
            "stops_until_target": float(stops), "last_event_age_s": np.nan,
            "last_position_age_s": np.nan, "last_speed": np.nan, "distance_to_target_m": np.nan,
            "distance_change_5m_m": np.nan, "speed_change_1m": np.nan, "stop_duration_s": 0.0,
        }
        if len(times):
            f["last_event_age_s"] = _scalar_seconds(now_ns - times[-1])
            f["last_speed"] = _number(speed[-1])
            positions = np.flatnonzero(valid == True)
            if len(positions):
                last = left + positions[-1]
                f["last_position_age_s"] = _scalar_seconds(now_ns - events.ticks[last])
                f["distance_to_target_m"] = float(haversine_m(events.lon[last], events.lat[last], target_lon, target_lat))
                recent_positions = positions[times[positions] >= now_ns - 300_000_000_000]
                if len(recent_positions) >= 2:
                    first = left + recent_positions[0]
                    f["distance_change_5m_m"] = f["distance_to_target_m"] - float(haversine_m(events.lon[first], events.lat[first], target_lon, target_lat))
            recent_speeds = speed[times.searchsorted(now_ns - 60_000_000_000):]
            recent_speeds = recent_speeds[~np.isnan(recent_speeds)]
            if len(recent_speeds) >= 2:
                f["speed_change_1m"] = float(recent_speeds[-1] - recent_speeds[0])
            stopped = speed < 3
            if stopped[-1]:
                moving = np.flatnonzero(~stopped)
                start = int(moving[-1] + 1) if len(moving) else 0
                disconnects = np.flatnonzero(np.diff(times) > 60_000_000_000) + 1
                if len(disconnects):
                    start = max(start, int(disconnects[-1]))
                f["stop_duration_s"] = _scalar_seconds(times[-1] - times[start])
        for seconds in WINDOWS:
            boundary = now_ns - seconds * 1_000_000_000
            begin = left + times.searchsorted(boundary, side="left")
            window_times = events.ticks[begin:right]
            window_speed = events.speed[begin:right]
            speeds = window_speed[~np.isnan(window_speed)].astype(float, copy=False)
            count = len(window_times)
            heading = events.heading[begin:right]
            heading = np.radians(heading[~np.isnan(heading)].astype(float, copy=False))
            gaps = np.diff(window_times) / 1_000_000_000
            # Add boundary gaps: an empty or sparse window cannot look healthy.
            if count:
                gaps = np.r_[gaps, _scalar_seconds(window_times[0] - boundary), _scalar_seconds(now_ns - window_times[-1])]
            # Duration weights avoid overweighting bursts of identical packets.
            if count >= 2:
                weights = np.minimum(gaps[:-2], 60)
                speed_for_weights = window_speed[:-1]
                known = np.isfinite(speed_for_weights)
                denominator = weights[known].sum()
                stopped_fraction = float(weights[known & (speed_for_weights < 3)].sum() / denominator) if denominator > 0 else np.nan
            else:
                stopped_fraction = np.nan
            window_valid = events.valid[begin:right]
            known_valid = window_valid[~np.isnan(window_valid)]
            values = {
                "point_count": float(count), "speed_mean": float(speeds.mean()) if len(speeds) else np.nan,
                "speed_median": float(np.median(speeds)) if len(speeds) else np.nan,
                "speed_std": float(speeds.std()) if len(speeds) else np.nan,
                "stopped_fraction": stopped_fraction,
                "invalid_location_fraction": float(1 - known_valid.mean()) if len(known_valid) else np.nan,
                "missing_speed_fraction": float(1 - len(speeds) / count) if count else np.nan,
                "max_gap_s": float(np.max(gaps)) if count else float(seconds),
                "heading_consistency": float(abs(np.exp(1j * heading).mean())) if len(heading) else np.nan,
            }
            f.update({f"{name}_{seconds}s": value for name, value in values.items()})
        return {name: float(f[name]) for name in FEATURE_NAMES}

    def build_many(self, points: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame([self.build(point) for point in points.to_dict("records")], columns=FEATURE_NAMES)


def build_features(events: pd.DataFrame | Iterable[Mapping[str, Any]],
                   plan: pd.DataFrame | Iterable[Mapping[str, Any]],
                   point: Mapping[str, Any]) -> dict[str, float]:
    """Pure online interface; supply only already received events in live mode."""
    return FeatureBuilder(events, plan).build(point)
