# Frozen pre-optimization reference, 2026-09-26. Test-only: do not synchronize with production.
"""Causal features shared by the HTTP application and offline experiments.

Identifiers only select input records; neither identifiers nor observed target
arrivals become features. The caller controls arrival-time availability in live
mode; event-time truncation is always enforced here as a second boundary.
"""

from collections.abc import Iterable, Mapping
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
    frame = events.copy() if isinstance(events, pd.DataFrame) else pd.DataFrame(events)
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
    frame["location_valid"] = frame["location_valid"].map(_valid) & frame["lon"].between(-180, 180) & frame["lat"].between(-90, 90)
    frame.loc[~frame["location_valid"], ["lon", "lat"]] = np.nan
    frame.loc[~frame["heading"].between(0, 360), "heading"] = np.nan
    return frame.dropna(subset=["event_time"]).sort_values("event_time", kind="stable").reset_index(drop=True)


class FeatureBuilder:
    """Index immutable history once; use the same extraction for every point."""

    def __init__(self, events: pd.DataFrame | Iterable[Mapping[str, Any]],
                 plan: pd.DataFrame | Iterable[Mapping[str, Any]]) -> None:
        events = _prepare_events(events)
        self.events = {str(key): group.reset_index(drop=True) for key, group in events.groupby("tr_id", sort=False)}
        self.empty_events = events.iloc[:0]
        raw_plan = plan.copy() if isinstance(plan, pd.DataFrame) else pd.DataFrame(plan)
        if raw_plan.empty:
            raw_plan = pd.DataFrame(columns=["tt_action_item_id", "tr_id", "time_begin", "geom", "building_address"])
        plan_frame = safe_plan(raw_plan)
        self.plan = {str(key): group for key, group in plan_frame.groupby("tr_id", sort=False)}
        self.visits = {(str(row["tr_id"]), str(row["tt_action_item_id"])): row for row in plan_frame.to_dict("records")}

    def build(self, point: Mapping[str, Any]) -> dict[str, float]:
        now, target_time = timestamp(point["T"]), timestamp(point["target_time_begin"])
        horizon = (target_time - now).total_seconds()
        if not 600 < horizon <= 900:
            raise ValueError("Target must fall in (T+600s, T+900s]")
        tr_id = str(point["tr_id"])
        history = self.events.get(tr_id, self.empty_events)
        times = history["event_time"]
        left = times.searchsorted(now - pd.Timedelta(seconds=900), side="left")
        right = times.searchsorted(now, side="right")
        history = history.iloc[left:right]
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
        stops = 0 if vehicle_plan is None else int(((vehicle_plan["time_begin"] > now) & (vehicle_plan["time_begin"] < target_time)).sum())
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
        if not history.empty:
            f["last_event_age_s"] = (now - history.iloc[-1]["event_time"]).total_seconds()
            f["last_speed"] = _number(history.iloc[-1]["speed"])
            positions = history[history["location_valid"]]
            if not positions.empty:
                last = positions.iloc[-1]
                f["last_position_age_s"] = (now - last["event_time"]).total_seconds()
                f["distance_to_target_m"] = float(haversine_m(last["lon"], last["lat"], target_lon, target_lat))
                recent_positions = positions[positions["event_time"] >= now - pd.Timedelta(minutes=5)]
                if len(recent_positions) >= 2:
                    first = recent_positions.iloc[0]
                    f["distance_change_5m_m"] = f["distance_to_target_m"] - float(haversine_m(first["lon"], first["lat"], target_lon, target_lat))
            recent_speeds = history.loc[history["event_time"] >= now - pd.Timedelta(minutes=1), "speed"].dropna()
            if len(recent_speeds) >= 2:
                f["speed_change_1m"] = float(recent_speeds.iloc[-1] - recent_speeds.iloc[0])
            stopped = history["speed"].lt(3).to_numpy()
            if stopped[-1]:
                moving = np.flatnonzero(~stopped)
                start = int(moving[-1] + 1) if len(moving) else 0
                gaps = history["event_time"].diff().dt.total_seconds().to_numpy()
                disconnects = np.flatnonzero(gaps > 60)
                if len(disconnects):
                    start = max(start, int(disconnects[-1]))
                f["stop_duration_s"] = (history.iloc[-1]["event_time"] - history.iloc[start]["event_time"]).total_seconds()
        for seconds in WINDOWS:
            window = history[history["event_time"] >= now - pd.Timedelta(seconds=seconds)]
            speeds = window["speed"].dropna().to_numpy(dtype=float)
            count = len(window)
            heading = np.radians(window["heading"].dropna().to_numpy(dtype=float))
            gaps = window["event_time"].diff().dt.total_seconds().dropna().to_numpy()
            # Add boundary gaps: an empty or sparse window cannot look healthy.
            if count:
                gaps = np.r_[gaps, (window.iloc[0]["event_time"] - (now - pd.Timedelta(seconds=seconds))).total_seconds(), (now - window.iloc[-1]["event_time"]).total_seconds()]
            # Duration weights avoid overweighting bursts of identical packets.
            if count >= 2:
                weights = np.minimum(window["event_time"].diff().dt.total_seconds().dropna().to_numpy(), 60)
                speed_for_weights = window["speed"].iloc[:-1].to_numpy()
                known = np.isfinite(speed_for_weights)
                denominator = weights[known].sum()
                stopped_fraction = float(weights[known & (speed_for_weights < 3)].sum() / denominator) if denominator > 0 else np.nan
            else:
                stopped_fraction = np.nan
            values = {
                "point_count": float(count), "speed_mean": float(speeds.mean()) if len(speeds) else np.nan,
                "speed_median": float(np.median(speeds)) if len(speeds) else np.nan,
                "speed_std": float(speeds.std()) if len(speeds) else np.nan,
                "stopped_fraction": stopped_fraction,
                "invalid_location_fraction": float(1 - window["location_valid"].mean()) if count else np.nan,
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
