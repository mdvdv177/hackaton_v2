"""Receive-time features for the dispatcher profile; official schema stays frozen."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

import numpy as np
import pandas as pd

from predictor.data import safe_plan, timestamp_column
from predictor.features import FEATURE_NAMES, FeatureBuilder
from predictor.observer import CausalObserver, epoch

STREAM_SCHEMA_VERSION = "2.0.0"
STREAM_FEATURE_NAMES = FEATURE_NAMES + ["segment_speed_kmh", "segment_speed_missing",
    "segment_speed_coverage", "segment_elapsed_s", "segment_max_gap_s"]


def enrich_stream_features(base: dict, segment: dict) -> dict[str, float]:
    value = segment.get("segment_speed_kmh")
    additions = {"segment_speed_kmh": value, "segment_speed_missing": float(value is None),
                 "segment_speed_coverage": segment.get("speed_coverage", 0),
                 "segment_elapsed_s": segment.get("elapsed_s", 0),
                 "segment_max_gap_s": segment.get("max_gap_s")}
    return {**base, **{key: float(item) if item is not None else np.nan for key, item in additions.items()}}


def build_stream_dataset(traffic: pd.DataFrame, plan: pd.DataFrame, points: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Strict causal reducer, ordered by max(receive_time, event_time).

    Point labels and supplied cur_dev_s are never passed to the observer or
    extractor. Each point sees exactly the events available at its own T.
    """
    plan = safe_plan(plan)
    rows = traffic.copy()
    rows["event_time"] = timestamp_column(rows["event_time"])
    receive_name = "receive_time" if "receive_time" in rows else "received_at"
    rows[receive_name] = timestamp_column(rows[receive_name]).fillna(rows["event_time"])
    rows["_available"] = rows[["event_time", receive_name]].max(axis=1)
    rows = rows.sort_values(["_available", "packet_id"], kind="stable")
    available = rows["_available"].map(epoch).to_numpy()
    events = rows.drop(columns="_available").to_dict("records")
    observer = CausalObserver(plan.to_dict("records"))
    plans = {str(key): frame for key, frame in plan.groupby("tr_id", sort=False)}
    history: dict[str, deque] = defaultdict(deque)
    seen: set[tuple[str, str]] = set()
    cursor, output, diagnostics = 0, {}, {}
    for original in points.sort_values("T", kind="stable").to_dict("records"):
        tick, tr_id = epoch(original["T"]), str(original["tr_id"])
        while cursor < len(events) and available[cursor] <= tick:
            event = events[cursor]
            cursor += 1
            vehicle = str(event["tr_id"])
            key = (vehicle, str(event["packet_id"]))
            if key in seen:
                continue
            seen.add(key)
            observer.observe(event, at=available[cursor - 1])
            history[vehicle].append(event)
        # Out-of-order delivery is allowed; filter all rows, not just the head.
        for vehicle, values in history.items():
            history[vehicle] = deque(row for row in values if epoch(row["event_time"]) >= tick - 900)
        hint = observer.hints(tr_id, tick)
        point = {name: original[name] for name in ("sample_id", "tr_id", "T", "target_stop_id", "target_time_begin")}
        point.update(cur_dev_s=hint["value"], cur_dev_source=hint["source"], cur_dev_observed_at=hint["observed_at"])
        base = FeatureBuilder(list(history[tr_id]), plans.get(tr_id, plan.iloc[:0])).build(point)
        features = enrich_stream_features(base, observer.segment(tr_id, tick))
        output[str(original["sample_id"])] = features
        diagnostics[str(original["sample_id"])] = {"sample_id": str(original["sample_id"]),
            "estimated_cur_dev_s": hint["value"], "hint_source": hint["source"],
            "fresh": bool(np.isfinite(base["last_event_age_s"]) and base["last_event_age_s"] <= 60),
            "target_already_observed": observer.has_visited(tr_id, str(original["target_stop_id"])),
            "confirmed_visits": observer.confirmed_count}
    order = points["sample_id"].astype(str).tolist()
    return pd.DataFrame([output[key] for key in order], columns=STREAM_FEATURE_NAMES), [diagnostics[key] for key in order]
