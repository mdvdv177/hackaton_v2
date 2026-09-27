"""Prepared runtime histories remain immutable and numerically identical."""
from collections import defaultdict
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backend.data import read_traffic
from predictor.data import load_plan, load_points, timestamp
from predictor.features import FEATURE_NAMES, FeatureBuilder, prepare_event
from reference_features import FeatureBuilder as ReferenceBuilder


def equal(reference, actual):
    assert list(reference) == list(actual) == FEATURE_NAMES
    for key, value in reference.items():
        assert value == actual[key] or np.isnan(value) and np.isnan(actual[key]), (key, value, actual[key])


def test_prepared_cutoffs_stable_duplicates_and_immutable_snapshots():
    now = pd.Timestamp("2026-01-06T12:00:00.123456789Z")
    plan = [{"tr_id": "v", "tt_action_item_id": "target", "time_begin": now + pd.Timedelta(seconds=720), "geom": "POINT (37.6 55.7)"}]
    rows = []
    for index, (event_delta, received_delta) in enumerate([
        (-900_000_000_001, -900_000_000_001), (-900_000_000_000, -900_000_000_000),
        (-60_000_000_001, -50_000_000_000), (-60_000_000_000, -60_000_000_000),
        (-30_000_000_000, -30_000_000_000), (-30_000_000_000, -20_000_000_000),
        (-1, 1), (1, -1), (0, 0),
    ]):
        rows.append({"tr_id": "v", "event_time": (now + pd.Timedelta(event_delta, unit="ns")).isoformat(),
                     "received_at": (now + pd.Timedelta(received_delta, unit="ns")).isoformat(),
                     "speed": [20., None, 0., 201., -1.][index % 5], "heading": 361. if index % 2 else 180.,
                     "lon": 37.6 + index / 100, "lat": 55.7, "location_valid": index % 3 != 0})
    rows = [rows[index] for index in (3, 0, 1, 2, 4, 5, 6, 7, 8)]  # Real late arrival, stable duplicate order.
    prepared = [prepare_event(row) for row in rows]
    plan_builder = FeatureBuilder([], plan)
    builder = plan_builder.with_prepared_histories({"v": tuple(prepared)}, now)
    eligible = [row for row, item in zip(rows, prepared) if now.value - 900_000_000_000 <= item.event_ns <= now.value and item.received_ns <= now.value]
    point = {"tr_id": "v", "T": now, "target_stop_id": "target", "target_time_begin": now + pd.Timedelta(seconds=720)}
    reference = ReferenceBuilder(eligible, plan).build(point)
    equal(reference, builder.build(point))
    assert len(builder.events["v"].ticks) == len(eligible)
    with pytest.raises(FrozenInstanceError):
        prepared[0].speed = 199
    with pytest.raises(ValueError):
        builder.events["v"].speed[0] = 199
    for row in rows:
        row["speed"] = 199
    prepared.append(prepare_event({**rows[0], "event_time": (now + pd.Timedelta(seconds=1)).isoformat()}))
    equal(reference, builder.build(point))
    assert plan_builder.events == {}
    # An earlier query must not reuse rows that arrived between its T and this
    # builder's cutoff. A later query must not reuse a shortened history window.
    for change in (-1, 1):
        with pytest.raises(ValueError, match="cutoff must match"):
            builder.build({**point, "T": now + pd.Timedelta(change, unit="ns")})


def test_every_test_point_matches_reference_after_real_receive_cutoff():
    dataset = Path(__file__).resolve().parents[1] / "dataset"
    plan = load_plan(dataset / "test/schedule.csv")
    points = load_points(dataset / "labels/labels_test.csv")
    histories = defaultdict(list)
    for row in read_traffic(dataset / "test/traffic.csv"):
        histories[row["tr_id"]].append((prepare_event(row), row))
    indexed_plan = FeatureBuilder([], plan)
    plans = {str(key): group for key, group in plan.groupby("tr_id", sort=False)}
    for point in points.to_dict("records"):
        tr_id, cutoff = str(point["tr_id"]), timestamp(point["T"]).value
        history = histories[tr_id]
        reference_rows = [row for item, row in history if cutoff - 900_000_000_000 <= item.event_ns <= cutoff and item.received_ns <= cutoff]
        actual = indexed_plan.with_prepared_histories({tr_id: [item for item, _ in history]}, point["T"]).build(point)
        reference = ReferenceBuilder(reference_rows, plans[tr_id]).build(point)
        equal(reference, actual)
