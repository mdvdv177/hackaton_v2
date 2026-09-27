"""Numerical contract against the frozen pandas extractor used to train both models."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from predictor.data import load_plan, load_points, load_traffic
from predictor.features import FEATURE_NAMES, FeatureBuilder
from reference_features import FeatureBuilder as ReferenceBuilder


def assert_exact(reference, actual):
    assert list(reference) == list(actual) == FEATURE_NAMES
    for name in FEATURE_NAMES:
        old, new = reference[name], actual[name]
        assert old == new or (np.isnan(old) and np.isnan(new)), (name, old, new)


@pytest.mark.parametrize("resolution", ["ns", "us", "ms", "s"])
def test_exact_reference_with_duplicates_boundaries_missing_and_stop_gaps(resolution):
    now = pd.Timestamp("2026-01-06T12:00:00.123456789Z").as_unit(resolution)
    # Includes boundary ±1ns and a >60s stopped gap; duplicated timestamps retain
    # input order (which controls the final speed, position, and stopped duration).
    offsets = [-901e9, -900e9-1, -900e9, -900e9+1, -300e9-1, -300e9, -180e9,
               -120e9-1, -60e9-1, -60e9, -30e9, -30e9, -1, 0, 1, 90e9]
    rows = []
    for index, offset in enumerate(offsets):
        rows.append({"tr_id": "v", "event_time": now + pd.Timedelta(int(offset), unit="ns"),
                     "lon": 37.5 + index / 1000, "lat": 55.5, "speed": [None, 0, 2, 3, 40, 999, -1][index % 7],
                     "heading": [None, -1, 0, 90, 360, 361][index % 6],
                     "location_valid": [True, "True", "1", False, None][index % 5]})
    rows += [{**rows[0], "tr_id": "other", "event_time": now, "speed": 199}]
    frame = pd.DataFrame(rows)
    frame["event_time"] = pd.to_datetime(frame["event_time"], utc=True).dt.as_unit(resolution)
    plan = [{"tr_id": "v", "tt_action_item_id": str(i), "time_begin": now + pd.Timedelta(seconds=offset),
             "geom": "POINT (37.6 55.7)"} for i, offset in enumerate([0, 1, 601, 720, 721])]
    reference, actual = ReferenceBuilder(frame, plan), FeatureBuilder(frame, plan)
    for delta in [0, 1, 30, 61, 180, 300, 900, 1800]:
        for vehicle in ("v", "other", "absent"):
            point = {"tr_id": vehicle, "T": now + pd.Timedelta(seconds=delta), "target_stop_id": "3",
                     "target_time_begin": now + pd.Timedelta(seconds=delta + 720),
                     "cur_dev_s": -15, "cur_dev_source": "estimated", "cur_dev_observed_at": now}
            assert_exact(reference.build(point), actual.build(point))


def test_exact_reference_randomized_unsorted_history_and_explicit_stop_runs():
    rng = np.random.default_rng(731)
    now = pd.Timestamp("2026-01-06T12:00:00.555555555Z")
    rows = []
    for vehicle in ("a", "b", "c"):
        for index in range(400):
            at = now + pd.Timedelta(int(rng.integers(-1_000_000_000_000, 100_000_000_000)), unit="ns")
            rows.append({"tr_id": vehicle, "event_time": at.isoformat(),
                         "lon": 37 + rng.random(), "lat": 55 + rng.random(),
                         "speed": rng.choice([np.nan, 0., 2.9, 3., 25., 200., 201.]),
                         "heading": rng.choice([np.nan, 0., 123.5, 360.]), "location_valid": bool(index % 3)})
    for offset in (-180_000_000_001, -120_000_000_000, -60_000_000_001, -1, 0):
        rows.append({"tr_id": "stopped", "event_time": (now + pd.Timedelta(offset, unit="ns")).isoformat(), "speed": 0})
    rng.shuffle(rows)
    plan = [{"tr_id": vehicle, "tt_action_item_id": "target", "time_begin": now + pd.Timedelta(seconds=720),
             "geom": "POINT (37.5 55.5)"} for vehicle in ("a", "b", "c", "stopped")]
    reference, actual = ReferenceBuilder(rows, plan), FeatureBuilder(rows, plan)
    for vehicle in ("a", "b", "c", "stopped"):
        for age in (0, 59, 300, 950):
            point = {"tr_id": vehicle, "T": now + pd.Timedelta(seconds=age), "target_stop_id": "target",
                     "target_time_begin": now + pd.Timedelta(seconds=age + 720), "cur_dev_s": None}
            assert_exact(reference.build(point), actual.build(point))


@pytest.mark.parametrize("dtype", ["float32", "float64", "Float64"])
def test_dataframe_numeric_precision_and_nullable_coordinates_match_reference(dtype):
    events = pd.DataFrame({"tr_id": ["v", "v"], "event_time": ["2026-01-06T12:00:00Z", "2026-01-06T12:00:01Z"],
        "lon": [37.61, 37.62], "lat": [55.75, 55.751], "speed": [.000001, 200.], "heading": [12.34, 23.45],
        "location_valid": [True, True]})
    for column in ("lon", "lat", "speed", "heading"):
        events[column] = events[column].astype(dtype)
    if dtype == "Float64":
        events.loc[0, "lon"] = pd.NA
    plan = [{"tr_id": "v", "tt_action_item_id": "target", "time_begin": "2026-01-06T12:12:00Z", "geom": "POINT (37.6 55.7)"}]
    point = {"tr_id": "v", "T": "2026-01-06T12:00:01Z", "target_stop_id": "target", "target_time_begin": "2026-01-06T12:12:00Z"}
    assert_exact(ReferenceBuilder(events, plan).build(point), FeatureBuilder(events, plan).build(point))


@pytest.mark.parametrize("split,points_file", [("train", "labels/labels_train.csv"), ("test", "labels/labels_test.csv")])
def test_all_supplied_real_points_exactly_match_frozen_training_extractor(split, points_file):
    dataset = Path(__file__).resolve().parents[1] / "dataset"
    events, plan = load_traffic(dataset / split / "traffic.csv"), load_plan(dataset / split / "schedule.csv")
    points = load_points(dataset / points_file)
    reference, actual = ReferenceBuilder(events, plan), FeatureBuilder(events, plan)
    for point in points.to_dict("records"):
        assert_exact(reference.build(point), actual.build(point))
