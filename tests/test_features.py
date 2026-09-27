"""Feature contracts: chronology, quality, identifier isolation and online parity."""

import numpy as np
import pandas as pd
import pytest

from predictor.data import safe_plan
from predictor.features import FEATURE_NAMES, FeatureBuilder, build_features


@pytest.fixture
def point():
    return {"tr_id": "bus", "T": "2026-01-06T12:00:00Z", "target_stop_id": "stop",
            "target_time_begin": "2026-01-06T12:12:00Z", "cur_dev_s": -45}


@pytest.fixture
def plan():
    return [{"tr_id": "bus", "tt_action_item_id": "stop", "time_begin": "2026-01-06T12:12:00Z",
             "geom": "POINT (37.6 55.7)", "time_fact_begin": "2026-01-06T12:13:00Z"}]


@pytest.fixture
def events():
    return [{"tr_id": "bus", "event_time": f"2026-01-06T11:{minute}:00Z", "lat": 55.7,
             "lon": 37.61, "speed": speed, "heading": 90, "location_valid": True}
            for minute, speed in [(56, 20), (57, 10), (58, 0), (59, 0)]]


def assert_features_equal(left, right):
    np.testing.assert_allclose(list(left.values()), list(right.values()), equal_nan=True)
    assert list(left) == list(right) == FEATURE_NAMES


def test_future_telemetry_and_target_facts_cannot_change_features(point, plan, events):
    reference = build_features(events, plan, point)
    future = {**events[-1], "event_time": "2026-01-06T12:00:00.000001Z", "speed": 150, "lon": 180}
    changed_plan = [{**plan[0], "time_fact_begin": "2030-01-01", "target_delay_s": 99999}]
    changed_point = {**point, "sample_id": "secret", "target_delay_s": 99999, "target_class": "late"}
    actual = build_features(events + [future], changed_plan, changed_point)
    assert_features_equal(reference, actual)
    assert not {"time_fact_begin", "sample_id", "target_delay_s", "target_stop_id", "tr_id"} & set(actual)


def test_online_offline_features_match_at_cutoff(point, plan, events):
    # Offline sees the entire file; the online adapter supplies only arrived history.
    future = {**events[-1], "event_time": "2026-01-06T12:01:00Z", "speed": 150}
    offline = FeatureBuilder(pd.DataFrame(events + [future]), pd.DataFrame(plan)).build(point)
    online = build_features(events, plan, point)
    assert_features_equal(offline, online)


def test_missing_telemetry_and_hint_are_explicit(point, plan):
    actual = build_features([], plan, {**point, "cur_dev_s": None})
    assert actual["point_count_900s"] == 0
    assert actual["max_gap_s_900s"] == 900
    assert actual["cur_dev_missing"] == 1
    assert np.isnan(actual["cur_dev_s"])
    assert np.isnan(actual["distance_to_target_m"])


def test_invalid_coordinates_never_become_zero_position(point, plan, events):
    actual = build_features([{**event, "location_valid": False, "lat": 0, "lon": 0} for event in events], plan, point)
    assert np.isnan(actual["distance_to_target_m"])
    assert actual["invalid_location_fraction_300s"] == 1


def test_microsecond_timestamps_use_seconds_for_gaps(point, plan, events):
    features = build_features(events, plan, point)
    assert features["max_gap_s_300s"] == 60
    assert features["stopped_fraction_300s"] == pytest.approx(1 / 3)
    assert features["stop_duration_s"] == 60


@pytest.mark.parametrize("seconds,accepted", [(600, False), (601, True), (900, True), (901, False)])
def test_target_window_is_left_open_right_closed(point, plan, seconds, accepted):
    point["target_time_begin"] = pd.Timestamp(point["T"]) + pd.Timedelta(seconds=seconds)
    if accepted:
        assert build_features([], plan, point)["horizon_s"] == seconds
    else:
        with pytest.raises(ValueError, match="Target must"):
            build_features([], plan, point)


def test_midnight_window_retains_previous_day_events(plan):
    point = {"tr_id": "bus", "T": "2026-01-07T00:01:00Z", "target_stop_id": "stop",
             "target_time_begin": "2026-01-07T00:12:00Z", "cur_dev_s": 0}
    events = [{"tr_id": "bus", "event_time": "2026-01-06T23:59:00Z", "speed": 12,
               "lat": 55, "lon": 37, "location_valid": True}]
    assert build_features(events, plan, point)["point_count_300s"] == 1


def test_schedule_allowlist_discards_outcomes(plan):
    cleaned = safe_plan(pd.DataFrame(plan))
    assert "time_fact_begin" not in cleaned
    assert set(cleaned) == {"tt_action_item_id", "tr_id", "time_begin", "geom", "building_address"}


def test_identifiers_select_rows_without_becoming_features(point, plan, events):
    original = build_features(events, plan, point)
    renamed_events = [{**event, "tr_id": "different_bus"} for event in events]
    renamed_plan = [{**plan[0], "tr_id": "different_bus", "tt_action_item_id": "different_visit"}]
    renamed_point = {**point, "tr_id": "different_bus", "target_stop_id": "different_visit"}
    assert_features_equal(original, build_features(renamed_events, renamed_plan, renamed_point))
