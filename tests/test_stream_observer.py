from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from predictor.features import FeatureBuilder
from predictor.observer import CausalObserver
from predictor.stream_features import build_stream_dataset, enrich_stream_features


ORIGIN = datetime(2026, 1, 6, 7, tzinfo=timezone.utc)


def at(seconds):
    return (ORIGIN + timedelta(seconds=seconds)).isoformat()


def plan():
    return [
        {"tt_action_item_id": "a", "tr_id": "v", "time_begin": at(-120), "geom": "POINT (37.6 55.7)", "building_address": "A"},
        {"tt_action_item_id": "b", "tr_id": "v", "time_begin": at(780), "geom": "POINT (37.61 55.71)", "building_address": "B"},
    ]


def event(seconds, *, speed=30, lat=55.7, lon=37.6, received=None, valid=True):
    return {"packet_id": str(seconds), "tr_id": "v", "event_time": at(seconds),
            "receive_time": at(seconds if received is None else received), "location_valid": valid,
            "speed": speed, "lat": lat, "lon": lon, "heading": 0}


def arrive(observer):
    assert observer.observe(event(-60), at(-60)) is None
    result = observer.observe(event(-45), at(-45))
    assert result["target_visit_id"] == "a"
    assert observer.hints("v", at(0))["value"] == 60


def test_future_unreceived_late_and_missing_gps_cannot_move_observer():
    observer = CausalObserver(plan())
    assert observer.observe(event(-60, received=10), at(0)) is None
    assert observer.observe(event(10), at(0)) is None
    assert not observer.candidates
    arrive(observer)
    before = observer.dump_state()
    assert observer.observe(event(-50, lat=55.8), at(0)) is None
    assert observer.dump_state() == before
    observer.observe(event(0, valid=False), at(0))
    assert observer.visits["v"]["departed"] is False
    assert observer.has_visited("v", "a")
    assert not observer.has_visited("v", "b")
    assert observer.hints("v", at(901))["source"] == "missing"


def test_segment_uses_duration_weights_and_requires_coverage():
    observer = CausalObserver(plan(), [{"id": "route-a-b", "from_visit_id": "a", "to_visit_id": "b"}])
    arrive(observer)
    observer.observe(event(0, lat=55.704, speed=30))
    observer.observe(event(10, lat=55.7045, speed=60))
    assert observer.segment("v", at(10))["segment_speed_kmh"] is None
    observer.observe(event(40, lat=55.705, speed=40))
    segment = observer.segment("v", at(40))
    assert segment["segment_id"] == "route-a-b"
    assert segment["segment_speed_kmh"] == pytest.approx(52.5)
    assert segment["speed_coverage"] == 1
    assert observer.segment("v", at(105))["segment_speed_kmh"] is None
    observer.observe(event(120, lat=55.706, speed=10))
    assert observer.segment("v", at(120))["segment_speed_kmh"] is None


def test_observer_restart_is_equivalent_and_repeated_stop_requires_exit():
    observer = CausalObserver(plan())
    arrive(observer)
    for second in (0, 10, 20):
        observer.observe(event(second))
    assert observer.confirmed_count == 1
    assert observer.segment("v", at(20))["segment_id"] is None
    observer.observe(event(30, lat=55.704))
    recovered = CausalObserver(plan())
    recovered.restore_state(observer.dump_state())
    for second in (45, 60):
        observer.observe(event(second, lat=55.705))
        recovered.observe(event(second, lat=55.705))
    assert recovered.dump_state() == observer.dump_state()
    assert recovered.segment("v", at(60)) == observer.segment("v", at(60))


def test_stream_extractor_ignores_supplied_hint_labels_facts_and_future():
    rows = [event(-60), event(-45), event(0, lat=55.704), event(10, lat=55.705), event(40, lat=55.706)]
    point = {"sample_id": "p", "tr_id": "v", "T": at(40), "target_stop_id": "b", "target_time_begin": at(780),
             "cur_dev_s": 999999, "target_delay_s": 999999, "target_class": "late"}
    x, diagnostics = build_stream_dataset(pd.DataFrame(rows), pd.DataFrame(plan()), pd.DataFrame([point]))
    changed = {**point, "cur_dev_s": -999999, "target_delay_s": -999999}
    poisoned_plan = [{**row, "time_fact_begin": "2099-01-01"} for row in plan()]
    extra = [event(41, speed=190), event(39, speed=190, received=1000)]
    changed_x, _ = build_stream_dataset(pd.DataFrame(rows + extra), pd.DataFrame(poisoned_plan), pd.DataFrame([changed]))
    pd.testing.assert_frame_equal(x, changed_x)
    assert x.iloc[0]["cur_dev_s"] == 60
    assert diagnostics[0]["hint_source"] == "estimated"
    # The exact same observer and feature calls are available to the Backend.
    observer = CausalObserver(plan())
    for row in rows:
        observer.observe(row, row["receive_time"])
    hint = observer.hints("v", point["T"])
    online_point = {**point, "cur_dev_s": hint["value"], "cur_dev_source": hint["source"], "cur_dev_observed_at": hint["observed_at"]}
    online = enrich_stream_features(FeatureBuilder(rows, plan()).build(online_point), observer.segment("v", point["T"]))
    np.testing.assert_allclose(x.iloc[0].to_numpy(), np.array(list(online.values())), equal_nan=True)


def test_ambiguous_colocated_visits_are_not_confirmed():
    rows = plan()
    rows.insert(1, {**rows[0], "tt_action_item_id": "ambiguous", "time_begin": at(-100)})
    observer = CausalObserver(rows)
    observer.observe(event(-60))
    observer.observe(event(-45))
    assert not observer.visits


def test_previous_hint_cannot_shift_match_beyond_absolute_plan_tolerance():
    rows = plan()
    rows[1]["time_begin"] = at(1500)
    observer = CausalObserver(rows)
    observer.deviations["v"] = {"value": -800, "source": "estimated", "observed_at": at(-10)}
    observer.observe(event(0, lat=55.71, lon=37.61))
    observer.observe(event(15, lat=55.71, lon=37.61))
    assert not observer.visits
