import copy
import math
import asyncio

import pytest

from scripts.check_warm_history import HISTORY_POINTS, WARM_RATE_HZ, WARM_SECONDS, history_checks, warm_event_offset
from scripts import check_warm_history as warm


def rows():
    return [{"id": f"{vehicle}-{cycle}", "tr_id": f"load-{vehicle:03}", "run_id": "warm-run",
             "history_points": 900, "horizon_s": 750} for vehicle in range(2) for cycle in range(3)]


def test_catchup_events_are_always_causal_and_fill_history_at_live_boundary():
    assert HISTORY_POINTS == 900 and WARM_SECONDS == 180
    assert all(warm_event_offset(index) <= index / WARM_RATE_HZ for index in range(HISTORY_POINTS))
    assert warm_event_offset(0) == -720
    assert warm_event_offset(899) == 179
    # NDTP timestamps use integer seconds. At the first live tick, 900
    # distinct source seconds remain in the actual 900-second feature window.
    warm_start = 10_000.123
    cutoff = warm_start + WARM_SECONDS
    times = [math.floor(warm_start + warm_event_offset(index)) for index in range(HISTORY_POINTS)]
    times.append(math.floor(cutoff))
    assert sum(cutoff - 900 < value <= cutoff for value in times) == 900


def test_multiple_full_history_predictions_for_every_vehicle_pass():
    checks, evidence = history_checks(rows(), "warm-run", 2)
    assert all(checks.values())
    assert evidence["load-001"] == {"model_renders": 3, "min_history_points": 900, "max_history_points": 900}


def test_frozen_single_vehicle_cannot_hide_behind_aggregate_model_count():
    values = [row for row in rows() if row["tr_id"] == "load-000"] * 100
    assert not history_checks(values, "warm-run", 2)[0]["each_vehicle_has_three_postwarm_model_renders"]


def test_repeated_render_of_same_prediction_does_not_count_as_multiple_updates():
    values = [row for row in rows() if row["id"].endswith("-0")] * 100
    assert not history_checks(values, "warm-run", 2)[0]["each_vehicle_has_three_postwarm_model_renders"]


@pytest.mark.asyncio
async def test_live_pacing_uses_actual_wall_time_after_backward_clock_adjustment(monkeypatch):
    clock = {"wall": 1_000.0, "monotonic": 100.0}
    monkeypatch.setattr(warm.time, "time", lambda: clock["wall"])
    monkeypatch.setattr(warm.time, "monotonic", lambda: clock["monotonic"])
    async def pace(deadline):
        clock.update(wall=1_000.999, monotonic=deadline)
    monkeypatch.setattr(warm, "wait_until", pace)
    class Sender:
        async def tick_at(self, source_elapsed_s, event_at):
            assert event_at <= clock["wall"]
            assert event_at == 1_000.999  # Scheduled wall start+1 would be future.
            assert source_elapsed_s == event_at - 900
    assert await warm.live_tick(Sender(), 900, 101) == 0


@pytest.mark.asyncio
async def test_failed_sender_does_not_abort_cleanup():
    async def fail():
        raise ValueError("sender failed")
    task = asyncio.create_task(fail())
    await asyncio.sleep(0)
    await warm.cancel_sender_task(task)
    assert task.done()


@pytest.mark.parametrize("field,value,key", [
    ("history_points", 200, "all_postwarm_models_have_900s_history"),
    ("history_points", None, "all_postwarm_models_have_900s_history"),
    ("horizon_s", 600, "all_publication_horizons_valid"),
    ("horizon_s", 901, "all_publication_horizons_valid"),
    ("run_id", "previous", "all_model_run_ids_match"),
])
def test_missing_history_or_wrong_forecasts_fail(field, value, key):
    values = copy.deepcopy(rows())
    values[0][field] = value
    assert not history_checks(values, "warm-run", 2)[0][key]
