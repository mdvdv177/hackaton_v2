"""The training reducer and actual Backend receive the same available events."""

import asyncio
import csv

import numpy as np

from backend.data import iso, timestamp
from backend.service import Dispatcher
from backend.storage import Store
from predictor.data import load_plan, load_points, load_traffic
from predictor.features import build_features
from predictor.stream_features import STREAM_FEATURE_NAMES, build_stream_dataset, enrich_stream_features
from test_backend import miniature_dataset


def test_real_backend_stream_features_equal_training_reducer(miniature_dataset):
    path = miniature_dataset / "test/traffic.csv"
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    # A received-future point must wait; a late point must not rewind matching.
    rows += [{**rows[0], "packet_id": "future", "event_time": "2026-01-06 07:00:10", "receive_time": "2026-01-06 06:59:00", "speed": "199"},
             {**rows[0], "packet_id": "late", "event_time": "2026-01-06 06:59:05", "receive_time": "2026-01-06 06:59:59", "speed": "10"}]
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    points = load_points(miniature_dataset / "labels/labels_test.csv")
    expected, diagnostics = build_stream_dataset(load_traffic(path), load_plan(miniature_dataset / "test/schedule.csv"), points)

    async def scenario():
        store = Store("sqlite:///:memory:")
        dispatcher = Dispatcher(store, miniature_dataset, "http://unused.invalid")
        try:
            dispatcher.load_data("test")
            dispatcher._sort_traffic("dispatcher")
            point = dispatcher.points[0]
            dispatcher.run = {"id": "parity", "mode": "dispatcher", "source": "test", "status": "paused", "speed": 1,
                              "created_at": point["T"], "virtual_time": point["T"]}
            cutoff = timestamp(point["T"])
            for event in dispatcher.traffic:
                available = max(timestamp(event["received_at"]), timestamp(event["event_time"]))
                if available > cutoff:
                    break
                dispatcher.run["virtual_time"] = iso(available)
                dispatcher._accept(event)
            dispatcher.run["virtual_time"] = iso(cutoff)
            hint = dispatcher._deviation(point["tr_id"], cutoff)
            query = {**point, "cur_dev_s": hint["value"], "cur_dev_source": hint["source"], "cur_dev_observed_at": hint["observed_at"]}
            base = build_features(list(dispatcher.events[point["tr_id"]]), dispatcher.plans[point["tr_id"]], query)
            actual = enrich_stream_features(base, dispatcher.observer.segment(point["tr_id"], cutoff))
            np.testing.assert_allclose(np.array([actual[name] for name in STREAM_FEATURE_NAMES]), expected.iloc[0].to_numpy(), equal_nan=True)
            assert hint["value"] == diagnostics[0]["estimated_cur_dev_s"]
        finally:
            dispatcher.run = None
            await dispatcher.close()
            store.engine.dispose()
    asyncio.run(scenario())
