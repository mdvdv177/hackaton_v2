"""Score a frozen profile with the shared receive-time causal observer.

Defaults to the stream model. Pass --model-dir artifacts/models for the legacy
model. This command never retrains or writes the official report/submission.
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
from sklearn.metrics import brier_score_loss, mean_absolute_error

from ml.model import DelayModel
from predictor.data import load_plan, load_points, load_traffic
from predictor.observer import OBSERVER_VERSION
from predictor.stream_features import build_stream_dataset


async def evaluate(data_dir: Path, model_dir: Path, output: Path) -> dict:
    started = time.perf_counter()
    points = load_points(data_dir / "labels/labels_test.csv", labels=False)
    features, diagnostics = build_stream_dataset(load_traffic(data_dir / "test/traffic.csv"),
        load_plan(data_dir / "test/schedule.csv"), points)
    model = DelayModel(model_dir)
    results = model.predict(features.loc[:, model.feature_names])
    predictions = np.array([item["prediction_delay_s"] for item in results])
    probabilities = np.array([item["p_late"] for item in results])
    # Ground truth is opened only after feature extraction/inference.
    labels = load_points(data_dir / "labels/labels_test.csv", labels=True).set_index("sample_id")
    target = np.array([labels.loc[item["sample_id"], "target_delay_s"] for item in diagnostics])
    fresh = np.array([item["fresh"] for item in diagnostics])
    observed = np.array([item["target_already_observed"] for item in diagnostics])
    estimated = np.array([item["hint_source"] == "estimated" for item in diagnostics])
    hint = features["cur_dev_s"].fillna(0).to_numpy()

    def metrics(mask: np.ndarray) -> dict:
        if not mask.any():
            return {"rows": 0, "mae_s": None, "brier_score": None}
        return {"rows": int(mask.sum()), "mae_s": float(mean_absolute_error(target[mask], predictions[mask])),
                "brier_score": float(brier_score_loss(target[mask] > 120, probabilities[mask])),
                "zero_baseline_mae_s": float(np.abs(target[mask]).mean()),
                "estimated_cur_dev_baseline_mae_s": float(mean_absolute_error(target[mask], hint[mask]))}

    report = {"created_at": datetime.now(timezone.utc).isoformat(), "profile": model.profile,
        "model_version": model.model_version, "feature_schema_version": model.schema_version,
        "protocol": {"sample_points": "supplied test targets and T, not every dispatcher tick",
            "telemetry_availability": "receive_time <= T AND event_time <= T",
            "cur_dev_source": "shared observer, or missing; never supplied hints",
            "observer_version": OBSERVER_VERSION, "retrained_or_tuned": False},
        "all_points_diagnostic": metrics(np.ones(len(target), dtype=bool)),
        "dispatcher_eligible_points": metrics(fresh & ~observed),
        "dispatcher_fresh_points": metrics(fresh), "stale_or_missing_points": metrics(~fresh),
        "estimated_hint_points": metrics(estimated), "missing_hint_points": metrics(~estimated),
        "coverage": {"test_points": len(target), "fresh_points": int(fresh.sum()),
            "fresh_fraction": float(fresh.mean()), "estimated_hint_points": int(estimated.sum()),
            "estimated_hint_fraction": float(estimated.mean()), "already_observed_targets": int(observed.sum()),
            "eligible_points": int((fresh & ~observed).sum()),
            "observed_segment_speed_points": int(features["segment_speed_kmh"].notna().sum())},
        "elapsed_s": time.perf_counter() - started,
        "limitations": ["All-point diagnostics include stale/already-observed examples; dispatcher suppresses these.",
            "This point-level protocol is not the continuous NDTP 30-second cadence test.",
            "The same-period test set does not establish unseen-day generalization.",
            "Stop matching remains heuristic; estimated hints may refer to different prior stops."]}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("profile", "all_points_diagnostic", "dispatcher_eligible_points", "coverage")}, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--model-dir", type=Path, default=Path("artifacts/models/stream"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/stream_evaluation.json"))
    args = parser.parse_args()
    asyncio.run(evaluate(args.data_dir, args.model_dir, args.output))


if __name__ == "__main__":
    main()
