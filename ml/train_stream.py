"""Train a separate dispatcher model with causal receive-time observer hints.

The official artifacts and submission are never written by this command.
Usage: python -m ml.train_stream --data-dir dataset --output-dir artifacts
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

from catboost import CatBoostClassifier, CatBoostRegressor
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, mean_absolute_error

from ml.model import DelayModel
from ml.reporting import calibration_figure
from ml.train import (CLASSIFIER_SETTINGS, REGRESSION_SETTINGS, SEED, _calibration_bins,
                      _group_metrics, real_mask, sha256, temporal_folds)
from predictor.data import load_plan, load_points, load_traffic
from predictor.features import FEATURE_NAMES
from predictor.observer import OBSERVER_VERSION
from predictor.stream_features import STREAM_FEATURE_NAMES, STREAM_SCHEMA_VERSION, build_stream_dataset


def _cohort_features(data_dir: Path, split: str, points: pd.DataFrame):
    traffic = load_traffic(data_dir / split / "traffic.csv")
    traffic = traffic.loc[traffic["tr_id"].isin(set(points["tr_id"]))]
    plan = load_plan(data_dir / split / "schedule.csv")
    plan = plan.loc[plan["tr_id"].isin(set(points["tr_id"]))]
    return build_stream_dataset(traffic, plan, points)


def train(data_dir: Path, output_dir: Path) -> dict:
    started = time.perf_counter()
    source_paths = ["train/traffic.csv", "train/schedule.csv", "labels/labels_train.csv"]
    points = load_points(data_dir / "labels/labels_train.csv", labels=True)
    points = points.loc[real_mask(points)].reset_index(drop=True)
    print(f"Building streaming train features for {len(points)} real points", flush=True)
    x, diagnostics = _cohort_features(data_dir, "train", points)
    y = points["target_delay_s"].to_numpy(float)
    current = x["cur_dev_s"].fillna(0).to_numpy()
    folds = temporal_folds(points)
    experiments = []
    for mode in ("zero", "cur_dev", "direct", "residual"):
        errors, fold_scores = [], []
        for fitting, validation, _ in folds:
            if mode == "zero":
                predicted = np.zeros(validation.sum())
            elif mode == "cur_dev":
                predicted = current[validation]
            else:
                model = CatBoostRegressor(**REGRESSION_SETTINGS)
                model.fit(x.loc[fitting], (y - current if mode == "residual" else y)[fitting])
                predicted = model.predict(x.loc[validation], thread_count=2)
                if mode == "residual":
                    predicted += current[validation]
            errors.extend(np.abs(predicted - y[validation]))
            fold_scores.append(float(mean_absolute_error(y[validation], predicted)))
        row = {"mode": mode, "oof_mae_s": float(np.mean(errors)), "fold_mae_s": fold_scores}
        experiments.append(row)
        print(f"stream/{mode}: chronological MAE {row['oof_mae_s']:.3f}s", flush=True)
    selected = min((row for row in experiments if row["mode"] in {"direct", "residual"}), key=lambda row: row["oof_mae_s"])
    mode = selected["mode"]
    baseline_oof = min(row["oof_mae_s"] for row in experiments if row["mode"] in {"zero", "cur_dev"})
    quality = "validated_on_train_folds" if selected["oof_mae_s"] < baseline_oof else "experimental"
    model_dir = output_dir / "models" / "stream"
    model_dir.mkdir(parents=True, exist_ok=True)
    regressor = CatBoostRegressor(**REGRESSION_SETTINGS)
    regressor.fit(x, y - current if mode == "residual" else y)
    regressor.save_model(str(model_dir / "regressor.cbm"))
    raw_oof, mask = np.full(len(points), np.nan), np.zeros(len(points), dtype=bool)
    for fitting, validation, _ in folds:
        classifier = CatBoostClassifier(**CLASSIFIER_SETTINGS)
        classifier.fit(x.loc[fitting], (y[fitting] > 120).astype(int))
        raw_oof[validation] = classifier.predict(x.loc[validation], prediction_type="RawFormulaVal", thread_count=2)
        mask |= validation
    calibrator = LogisticRegression(C=1.0, random_state=SEED)
    calibrator.fit(raw_oof[mask].reshape(-1, 1), (y[mask] > 120).astype(int))
    classifier = CatBoostClassifier(**CLASSIFIER_SETTINGS)
    classifier.fit(x, (y > 120).astype(int))
    classifier.save_model(str(model_dir / "classifier.cbm"))
    hashes = {name: sha256(data_dir / name) for name in source_paths}
    fingerprint = hashlib.sha256(json.dumps({"hashes": hashes, "schema": STREAM_SCHEMA_VERSION,
        "observer": OBSERVER_VERSION, "mode": mode, "settings": REGRESSION_SETTINGS}, sort_keys=True).encode()).hexdigest()[:12]
    metadata = {"profile": "stream", "model_version": f"stream-{fingerprint}", "feature_schema_version": STREAM_SCHEMA_VERSION,
        "feature_names": STREAM_FEATURE_NAMES, "observer_version": OBSERVER_VERSION, "prediction_mode": mode,
        "training_rows": len(points), "training_cohort": "real-only", "quality_status": quality,
        "created_at": datetime.now(timezone.utc).isoformat(), "source_sha256": hashes,
        "regression_settings": REGRESSION_SETTINGS, "classifier_settings": CLASSIFIER_SETTINGS,
        "probability_event": "target_delay_s > 120", "probability_calibration": {
            "method": "sigmoid on chronological out-of-fold predictions; train only",
            "slope": float(calibrator.coef_[0, 0]), "intercept": float(calibrator.intercept_[0]),
            "calibration_rows": int(mask.sum())},
        "feature_policy": {"event_cutoff": "event_time <= T AND receive_time <= T", "history_s": 900,
            "cur_dev_source": "shared causal observer only; supplied cur_dev_s excluded",
            "segment_speed": "duration weighted; >=30s, >=80% coverage, max gap 60s", "future_interpolation": False},
        "dependencies": {name: importlib.metadata.version(name) for name in ("catboost", "numpy", "pandas", "scikit-learn")},
        "time_zone": "UTC", "seed": SEED}
    (model_dir / "manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # Test labels are opened only once the training decision and artifacts are fixed.
    print("Streaming candidate frozen; evaluating supplied test", flush=True)
    test_points = load_points(data_dir / "labels/labels_test.csv", labels=True)
    if set(points["target_stop_id"]) & set(test_points["target_stop_id"]):
        raise ValueError("Train/test visits overlap")
    test_x, test_diagnostics = _cohort_features(data_dir, "test", test_points)
    result = DelayModel(model_dir).predict(test_x)
    prediction = np.array([row["prediction_delay_s"] for row in result])
    probabilities = np.array([row["p_late"] for row in result])
    target = test_points["target_delay_s"].to_numpy(float)
    estimated = test_x["cur_dev_s"].fillna(0).to_numpy()
    fresh = np.array([row["fresh"] for row in test_diagnostics])
    unobserved = ~np.array([row["target_already_observed"] for row in test_diagnostics])
    official = DelayModel(output_dir / "models")
    legacy_result = official.predict(test_x.loc[:, FEATURE_NAMES])
    legacy_prediction = np.array([row["prediction_delay_s"] for row in legacy_result])

    def metrics(subset: np.ndarray) -> dict:
        if not subset.any():
            return {"rows": 0, "mae_s": None, "brier_score": None}
        return {"rows": int(subset.sum()), "mae_s": float(mean_absolute_error(target[subset], prediction[subset])),
            "zero_baseline_mae_s": float(np.abs(target[subset]).mean()),
            "estimated_cur_dev_baseline_mae_s": float(mean_absolute_error(target[subset], estimated[subset])),
            "legacy_official_model_estimated_hint_mae_s": float(mean_absolute_error(target[subset], legacy_prediction[subset])),
            "brier_score": float(brier_score_loss(target[subset] > 120, probabilities[subset]))}

    all_metrics = metrics(np.ones(len(target), dtype=bool))
    all_metrics.update(calibration_bins=_calibration_bins(target > 120, probabilities),
        groups=_group_metrics(test_points, test_x, prediction),
        prevalence_baseline_brier=float(brier_score_loss(target > 120, np.full(len(target), (y > 120).mean()))))
    report = {"profile": "stream", "model_version": metadata["model_version"], "quality_status": quality,
        "created_at": metadata["created_at"], "selection": {"selected": selected, "experiments": experiments,
            "folds": [row[2] for row in folds], "test_used_for_selection": False,
            "synthetic_training_used": False, "provided_cur_dev_used": False},
        "test": all_metrics, "dispatcher_eligible_points": metrics(fresh & unobserved),
        "fresh_points": metrics(fresh), "coverage": {"train_points": len(points), "test_points": len(test_points),
            "train_estimated_hints": sum(row["hint_source"] == "estimated" for row in diagnostics),
            "test_estimated_hints": sum(row["hint_source"] == "estimated" for row in test_diagnostics),
            "test_observed_segment_speeds": int(test_x["segment_speed_kmh"].notna().sum()),
            "test_already_observed_targets": int((~unobserved).sum()), "test_fresh_points": int(fresh.sum())},
        "acceptance": {"beats_zero_baseline": all_metrics["mae_s"] < all_metrics["zero_baseline_mae_s"],
            "beats_estimated_hint_baseline": all_metrics["mae_s"] < all_metrics["estimated_cur_dev_baseline_mae_s"],
            "beats_legacy_stream_behavior": all_metrics["mae_s"] < all_metrics["legacy_official_model_estimated_hint_mae_s"]},
        "limitations": ["Same-period held-out visits; unseen-day quality is not established.",
            "Test/validate telemetry overlap. Validate outcomes are never read.",
            "Test is reporting-only; model, thresholds and calibrator were selected on chronological train folds.",
            "Missing estimated hints use zero only in the explicitly named hint baseline; model gets missing indicators.",
            "Point-level diagnostics include stale/already-observed targets; dispatcher eligibility is reported separately.",
            "Stop matching is heuristic and model factors are not proven causes."],
        "elapsed_s": time.perf_counter() - started}
    (output_dir / "stream_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame({"sample_id": test_points["sample_id"], "target_delay_s": target,
        "prediction_delay_s": prediction, "p_late": probabilities,
        "estimated_cur_dev_s": test_x["cur_dev_s"], "fresh": fresh,
        "target_already_observed": ~unobserved}).to_csv(output_dir / "stream_test_predictions.csv", index=False)
    calibration_figure(report, output_dir / "stream_calibration.png")
    print(json.dumps({"test": {key: value for key, value in all_metrics.items() if key not in {"groups", "calibration_bins"}},
                      "acceptance": report["acceptance"], "coverage": report["coverage"]}, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    train(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
