"""Reproducible temporal model selection and a single untouched-test evaluation.

Usage: python -m ml.train --data-dir dataset --output-dir artifacts
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time
from typing import Any

from catboost import CatBoostClassifier, CatBoostRegressor
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, mean_absolute_error

from ml.model import DelayModel
from ml.reporting import calibration_figure
from predictor.data import load_plan, load_points, load_traffic
from predictor.features import FEATURE_NAMES, FEATURE_SCHEMA_VERSION, FeatureBuilder

SEED = 20260926
REGRESSION_SETTINGS = {"iterations": 400, "depth": 5, "learning_rate": 0.045, "loss_function": "MAE",
                       "random_seed": SEED, "thread_count": 2, "verbose": False, "allow_writing_files": False}
CLASSIFIER_SETTINGS = {**REGRESSION_SETTINGS, "iterations": 300, "depth": 4, "loss_function": "Logloss"}


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def real_mask(points: pd.DataFrame) -> np.ndarray:
    """Dataset-specific synthetic range, kept explicit rather than a model feature."""
    ids = pd.to_numeric(points["tr_id"], errors="coerce")
    return (~ids.between(9_000_000, 9_999_999)).to_numpy()


def temporal_folds(points: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray, dict[str, Any]]]:
    real = real_mask(points)
    times = points.loc[real, "T"]
    boundaries = [times.quantile(fraction) for fraction in (0.45, 0.63, 0.81)]
    folds = []
    for index, start in enumerate(boundaries):
        end = boundaries[index + 1] if index + 1 < len(boundaries) else times.max() + pd.Timedelta(seconds=1)
        validation = real & (points["T"] >= start).to_numpy() & (points["T"] < end).to_numpy()
        forbidden = set(points.loc[validation, "target_stop_id"])
        cutoff = start - pd.Timedelta(minutes=30)
        training = (points["T"] < cutoff).to_numpy() & ~points["target_stop_id"].isin(forbidden).to_numpy()
        # Training outcomes must also have become observable before the purge boundary.
        arrivals = points["target_time_begin"] + pd.to_timedelta(points["target_delay_s"], unit="s")
        training &= (arrivals < cutoff).to_numpy()
        if (training & real).sum() < 30 or validation.sum() < 10:
            raise ValueError("Not enough data for the fixed temporal validation protocol")
        folds.append((training, validation, {"validation_start": start.isoformat(), "validation_end": end.isoformat(),
                     "training_cutoff": cutoff.isoformat(), "real_training_rows": int((training & real).sum()),
                     "augmented_training_rows": int(training.sum()), "validation_rows": int(validation.sum()),
                     "target_visits_disjoint": True, "purge_s": 1800}))
    return folds


def _group_metrics(points: pd.DataFrame, features: pd.DataFrame, predictions: np.ndarray) -> dict[str, Any]:
    y = points["target_delay_s"].to_numpy(float)
    keys = {
        "vehicle": points["tr_id"].astype(str),
        "class": points["target_class"].astype(str),
        "cur_dev": features["cur_dev_missing"].map({0.0: "present", 1.0: "missing"}),
        "telemetry": pd.Series(np.where(features["point_count_900s"] == 0, "missing",
                                        np.where(features["invalid_location_fraction_900s"].fillna(1) > 0.25, "poor", "good"))),
    }
    output = {}
    for dimension, values in keys.items():
        output[dimension] = {}
        for key in values.unique():
            mask = (values == key).to_numpy()
            output[dimension][str(key)] = {"rows": int(mask.sum()), "mae_s": float(mean_absolute_error(y[mask], predictions[mask]))}
    return output


def _calibration_bins(y: np.ndarray, p: np.ndarray) -> list[dict[str, float | int]]:
    output = []
    for lower in np.arange(0, 1, 0.1):
        selected = (p >= lower) & (p < lower + 0.1 if lower < 0.89 else p <= 1)
        if selected.any():
            output.append({"lower": round(float(lower), 1), "upper": round(float(lower + 0.1), 1),
                           "rows": int(selected.sum()), "mean_probability": float(p[selected].mean()),
                           "observed_late_rate": float(y[selected].mean())})
    return output


def train(data_dir: Path, output_dir: Path) -> dict[str, Any]:
    started = time.perf_counter()
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    train_points = load_points(data_dir / "labels/labels_train.csv", labels=True)
    real = real_mask(train_points)
    print(f"Build causal features: {len(train_points)} train points ({int(real.sum())} real)", flush=True)
    train_traffic = load_traffic(data_dir / "train/traffic.csv")
    train_plan = load_plan(data_dir / "train/schedule.csv")
    x = FeatureBuilder(train_traffic, train_plan).build_many(train_points)
    y = train_points["target_delay_s"].to_numpy(float)
    current = x["cur_dev_s"].fillna(0).to_numpy()
    folds = temporal_folds(train_points)
    experiments = []
    oof_raw = np.full(len(train_points), np.nan)
    oof_mask = np.zeros(len(train_points), dtype=bool)
    candidates = [("real", "zero"), ("real", "cur_dev"), ("real", "direct"), ("real", "residual"),
                  ("augmented", "direct"), ("augmented", "residual")]
    for cohort, mode in candidates:
        errors = []
        fold_scores = []
        for training, validation, _ in folds:
            fit_mask = training & real if cohort == "real" else training
            if mode == "zero":
                prediction = np.zeros(validation.sum())
            elif mode == "cur_dev":
                prediction = current[validation]
            else:
                estimator = CatBoostRegressor(**REGRESSION_SETTINGS)
                target = y - current if mode == "residual" else y
                estimator.fit(x.loc[fit_mask], target[fit_mask])
                prediction = estimator.predict(x.loc[validation], thread_count=2)
                if mode == "residual":
                    prediction += current[validation]
            errors.extend(np.abs(prediction - y[validation]).tolist())
            fold_scores.append(float(mean_absolute_error(y[validation], prediction)))
        eligible = cohort == "real"
        experiment = {"cohort": cohort, "mode": mode, "oof_mae_s": float(np.mean(errors)),
                      "fold_mae_s": fold_scores, "eligible_for_selection": eligible}
        if not eligible:
            experiment["limitation"] = "Synthetic lineage is not supplied; augmentation is diagnostic only and is excluded from production selection."
        experiments.append(experiment)
        print(f"{cohort}/{mode}: temporal MAE {experiment['oof_mae_s']:.3f}s; eligible={eligible}", flush=True)
    selected = min((item for item in experiments if item["eligible_for_selection"]), key=lambda item: item["oof_mae_s"])
    mode = selected["mode"]
    if mode in {"direct", "residual"}:
        regressor = CatBoostRegressor(**REGRESSION_SETTINGS)
        regressor.fit(x.loc[real], (y - current if mode == "residual" else y)[real])
        regressor.save_model(str(model_dir / "regressor.cbm"))
    for training, validation, _ in folds:
        classifier = CatBoostClassifier(**CLASSIFIER_SETTINGS)
        classifier.fit(x.loc[training & real], (y[training & real] > 120).astype(int))
        oof_raw[validation] = classifier.predict(x.loc[validation], prediction_type="RawFormulaVal", thread_count=2)
        oof_mask |= validation
    calibrator = LogisticRegression(C=1.0, random_state=SEED)
    calibrator.fit(oof_raw[oof_mask].reshape(-1, 1), (y[oof_mask] > 120).astype(int))
    classifier = CatBoostClassifier(**CLASSIFIER_SETTINGS)
    classifier.fit(x.loc[real], (y[real] > 120).astype(int))
    classifier.save_model(str(model_dir / "classifier.cbm"))
    source_paths = ["train/traffic.csv", "train/schedule.csv", "labels/labels_train.csv"]
    source_hashes = {name: sha256(data_dir / name) for name in source_paths}
    fingerprint = hashlib.sha256(json.dumps({"hashes": source_hashes, "settings": REGRESSION_SETTINGS,
                    "schema": FEATURE_SCHEMA_VERSION, "mode": mode}, sort_keys=True).encode()).hexdigest()[:12]
    metadata = {"model_version": f"delay-{fingerprint}", "feature_schema_version": FEATURE_SCHEMA_VERSION,
                "feature_names": FEATURE_NAMES, "prediction_mode": mode, "training_rows": int(real.sum()),
                "training_cohort": "real-only", "created_at": datetime.now(timezone.utc).isoformat(),
                "regression_settings": REGRESSION_SETTINGS, "classifier_settings": CLASSIFIER_SETTINGS,
                "probability_event": "target_delay_s > 120", "probability_calibration": {
                    "method": "sigmoid on out-of-fold chronological predictions", "slope": float(calibrator.coef_[0, 0]),
                    "intercept": float(calibrator.intercept_[0]), "calibration_rows": int(oof_mask.sum())},
                "source_sha256": source_hashes, "dependencies": {name: importlib.metadata.version(name) for name in ("catboost", "numpy", "pandas", "scikit-learn")},
                "python_version": platform.python_version(), "time_zone": "UTC", "seed": SEED,
                "feature_policy": {"history_s": 900, "event_cutoff": "event_time <= T", "future_interpolation": False,
                                   "plan_columns": ["tt_action_item_id", "tr_id", "time_begin", "geom", "building_address"],
                                   "synthetic_id_range": [9_000_000, 9_999_999]}}
    (model_dir / "manifest.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    # The test labels are first opened here, after model selection and artifacts are fixed.
    print("Selected model fixed; evaluate supplied test once", flush=True)
    test_points = load_points(data_dir / "labels/labels_test.csv", labels=True)
    if set(test_points["target_stop_id"]) & set(train_points.loc[real, "target_stop_id"]):
        raise ValueError("Train/test target visits overlap: independent visit evaluation is invalid")
    test_traffic = load_traffic(data_dir / "test/traffic.csv")
    test_plan = load_plan(data_dir / "test/schedule.csv")
    test_x = FeatureBuilder(test_traffic, test_plan).build_many(test_points)
    fitted = DelayModel(model_dir)
    infer_started = time.perf_counter()
    results = fitted.predict(test_x)
    inference_s = time.perf_counter() - infer_started
    prediction = np.array([result["prediction_delay_s"] for result in results])
    p_late = np.array([result["p_late"] for result in results])
    test_y = test_points["target_delay_s"].to_numpy(float)
    zero_mae = float(mean_absolute_error(test_y, np.zeros(len(test_y))))
    cur_mae = float(mean_absolute_error(test_y, test_x["cur_dev_s"].fillna(0)))
    model_mae = float(mean_absolute_error(test_y, prediction))
    report = {"model_version": metadata["model_version"], "created_at": metadata["created_at"],
              "selection": {"selected": selected, "experiments": experiments, "folds": [fold[2] for fold in folds],
                            "test_used_for_selection": False, "synthetic_lineage_verified": False},
              "test": {"rows": len(test_points), "mae_s": model_mae, "zero_baseline_mae_s": zero_mae,
                       "cur_dev_baseline_mae_s": cur_mae, "improves_cur_dev_baseline": bool(model_mae < cur_mae),
                       "brier_score": float(brier_score_loss(test_y > 120, p_late)),
                       "prevalence_baseline_brier": float(brier_score_loss(test_y > 120, np.full(len(test_y), (y[real] > 120).mean()))),
                       "calibration_bins": _calibration_bins(test_y > 120, p_late),
                       "groups": _group_metrics(test_points, test_x, prediction),
                       "batch_inference_s": inference_s},
              "data_audit": {"train_traffic_rows": len(train_traffic), "test_traffic_rows": len(test_traffic),
                             "train_points": len(train_points), "real_train_points": int(real.sum()),
                             "test_validate_traffic_identical": sha256(data_dir / "test/traffic.csv") == sha256(data_dir / "validate/traffic.csv"),
                             "train_test_target_overlap": 0, "validate_schedule_facts_read": False},
              "limitations": ["Test and validate contain identical telemetry and are not independent periods.",
                              "Real train/test labels cover the same calendar period; the test measures held-out visits, not unseen-day generalization.",
                              "Synthetic lineage is unknown; augmented experiments are diagnostic and synthetic data is excluded from the released model.",
                              "Online cur_dev_s is estimated; official metrics use provided cur_dev_s and do not establish online accuracy.",
                              "Probability calibration is based on temporal out-of-fold training predictions; distribution shift can impair calibration.",
                              "MAE_TARGET was not provided, so an official score is not calculated."],
              "elapsed_s": time.perf_counter() - started}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    calibration_figure(report, output_dir / "calibration.png")
    pd.DataFrame({"sample_id": test_points["sample_id"], "target_delay_s": test_y,
                  "prediction_delay_s": prediction, "p_late": p_late}).to_csv(output_dir / "test_predictions.csv", index=False)
    print(f"Test MAE={model_mae:.3f}s; zero={zero_mae:.3f}s; cur_dev={cur_mae:.3f}s; Brier={report['test']['brier_score']:.4f}", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    train(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
