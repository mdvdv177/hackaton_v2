"""Isolated, reproducible official-profile CatBoost tuning without test selection.

The first two existing chronological folds select a candidate. Its settings and
tree count are frozen before the third fold is scored once. Only a confirmed
candidate is trained on all real train rows and exported into the new run folder.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import itertools
import json
from pathlib import Path
import platform
import shutil
import time
from typing import Any, Callable

from catboost import CatBoostRegressor
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from ml.model import DelayModel
from ml.submit import generate as generate_submission
from ml.train import REGRESSION_SETTINGS, SEED, _group_metrics, real_mask, sha256, temporal_folds
from predictor.data import load_plan, load_points, load_traffic
from predictor.features import FEATURE_NAMES, FEATURE_SCHEMA_VERSION, FeatureBuilder

SEARCH_SPACE = {
    "depth": [3, 4, 5, 6, 7],
    "learning_rate": [0.02, 0.03, 0.045, 0.07, 0.1],
    "l2_leaf_reg": [1, 3, 5, 10, 20],
    "random_strength": [0.0, 0.5, 1.0, 2.0],
    "subsample": [0.6, 0.8, 1.0],
}
MIN_IMPROVEMENT_S = 1.0
TIE_TOLERANCE_S = 1e-9
BASELINE_TOLERANCE_S = 0.01
MAX_ITERATIONS = 2000
PATIENCE = 100


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def generate_configs(trials: int = 50, seed: int = SEED,
                     *, base_params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Include the untouched control, its early-stop variant, then unique grid draws."""
    maximum = int(np.prod([len(values) for values in SEARCH_SPACE.values()])) + 1
    if not 2 <= trials <= maximum:
        raise ValueError(f"trials must be between 2 and {maximum}, including the two controls")
    baseline = deepcopy(base_params or REGRESSION_SETTINGS)
    baseline.update(random_seed=seed, thread_count=2, loss_function="MAE", verbose=False, allow_writing_files=False)
    for key, value in {"bootstrap_type": "MVS", "subsample": 0.8, "l2_leaf_reg": 3, "random_strength": 1.0}.items():
        baseline.setdefault(key, value)
    if baseline.get("iterations") != 400:
        raise ValueError("The accepted control must retain its original 400 iterations")
    current = {**baseline, "bootstrap_type": "MVS", "subsample": 0.8,
               "l2_leaf_reg": 3, "random_strength": 1.0, "iterations": MAX_ITERATIONS}
    configs = [
        {"id": "control", "kind": "control", "params": baseline},
        {"id": "current_early_stop", "kind": "candidate", "params": current},
    ]
    keys = list(SEARCH_SPACE)
    current_tuple = tuple(current[key] for key in keys)
    choices = [values for values in itertools.product(*(SEARCH_SPACE[key] for key in keys)) if values != current_tuple]
    if trials - 2 > len(choices):
        raise ValueError("trials exceeds the number of unique configurations")
    indices = np.random.default_rng(seed).choice(len(choices), size=trials - 2, replace=False)
    for index, choice in enumerate(indices, start=1):
        params = {**current, **dict(zip(keys, choices[int(choice)], strict=True))}
        configs.append({"id": f"trial_{index:03d}", "kind": "candidate", "params": params})
    return configs


def make_inner_split(points: pd.DataFrame, outer_train_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Last 20% of outer training times stop trees; purge before the first stop time."""
    outer = np.asarray(outer_train_mask, dtype=bool) & real_mask(points)
    eligible = points.loc[outer]
    if len(eligible) < 50:
        raise ValueError("Need at least 50 real outer-training rows for an inner split")
    boundary = eligible["T"].quantile(0.8)
    stop = outer & (points["T"] >= boundary).to_numpy()
    if not stop.any():
        raise ValueError("Inner stopping block is empty")
    stop_start = points.loc[stop, "T"].min()
    cutoff = stop_start - pd.Timedelta(minutes=30)
    forbidden = set(points.loc[stop, "target_stop_id"])
    arrivals = points["target_time_begin"] + pd.to_timedelta(points["target_delay_s"], unit="s")
    fit = outer & (points["T"] < cutoff).to_numpy() & (arrivals < cutoff).to_numpy()
    fit &= ~points["target_stop_id"].isin(forbidden).to_numpy()
    if fit.sum() < 30 or stop.sum() < 10:
        raise ValueError("Inner split needs at least 30 fit and 10 stopping rows after purging")
    metadata = {"fit_rows": int(fit.sum()), "stop_rows": int(stop.sum()),
                "stop_start": stop_start.isoformat(), "fit_cutoff": cutoff.isoformat(), "purge_s": 1800,
                "target_visits_disjoint": True, "observed_outcomes_before_cutoff": True}
    return fit, stop, metadata


def fit_candidate_fold(config: dict[str, Any], seed: int, x: pd.DataFrame, points: pd.DataFrame,
                       outer_train: np.ndarray, outer_valid: np.ndarray, fold: int,
                       *, estimator_factory: Callable[..., Any] | None = None
                       ) -> tuple[dict[str, Any], pd.DataFrame]:
    """Fit an inner early-stop model, then a fresh full-outer model for scoring."""
    factory = estimator_factory or CatBoostRegressor
    outer_train = np.asarray(outer_train, dtype=bool) & real_mask(points)
    outer_valid = np.asarray(outer_valid, dtype=bool) & real_mask(points)
    if np.any(outer_train & outer_valid) or not outer_valid.any():
        raise ValueError("Outer training and validation must be non-overlapping and nonempty")
    current = x["cur_dev_s"].fillna(0).to_numpy(dtype=float)
    y = points["target_delay_s"].to_numpy(dtype=float)
    residual = y - current
    params = {**config["params"], "random_seed": seed, "thread_count": 2,
              "verbose": False, "allow_writing_files": False}
    inner_metadata = None
    if config["kind"] == "control":
        iterations = int(params["iterations"])
    else:
        fit, stop, inner_metadata = make_inner_split(points, outer_train)
        inner_model = factory(**{**params, "iterations": MAX_ITERATIONS})
        inner_model.fit(x.loc[fit], residual[fit], eval_set=(x.loc[stop], residual[stop]),
                        early_stopping_rounds=PATIENCE, use_best_model=True)
        best = inner_model.get_best_iteration()
        if best is None or not 0 <= int(best) < MAX_ITERATIONS:
            raise ValueError("Early stopping did not provide a valid best iteration")
        iterations = int(best) + 1
    model = factory(**{**params, "iterations": iterations})
    # No outer eval_set: its targets never select the number of trees.
    model.fit(x.loc[outer_train], residual[outer_train])
    prediction = np.asarray(model.predict(x.loc[outer_valid], thread_count=2), dtype=float) + current[outer_valid]
    if not np.isfinite(prediction).all():
        raise ValueError("Candidate produced non-finite predictions")
    record = {"fold": int(fold), "mae_s": float(mean_absolute_error(y[outer_valid], prediction)),
              "n": int(outer_valid.sum()), "iterations": iterations,
              "inner_iterations": iterations if inner_metadata else None, "inner_split": inner_metadata}
    predictions = pd.DataFrame({"sample_id": points.loc[outer_valid, "sample_id"].to_numpy(),
                                "target_delay_s": y[outer_valid], "cur_dev_s": current[outer_valid],
                                "prediction_delay_s": prediction, "fold": fold})
    return record, predictions


def _pooled(folds: list[dict[str, Any]]) -> float:
    return float(sum(item["mae_s"] * item["n"] for item in folds) / sum(item["n"] for item in folds))


def evaluate_config(config: dict[str, Any], seed: int, x: pd.DataFrame, points: pd.DataFrame,
                    folds: list[tuple[np.ndarray, np.ndarray, dict[str, Any]]],
                    fold_ids: tuple[int, ...] = (0, 1)) -> tuple[dict[str, Any], pd.DataFrame]:
    started = time.perf_counter()
    records, predictions = [], []
    for fold in fold_ids:
        training, validation, _ = folds[fold]
        record, frame = fit_candidate_fold(config, seed, x, points, training, validation, fold)
        records.append(record)
        predictions.append(frame)
    return {"config_id": config["id"], "seed": int(seed), "status": "complete", "folds": records,
            "pooled_mae_s": _pooled(records), "elapsed_s": time.perf_counter() - started}, pd.concat(predictions, ignore_index=True)


def _complete_pair(record: dict[str, Any]) -> bool:
    if record.get("status") != "complete":
        return False
    folds = record.get("folds", [])
    selected = [item for item in folds if item.get("fold") in (0, 1)]
    return (len(selected) == 2 and {item["fold"] for item in selected} == {0, 1}
            and all(item.get("n", 0) > 0 and np.isfinite(item.get("mae_s", np.nan)) for item in selected))


def _first_pair(record: dict[str, Any]) -> list[dict[str, Any]]:
    return sorted((item for item in record["folds"] if item["fold"] in (0, 1)), key=lambda item: item["fold"])


def choose_finalist(results: list[dict[str, Any]], control_results: list[dict[str, Any]],
                    configs: list[dict[str, Any]], *, seed: int = SEED) -> dict[str, Any] | None:
    """Apply the predeclared three-seed nomination gate on folds zero and one."""
    seeds = (seed, seed + 1, seed + 2)
    controls = {record["seed"]: record for record in control_results if _complete_pair(record)}
    if not all(value in controls for value in seeds):
        return None
    baseline_scores = np.array([_pooled(_first_pair(controls[value])) for value in seeds])
    baseline_folds = np.mean([[item["mae_s"] for item in _first_pair(controls[value])] for value in seeds], axis=0)
    eligible = []
    for config in configs:
        if config["kind"] == "control":
            continue
        runs = {record["seed"]: record for record in results if record.get("config_id") == config["id"] and _complete_pair(record)}
        if not all(value in runs for value in seeds):
            continue
        pairs = [_first_pair(runs[value]) for value in seeds]
        scores = np.array([_pooled(pair) for pair in pairs])
        fold_means = np.mean([[item["mae_s"] for item in pair] for pair in pairs], axis=0)
        improvement = float(baseline_scores.mean() - scores.mean())
        seed_wins = int(np.sum(scores < baseline_scores))
        if improvement < MIN_IMPROVEMENT_S or np.any(fold_means > baseline_folds) or seed_wins < 2:
            continue
        iterations = [int(item["iterations"]) for pair in pairs for item in pair]
        eligible.append({"config_id": config["id"], "mean_pooled_mae_s": float(scores.mean()),
                         "baseline_mean_pooled_mae_s": float(baseline_scores.mean()), "improvement_s": improvement,
                         "fold_means": fold_means.tolist(), "baseline_fold_means": baseline_folds.tolist(),
                         "seed_wins": seed_wins, "seeds": list(seeds),
                         "iterations": int(np.floor(np.median(iterations) + 0.5)), "inner_iterations": iterations,
                         "seed": int(seed), "params": deepcopy(config["params"])})
    if not eligible:
        return None
    best = min(item["mean_pooled_mae_s"] for item in eligible)
    tied = [item for item in eligible if item["mean_pooled_mae_s"] - best <= TIE_TOLERANCE_S]
    winner = min(tied, key=lambda item: (item["params"]["depth"], item["iterations"], item["mean_pooled_mae_s"], item["config_id"]))
    winner["params"].update(iterations=winner["iterations"], random_seed=seed)
    return winner


def protected_fingerprints(data_dir: Path, baseline_model_dir: Path) -> dict[str, str]:
    """Hash immutable inputs, both deployed profiles and existing submissions."""
    data_dir, baseline_model_dir = Path(data_dir).resolve(), Path(baseline_model_dir).resolve()
    files = {path.resolve() for root in (data_dir, baseline_model_dir) for path in root.rglob("*") if path.is_file()}
    project = Path(__file__).resolve().parents[1]
    for root in (project / "dataset", project / "artifacts/models", project / "artifacts/models_stream", project / "artifacts/stream_models"):
        if root.exists():
            files.update(path.resolve() for path in root.rglob("*") if path.is_file())
    for path in (project / "submission.csv", project / "artifacts/submission.csv", baseline_model_dir.parent / "submission.csv",
                 baseline_model_dir.parent / "report.json"):
        if path.is_file():
            files.add(path.resolve())
    for folder in (project, project / "artifacts", baseline_model_dir.parent):
        files.update(path.resolve() for path in folder.glob("*submission*.csv") if path.is_file())
    return {str(path): sha256(path) for path in sorted(files)}


def _prepare_output(output_dir: Path, data_dir: Path, baseline_model_dir: Path) -> Path:
    output = output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Output directory already exists; choose a new run directory: {output}")
    project = Path(__file__).resolve().parents[1]
    forbidden = tuple(path.resolve() for path in (data_dir, baseline_model_dir, project / "dataset", project / "artifacts/models",
                                                 project / "artifacts/models_stream", project / "artifacts/stream_models"))
    if any(output == path or path in output.parents or output in path.parents for path in forbidden):
        raise ValueError("Output must be separate from the dataset and active model directories")
    output.mkdir(parents=True, exist_ok=False)
    return output


def _save_predictions(output: Path, config_id: str, seed: int, frame: pd.DataFrame) -> None:
    path = output / "oof" / config_id / f"seed-{seed}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8")


def _split_report(points: pd.DataFrame, folds: list) -> list[dict[str, Any]]:
    result = []
    for index, (train, validation, metadata) in enumerate(folds):
        fit, stop, inner = make_inner_split(points, train)
        result.append({"fold": index, "role": "search" if index < 2 else "one_time_confirmation", **metadata,
                       "training_ids": points.loc[train & real_mask(points), "sample_id"].tolist(),
                       "scoring_ids": points.loc[validation, "sample_id"].tolist(),
                       "inner": {**inner, "fit_ids": points.loc[fit, "sample_id"].tolist(),
                                 "stop_ids": points.loc[stop, "sample_id"].tolist()}})
    return result


def _confirm_frozen(finalist: dict[str, Any], x: pd.DataFrame, points: pd.DataFrame,
                    fold: tuple[np.ndarray, np.ndarray, dict[str, Any]], baseline_mae_s: float,
                    output: Path) -> dict[str, Any]:
    frozen_path = output / "finalist.json"
    if not frozen_path.is_file():
        raise RuntimeError("Finalist must be persisted before confirmation")
    if json.loads(frozen_path.read_text()) != finalist:
        raise ValueError("Confirmation settings differ from the persisted frozen finalist")
    attempt_path = output / "confirmation_attempt.json"
    if (output / "confirmation.json").exists() or attempt_path.exists():
        raise FileExistsError("The single allowed confirmation attempt was already started")
    with attempt_path.open("x", encoding="utf-8") as stream:
        json.dump({"started_at": _now(), "finalist_sha256": sha256(frozen_path)}, stream)
    # kind=control means fixed iterations with no early stopping, not the baseline's hyperparameters.
    fixed = {"id": finalist["config_id"], "kind": "control", "params": finalist["params"]}
    record, predictions = fit_candidate_fold(fixed, finalist["seed"], x, points, fold[0], fold[1], 2)
    predictions.to_csv(output / "confirmation_predictions.csv", index=False, encoding="utf-8")
    improvement = float(baseline_mae_s - record["mae_s"])
    report = {"finalist_sha256": sha256(frozen_path), "fold": 2, "candidate_mae_s": record["mae_s"],
              "baseline_mae_s": baseline_mae_s, "improvement_s": improvement, "required_improvement_s": MIN_IMPROVEMENT_S,
              "confirmed": bool(improvement >= MIN_IMPROVEMENT_S), "iterations": record["iterations"],
              "seed": finalist["seed"], "rows": record["n"], "completed_at": _now()}
    _write_json(output / "confirmation.json", report)
    return report


def _export_candidate(finalist: dict[str, Any], x: pd.DataFrame, points: pd.DataFrame,
                      metadata: dict[str, Any], baseline_model_dir: Path, output: Path,
                      confirmation: dict[str, Any]) -> dict[str, Any]:
    if not confirmation.get("confirmed"):
        raise ValueError("Only a confirmed candidate can be exported")
    real = real_mask(points)
    residual = points["target_delay_s"].to_numpy(float) - x["cur_dev_s"].fillna(0).to_numpy(float)
    model = CatBoostRegressor(**finalist["params"])
    model.fit(x.loc[real], residual[real])
    model_dir = output / "models"
    model_dir.mkdir(exist_ok=False)
    model.save_model(str(model_dir / "regressor.cbm"))
    shutil.copy2(baseline_model_dir / "classifier.cbm", model_dir / "classifier.cbm")
    if sha256(model_dir / "classifier.cbm") != sha256(baseline_model_dir / "classifier.cbm"):
        raise RuntimeError("Classifier copy does not match the immutable baseline")
    manifest = deepcopy(metadata)
    identity = hashlib.sha256(json.dumps({"baseline": metadata["model_version"], "finalist": finalist,
                              "confirmation": confirmation}, sort_keys=True).encode()).hexdigest()[:12]
    manifest.update(model_version=f"official-tuned-{identity}", created_at=_now(), profile="official",
                    prediction_mode="residual", training_cohort="real-only", training_rows=int(real.sum()),
                    regression_settings=deepcopy(finalist["params"]), parent_model_version=metadata["model_version"],
                    tuning={"protocol": "first_two_folds_search_third_frozen_confirmation",
                            "confirmation": confirmation, "classifier_and_calibration": "copied_unchanged",
                            "active_model_replaced": False})
    manifest["artifact_sha256"] = {name: sha256(model_dir / name) for name in ("regressor.cbm", "classifier.cbm")}
    manifest["dependencies"] = {name: importlib.metadata.version(name) for name in ("catboost", "numpy", "pandas", "scikit-learn")}
    manifest["python_version"] = platform.python_version()
    project = Path(__file__).resolve().parents[1]
    manifest["tuning"]["provenance"] = {
        "baseline_manifest_sha256": sha256(baseline_model_dir / "manifest.json"),
        "source_sha256": {name: sha256(project / name) for name in ("ml/tune.py", "predictor/features.py", "predictor/data.py")},
        "run_file_sha256": {name: sha256(output / name) for name in ("configs.json", "splits.json", "finalist.json") if (output / name).is_file()},
        "feature_names_sha256": hashlib.sha256(json.dumps(FEATURE_NAMES).encode()).hexdigest(),
        "protected_inputs": json.loads((output / "integrity_before.json").read_text()) if (output / "integrity_before.json").exists() else {},
    }
    _write_json(model_dir / "manifest.json", manifest)
    loaded = DelayModel(model_dir)
    expected = np.asarray(model.predict(x.loc[real], thread_count=2)) + x.loc[real, "cur_dev_s"].fillna(0).to_numpy(float)
    actual = np.asarray([item["prediction_delay_s"] for item in loaded.predict(x.loc[real])])
    if not np.allclose(expected, actual, rtol=0, atol=1e-9):
        raise RuntimeError("Saved/reloaded regressor predictions differ from the fitted model")
    if loaded.metadata["probability_calibration"] != metadata["probability_calibration"]:
        raise RuntimeError("Probability calibration must remain unchanged")
    _write_json(output / "export_verification.json", {"rows": int(real.sum()), "max_prediction_difference_s": float(np.max(np.abs(expected - actual))),
                "classifier_unchanged": True, "probability_calibration_unchanged": True})
    return manifest


def _report_test(data_dir: Path, model_dir: Path, output: Path) -> dict[str, Any]:
    """Reporting-only test access; callers must already have passed confirmation."""
    points = load_points(data_dir / "labels/labels_test.csv", labels=True)
    x = FeatureBuilder(load_traffic(data_dir / "test/traffic.csv"), load_plan(data_dir / "test/schedule.csv")).build_many(points)
    results = DelayModel(model_dir).predict(x)
    predictions = np.asarray([item["prediction_delay_s"] for item in results])
    target = points["target_delay_s"].to_numpy(float)
    pd.DataFrame({"sample_id": points["sample_id"], "target_delay_s": target,
                  "prediction_delay_s": predictions, "p_late": [item["p_late"] for item in results]}).to_csv(output / "test_predictions.csv", index=False)
    report = {"rows": len(points), "mae_s": float(mean_absolute_error(target, predictions)),
              "zero_baseline_mae_s": float(np.mean(np.abs(target))),
              "cur_dev_baseline_mae_s": float(mean_absolute_error(target, x["cur_dev_s"].fillna(0))),
              "groups": _group_metrics(points, x, predictions), "used_for_selection": False}
    _write_json(output / "test_report.json", report)
    return report


def run_tuning(data_dir: Path, baseline_model_dir: Path, output_dir: Path,
               trials: int = 50, seed: int = SEED) -> dict[str, Any]:
    data_dir, baseline_model_dir = Path(data_dir).resolve(), Path(baseline_model_dir).resolve()
    # Resolve/validate paths before making even a temporary output file.
    output = _prepare_output(Path(output_dir), data_dir, baseline_model_dir)
    started = time.perf_counter()
    fingerprints = protected_fingerprints(data_dir, baseline_model_dir)
    _write_json(output / "integrity_before.json", fingerprints)
    report: dict[str, Any] = {"status": "RUNNING", "phase": "baseline_validation", "started_at": _now(),
                              "trials": trials, "seed": seed, "test_read": False, "validate_read": False,
                              "active_model_replaced": False,
                              "limitations": ["The third fold was already used in the original baseline development; it is reserved for this search, not a pristine holdout.",
                                               "Three random seeds test optimization stability; they are not independent data samples.",
                                               "A one-second acceptance margin is an operational rule, not statistical significance.",
                                               "Real labels cover one period; no new-day generalization is established."]}
    _write_json(output / "report.json", report)
    results: list[dict[str, Any]] = []
    controls: list[dict[str, Any]] = []
    error: BaseException | None = None
    try:
        project = Path(__file__).resolve().parents[1]
        provenance = {
            "created_at": _now(), "python_version": platform.python_version(),
            "dependencies": {name: importlib.metadata.version(name) for name in ("catboost", "numpy", "pandas", "scikit-learn")},
            "source_sha256": {name: sha256(project / name) for name in (
                "ml/tune.py", "ml/train.py", "ml/model.py", "ml/submit.py", "predictor/features.py", "predictor/data.py")},
            "seed": seed, "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "feature_names": FEATURE_NAMES,
        }
        _write_json(output / "provenance.json", provenance)
        metadata = json.loads((baseline_model_dir / "manifest.json").read_text())
        if metadata.get("profile", "official") != "official" or metadata["prediction_mode"] != "residual":
            raise ValueError("Tuning requires the official residual baseline")
        if metadata["feature_names"] != FEATURE_NAMES or metadata["feature_schema_version"] != FEATURE_SCHEMA_VERSION:
            raise ValueError("Official baseline feature schema does not match the shared builder")
        provenance["baseline"] = {
            "model_version": metadata["model_version"], "manifest_sha256": sha256(baseline_model_dir / "manifest.json"),
            "declared_regression_settings": metadata["regression_settings"],
            "effective_regression_settings": DelayModel(baseline_model_dir).regressor.get_all_params(),
        }
        _write_json(output / "provenance.json", provenance)
        baseline_report_path = baseline_model_dir.parent / "report.json"
        baseline_report = json.loads(baseline_report_path.read_text())
        expected = baseline_report["selection"]["selected"]
        configs = generate_configs(trials, seed, base_params=metadata["regression_settings"])
        _write_json(output / "configs.json", {"seed": seed, "space": SEARCH_SPACE, "configs": configs,
                    "cpu_threads_per_fit": 2, "early_stop_max_iterations": MAX_ITERATIONS, "early_stop_patience": PATIENCE,
                    "nomination_margin_s": MIN_IMPROVEMENT_S, "tie_tolerance_s": TIE_TOLERANCE_S})
        all_points = load_points(data_dir / "labels/labels_train.csv", labels=True)
        points = all_points.loc[real_mask(all_points)].reset_index(drop=True)
        if len(points) != metadata["training_rows"]:
            raise ValueError("Real training row count differs from the baseline manifest")
        train_traffic = load_traffic(data_dir / "train/traffic.csv")
        train_traffic = train_traffic.loc[train_traffic["tr_id"].isin(set(points["tr_id"]))]
        x = FeatureBuilder(train_traffic, load_plan(data_dir / "train/schedule.csv")).build_many(points)
        if list(x.columns) != FEATURE_NAMES:
            raise ValueError("Feature columns or ordering changed")
        provenance["feature_matrix"] = {
            "rows": len(x), "columns": len(FEATURE_NAMES),
            "serialization": "UTF-8 CSV; column order FEATURE_NAMES; float_format=%.17g; na_rep=NaN; index=False",
            "sha256": hashlib.sha256(x.to_csv(index=False, float_format="%.17g", na_rep="NaN").encode("utf-8")).hexdigest(),
            "ordered_sample_ids_sha256": hashlib.sha256("\n".join(points["sample_id"].astype(str)).encode("utf-8")).hexdigest(),
        }
        _write_json(output / "provenance.json", provenance)
        folds = temporal_folds(points)
        if len(folds) != 3:
            raise ValueError("Expected exactly the three existing temporal folds")
        _write_json(output / "splits.json", _split_report(points, folds))
        control, predictions = evaluate_config(configs[0], seed, x, points, folds, (0, 1, 2))
        _save_predictions(output, "control", seed, predictions)
        expected_scores = expected["fold_mae_s"]
        differences = [abs(actual["mae_s"] - reference) for actual, reference in zip(control["folds"], expected_scores, strict=True)]
        differences.append(abs(control["pooled_mae_s"] - expected["oof_mae_s"]))
        _write_json(output / "control.json", {"result": control, "expected": expected,
                    "tolerance_s": BASELINE_TOLERANCE_S, "max_difference_s": max(differences)})
        if max(differences) > BASELINE_TOLERANCE_S:
            raise ValueError(f"Baseline reproduction failed: difference {max(differences):.6f}s exceeds {BASELINE_TOLERANCE_S}s")
        controls.append(control)
        results.append(control)
        _write_json(output / "results.json", results)
        report.update(phase="search", real_training_rows=len(points), feature_count=len(FEATURE_NAMES),
                      baseline_oof_mae_s=control["pooled_mae_s"])
        _write_json(output / "report.json", report)

        def evaluate_and_save(config: dict[str, Any], current_seed: int) -> dict[str, Any]:
            try:
                result, predictions = evaluate_config(config, current_seed, x, points, folds)
                _save_predictions(output, config["id"], current_seed, predictions)
            except Exception as exc:
                result = {"config_id": config["id"], "seed": current_seed, "status": "failed", "error": str(exc)}
            results.append(result)
            _write_json(output / "results.json", results)
            print(json.dumps({key: result[key] for key in ("config_id", "seed", "status", "pooled_mae_s", "error") if key in result}), flush=True)
            return result

        for config in configs[1:]:
            evaluate_and_save(config, seed)
        completed = [item for item in results if item.get("config_id") != "control" and _complete_pair(item)]
        config_map = {item["id"]: item for item in configs}
        completed.sort(key=lambda item: (_pooled(_first_pair(item)), config_map[item["config_id"]]["params"]["depth"], item["config_id"]))
        shortlist = [config_map[item["config_id"]] for item in completed[:3]]
        report.update(phase="seed_confirmation", shortlisted_config_ids=[item["id"] for item in shortlist])
        _write_json(output / "report.json", report)
        for current_seed in (seed + 1, seed + 2):
            controls.append(evaluate_and_save(configs[0], current_seed))
            for config in shortlist:
                evaluate_and_save(config, current_seed)
        finalist = choose_finalist(results, controls, shortlist, seed=seed)
        if finalist is None:
            report.update(status="NO_CONFIRMED_IMPROVEMENT", phase="completed", reason="No complete candidate passed the predeclared first-two-fold three-seed gate")
        else:
            finalist.update(frozen_at=_now(), feature_schema_version=FEATURE_SCHEMA_VERSION,
                            prediction_mode="residual", training_cohort="real-only", confirmation_attempts_allowed=1)
            _write_json(output / "finalist.json", finalist)
            report.update(phase="frozen_confirmation", finalist=finalist)
            _write_json(output / "report.json", report)
            confirmation = _confirm_frozen(finalist, x, points, folds[2], control["folds"][2]["mae_s"], output)
            report["confirmation"] = confirmation
            if not confirmation["confirmed"]:
                report.update(status="NO_CONFIRMED_IMPROVEMENT", phase="completed", reason="Frozen candidate failed the one-time third-fold margin; no runner-up evaluated")
            else:
                report.update(phase="final_fit_and_export")
                _write_json(output / "report.json", report)
                manifest = _export_candidate(finalist, x, points, metadata, baseline_model_dir, output, confirmation)
                report.update(phase="test_reporting", model_version=manifest["model_version"], test_read=True)
                _write_json(output / "report.json", report)
                report["test"] = _report_test(data_dir, output / "models", output)
                report.update(phase="submission", validate_read=True)
                _write_json(output / "report.json", report)
                submission = generate_submission(data_dir, output / "models", output / "submission.csv")
                report.update(status="CONFIRMED_IMPROVEMENT", phase="completed", submission_rows=len(submission))
    except KeyboardInterrupt as exc:
        error = exc
        report.update(status="INTERRUPTED", phase="interrupted", error="Interrupted by user")
    except Exception as exc:
        error = exc
        report.update(status="FAILED", phase="failed", error=str(exc))
    finally:
        after = protected_fingerprints(data_dir, baseline_model_dir)
        changed = [path for path, digest in fingerprints.items() if after.get(path) != digest]
        added = sorted(set(after) - set(fingerprints))
        integrity = {"unchanged": not changed and not added, "changed_or_missing": changed,
                     "unexpected_added": added, "before": fingerprints, "after": after}
        _write_json(output / "integrity.json", integrity)
        if not integrity["unchanged"]:
            error = RuntimeError("Immutable inputs or deployed artifacts changed during tuning")
            report.update(status="FAILED", phase="integrity_failed", error=str(error))
        report.update(finished_at=_now(), elapsed_s=time.perf_counter() - started, integrity_unchanged=integrity["unchanged"])
        _write_json(output / "report.json", report)
    if error is not None:
        raise error
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--baseline-model-dir", type=Path, default=Path("artifacts/models"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    report = run_tuning(args.data_dir, args.baseline_model_dir, args.output_dir, args.trials, args.seed)
    print(json.dumps({key: report[key] for key in ("status", "reason", "model_version", "elapsed_s") if key in report}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
