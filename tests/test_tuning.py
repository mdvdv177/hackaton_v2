"""Train-only CatBoost tuning protocol, selection gates and artifact isolation."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml import tune
from ml.model import DelayModel
from ml.submit import validate_submission
from ml.train import REGRESSION_SETTINGS
from predictor.features import FEATURE_NAMES, FEATURE_SCHEMA_VERSION


SEEDS = (20260926, 20260927, 20260928)


def points_fixture(rows=220, tied=False):
    times = pd.date_range("2026-01-01", periods=(rows + 1) // 2 if tied else rows, freq="5min", tz="UTC")
    if tied:
        times = times.repeat(2)[:rows]
    return pd.DataFrame({"sample_id": [f"sample-{i}" for i in range(rows)], "tr_id": "100",
        "T": times, "target_stop_id": [f"visit-{i}" for i in range(rows)],
        "target_time_begin": times + pd.Timedelta(minutes=12),
        "cur_dev_s": np.linspace(-30, 90, rows), "target_delay_s": np.linspace(-10, 110, rows),
        "target_class": ["late" if i % 3 == 0 else "ontime" for i in range(rows)]})


def feature_fixture(points):
    frame = pd.DataFrame(0., index=points.index, columns=FEATURE_NAMES)
    frame["cur_dev_s"] = points["cur_dev_s"]
    frame["horizon_s"] = 720.
    return frame


def run_record(config_id, seed, scores=(8., 8.), counts=(10, 90), iterations=(10, 20), status="complete"):
    folds = [{"fold": fold, "mae_s": score, "n": count, "iterations": count_iterations,
              "inner_iterations": count_iterations}
             for fold, (score, count, count_iterations) in enumerate(zip(scores, counts, iterations))]
    return {"config_id": config_id, "seed": seed, "status": status, "folds": folds,
            "pooled_mae_s": float(np.average(scores, weights=counts))}


def test_deterministic_fifty_configurations_match_the_accepted_residual_space():
    configs = tune.generate_configs(trials=50, seed=SEEDS[0])
    assert configs == tune.generate_configs(trials=50, seed=SEEDS[0])
    assert configs != tune.generate_configs(trials=50, seed=SEEDS[0] + 1)
    assert len(configs) == 50
    assert configs[0]["kind"] == "control"
    assert all(item["kind"] == "candidate" for item in configs[1:])
    assert len({item["id"] for item in configs}) == 50
    spaces = {"depth": {3, 4, 5, 6, 7}, "learning_rate": {.02, .03, .045, .07, .1},
              "l2_leaf_reg": {1, 3, 5, 10, 20}, "random_strength": {0, .5, 1, 2}, "subsample": {.6, .8, 1}}
    combinations = [tuple(item["params"][name] for name in spaces) for item in configs[1:]]
    assert len(set(combinations)) == 49
    for item in configs:
        params = item["params"]
        assert params["loss_function"] == "MAE"
        assert params["thread_count"] == 2
        assert params.get("task_type", "CPU") == "CPU"
        if item["kind"] == "candidate":
            assert params["bootstrap_type"] == "MVS"
            for name, allowed in spaces.items():
                assert params[name] in allowed
    assert configs[0]["params"]["depth"] == configs[1]["params"]["depth"] == 5
    assert configs[0]["params"]["learning_rate"] == configs[1]["params"]["learning_rate"] == .045


def test_inner_split_keeps_timestamp_ties_purges_visits_and_waits_for_outcomes():
    points = points_fixture(tied=True)
    outer = np.arange(len(points)) < 180
    _, preliminary_stop, _ = tune.make_inner_split(points, outer)
    stop_ids = points.index[preliminary_stop]
    points.loc[0, "target_stop_id"] = points.loc[stop_ids[0], "target_stop_id"]
    points.loc[1, "target_delay_s"] = 1_000_000  # Outcome is unavailable at the inner cutoff.
    points.loc[2, "tr_id"] = "9000001"  # Synthetic lineage must not enter tuning.
    fit, stop, metadata = tune.make_inner_split(points, outer)
    assert not (fit & stop).any()
    assert (outer | ~(fit | stop)).all()
    assert 34 <= int(stop.sum()) <= 38
    first_stop = points.loc[stop, "T"].min()
    cutoff = first_stop - pd.Timedelta(minutes=30)
    assert (points.loc[fit, "T"] < cutoff).all()
    observable = points["target_time_begin"] + pd.to_timedelta(points["target_delay_s"], unit="s")
    assert (observable.loc[fit] < cutoff).all()
    assert not set(points.loc[fit, "target_stop_id"]) & set(points.loc[stop, "target_stop_id"])
    assert not fit[0] and not fit[1] and not fit[2] and not stop[2]
    # A boundary timestamp is assigned as one group, not split by row position.
    tied = outer & (points["T"] == first_stop).to_numpy()
    assert stop[tied].all()
    assert metadata


class RecordingRegressor:
    instances = []
    best_iteration = 6
    residual_prediction = -7.5

    def __init__(self, **settings):
        self.settings = settings
        self.fit_calls = []
        type(self).instances.append(self)

    def fit(self, features, target, **kwargs):
        self.fit_calls.append((features.copy(), np.asarray(target).copy(), kwargs))
        return self

    def get_best_iteration(self):
        return self.best_iteration

    def predict(self, features, **kwargs):
        return np.full(len(features), self.residual_prediction)


def test_early_stop_uses_best_plus_one_then_refits_full_outer_train_with_signed_residuals():
    RecordingRegressor.instances = []
    points = points_fixture()
    x = feature_fixture(points)
    outer_train = np.arange(len(points)) < 180
    outer_valid = ~outer_train
    config = tune.generate_configs(trials=2, seed=SEEDS[0])[1]
    record, predictions = tune.fit_candidate_fold(config, SEEDS[0], x, points,
        outer_train, outer_valid, 0, estimator_factory=RecordingRegressor)
    assert len(RecordingRegressor.instances) == 2
    early, refitted = RecordingRegressor.instances
    fit_x, fit_y, fit_kwargs = early.fit_calls[0]
    final_x, final_y, final_kwargs = refitted.fit_calls[0]
    assert len(fit_x) < int(outer_train.sum())
    assert "eval_set" in fit_kwargs
    assert refitted.settings["iterations"] == RecordingRegressor.best_iteration + 1
    assert record["iterations"] == RecordingRegressor.best_iteration + 1
    assert set(final_x.index) == set(points.index[outer_train])
    assert not final_kwargs.get("eval_set")
    target = points["target_delay_s"] - x["cur_dev_s"].fillna(0)
    np.testing.assert_array_equal(fit_y, target.loc[fit_x.index].to_numpy())
    np.testing.assert_array_equal(final_y, target.loc[outer_train].to_numpy())
    np.testing.assert_array_equal(predictions["prediction_delay_s"].to_numpy(),
                                  x.loc[outer_valid, "cur_dev_s"].to_numpy() + RecordingRegressor.residual_prediction)


def test_control_is_fixed_400_without_inner_early_stopping():
    RecordingRegressor.instances = []
    points = points_fixture()
    mask = np.arange(len(points)) < 180
    config = tune.generate_configs(trials=2, seed=SEEDS[0])[0]
    tune.fit_candidate_fold(config, SEEDS[0], feature_fixture(points), points, mask, ~mask, 0,
                            estimator_factory=RecordingRegressor)
    assert len(RecordingRegressor.instances) == 1
    estimator = RecordingRegressor.instances[0]
    assert estimator.settings["iterations"] == 400
    assert "eval_set" not in estimator.fit_calls[0][2]


def test_search_scores_only_two_folds_and_pools_individual_absolute_errors(monkeypatch):
    calls = []

    def fold(config, seed, x, points, train, valid, index, **kwargs):
        calls.append(index)
        count, error = [(2, 1.), (8, 10.), (3, 999.)][index]
        predictions = pd.DataFrame({"sample_id": [f"{index}-{i}" for i in range(count)], "target_delay_s": 0.,
            "cur_dev_s": 0., "prediction_delay_s": error, "fold": index})
        return {"fold": index, "mae_s": error, "n": count, "iterations": 7, "inner_iterations": 7}, predictions

    monkeypatch.setattr(tune, "fit_candidate_fold", fold)
    points = points_fixture()
    mask = np.arange(len(points)) < 180
    folds = [(mask, ~mask, {})] * 3
    config = tune.generate_configs(trials=2, seed=SEEDS[0])[1]
    record, predictions = tune.evaluate_config(config, SEEDS[0], feature_fixture(points), points, folds)
    assert calls == [0, 1]
    assert set(predictions["fold"]) == {0, 1}
    assert record["pooled_mae_s"] == pytest.approx(8.2)
    assert record["pooled_mae_s"] != pytest.approx(5.5)


def gate_case(scores, *, counts=(10, 90)):
    configs = tune.generate_configs(trials=2, seed=SEEDS[0])
    controls = [run_record(configs[0]["id"], seed, (10., 10.), counts) for seed in SEEDS]
    results = [run_record(configs[1]["id"], seed, row, counts, iterations=(iteration, iteration))
               for seed, row, iteration in zip(SEEDS, scores, (10, 20, 30))]
    return configs, controls, results


def test_nomination_requires_three_seeds_one_second_both_folds_and_two_wins():
    configs, control, results = gate_case([(8., 8.), (9., 9.), (10., 10.)])
    winner = tune.choose_finalist(results, control, configs)
    assert winner is not None and winner["config_id"] == configs[1]["id"]
    assert winner["mean_pooled_mae_s"] == 9.
    assert winner["seed_wins"] == 2 and winner["iterations"] == 20
    assert tune.choose_finalist(results[:2], control, configs) is None
    broken = copy.deepcopy(results)
    broken[-1]["status"] = "failed"
    assert tune.choose_finalist(broken, control, configs) is None
    incomplete = copy.deepcopy(results)
    incomplete[-1]["folds"] = incomplete[-1]["folds"][:1]
    assert tune.choose_finalist(incomplete, control, configs) is None
    _, control, less_than_one_second = gate_case([(9.01, 9.01)] * 3)
    assert tune.choose_finalist(less_than_one_second, control, configs) is None
    _, control, one_seed_only = gate_case([(1., 1.), (11., 11.), (11., 11.)])
    assert tune.choose_finalist(one_seed_only, control, configs) is None
    _, control, worse_second_fold = gate_case([(1., 11.)] * 3, counts=(90, 10))
    assert tune.choose_finalist(worse_second_fold, control, configs) is None


def test_reserved_third_control_fold_cannot_change_finalist_choice():
    configs, control, results = gate_case([(8., 8.), (9., 9.), (10., 10.)])
    expected = tune.choose_finalist(results, control, configs)
    for record in control:
        record["folds"].append({"fold": 2, "mae_s": 1_000_000, "n": 5000, "iterations": 400, "inner_iterations": None})
    assert tune.choose_finalist(results, control, configs) == expected


@pytest.mark.parametrize("advantage,expected_complex", [(.4, True), (2e-9, True), (1e-10, False)])
def test_complexity_tiebreak_is_only_for_numeric_equality_not_a_half_second(advantage, expected_complex):
    configs = tune.generate_configs(trials=3, seed=SEEDS[0])
    configs[1]["params"]["depth"] = 3
    configs[2]["params"]["depth"] = 7
    controls = [run_record(configs[0]["id"], seed, (10., 10.)) for seed in SEEDS]
    simpler = [run_record(configs[1]["id"], seed, (8., 8.), iterations=(10, 10)) for seed in SEEDS]
    complex_results = [run_record(configs[2]["id"], seed, (8. - advantage, 8. - advantage), iterations=(30, 30)) for seed in SEEDS]
    finalist = tune.choose_finalist(simpler + complex_results, controls, configs)
    assert finalist["config_id"] == configs[2 if expected_complex else 1]["id"]


def tree_signature(root):
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "directory"
            for path in root.rglob("*")}


@pytest.fixture
def tuning_case(tmp_path):
    """Tiny real model files; expensive search scores are injected independently."""
    from catboost import CatBoostClassifier, CatBoostRegressor

    dataset, baseline = tmp_path / "dataset", tmp_path / "baseline/models"
    for folder in ("train", "test", "validate", "labels"):
        (dataset / folder).mkdir(parents=True)
    baseline.mkdir(parents=True)
    train = points_fixture()
    train["target_delay_s"] = train["cur_dev_s"] + 20 + np.sin(np.arange(len(train))) * 3
    synthetic = train.iloc[:10].copy()
    synthetic["tr_id"] = "9000001"
    synthetic["sample_id"] = [f"synthetic-{i}" for i in range(len(synthetic))]
    synthetic["target_stop_id"] = [f"synthetic-visit-{i}" for i in range(len(synthetic))]
    synthetic["target_delay_s"] = 1_000_000
    pd.concat([train, synthetic], ignore_index=True).to_csv(dataset / "labels/labels_train.csv", index=False)
    test = points_fixture(8)
    test["sample_id"] = [f"test-{i}" for i in range(len(test))]
    test["target_stop_id"] = [f"test-visit-{i}" for i in range(len(test))]
    test.to_csv(dataset / "labels/labels_test.csv", index=False)
    validate = test.drop(columns=["target_delay_s", "target_class"]).iloc[:6].copy()
    validate["sample_id"] = [f"проверка-{i}" for i in range(len(validate))]
    validate["cur_dev_s"] = -90.
    validate.to_csv(dataset / "validate/points.csv", index=False)
    for split, points in (("train", train), ("test", test), ("validate", validate)):
        schedule = pd.DataFrame({"tr_id": points["tr_id"], "tt_action_item_id": points["target_stop_id"],
                                 "time_begin": points["target_time_begin"], "geom": "POINT (37.6 55.7)",
                                 "building_address": "Остановка"})
        schedule.to_csv(dataset / split / ("schedule_plan.csv" if split == "validate" else "schedule.csv"), index=False)
        traffic = pd.DataFrame({"packet_id": [f"packet-{i}" for i in range(len(points))], "tr_id": points["tr_id"],
            "unit_id": "123", "event_time": points["T"], "receive_time": points["T"],
            "location_valid": True, "lat": 55.7, "lon": 37.6, "speed": 15., "heading": 90.})
        traffic.to_csv(dataset / split / "traffic.csv", index=False)
    template_ids = validate["sample_id"].tolist()[::-1]
    pd.DataFrame({"sample_id": template_ids, "prediction": 0}).to_csv(dataset / "sample_submission.csv", sep=";", index=False)
    tiny_x = feature_fixture(train.iloc[:16])
    tiny_settings = {"iterations": 3, "depth": 2, "thread_count": 2, "random_seed": SEEDS[0],
                     "verbose": False, "allow_writing_files": False}
    CatBoostRegressor(**tiny_settings, loss_function="MAE").fit(tiny_x, np.linspace(-3, 8, len(tiny_x))).save_model(str(baseline / "regressor.cbm"))
    CatBoostClassifier(**tiny_settings, loss_function="Logloss").fit(tiny_x, np.arange(len(tiny_x)) % 2).save_model(str(baseline / "classifier.cbm"))
    metadata = {"profile": "official", "model_version": "tiny-official-baseline", "prediction_mode": "residual",
        "feature_schema_version": FEATURE_SCHEMA_VERSION, "feature_names": FEATURE_NAMES,
        "training_rows": len(train), "training_cohort": "real-only", "regression_settings": REGRESSION_SETTINGS,
        "probability_calibration": {"method": "fixture frozen sigmoid", "slope": .8, "intercept": -.2}}
    (baseline / "manifest.json").write_text(json.dumps(metadata), encoding="utf-8")
    (baseline.parent / "report.json").write_text(json.dumps({"selection": {"selected": {
        "cohort": "real", "mode": "residual", "oof_mae_s": 10., "fold_mae_s": [10., 10., 10.]}}}), encoding="utf-8")
    (baseline.parent / "submission.csv").write_text("sample_id;prediction\nprotected;7\n", encoding="utf-8")
    return {"dataset": dataset, "baseline": baseline, "output": tmp_path / "isolated-candidate",
            "train": train, "template_ids": template_ids}


def mock_search(monkeypatch, case, *, failure=None):
    """Inject search scores while keeping real split/export/load/submission behavior."""
    events = []
    original_loader = tune.load_points
    output = case["output"]

    def load_points(path, **kwargs):
        path = Path(path)
        if path.name == "labels_test.csv":
            assert (output / "models/manifest.json").is_file()
            assert json.loads((output / "confirmation.json").read_text())["confirmed"]
            events.append("test_read")
        return original_loader(path, **kwargs)

    def evaluate(config, seed, x, points, folds, fold_ids=(0, 1)):
        ids = tuple(fold_ids)
        events.append(("score", config["id"], seed, ids))
        assert points["tr_id"].eq("100").all()
        if config["kind"] != "control":
            assert ids == (0, 1)
            assert not (output / "finalist.json").exists()
            if failure == "incomplete":
                raise RuntimeError("Injected interrupted trial")
        error = 10. if config["kind"] == "control" else 9.5 if failure == "gate" else 8.
        records, predictions = [], []
        for index in ids:
            mask = folds[index][1]
            records.append({"fold": index, "mae_s": error, "n": int(mask.sum()), "iterations": 400 if config["kind"] == "control" else 7,
                            "inner_iterations": None if config["kind"] == "control" else 7})
            predictions.append(pd.DataFrame({"sample_id": points.loc[mask, "sample_id"].to_numpy(),
                "target_delay_s": points.loc[mask, "target_delay_s"].to_numpy(), "cur_dev_s": x.loc[mask, "cur_dev_s"].to_numpy(),
                "prediction_delay_s": points.loc[mask, "target_delay_s"].to_numpy() + error, "fold": index}))
        return {"config_id": config["id"], "seed": seed, "status": "complete", "folds": records,
                "pooled_mae_s": error}, pd.concat(predictions, ignore_index=True)

    def confirmation(config, seed, x, points, training, valid, fold, **kwargs):
        assert fold == 2 and config["id"] != "control"
        frozen = json.loads((output / "finalist.json").read_text())
        phase = json.loads((output / "report.json").read_text())["phase"]
        assert phase == "frozen_confirmation" and frozen["params"] == config["params"]
        assert config["kind"] == "control" and config["params"]["iterations"] == 7
        events.append(("reserved_confirmation", config["id"]))
        error = 9.5 if failure == "confirmation" else 8.
        frame = pd.DataFrame({"sample_id": points.loc[valid, "sample_id"].to_numpy(),
            "target_delay_s": points.loc[valid, "target_delay_s"].to_numpy(), "cur_dev_s": x.loc[valid, "cur_dev_s"].to_numpy(),
            "prediction_delay_s": points.loc[valid, "target_delay_s"].to_numpy() + error, "fold": 2})
        return {"fold": 2, "mae_s": error, "n": int(valid.sum()), "iterations": 7}, frame

    monkeypatch.setattr(tune, "load_points", load_points)
    monkeypatch.setattr(tune, "evaluate_config", evaluate)
    monkeypatch.setattr(tune, "fit_candidate_fold", confirmation)
    return events


@pytest.mark.parametrize("location", ["dataset_child", "model_child", "existing_output"])
def test_unsafe_output_paths_are_rejected_before_any_write(tuning_case, location):
    case = tuning_case
    root = case["dataset"].parent
    output = (case["dataset"] / "candidate" if location == "dataset_child" else
              case["baseline"] / "candidate" if location == "model_child" else case["output"])
    if location == "existing_output":
        output.mkdir()
        (output / "keep.txt").write_text("existing result", encoding="utf-8")
    before = tree_signature(root)
    with pytest.raises((ValueError, FileExistsError)):
        tune.run_tuning(case["dataset"], case["baseline"], output, trials=2)
    assert tree_signature(root) == before


@pytest.mark.parametrize("failure", ["gate", "incomplete", "confirmation"])
def test_failed_or_incomplete_search_cannot_read_test_export_model_or_write_submission(tuning_case, monkeypatch, failure):
    case = tuning_case
    before_dataset, before_baseline = tree_signature(case["dataset"]), tree_signature(case["baseline"].parent)
    events = mock_search(monkeypatch, case, failure=failure)

    def forbidden(*args, **kwargs):
        pytest.fail("A failed candidate must not open test/validate or export model artifacts")

    monkeypatch.setattr(tune, "_report_test", forbidden)
    monkeypatch.setattr(tune, "_export_candidate", forbidden)
    monkeypatch.setattr(tune, "generate_submission", forbidden)
    report = tune.run_tuning(case["dataset"], case["baseline"], case["output"], trials=2)
    assert report["status"] == "NO_CONFIRMED_IMPROVEMENT"
    assert not report["test_read"] and not report["validate_read"]
    assert "test_read" not in events
    assert not (case["output"] / "models").exists()
    assert not (case["output"] / "submission.csv").exists()
    confirmations = [item for item in events if isinstance(item, tuple) and item[0] == "reserved_confirmation"]
    assert len(confirmations) == (1 if failure == "confirmation" else 0)
    assert tree_signature(case["dataset"]) == before_dataset
    assert tree_signature(case["baseline"].parent) == before_baseline
    assert json.loads((case["output"] / "integrity.json").read_text())["unchanged"]


def test_confirmed_candidate_refits_all_real_train_reloads_and_preserves_probabilities_and_submission(tuning_case, monkeypatch):
    from catboost import CatBoostRegressor

    case = tuning_case
    before_dataset, before_baseline = tree_signature(case["dataset"]), tree_signature(case["baseline"].parent)
    events = mock_search(monkeypatch, case)
    fits = []

    class TinyFinalRegressor:
        def __init__(self, **settings):
            assert settings["iterations"] == 7
            self.model = CatBoostRegressor(**settings)

        def fit(self, features, target, **kwargs):
            fits.append((features.copy(), np.asarray(target).copy(), kwargs))
            self.model.fit(features, target, **kwargs)
            return self

        def save_model(self, path):
            self.model.save_model(path)

        def predict(self, features, **kwargs):
            return self.model.predict(features, **kwargs)

    monkeypatch.setattr(tune, "CatBoostRegressor", TinyFinalRegressor)
    report = tune.run_tuning(case["dataset"], case["baseline"], case["output"], trials=2)
    assert report["status"] == "CONFIRMED_IMPROVEMENT" and report["test_read"] and report["validate_read"]
    assert len(fits) == 1 and len(fits[0][0]) == len(case["train"])
    actual_train = pd.read_csv(case["dataset"] / "labels/labels_train.csv").iloc[:len(case["train"])]
    np.testing.assert_array_equal(fits[0][1], actual_train["target_delay_s"].to_numpy() - fits[0][0]["cur_dev_s"].fillna(0).to_numpy())
    assert not fits[0][2]
    assert len([item for item in events if isinstance(item, tuple) and item[0] == "reserved_confirmation"]) == 1
    assert events.index("test_read") > next(index for index, item in enumerate(events) if isinstance(item, tuple) and item[0] == "reserved_confirmation")
    model_dir = case["output"] / "models"
    baseline, loaded, reloaded = DelayModel(case["baseline"]), DelayModel(model_dir), DelayModel(model_dir)
    probe = fits[0][0].iloc[:8]
    first, second = loaded.predict(probe, explain=True), reloaded.predict(probe, explain=True)
    assert first == second
    assert [item["p_late"] for item in first] == [item["p_late"] for item in baseline.predict(probe)]
    assert (model_dir / "classifier.cbm").read_bytes() == (case["baseline"] / "classifier.cbm").read_bytes()
    assert loaded.metadata["probability_calibration"] == baseline.metadata["probability_calibration"]
    assert loaded.mode == "residual" and loaded.metadata["training_rows"] == len(case["train"])
    raw = (case["output"] / "submission.csv").read_bytes()
    assert raw.decode("utf-8").splitlines()[0] == "sample_id;prediction"
    submission = pd.read_csv(case["output"] / "submission.csv", sep=";", dtype={"sample_id": str})
    validate_submission(submission, case["template_ids"])
    assert (submission["prediction"] < 0).any()
    assert tree_signature(case["dataset"]) == before_dataset
    assert tree_signature(case["baseline"].parent) == before_baseline
