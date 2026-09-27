"""Model selection split safety, inference contracts and submission validity."""

from pathlib import Path

from fastapi.testclient import TestClient
import numpy as np
import pandas as pd
import pytest

from ml.app import app
from ml.submit import validate_submission
from ml.train import temporal_folds
from predictor.features import FEATURE_NAMES, FEATURE_SCHEMA_VERSION


def test_temporal_folds_purge_targets_and_observed_outcomes():
    times = pd.date_range("2026-01-06", periods=220, freq="5min", tz="UTC")
    points = pd.DataFrame({"tr_id": "100", "T": times, "target_stop_id": [str(i // 2) for i in range(len(times))],
                           "target_time_begin": times + pd.Timedelta(minutes=12), "target_delay_s": 3600})
    for training, validation, metadata in temporal_folds(points):
        assert not (set(points.loc[training, "target_stop_id"]) & set(points.loc[validation, "target_stop_id"]))
        cutoff = pd.Timestamp(metadata["training_cutoff"])
        assert (points.loc[training, "T"] < cutoff).all()
        observed_at = points.loc[training, "target_time_begin"] + pd.Timedelta(hours=1)
        assert (observed_at < cutoff).all()
        assert pd.Timestamp(metadata["validation_start"]) - cutoff == pd.Timedelta(minutes=30)


def test_submission_preserves_negative_predictions_and_template_order():
    valid = pd.DataFrame({"sample_id": ["b", "a"], "prediction": [-80.5, 12]})
    validate_submission(valid, ["b", "a"])
    with pytest.raises(ValueError, match="IDs/order"):
        validate_submission(valid, ["a", "b"])
    with pytest.raises(ValueError, match="exactly once"):
        validate_submission(pd.concat([valid, valid]), ["b", "a"])
    with pytest.raises(ValueError, match="finite"):
        validate_submission(valid.assign(prediction=[np.inf, 1]), ["b", "a"])


def test_missing_artifact_is_unready_but_process_alive(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    with TestClient(app) as client:
        assert client.get("/health/live").status_code == 200
        assert client.get("/health/ready").status_code == 503
        assert client.post("/v1/predict", json={"feature_schema_version": FEATURE_SCHEMA_VERSION,
                    "items": [{"request_id": "a", "features": {}}]}).status_code == 503


def test_artifact_http_batch_contract(monkeypatch):
    model_dir = Path(__file__).parents[1] / "artifacts/models"
    if not (model_dir / "manifest.json").exists():
        pytest.skip("Run python -m ml.train to generate integration model")
    monkeypatch.setenv("MODEL_DIR", str(model_dir))
    features = {name: None for name in FEATURE_NAMES}
    features.update(cur_dev_s=-45, cur_dev_missing=0, horizon_s=720)
    payload = {"feature_schema_version": FEATURE_SCHEMA_VERSION, "items": [{"request_id": "a", "features": features}]}
    with TestClient(app) as client:
        assert client.get("/health/ready").status_code == 200
        response = client.post("/v1/predict", json=payload)
        assert response.status_code == 200
        result = response.json()["items"][0]
        assert result["request_id"] == "a"
        assert np.isfinite(result["prediction_delay_s"])
        assert 0 <= result["p_late"] <= 1
        assert result["feature_schema_version"] == FEATURE_SCHEMA_VERSION
        assert client.post("/v1/predict", json={**payload, "feature_schema_version": "broken"}).status_code == 422
        assert client.post("/v1/predict", json={**payload, "items": [payload["items"][0]] * 2}).status_code == 422
        features.pop("cur_dev_s")
        assert client.post("/v1/predict", json=payload).status_code == 422
