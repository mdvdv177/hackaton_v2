from pathlib import Path

from fastapi.testclient import TestClient
import numpy as np
import pytest

from ml.app import app
from ml.model import DelayModel
from predictor.features import FEATURE_NAMES, FEATURE_SCHEMA_VERSION
from predictor.stream_features import STREAM_FEATURE_NAMES, STREAM_SCHEMA_VERSION


def test_stream_profile_is_explicit_and_keeps_official_compatible(monkeypatch):
    directory = Path(__file__).parents[1] / "artifacts/models"
    if not (directory / "stream/manifest.json").exists():
        pytest.skip("Run python -m ml.train_stream first")
    monkeypatch.setenv("MODEL_DIR", str(directory))
    features = {name: None for name in STREAM_FEATURE_NAMES}
    features.update(cur_dev_s=60, cur_dev_missing=0, cur_dev_estimated=1, horizon_s=780)
    payload = {"profile": "stream", "feature_schema_version": STREAM_SCHEMA_VERSION,
               "items": [{"request_id": "a", "features": features}]}
    with TestClient(app) as client:
        assert client.get("/health/ready?profile=stream").status_code == 200
        assert set(client.get("/v1/profiles").json()["profiles"]) == {"official", "stream"}
        response = client.post("/v1/predict", json=payload)
        assert response.status_code == 200, response.text
        result = response.json()["items"][0]
        assert result["profile"] == "stream"
        assert np.isfinite(result["prediction_delay_s"])
        assert 0 <= result["p_late"] <= 1
        assert client.post("/v1/predict", json={**payload, "profile": "official"}).status_code == 422
        assert client.post("/v1/predict", json={**payload, "feature_schema_version": FEATURE_SCHEMA_VERSION}).status_code == 422
        official = {"feature_schema_version": FEATURE_SCHEMA_VERSION,
                    "items": [{"request_id": "b", "features": {name: features[name] for name in FEATURE_NAMES}}]}
        assert client.post("/v1/predict", json=official).status_code == 200


def test_stream_training_report_preserves_selection_boundary():
    import json
    path = Path(__file__).parents[1] / "artifacts/stream_report.json"
    if not path.exists():
        pytest.skip("Run streaming training first")
    report = json.loads(path.read_text())
    assert report["selection"]["test_used_for_selection"] is False
    assert report["selection"]["provided_cur_dev_used"] is False
    assert report["selection"]["synthetic_training_used"] is False
    assert report["test"]["rows"] == 353
    assert all(fold["target_visits_disjoint"] and fold["purge_s"] == 1800 for fold in report["selection"]["folds"])


def test_required_stream_profile_readiness_and_no_implicit_fallback(monkeypatch, tmp_path):
    directory = Path(__file__).parents[1] / "artifacts/models"
    monkeypatch.setenv("MODEL_DIR", str(directory))
    monkeypatch.setenv("STREAM_MODEL_DIR", str(tmp_path))
    monkeypatch.setenv("REQUIRED_MODEL_PROFILES", "official,stream")
    with TestClient(app) as client:
        assert client.get("/health/live").status_code == 200
        assert client.get("/health/ready").status_code == 503
        response = client.post("/v1/predict", json={"profile": "stream", "feature_schema_version": STREAM_SCHEMA_VERSION,
            "items": [{"request_id": "x", "features": {name: None for name in STREAM_FEATURE_NAMES}}]})
        assert response.status_code == 503
