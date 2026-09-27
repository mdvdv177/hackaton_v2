"""Versioned CPU inference and a JSON-serializable probability calibrator."""

import json
from pathlib import Path
from typing import Any

from catboost import CatBoostClassifier, CatBoostRegressor, Pool
import numpy as np
import pandas as pd

from predictor.features import FEATURE_NAMES, FEATURE_SCHEMA_VERSION
from predictor.stream_features import STREAM_FEATURE_NAMES, STREAM_SCHEMA_VERSION

FACTOR_LABELS = {
    "cur_dev_s": "Текущее отклонение от расписания",
    "last_speed": "Последняя скорость",
    "distance_to_target_m": "Расстояние по прямой до цели",
    "distance_change_5m_m": "Изменение расстояния до цели",
    "stop_duration_s": "Длительность наблюдаемого простоя",
    "horizon_s": "Время до планового прибытия",
    "stops_until_target": "Число промежуточных посещений",
    "hour_sin": "Время суток", "hour_cos": "Время суток",
    "target_lon": "Расположение целевой остановки", "target_lat": "Расположение целевой остановки",
    "segment_speed_kmh": "Средняя скорость на наблюдаемом участке",
    "segment_speed_coverage": "Полнота наблюдений на участке",
    "segment_elapsed_s": "Время на наблюдаемом участке",
}


class DelayModel:
    def __init__(self, model_dir: str | Path):
        self.path = Path(model_dir)
        self.metadata = json.loads((self.path / "manifest.json").read_text())
        self.schema_version = self.metadata["feature_schema_version"]
        schemas = {FEATURE_SCHEMA_VERSION: FEATURE_NAMES, STREAM_SCHEMA_VERSION: STREAM_FEATURE_NAMES}
        if self.schema_version not in schemas:
            raise ValueError("Model feature schema version does not match this application")
        self.feature_names = schemas[self.schema_version]
        if self.metadata["feature_names"] != self.feature_names:
            raise ValueError("Model feature names/order do not match this application")
        self.profile = self.metadata.get("profile", "official")
        self.model_version = self.metadata["model_version"]
        self.mode = self.metadata["prediction_mode"]
        self.regressor: CatBoostRegressor | None = None
        if self.mode in {"direct", "residual"}:
            self.regressor = CatBoostRegressor()
            self.regressor.load_model(str(self.path / "regressor.cbm"))
        self.classifier = CatBoostClassifier()
        self.classifier.load_model(str(self.path / "classifier.cbm"))

    def predict(self, features: pd.DataFrame | list[dict[str, Any]], *, explain: bool = False) -> list[dict[str, Any]]:
        frame = features.copy() if isinstance(features, pd.DataFrame) else pd.DataFrame(features)
        missing = set(self.feature_names) - set(frame.columns)
        if missing:
            raise ValueError(f"Missing required features: {sorted(missing)}")
        frame = frame.loc[:, self.feature_names].astype(float).replace([np.inf, -np.inf], np.nan)
        current = frame["cur_dev_s"].fillna(0).to_numpy()
        if self.mode == "zero":
            prediction = np.zeros(len(frame))
        elif self.mode == "cur_dev":
            prediction = current
        else:
            prediction = self.regressor.predict(frame, thread_count=2)
            if self.mode == "residual":
                prediction += current
        raw = np.asarray(self.classifier.predict(frame, prediction_type="RawFormulaVal", thread_count=2))
        calibration = self.metadata["probability_calibration"]
        logits = np.clip(calibration["slope"] * raw + calibration["intercept"], -40, 40)
        probabilities = 1 / (1 + np.exp(-logits))
        shap_values = None
        if explain and self.regressor is not None:
            shap_values = self.regressor.get_feature_importance(Pool(frame), type="ShapValues", thread_count=2)[:, :-1]
        results = []
        for index in range(len(frame)):
            factors = []
            if shap_values is not None:
                for feature_index in np.argsort(np.abs(shap_values[index]))[-3:][::-1]:
                    name = self.feature_names[feature_index]
                    value = frame.iloc[index][name]
                    factors.append({"feature": name, "label": FACTOR_LABELS.get(name, name),
                                    "contribution_s": round(float(shap_values[index][feature_index]), 3),
                                    "value": float(value) if np.isfinite(value) else None})
            results.append({"prediction_delay_s": float(prediction[index]), "p_late": float(probabilities[index]),
                            "factors": factors, "model_version": self.model_version,
                            "feature_schema_version": self.schema_version, "profile": self.profile,
                            "status": "ok", "source": "model" if self.regressor is not None else "baseline"})
        return results
