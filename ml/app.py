"""Independent inference service: no labels, telemetry store or schedule access."""

from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path
from typing import Any, Literal

from catboost import CatBoostError
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool

from ml.model import DelayModel

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(application: FastAPI):
    official_path = Path(os.getenv("MODEL_DIR", "artifacts/models"))
    paths = {"official": official_path, "stream": Path(os.getenv("STREAM_MODEL_DIR", str(official_path / "stream")))}
    application.state.models, application.state.load_errors = {}, {}
    for profile, path in paths.items():
        try:
            model = DelayModel(path)
            if model.profile != profile:
                raise ValueError(f"Artifact profile {model.profile} cannot serve {profile}")
            application.state.models[profile] = model
        except (OSError, ValueError, KeyError, CatBoostError) as exc:
            application.state.load_errors[profile] = str(exc)
            logger.error("Model profile %s not ready: %s", profile, exc)
    application.state.model = application.state.models.get("official")
    application.state.load_error = application.state.load_errors.get("official")
    yield


app = FastAPI(title="Transport Delay ML", version="1.0.0", lifespan=lifespan)


class PredictionItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    request_id: str
    features: dict[str, float | None]


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    feature_schema_version: str
    profile: Literal["official", "stream"] = "official"
    items: list[PredictionItem] = Field(min_length=1, max_length=256)


@app.get("/health/live")
def live() -> dict[str, str]:
    return {"status": "alive"}


@app.get("/health/ready")
def ready(profile: Literal["official", "stream"] = "official") -> dict[str, Any]:
    required = {name.strip() for name in os.getenv("REQUIRED_MODEL_PROFILES", "official").split(",") if name.strip()}
    missing = required - set(getattr(app.state, "models", {}))
    if missing:
        raise HTTPException(503, f"Required model profiles unavailable: {', '.join(sorted(missing))}")
    model = getattr(app.state, "models", {}).get(profile)
    if model is None:
        raise HTTPException(503, "No valid model artifact loaded")
    return {"status": "ready", "profile": profile, "model_version": model.model_version,
            "feature_schema_version": model.schema_version,
            "profiles": {name: "ready" if name in app.state.models else "unavailable" for name in ("official", "stream")}}


@app.get("/v1/profiles")
def profiles() -> dict[str, Any]:
    return {"profiles": {name: {"status": "ready", "model_version": model.model_version,
                "feature_schema_version": model.schema_version, "feature_names": model.feature_names,
                "quality_status": model.metadata.get("quality_status", "official_protocol")}
            for name, model in getattr(app.state, "models", {}).items()},
            "unavailable": list(getattr(app.state, "load_errors", {}))}


@app.post("/v1/predict")
async def predict(request: PredictionRequest) -> dict[str, Any]:
    model = getattr(app.state, "models", {}).get(request.profile)
    if model is None:
        raise HTTPException(503, "No valid model artifact loaded")
    if request.feature_schema_version != model.schema_version:
        raise HTTPException(422, "Incompatible feature_schema_version")
    if len({item.request_id for item in request.items}) != len(request.items):
        raise HTTPException(422, "request_id must be unique within a batch")
    for item in request.items:
        if set(item.features) != set(model.feature_names):
            raise HTTPException(422, f"Invalid feature fields for request {item.request_id}")
    try:
        results = await run_in_threadpool(model.predict, [item.features for item in request.items], explain=True)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return {"model_version": model.model_version, "feature_schema_version": model.schema_version, "profile": request.profile,
            "items": [{"request_id": item.request_id, **result} for item, result in zip(request.items, results, strict=True)]}
