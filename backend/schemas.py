"""Versioned public response shapes exposed by OpenAPI."""
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


class PublicModel(BaseModel):
    model_config = ConfigDict(extra="allow")


class RunOutput(PublicModel):
    id: str
    mode: Literal["dispatcher", "evaluation", "ndtp"]
    source: str
    status: str
    virtual_time: str
    speed: int
    scenario_id: str | None = None
    scenario_name: str | None = None
    schedule_mode: Literal["as_is", "demo_rebased"] | None = None
    manifest: dict[str, Any] | None = None


class PredictionOutput(PublicModel):
    id: str | None = None
    prediction_time: str
    feature_cutoff_at: str | None = None
    generated_at: str | None = None
    triggered_at: str | None = None
    published_at: str | None = None
    published_scenario_at: str | None = None
    publication_horizon_s: float | None = None
    timing_status: str = "unknown_legacy"
    current_prediction: bool = False
    display_state: str = "stale"
    horizon_s: float | None = None
    current_segment: dict[str, Any] | None = None
    target_segment: dict[str, Any] | None = None
    target_visit_id: str
    target_name: str
    target_time_begin: str
    prediction_delay_s: float
    predicted_arrival_at: str
    p_late: float | None
    risk: Literal["red", "yellow", "green", "gray"]
    source: Literal["model", "baseline"]
    model_version: str | None
    factors: list[dict[str, Any]]


class VehicleOutput(PublicModel):
    id: str
    lat: float | None
    lon: float | None
    speed: float | None
    event_time: str
    received_at: str | None = None
    packet_id: str | None = None
    telemetry_age_s: float | None = None
    position_age_s: float | None
    stale: bool
    cur_dev_s: float | None
    cur_dev_source: str
    prediction: PredictionOutput | None


class VehicleDetail(VehicleOutput):
    telemetry: list[dict[str, Any]]
    history: list[PredictionOutput]
    planned_visits: list[dict[str, Any]]


class IncidentOutput(PublicModel):
    id: str
    tr_id: str
    target_visit_id: str
    risk: Literal["red", "yellow", "green", "gray"]
    status: Literal["active", "monitoring", "resolved", "data_stale", "expired"]
    acknowledged: bool
    prediction_delay_s: float
    p_late: float | None
    target_name: str
    prediction_time: str
    reason: str
    recommendation: str
    prediction_id: str | None = None
    prediction: PredictionOutput | None = None
    first_alert_at: str | None = None
    first_alert_scenario_at: str | None = None
    first_alert_horizon_s: float | None = None


class SnapshotOutput(PublicModel):
    run: RunOutput | None
    vehicles: list[VehicleOutput]
    incidents: list[IncidentOutput]
    system: dict[str, Any]
    event_id: int
    published_at: str | None = None
    network_version: str | None = None
    segment_risks: dict[str, str] = {}
    incidents_total: int = 0
