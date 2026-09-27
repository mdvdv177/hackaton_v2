"""REST and SSE adapter for the dispatcher service."""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel, field_validator

from backend.schemas import IncidentOutput, SnapshotOutput, VehicleDetail
from backend.service import Dispatcher
from backend.storage import Store

logger = logging.getLogger(__name__)


class ReplayStart(BaseModel):
    mode: Literal["dispatcher", "evaluation"] = "dispatcher"
    speed: Literal[1, 5, 20] = 20
    start_time: str | None = None
    source: Literal["test", "validate"] = "test"


class ReplaySpeed(BaseModel):
    speed: Literal[1, 5, 20]


class LiveStart(BaseModel):
    scenario_id: str | None = None
    schedule_mode: Literal["as_is", "demo_rebased"] = "as_is"
    source_time: str | None = None


def create_app(dispatcher: Dispatcher | None = None, start_listener: bool = True) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        instance = dispatcher
        if instance is None:
            database_url = os.getenv("DATABASE_URL", "sqlite:///artifacts/dispatcher.db")
            # PostgreSQL Docker services can need a short initialization window.
            for attempt in range(30):
                try:
                    store = Store(database_url)
                    break
                except Exception:
                    if attempt == 29:
                        raise
                    await asyncio.sleep(1)
            instance = Dispatcher(store, Path(os.getenv("DATA_DIR", "dataset")), os.getenv("ML_URL", "http://127.0.0.1:8001"))
        application.state.dispatcher = instance
        await instance.recover()
        server = None
        if start_listener and os.getenv("NDTP_ENABLED", "true").lower() != "false":
            from backend.ndtp import start_server
            server = await start_server(instance.ingest_live, host=os.getenv("NDTP_HOST", "0.0.0.0"), port=int(os.getenv("NDTP_PORT", "9201")))
        try:
            yield
        finally:
            if server:
                server.close()
                await server.wait_closed()
            await instance.close()

    application = FastAPI(title="Transport Dispatcher", version="1.0.0", lifespan=lifespan)
    origins = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")
    application.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=False,
                               allow_methods=["GET", "POST"], allow_headers=["Content-Type", "Last-Event-ID"])

    def service() -> Dispatcher:
        return application.state.dispatcher

    @application.get("/health/live")
    async def live():
        return {"status": "ok"}

    @application.get("/health/ready")
    async def ready():
        current = service()
        if not current.database_ready or not await asyncio.to_thread(current.store.healthy):
            raise HTTPException(503, "Database unavailable")
        return {"status": "ready", "ml_status": current.ml_status}

    @application.get("/api/v1/snapshot", response_model=SnapshotOutput)
    async def snapshot():
        return service().snapshot()

    @application.get("/api/v1/vehicles/{tr_id}", response_model=VehicleDetail)
    async def vehicle(tr_id: str):
        try:
            return service().detail(tr_id)
        except KeyError as error:
            raise HTTPException(404, "Vehicle not found in current run") from error

    @application.get("/api/v1/incidents", response_model=list[IncidentOutput])
    async def incidents(risk: str | None = None, status: str | None = None, acknowledged: bool | None = None,
                        limit: int = Query(100, ge=1, le=500), offset: int = Query(0, ge=0)):
        values = sorted((service().public_incident(item) for item in service().incidents.values()), key=lambda item: item["prediction_time"], reverse=True)
        filtered = [item for item in values if (risk is None or item["risk"] == risk)
                and (status is None or item["status"] == status)
                and (acknowledged is None or item["acknowledged"] == acknowledged)]
        return filtered[offset:offset + limit]

    @application.get("/api/v1/incidents/{identifier}", response_model=IncidentOutput)
    async def incident(identifier: str):
        try:
            return service().incident_detail(identifier)
        except KeyError as error:
            raise HTTPException(404, "Incident not found in current run") from error

    @application.get("/api/v1/network")
    async def network():
        return service().network_snapshot()

    @application.get("/api/v1/scenarios")
    async def scenarios():
        packages = await asyncio.to_thread(service().store.list_scenarios)
        return [{key: item.get(key) for key in ("id", "source_id", "name", "version", "geometry_kind", "coverage", "starts_at", "ends_at", "manifest")} for item in packages]

    @application.post("/api/v1/scenarios/import")
    async def import_scenario(body: dict, dry_run: bool = Query(False)):
        from backend.scenarios import validate_package
        try:
            package = await asyncio.to_thread(validate_package, body)
        except (ValueError, TypeError, KeyError) as error:
            raise HTTPException(422, str(error)) from error
        if not dry_run:
            await asyncio.to_thread(service().store.save_scenario, package)
        return {"valid": True, "dry_run": dry_run, "scenario": {
            key: package.get(key) for key in ("id", "source_id", "name", "version", "geometry_kind", "coverage", "starts_at", "ends_at", "manifest")}}

    @application.post("/api/v1/incidents/{identifier}/ack", response_model=IncidentOutput)
    async def acknowledge(identifier: str):
        try:
            return await service().acknowledge(identifier)
        except KeyError as error:
            raise HTTPException(404, "Incident not found in current run") from error

    @application.post("/api/v1/replay/start", response_model=SnapshotOutput)
    async def start(body: ReplayStart):
        try:
            return await service().start_replay(body.mode, body.speed, body.start_time, body.source)
        except ValueError as error:
            raise HTTPException(422, str(error)) from error

    @application.post("/api/v1/replay/pause", response_model=SnapshotOutput)
    async def pause():
        try:
            return await service().set_status("paused")
        except ValueError as error:
            raise HTTPException(409, str(error)) from error

    @application.post("/api/v1/replay/resume", response_model=SnapshotOutput)
    async def resume():
        try:
            return await service().set_status("running")
        except ValueError as error:
            raise HTTPException(409, str(error)) from error

    @application.post("/api/v1/replay/speed", response_model=SnapshotOutput)
    async def speed(body: ReplaySpeed):
        try:
            return await service().set_status(speed=body.speed)
        except ValueError as error:
            raise HTTPException(409, str(error)) from error

    @application.post("/api/v1/live/start", response_model=SnapshotOutput)
    async def start_live(body: LiveStart | None = None):
        try:
            settings = body or LiveStart()
            return await service().start_live(settings.scenario_id, settings.schedule_mode, settings.source_time)
        except (ValueError, KeyError) as error:
            raise HTTPException(409, str(error)) from error

    @application.get("/api/v1/system/status")
    async def status():
        return {"run": service().public_run(), **service().system()}

    @application.get("/metrics", response_class=PlainTextResponse)
    async def metrics():
        current = service()
        values = {key: value for key, value in current.system().items() if isinstance(value, (int, float))}
        values["database_ready"] = int(current.database_ready)
        return "\n".join(f"dispatcher_{name} {value}" for name, value in values.items()) + "\n"

    @application.get("/api/v1/events")
    async def events(request: Request, last_event_id: str | None = Header(None), since: int | None = Query(None)):
        current = service()
        try:
            cursor = int(last_event_id) if last_event_id is not None else since
        except ValueError as error:
            raise HTTPException(400, "Last-Event-ID must be an integer") from error

        async def stream():
            nonlocal cursor
            if cursor is None:
                cursor = current.sequence
                yield f"id: {cursor}\nevent: snapshot\ndata: {json.dumps(current.snapshot(), ensure_ascii=False)}\n\n"
            while not await request.is_disconnected():
                oldest = current.stream[0][0] if current.stream else current.sequence
                if cursor > current.sequence or cursor < oldest - 1:
                    cursor = current.sequence
                    yield f"id: {cursor}\nevent: reset\ndata: {{\"reason\":\"snapshot_required\"}}\n\n"
                pending = [(seq, item) for seq, item in current.stream if seq > cursor]
                for sequence, item in pending:
                    cursor = sequence
                    yield f"id: {sequence}\nevent: snapshot\ndata: {item}\n\n"
                try:
                    async with current.condition:
                        if cursor >= current.sequence:
                            await asyncio.wait_for(current.condition.wait(), timeout=15)
                except asyncio.TimeoutError:
                    yield ": heartbeat\n\n"
        return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    return application


app = create_app()
