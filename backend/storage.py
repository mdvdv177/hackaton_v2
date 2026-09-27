"""Small relational store; JSON payloads preserve public contract versions."""
from __future__ import annotations

from datetime import datetime, timezone
from contextlib import nullcontext
from functools import wraps
from pathlib import Path
from threading import RLock
from typing import Any

from sqlalchemy import JSON, Float, String, Integer, Index, create_engine, select, func, update, delete, inspect, text
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column
from sqlalchemy.pool import StaticPool


class Base(DeclarativeBase):
    pass


class Run(Base):
    __tablename__ = "runs"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    created_at: Mapped[str] = mapped_column(String, index=True)
    payload: Mapped[dict] = mapped_column(JSON)


class Telemetry(Base):
    __tablename__ = "telemetry_events"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    run_id: Mapped[str] = mapped_column(String, index=True)
    tr_id: Mapped[str] = mapped_column(String, index=True)
    event_at: Mapped[float] = mapped_column(Float, index=True)
    payload: Mapped[dict] = mapped_column(JSON)
    stored_at: Mapped[float | None] = mapped_column(Float, nullable=True, default=lambda: datetime.now(timezone.utc).timestamp())


class Plan(Base):
    __tablename__ = "planned_visits"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    source: Mapped[str] = mapped_column(String, index=True)
    payload: Mapped[dict] = mapped_column(JSON)


class Prediction(Base):
    __tablename__ = "predictions"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    run_id: Mapped[str] = mapped_column(String, index=True)
    tr_id: Mapped[str] = mapped_column(String, index=True)
    payload: Mapped[dict] = mapped_column(JSON)
    stored_at: Mapped[float | None] = mapped_column(Float, nullable=True, default=lambda: datetime.now(timezone.utc).timestamp())


class Incident(Base):
    __tablename__ = "incidents"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    run_id: Mapped[str] = mapped_column(String, index=True)
    payload: Mapped[dict] = mapped_column(JSON)
    stored_at: Mapped[float | None] = mapped_column(Float, nullable=True, default=lambda: datetime.now(timezone.utc).timestamp())


class ObservedVisit(Base):
    __tablename__ = "observed_visits"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    run_id: Mapped[str] = mapped_column(String, index=True)
    payload: Mapped[dict] = mapped_column(JSON)


class Vehicle(Base):
    __tablename__ = "vehicles"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    payload: Mapped[dict] = mapped_column(JSON)


class Binding(Base):
    __tablename__ = "device_bindings"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    payload: Mapped[dict] = mapped_column(JSON)


class ModelVersion(Base):
    __tablename__ = "model_versions"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    payload: Mapped[dict] = mapped_column(JSON)


class SchemaMigration(Base):
    __tablename__ = "schema_migrations"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    applied_at: Mapped[str] = mapped_column(String)


class Scenario(Base):
    __tablename__ = "scenarios"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    created_at: Mapped[str] = mapped_column(String, index=True)
    payload: Mapped[dict] = mapped_column(JSON)


class Inbox(Base):
    """Additive migration: legacy telemetry is already applied and has no inbox row."""
    __tablename__ = "telemetry_inbox"
    ingest_seq: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_id: Mapped[str] = mapped_column(String, unique=True)
    run_id: Mapped[str] = mapped_column(String)
    eligible_at: Mapped[float] = mapped_column(Float)
    committed_at: Mapped[float] = mapped_column(Float)
    applied_at: Mapped[float | None] = mapped_column(Float, nullable=True)
    __table_args__ = (Index("ix_inbox_ready", "run_id", "applied_at", "eligible_at", "ingest_seq"),)


def serialized(method):
    """SQLite tests share one connection; PostgreSQL uses independent transactions."""
    @wraps(method)
    def call(self, *args, **kwargs):
        with self._mutex if self.engine.dialect.name == "sqlite" else nullcontext():
            return method(self, *args, **kwargs)
    return call


class Store:
    def __init__(self, url: str):
        options: dict[str, Any] = {"pool_pre_ping": True}
        if url.startswith("sqlite"):
            options["connect_args"] = {"check_same_thread": False}
            if ":memory:" in url:
                options["poolclass"] = StaticPool
            else:
                path = url.split("///", 1)[-1]
                Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(url, **options)
        self._mutex = RLock()
        # Migration 1 preserves the original tables; migration 2 only adds tables
        # and indexes, so it is safe both for an existing database and a clean one.
        Base.metadata.create_all(self.engine)
        # Version 3 records retention in wall time, independently of replay dates.
        # Existing rows inherit their run's creation time, never a fabricated
        # prediction publication timestamp. No source payload is rewritten.
        with self.engine.begin() as connection:
            for model in (Telemetry, Prediction, Incident):
                table = model.__tablename__
                if "stored_at" not in {c["name"] for c in inspect(connection).get_columns(table)}:
                    connection.execute(text(f"ALTER TABLE {table} ADD COLUMN stored_at DOUBLE PRECISION"))
                clock_sql = "CAST(strftime('%s', runs.created_at) AS REAL)" if self.engine.dialect.name == "sqlite" else "EXTRACT(EPOCH FROM CAST(runs.created_at AS TIMESTAMPTZ))"
                connection.execute(text(f"UPDATE {table} SET stored_at = COALESCE((SELECT {clock_sql} FROM runs WHERE runs.id = {table}.run_id), :now) WHERE stored_at IS NULL"),
                                   {"now": datetime.now(timezone.utc).timestamp()})
                connection.execute(text(f"CREATE INDEX IF NOT EXISTS ix_{table}_stored_at ON {table} (stored_at)"))
        with Session(self.engine) as session:
            for version in ("001_initial", "002_scenarios_durable_inbox", "003_wall_time_retention"):
                if session.get(SchemaMigration, version) is None:
                    session.add(SchemaMigration(id=version, applied_at=datetime.now(timezone.utc).isoformat()))
            session.commit()

    @serialized
    def healthy(self) -> bool:
        try:
            with self.engine.connect() as connection:
                connection.execute(select(1))
            return True
        except Exception:
            return False

    @serialized
    def save_run(self, payload: dict) -> None:
        with Session(self.engine) as session:
            session.merge(Run(id=payload["id"], created_at=payload["created_at"], payload=payload))
            session.commit()

    @serialized
    def latest_run(self) -> dict | None:
        with Session(self.engine) as session:
            row = session.scalars(select(Run).order_by(Run.created_at.desc()).limit(1)).first()
            return row.payload if row else None

    def _insert(self, model):
        from sqlalchemy.dialects.postgresql import insert as pg_insert
        from sqlalchemy.dialects.sqlite import insert as sqlite_insert
        return (sqlite_insert if self.engine.dialect.name == "sqlite" else pg_insert)(model)

    def _upsert(self, session, model, rows):
        for offset in range(0, len(rows), 200):
            statement = self._insert(model).values(rows[offset:offset + 200])
            changes = {column.name: getattr(statement.excluded, column.name)
                       for column in model.__table__.columns if column.name != "id"}
            session.execute(statement.on_conflict_do_update(index_elements=["id"], set_=changes))

    @serialized
    def put_many(self, model: type[Base], rows: list[dict]) -> None:
        if not rows:
            return
        with Session(self.engine) as session:
            self._upsert(session, model, rows)
            session.commit()

    @serialized
    def insert_telemetry(self, rows: list[dict]) -> None:
        if not rows:
            return
        with Session(self.engine) as session:
            for offset in range(0, len(rows), 200):
                session.execute(self._insert(Telemetry).values(rows[offset:offset + 200]).on_conflict_do_nothing(index_elements=["id"]))
            session.commit()

    @serialized
    def rows(self, model: type[Base], run_id: str | None = None, tr_id: str | None = None) -> list[dict]:
        query = select(model)
        if run_id is not None:
            query = query.where(model.run_id == run_id)
        if tr_id is not None:
            query = query.where(model.tr_id == tr_id)
        with Session(self.engine) as session:
            return [row.payload for row in session.scalars(query)]

    @serialized
    def recent_telemetry(self, run_id: str, cutoff: float) -> list[dict]:
        with Session(self.engine) as session:
            query = select(Telemetry).where(Telemetry.run_id == run_id, Telemetry.event_at >= cutoff).order_by(Telemetry.event_at)
            return [row.payload for row in session.scalars(query)]

    @serialized
    def recent_applied_telemetry(self, run_id: str, cutoff: float) -> list[dict]:
        with Session(self.engine) as session:
            query = (select(Telemetry.payload).outerjoin(Inbox, Inbox.event_id == Telemetry.id)
                     .where(Telemetry.run_id == run_id, Telemetry.event_at >= cutoff,
                            (Inbox.ingest_seq.is_(None) | Inbox.applied_at.is_not(None)))
                     .order_by(Telemetry.event_at))
            return list(session.scalars(query))

    @serialized
    def enqueue_telemetry(self, rows: list[dict]) -> list[str]:
        """Commit immutable telemetry and its work item together. Return new IDs only."""
        if not rows:
            return []
        from backend.data import timestamp
        unique = {row["id"]: row for row in rows}
        ids = []
        committed = datetime.now(timezone.utc).timestamp()
        with Session(self.engine) as session:
            values = list(unique.values())
            for offset in range(0, len(values), 200):
                statement = self._insert(Telemetry).values(values[offset:offset + 200])
                ids.extend(session.scalars(statement.on_conflict_do_nothing(index_elements=["id"]).returning(Telemetry.id)))
            for offset in range(0, len(ids), 200):
                pending = []
                for identifier in ids[offset:offset + 200]:
                    row = unique[identifier]
                    event = row["payload"]
                    pending.append({"event_id": identifier, "run_id": row["run_id"],
                                    "eligible_at": max(row["event_at"], timestamp(event.get("received_at") or event["event_time"]).timestamp()),
                                    "committed_at": committed})
                session.execute(self._insert(Inbox).values(pending))
            session.commit()
        return ids

    @serialized
    def pending_telemetry(self, run_id: str, at: float, limit: int = 500) -> list[dict]:
        with Session(self.engine) as session:
            query = (select(Telemetry.id, Telemetry.payload, Inbox.ingest_seq, Inbox.committed_at)
                     .join(Inbox, Inbox.event_id == Telemetry.id)
                     .where(Inbox.run_id == run_id, Inbox.applied_at.is_(None), Inbox.eligible_at <= at)
                     .order_by(Inbox.eligible_at, Inbox.ingest_seq).limit(limit))
            return [{"id": identifier, "payload": payload, "ingest_seq": seq, "committed_at": committed}
                    for identifier, payload, seq, committed in session.execute(query)]

    @serialized
    def inbox_stats(self, run_id: str, at: float) -> dict:
        with Session(self.engine) as session:
            pending = session.scalar(select(func.count()).select_from(Inbox).where(Inbox.run_id == run_id, Inbox.applied_at.is_(None)))
            ready, oldest = session.execute(select(func.count(), func.min(Inbox.eligible_at)).where(
                Inbox.run_id == run_id, Inbox.applied_at.is_(None), Inbox.eligible_at <= at)).one()
            return {"pending_count": pending, "ready_count": ready,
                    "oldest_ready_age_s": max(0, at - oldest) if oldest is not None else 0}

    @serialized
    def commit_applied(self, ids: list[str], run: dict, observed: list[dict] | None = None) -> None:
        """Checkpoint, visit evidence and consumed flags form one transaction."""
        with Session(self.engine) as session:
            self._upsert(session, Run, [{"id": run["id"], "created_at": run["created_at"], "payload": run}])
            self._upsert(session, ObservedVisit, observed or [])
            for offset in range(0, len(ids), 500):
                session.execute(update(Inbox).where(Inbox.run_id == run["id"], Inbox.event_id.in_(ids[offset:offset + 500]),
                                                    Inbox.applied_at.is_(None)).values(applied_at=datetime.now(timezone.utc).timestamp()))
            session.commit()

    @serialized
    def commit_predictions(self, predictions: list[dict], incidents: list[dict]) -> None:
        with Session(self.engine) as session:
            self._upsert(session, Prediction, predictions)
            self._upsert(session, Incident, incidents)
            session.commit()

    @serialized
    def save_scenario(self, payload: dict) -> dict:
        with Session(self.engine) as session:
            session.execute(self._insert(Scenario).values(id=payload["id"], created_at=datetime.now(timezone.utc).isoformat(), payload=payload)
                            .on_conflict_do_nothing(index_elements=["id"]))
            existing = session.get(Scenario, payload["id"])
            if existing.payload != payload:
                raise ValueError("Scenario version is immutable")
            session.commit()
        return payload

    @serialized
    def load_scenario(self, identifier: str) -> dict | None:
        with Session(self.engine) as session:
            row = session.get(Scenario, identifier)
            return row.payload if row else None

    @serialized
    def list_scenarios(self) -> list[dict]:
        with Session(self.engine) as session:
            return [row.payload for row in session.scalars(select(Scenario).order_by(Scenario.created_at.desc()).limit(200))]

    @serialized
    def recent_predictions(self, run_id: str, per_vehicle: int = 240) -> list[dict]:
        rank = func.row_number().over(partition_by=Prediction.tr_id, order_by=Prediction.payload["prediction_time"].as_string().desc())
        query = select(Prediction.payload.label("payload"), rank.label("rank")).where(Prediction.run_id == run_id).subquery()
        with Session(self.engine) as session:
            return list(session.scalars(select(query.c.payload).where(query.c.rank <= per_vehicle)))

    @serialized
    def incident_page(self, run_id: str, limit: int = 100, offset: int = 0) -> list[dict]:
        with Session(self.engine) as session:
            return list(session.scalars(select(Incident.payload).where(Incident.run_id == run_id)
                         .order_by(Incident.payload["prediction_time"].as_string().desc()).offset(offset).limit(min(limit, 500))))

    @serialized
    def prune(self, now: float, active_run_id: str | None = None, active_cutoff: float | None = None, limit: int = 500) -> dict:
        """Bounded wall-time retention; never delete pending work or active recovery window."""
        with Session(self.engine) as session:
            query = (select(Telemetry.id).outerjoin(Inbox, Telemetry.id == Inbox.event_id)
                     .where(Telemetry.stored_at < now - 86400,
                            Inbox.ingest_seq.is_(None) | Inbox.applied_at.is_not(None)))
            if active_run_id and active_cutoff is not None:
                query = query.where((Telemetry.run_id != active_run_id) | (Telemetry.event_at < active_cutoff))
            ids = list(session.scalars(query.limit(limit)))
            if ids:
                session.execute(delete(Inbox).where(Inbox.event_id.in_(ids)))
                session.execute(delete(Telemetry).where(Telemetry.id.in_(ids)))
            removed = 0
            for model in (Prediction, Incident):
                query = select(model.id).where(model.stored_at < now - 7 * 86400)
                if model is Incident:
                    query = query.where(Incident.payload["status"].as_string().in_(["resolved", "expired", "closed"]))
                old_ids = list(session.scalars(query.limit(limit)))
                if old_ids:
                    removed += session.execute(delete(model).where(model.id.in_(old_ids))).rowcount
            session.commit()
            return {"telemetry_deleted": len(ids), "results_deleted": removed}
