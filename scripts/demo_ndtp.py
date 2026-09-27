"""Send original test telemetry through TCP NDTP at 1x using a rebased live plan.

Run `.venv/bin/python scripts/demo_ndtp.py --start-stack`; Ctrl+C closes only
the sender. The application remains running. Dataset files are read-only.
"""
from __future__ import annotations

import argparse
import asyncio
from bisect import bisect_left
import contextlib
import csv
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

import httpx
from dotenv import dotenv_values

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.data import number, timestamp
from scripts.ndtp_sender import handshake, navigation
from scripts.stack import ROOT, compose, doctor

DEFAULT_SOURCE_TIME = "2026-01-06T07:00:00Z"
MAX_WARMUP_EVENTS = 2000
MAX_WARMUP_RATE = 100


@dataclass(frozen=True, slots=True)
class RecordedPacket:
    request: int
    peer: int
    event_at: float
    received_at: float
    lat: float
    lon: float
    speed: float
    heading: float
    altitude: float
    valid: bool

    def encode(self, offset_s: float) -> bytes:
        return navigation(self.peer, self.request, self.lat, self.lon, self.speed,
                          at=self.event_at + offset_s, valid=self.valid,
                          heading=self.heading, altitude=self.altitude)


def load_packets(dataset: Path) -> tuple[list[RecordedPacket], dict]:
    """Read only the supplied test plan and telemetry, never labels/fact times."""
    with (dataset / "test/schedule.csv").open(encoding="utf-8", newline="") as stream:
        vehicles = {row["tr_id"] for row in csv.DictReader(stream)}
    packets, ignored, missing = [], 0, 0
    with (dataset / "test/traffic.csv").open(encoding="utf-8", newline="") as stream:
        for index, row in enumerate(csv.DictReader(stream), 1):
            if row["tr_id"] not in vehicles:
                ignored += 1
                continue
            lat, lon, speed = (number(row.get(key)) for key in ("lat", "lon", "speed"))
            valid = (row.get("location_valid", "").lower() in {"true", "1"}
                     and lat is not None and lon is not None
                     and -90 <= lat <= 90 and -180 <= lon <= 180)
            if speed is None or not 0 <= speed <= 200:
                speed = 65535  # Mandatory wire field; normalizer maps out-of-range to null.
                missing += 1
            packets.append(RecordedPacket(index, int(row["unit_id"]),
                timestamp(row["event_time"]).timestamp(), timestamp(row["receive_time"]).timestamp(),
                lat if valid else 0, lon if valid else 0, speed,
                number(row.get("heading")) or 0, number(row.get("alt")) or 0, bool(valid)))
    packets.sort(key=lambda packet: (packet.received_at, packet.request))
    if not packets:
        raise ValueError("No test telemetry matches the test plan")
    return packets, {"source_packets": len(packets), "devices": len({p.peer for p in packets}),
                     "missing_speed_packets": missing, "unbound_packets_skipped": ignored,
                     "source_sha256": hashlib.sha256((dataset / "test/schedule.csv").read_bytes()).hexdigest()}


def partition_packets(packets: list[RecordedPacket], source_at: float, warmup_s: float,
                      max_events: int = MAX_WARMUP_EVENTS) -> tuple[list[RecordedPacket], list[RecordedPacket], int]:
    """Optional bounded past-only bootstrap; normal live rows retain receive ordering."""
    if not 0 <= warmup_s <= 900 or not 1 <= max_events <= MAX_WARMUP_EVENTS:
        raise ValueError("Warmup is limited to 900 seconds and 2000 events")
    arrivals = [packet.received_at for packet in packets]
    boundary = bisect_left(arrivals, source_at)
    if boundary == len(packets) or source_at < arrivals[0]:
        raise ValueError("source-time must fall within the supplied test receive-time range")
    history = [p for p in packets[bisect_left(arrivals, source_at - warmup_s):boundary]
               if source_at - warmup_s <= p.event_at <= source_at]
    return history[-max_events:], packets[boundary:], max(0, len(history) - max_events)


def manifest_offset(run: dict, source_at: float, expected_sha256: str) -> float:
    manifest = run.get("manifest") or {}
    if run.get("mode") != "ndtp" or run.get("schedule_mode") != "demo_rebased":
        raise ValueError("Backend did not start the requested demo_rebased NDTP run")
    if manifest.get("source_sha256") != expected_sha256 or manifest.get("demonstration") is not True:
        raise ValueError("Backend and sender test schedules differ; refusing to shift unrelated telemetry")
    offset = float(manifest["offset_s"])
    declared_source = timestamp(manifest["source_time"]).timestamp()
    anchor = timestamp(manifest["wall_anchor"]).timestamp()
    if (not math.isfinite(offset) or abs(declared_source - source_at) > .001
            or abs(anchor - declared_source - offset) > .001):
        raise ValueError("Backend manifest has an inconsistent clock offset")
    return offset


class RecordedSender:
    """One device per TCP connection; retries keep the original packet identity."""
    def __init__(self, host: str, port: int):
        self.host, self.port = host, port
        self.connections: dict[int, tuple] = {}
        self.sent = self.reconnects = 0

    async def drop(self, peer: int):
        connection = self.connections.pop(peer, None)
        if connection:
            connection[1].close()
            with contextlib.suppress(ConnectionError, OSError, asyncio.TimeoutError):
                await asyncio.wait_for(connection[1].wait_closed(), timeout=2)

    async def send(self, packet: RecordedPacket, offset_s: float):
        wire = packet.encode(offset_s)
        for attempt in range(3):
            try:
                current = self.connections.get(packet.peer)
                if current and (current[0].at_eof() or current[1].is_closing()
                                or time.monotonic() - current[2] > 75):
                    await self.drop(packet.peer)
                if packet.peer not in self.connections:
                    reader, writer = await asyncio.wait_for(asyncio.open_connection(self.host, self.port), timeout=3)
                    self.connections[packet.peer] = (reader, writer, time.monotonic())
                    writer.write(handshake(packet.peer))
                    self.reconnects += 1
                reader, writer, _ = self.connections[packet.peer]
                writer.write(wire)
                await asyncio.wait_for(writer.drain(), timeout=3)
                self.connections[packet.peer] = (reader, writer, time.monotonic())
                self.sent += 1
                return
            except (OSError, ConnectionError, asyncio.TimeoutError):
                await self.drop(packet.peer)
                if attempt == 2:
                    raise
                await asyncio.sleep(1)

    async def close(self):
        await asyncio.gather(*(self.drop(peer) for peer in list(self.connections)))


async def wait_until(at: float, stop: asyncio.Event, *, clock=time.time, sleep=asyncio.sleep) -> bool:
    """The wall clock is never accelerated, including packets with future event times."""
    while not stop.is_set():
        remaining = at - clock()
        if remaining <= 0:
            return True
        await sleep(min(remaining, .25))
    return False


def log(event: str, **fields):
    print(json.dumps({"event": event, **fields}, ensure_ascii=False), flush=True)


async def run_demo(args, packets: list[RecordedPacket], metadata: dict) -> dict:
    source_at = timestamp(args.source_time).timestamp()
    warmup, live, truncated = partition_packets(packets, source_at, args.warmup_seconds)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for name in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(name, stop.set)
    sender = RecordedSender(args.ndtp_host, args.ndtp_port)
    state = {"reason": "source_eof", "max_arrival_lag_s": 0.0}
    monitor_task = None
    started_at = time.monotonic()
    async with httpx.AsyncClient(base_url=args.backend_url, timeout=15,
                                 limits=httpx.Limits(keepalive_expiry=1)) as client:
        try:
            ready = await client.get("/health/ready")
            ready.raise_for_status()
            response = await client.post("/api/v1/live/start", json={"schedule_mode": "demo_rebased", "source_time": args.source_time})
            response.raise_for_status()
            run = response.json()["run"]
            offset = manifest_offset(run, source_at, metadata["source_sha256"])
            log("started", run_id=run["id"], dashboard=args.dashboard_url,
                manifest=run["manifest"], **metadata, live_packets=len(live),
                warmup_packets=len(warmup), warmup_truncated=truncated,
                wire_precision="whole UTC seconds, integer km/h and heading; absent heading=0",
                note="1x real clock; Ctrl+C stops this sender and leaves the application running")

            async def monitor():
                failures = 0
                while not stop.is_set():
                    if args.duration is not None and time.monotonic() - started_at >= args.duration:
                        state["reason"] = "duration_reached"
                        stop.set()
                        return
                    try:
                        snapshot = await client.get("/api/v1/snapshot")
                        snapshot.raise_for_status()
                        value = snapshot.json()
                        current = value.get("run") or {}
                        if current.get("id") != run["id"] or current.get("status") != "running":
                            state["reason"] = "application_run_changed"
                            stop.set()
                            return
                        failures = 0
                        model_count = sum(v.get("prediction", {}).get("source") == "model"
                                          for v in value.get("vehicles", []) if v.get("prediction"))
                        log("progress", sent=sender.sent, connected_devices=len(sender.connections),
                            model_predictions=model_count, max_arrival_lag_s=round(state["max_arrival_lag_s"], 3))
                    except (httpx.HTTPError, ValueError) as error:
                        failures += 1
                        log("api_unavailable", attempts=failures, detail=str(error))
                        if failures >= 3:
                            state["reason"] = "api_unavailable"
                            stop.set()
                            return
                    for _ in range(40):
                        if stop.is_set() or (args.duration is not None and time.monotonic() - started_at >= args.duration):
                            break
                        await asyncio.sleep(.25)

            monitor_task = asyncio.create_task(monitor())
            # Backfill is explicitly optional. It uses <=20s at 100 packets/s for
            # at most 2000 past events; original arrival times are NOT asserted.
            warmup_started = time.monotonic()
            for index, packet in enumerate(warmup):
                if stop.is_set():
                    break
                await sender.send(packet, offset)
                await asyncio.sleep(max(0, warmup_started + (index + 1) / args.warmup_rate - time.monotonic()))
            if warmup:
                log("warmup_finished", sent=min(sender.sent, len(warmup)),
                    elapsed_s=round(time.monotonic() - warmup_started, 3),
                    note="Backfilled history was received now; startup packets may arrive late")
            # A cap also bounds catch-up after initial backfill or a TCP retry.
            next_send_at = 0.0
            for packet in live:
                due = packet.received_at + offset
                if not await wait_until(max(due, next_send_at), stop):
                    break
                state["max_arrival_lag_s"] = max(state["max_arrival_lag_s"], time.time() - due)
                await sender.send(packet, offset)
                next_send_at = time.time() + 1 / MAX_WARMUP_RATE
            if stop.is_set() and state["reason"] == "source_eof":
                state["reason"] = "interrupted"
            return {**state, "run_id": run["id"], "sent": sender.sent, "connections_opened": sender.reconnects}
        finally:
            stop.set()
            if monitor_task:
                monitor_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await monitor_task
            await sender.close()
            for name in (signal.SIGINT, signal.SIGTERM):
                with contextlib.suppress(NotImplementedError):
                    loop.remove_signal_handler(name)


def arguments(argv: list[str] | None = None):
    configured = {**{key: value for key, value in dotenv_values(ROOT / ".env").items() if value is not None}, **os.environ}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-stack", action="store_true", help="Start existing Compose images before sending; leave stack running on exit")
    parser.add_argument("--dataset", type=Path, default=ROOT / "dataset")
    parser.add_argument("--backend-url", default=f"http://127.0.0.1:{configured.get('BACKEND_PORT', '8010')}")
    parser.add_argument("--dashboard-url", default=f"http://127.0.0.1:{configured.get('DASHBOARD_PORT', '8088')}")
    parser.add_argument("--ndtp-host", default="127.0.0.1")
    parser.add_argument("--ndtp-port", type=int, default=int(configured.get("NDTP_PORT", "9211")))
    parser.add_argument("--source-time", default=DEFAULT_SOURCE_TIME, help="UTC anchor within original test receive times")
    parser.add_argument("--duration", type=float, help="Optional sender duration in seconds; default runs until original data ends")
    parser.add_argument("--warmup-seconds", type=float, default=0, help="Optional past-only backfill, 0..900s, capped at 2000 events (default: none)")
    parser.add_argument("--warmup-rate", type=float, default=MAX_WARMUP_RATE, help="Backfill packets/s, 1..100 (default: 100)")
    args = parser.parse_args(argv)
    if args.duration is not None and (not math.isfinite(args.duration) or args.duration <= 0):
        parser.error("--duration must be a finite positive number")
    if not 0 <= args.warmup_seconds <= 900 or not 1 <= args.warmup_rate <= MAX_WARMUP_RATE:
        parser.error("--warmup-seconds must be 0..900 and --warmup-rate 1..100")
    return args


def main():
    args = arguments()
    try:
        packets, metadata = load_packets(args.dataset)
        # Validate the requested range before changing the running application.
        partition_packets(packets, timestamp(args.source_time).timestamp(), args.warmup_seconds)
        if args.start_stack:
            doctor()
            compose("up", "-d", "--no-build", "--wait", "--wait-timeout", "90")
        result = asyncio.run(run_demo(args, packets, metadata))
        log("stopped", **result, application_left_running=True)
        if result["reason"] == "api_unavailable":
            raise SystemExit(1)
    except KeyboardInterrupt:
        log("stopped", reason="interrupted", application_left_running=True)
    except (OSError, ValueError, httpx.HTTPError, KeyError) as error:
        log("failed", detail=str(error), application_left_running=True)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
