"""Deterministic TCP NDTP source for integration/capacity checks; no hidden truth in API."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import math
import struct
import time

from backend.ndtp import NAV, NPH, NPL, wire_crc


def frame(body: bytes, peer: int, request: int, service: int = 1, kind: int = 101) -> bytes:
    payload = NPH.pack(service, kind, 1, request % (2 ** 32)) + body
    return NPL.pack(0x7E7E, len(payload), 0, wire_crc(payload), 2, peer, 0) + payload


def handshake(peer: int) -> bytes:
    return frame(struct.pack("<HHHIII", 6, 2, 0, peer, 65535, 0), peer, 0, 0, 100)


def navigation(peer: int, request: int, lat: float, lon: float, speed: float, at: float | None = None,
               *, valid: bool = True, heading: float = 0, altitude: float = 100) -> bytes:
    bits = (0x80 if valid else 0) | (0x20 if lat >= 0 else 0) | (0x40 if lon >= 0 else 0)
    nav = NAV.pack(int(time.time() if at is None else at), round(abs(lon) * 1e7), round(abs(lat) * 1e7),
                   bits, 100, max(0, min(65535, round(speed))), max(0, min(65535, round(speed))),
                   max(0, min(360, round(heading))), 0, max(0, min(65535, round(altitude))), 9, 2)
    return frame(bytes([0, 0]) + nav, peer, request)


def load_scenario(vehicles: int = 100, duration_s: int = 3600, anchor: datetime | None = None) -> dict:
    """Explicit synthetic demonstration, with 90s lateness and a stop every 180s."""
    anchor = anchor or datetime.now(timezone.utc)
    visits = []
    for vehicle in range(vehicles):
        for index in range(math.ceil((duration_s + 1200) / 180) + 1):
            lat, lon = position(vehicle, index * 180)[0:2]
            visits.append({"tt_action_item_id": f"load-{vehicle}-{index}", "tr_id": f"load-{vehicle:03}",
                "time_begin": (anchor + timedelta(seconds=index * 180 - 90)).isoformat(),
                "geom": f"POINT ({lon} {lat})", "building_address": f"Демо-остановка {index + 1}"})
    return {"schema_version": "1.0", "source_id": f"load-{anchor.strftime('%Y%m%dT%H%M%S%f')}",
            "name": f"Демонстрационная нагрузка · {vehicles} ТС", "timezone": "UTC",
            "planned_visits": visits, "device_bindings": {str(900000 + i): f"load-{i:03}" for i in range(vehicles)},
            "manifest": {"demonstration": True, "wall_anchor": anchor.isoformat()}}


def position(vehicle: int, elapsed: float) -> tuple[float, float, float]:
    step = max(0, elapsed) / 180
    segment, phase = int(step), step % 1
    progress = 0 if phase < 1 / 9 else (phase - 1 / 9) / (8 / 9)
    return 55.60 + (segment + progress) * .003, 37.55 + vehicle * .0002, (0 if progress == 0 else 7.5)


class FleetSender:
    def __init__(self, host: str, port: int, vehicles: int, anchor: float):
        self.host, self.port, self.vehicles, self.anchor = host, port, vehicles, anchor
        self.sent = 0
        self.connections = 0
        self.failures = 0
        self.enabled = True
        self.requests = [0] * vehicles
        self.writers = {}

    async def tick(self, elapsed: float):
        if not self.enabled:
            return
        async def send(index):
            peer = 900000 + index
            try:
                if index not in self.writers:
                    _, writer = await asyncio.wait_for(asyncio.open_connection(self.host, self.port), timeout=3)
                    self.writers[index] = writer
                    writer.write(handshake(peer))
                    self.connections += 1
                writer = self.writers[index]
                self.requests[index] += 1
                writer.write(navigation(peer, self.requests[index], *position(index, elapsed)))
                await asyncio.wait_for(writer.drain(), timeout=3)
                self.sent += 1
            except (OSError, asyncio.TimeoutError, ConnectionError):
                self.failures += 1
                writer = self.writers.pop(index, None)
                if writer:
                    writer.close()
        await asyncio.gather(*(send(index) for index in range(self.vehicles)))

    async def disconnect(self):
        writers, self.writers = self.writers, {}
        for writer in writers.values():
            writer.close()
        await asyncio.gather(*(writer.wait_closed() for writer in writers.values()), return_exceptions=True)
