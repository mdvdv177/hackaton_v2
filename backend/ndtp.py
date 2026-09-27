"""Bounded, streaming NDTP 6.2 receiver for the supplied telemetry emulator.

The emulator does not require a protocol acknowledgement. Only cell layouts
documented with exact lengths are accepted; unknown cells reject the entire frame.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from datetime import datetime, timezone
import hashlib
import logging
import struct
from typing import Awaitable, Callable

LOGGER = logging.getLogger(__name__)
NPL = struct.Struct("<HHHHBIH")
NPH = struct.Struct("<HHHI")
NAV = struct.Struct("<IIIBBHHHHHBB")
CELL_SIZES = {0: 26, 2: 26, 8: 6, 10: 37, 15: 50, 16: 8}
MAX_FRAME = 65535 + NPL.size
STATS: Counter = Counter()


class ProtocolError(ValueError):
    """A frame violates the documented NDTP contract."""


def crc16(data: bytes) -> int:
    """CRC-16/Modbus, before the NPL byte swap."""
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ (0xA001 if crc & 1 else 0)
    return crc


def wire_crc(data: bytes) -> int:
    crc = crc16(data)
    return ((crc & 0xFF) << 8) | (crc >> 8)


class FrameParser:
    """One parser per TCP connection; buffers incomplete frames only."""

    def __init__(self) -> None:
        self.buffer = bytearray()
        self.unit_id: int | None = None

    def feed(self, data: bytes) -> list[dict]:
        self.buffer.extend(data)
        events = []
        while len(self.buffer) >= NPL.size:
            if self.buffer[:2] != b"~~":
                offset = self.buffer.find(b"~~", 1)
                if offset < 0:
                    self.buffer[:] = self.buffer[-1:] if self.buffer[-1:] == b"~" else b""
                    STATS["invalid_signature"] += 1
                    break
                del self.buffer[:offset]
                STATS["invalid_signature"] += 1
                if len(self.buffer) < NPL.size:
                    break
            _, size, flags, checksum, kind, peer, _ = NPL.unpack_from(self.buffer)
            if size < NPH.size or kind != 2 or flags != 0:
                del self.buffer[:2]
                STATS["invalid_header"] += 1
                continue
            total = NPL.size + size
            if len(self.buffer) < total:
                break
            frame = bytes(self.buffer[:total])
            del self.buffer[:total]
            payload = frame[NPL.size:]
            if wire_crc(payload) != checksum:
                STATS["crc_errors"] += 1
                continue
            try:
                event = self._decode(payload, peer)
            except (ProtocolError, struct.error, OverflowError, OSError) as exc:
                STATS["invalid_frames"] += 1
                LOGGER.warning("NDTP frame rejected: %s", exc)
                continue
            STATS["frames"] += 1
            if event is not None:
                events.append(event)
        if len(self.buffer) > MAX_FRAME:
            self.buffer.clear()
            raise ProtocolError("NDTP buffer limit exceeded")
        return events

    def _decode(self, payload: bytes, peer: int) -> dict | None:
        service, kind, _, request_id = NPH.unpack_from(payload)
        body = payload[NPH.size:]
        if (service, kind) == (0, 100):
            if len(body) != 18:
                raise ProtocolError("Invalid handshake length")
            major, minor, flags, unit, _, _ = struct.unpack("<HHHIII", body)
            if major != 6 or minor != 2 or unit != peer or flags != 0:
                raise ProtocolError("Unsupported handshake")
            if self.unit_id is not None and self.unit_id != peer:
                raise ProtocolError("Device changed on an established connection")
            self.unit_id = peer
            STATS["handshakes"] += 1
            return None
        if (service, kind) != (1, 101):
            raise ProtocolError("Unsupported NPH message")
        if self.unit_id != peer:
            raise ProtocolError("Realtime frame without matching handshake")
        offset, navigation = 0, None
        while offset < len(body):
            if offset + 2 > len(body):
                raise ProtocolError("Truncated cell header")
            cell_type, _ = body[offset:offset + 2]
            size = CELL_SIZES.get(cell_type)
            if size is None:
                raise ProtocolError(f"Unknown cell length: type {cell_type}")
            end = offset + 2 + size
            if end > len(body):
                raise ProtocolError("Truncated cell payload")
            if cell_type == 0:
                if navigation is not None or offset != 0:
                    raise ProtocolError("Navigation must occur once and first")
                navigation = NAV.unpack(body[offset + 2:end])
            offset = end
        if navigation is None:
            raise ProtocolError("Missing navigation cell")
        timestamp, lon, lat, bits, _, speed, _, course, _, alt, _, _ = navigation
        lon, lat = lon / 1e7, lat / 1e7
        lon *= 1 if bits & 0x40 else -1
        lat *= 1 if bits & 0x20 else -1
        valid = bool(bits & 0x80) and -180 <= lon <= 180 and -90 <= lat <= 90
        if course > 360:
            raise ProtocolError("Invalid navigation course")
        return {
            "unit_id": str(peer),
            "packet_id": f"ndtp:{peer}:{request_id}:{hashlib.sha256(payload).hexdigest()[:20]}",
            "event_time": datetime.fromtimestamp(timestamp, timezone.utc).isoformat(),
            "received_at": datetime.now(timezone.utc).isoformat(),
            "lat": lat if valid else None,
            "lon": lon if valid else None,
            "alt": alt,
            "speed": float(speed),
            "heading": float(course),
            "location_valid": valid,
            "source": "ndtp",
        }


def get_stats() -> dict:
    return dict(STATS)


async def start_server(
    callback: Callable[[dict], Awaitable[None]],
    host: str = "0.0.0.0",
    port: int = 9201,
) -> asyncio.Server:
    """Start the NDTP listener; callback applies binding and run-mode rules."""

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        STATS["connections"] += 1
        STATS["active_connections"] += 1
        parser = FrameParser()
        try:
            while True:
                chunk = await asyncio.wait_for(reader.read(8192), timeout=90)
                if not chunk:
                    break
                for event in parser.feed(chunk):
                    await callback(event)
                    STATS["events"] += 1
        except (asyncio.TimeoutError, ConnectionError, ProtocolError):
            STATS["connection_errors"] += 1
        except Exception:
            STATS["callback_errors"] += 1
            LOGGER.exception("NDTP ingestion failed")
        finally:
            STATS["active_connections"] -= 1
            writer.close()
            try:
                await writer.wait_closed()
            except ConnectionError:
                pass

    return await asyncio.start_server(handle, host, port, limit=MAX_FRAME)
