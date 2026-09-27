"""Protocol tests use binary frames, independent of the shipped Docker image."""

import asyncio
import struct

import pytest

from backend.ndtp import FrameParser, NAV, NPH, NPL, crc16, start_server, wire_crc


def frame(body, service=1, kind=101, peer=123, request=1):
    payload = NPH.pack(service, kind, 1, request) + body
    return NPL.pack(0x7E7E, len(payload), 0, wire_crc(payload), 2, peer, 0) + payload


def handshake(peer=123):
    return frame(struct.pack("<HHHIII", 6, 2, 0, peer, 65535, 0), 0, 100, peer)


def nav(bits=0xE0):
    return bytes([0, 0]) + NAV.pack(1767670500, 376173210, 557551234, bits, 100, 27, 30, 180, 22, 140, 9, 2)


def test_crc_standard_vector():
    assert crc16(b"123456789") == 0x4B37


def test_fragmented_coalesced_handshake_and_signed_coordinates():
    parser = FrameParser()
    stream = handshake() + frame(nav()) + frame(nav(0x80), request=2)
    events = []
    for offset in range(0, len(stream), 7):
        events += parser.feed(stream[offset:offset + 7])
    assert len(events) == 2
    assert events[0]["lat"] == pytest.approx(55.7551234)
    assert events[0]["lon"] == pytest.approx(37.617321)
    assert events[1]["lat"] < 0 and events[1]["lon"] < 0
    assert events[0]["speed"] == 27


def test_corruption_unknown_cells_and_missing_handshake():
    parser = FrameParser()
    assert parser.feed(frame(nav())) == []
    parser.feed(handshake())
    bad = bytearray(frame(nav()))
    bad[-1] ^= 1
    assert parser.feed(bytes(bad)) == []
    assert parser.feed(frame(nav() + bytes([99, 0, 1]))) == []
    assert len(parser.feed(b"noise" + frame(nav()))) == 1


def test_invalid_location_is_missing_not_zero():
    parser = FrameParser()
    event = parser.feed(handshake() + frame(nav(0x60)))[0]
    assert not event["location_valid"]
    assert event["lat"] is None and event["lon"] is None


@pytest.mark.asyncio
async def test_server_reconnect_and_connection_isolation():
    events = []

    async def consume(event):
        events.append(event)

    server = await start_server(consume, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        for _ in range(2):
            _, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.write(handshake() + frame(nav()))
            await writer.drain()
            writer.close()
            await writer.wait_closed()
        for _ in range(50):
            if len(events) == 2:
                break
            await asyncio.sleep(0.01)
        assert len(events) == 2
        assert events[0]["packet_id"] == events[1]["packet_id"]
    finally:
        server.close()
        await server.wait_closed()
