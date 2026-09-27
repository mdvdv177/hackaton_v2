"""Real CSV demo invariants: one clock shift, missing values, bounded history."""

import asyncio
import csv
import hashlib
import time

import pytest

from backend.data import normalize_event
from backend.ndtp import FrameParser, start_server
from scripts.demo_ndtp import RecordedPacket, RecordedSender, arguments, load_packets, manifest_offset, partition_packets, wait_until
from scripts.ndtp_sender import handshake, navigation


def test_missing_and_invalid_wire_speeds_never_become_vehicle_speed():
    parser = FrameParser()
    missing = parser.feed(handshake(123) + navigation(123, 70000, 0, 0, 65535,
                                                    at=1767670500, valid=False))[0]
    event = normalize_event(missing)
    assert event["speed"] is None
    assert event["lat"] is None and event["lon"] is None
    assert not event["location_valid"]
    for value in (-1, 201, float("inf"), "", None):
        assert normalize_event({**missing, "speed": value})["speed"] is None
    for value in (0, 25, 200):
        assert normalize_event({**missing, "speed": value})["speed"] == value


def test_cli_uses_compose_env_ports_with_explicit_overrides(tmp_path, monkeypatch):
    monkeypatch.setattr("scripts.demo_ndtp.ROOT", tmp_path)
    (tmp_path / ".env").write_text("BACKEND_PORT=18765\nNDTP_PORT=19211\n", encoding="utf-8")
    monkeypatch.delenv("BACKEND_PORT", raising=False)
    monkeypatch.setenv("NDTP_PORT", "19212")
    args = arguments(["--dashboard-url", "http://other:18888"])
    assert args.backend_url == "http://127.0.0.1:18765"
    assert args.ndtp_port == 19212
    assert args.dashboard_url == "http://other:18888"


def packet(request, event_at, received_at, **overrides):
    return RecordedPacket(**{"request": request, "peer": 123, "event_at": event_at,
        "received_at": received_at, "lat": 55.75, "lon": 37.61, "speed": 20,
        "heading": 180, "altitude": 100, "valid": True, **overrides})


def test_original_files_are_read_only_and_receipt_order_is_preserved(tmp_path):
    directory = tmp_path / "test"
    directory.mkdir()
    plan = directory / "schedule.csv"
    plan.write_text("tr_id,time_fact_begin\nvehicle,DO_NOT_READ_AS_FEATURE\n", encoding="utf-8")
    traffic = directory / "traffic.csv"
    fields = ["tr_id", "unit_id", "event_time", "receive_time", "lat", "lon", "speed", "heading", "alt", "location_valid"]
    rows = [dict(zip(fields, values)) for values in [
        ["vehicle", "123", "2026-01-06T07:00:02Z", "2026-01-06T07:00:01Z", "55.75", "37.61", "20", "180", "100", "True"],
        ["vehicle", "123", "2026-01-06T07:00:00Z", "2026-01-06T07:00:00Z", "", "", "", "", "", "False"],
    ]]
    with traffic.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    before = {path: path.read_bytes() for path in (plan, traffic)}
    records, metadata = load_packets(tmp_path)
    assert [p.request for p in records] == [2, 1]
    assert records[0].speed == 65535 and not records[0].valid
    assert records[1].event_at > records[1].received_at  # Original early receipt remains early.
    assert metadata["source_sha256"] == hashlib.sha256(before[plan]).hexdigest()
    assert metadata["missing_speed_packets"] == 1
    assert all(path.read_bytes() == original for path, original in before.items())


def test_warmup_is_past_only_and_bounded_without_touching_live_rows():
    rows = [packet(i, i / 10, i / 10) for i in range(5000)]
    # A future-event row received early is not used as warmup history.
    rows.insert(3500, packet(99999, 999, 349.99))
    rows.sort(key=lambda p: p.received_at)
    warmup, live, omitted = partition_packets(rows, 400, 900)
    assert len(warmup) == 2000 and omitted == 2000
    assert all(p.event_at <= 400 and p.received_at < 400 for p in warmup)
    assert live == [p for p in rows if p.received_at >= 400]
    empty, same_live, omitted = partition_packets(rows, 400, 0)
    assert empty == [] and omitted == 0 and same_live == live
    with pytest.raises(ValueError, match="limited"):
        partition_packets(rows, 400, 901)


def test_manifest_is_the_only_clock_shift_and_wire_timestamp_is_rounded_down():
    source = 1767682800.25
    offset = 123456.789
    from backend.data import iso
    run = {"mode": "ndtp", "schedule_mode": "demo_rebased", "manifest": {
        "demonstration": True, "source_sha256": "a" * 64, "offset_s": offset,
        "source_time": iso(source), "wall_anchor": iso(source + offset)}}
    actual = manifest_offset(run, source, "a" * 64)
    record = packet(70000, source + 5, source + 7)
    parsed = FrameParser().feed(handshake(123) + record.encode(actual))[0]
    from backend.data import timestamp
    assert timestamp(parsed["event_time"]).timestamp() == int(record.event_at + offset)
    assert (record.received_at + actual) - (source + actual) == 7
    with pytest.raises(ValueError, match="schedules differ"):
        manifest_offset(run, source, "b" * 64)
    run["manifest"]["offset_s"] += 1
    with pytest.raises(ValueError, match="inconsistent"):
        manifest_offset(run, source, "a" * 64)


@pytest.mark.asyncio
async def test_receive_scheduler_waits_for_real_due_time_and_cancellation():
    clock = [100.0]
    sleeps = []

    async def sleep(seconds):
        sleeps.append(seconds)
        clock[0] += seconds

    stop = asyncio.Event()
    assert await wait_until(101.3, stop, clock=lambda: clock[0], sleep=sleep)
    assert clock[0] == pytest.approx(101.3)
    assert max(sleeps) <= .25
    stop.set()
    assert not await wait_until(110, stop, clock=lambda: clock[0], sleep=sleep)
    assert clock[0] == pytest.approx(101.3)


@pytest.mark.asyncio
async def test_recorded_sender_tcp_reconnect_keeps_identity_and_missing_data():
    events = []

    async def accept(event):
        events.append(normalize_event(event))

    server = await start_server(accept, "127.0.0.1", 0)
    sender = RecordedSender("127.0.0.1", server.sockets[0].getsockname()[1])
    record = packet(70000, int(time.time()) - 10, int(time.time()), speed=65535, valid=False)
    try:
        await sender.send(record, 0)
        await sender.drop(123)
        await sender.send(record, 0)
        for _ in range(50):
            if len(events) == 2:
                break
            await asyncio.sleep(.01)
        assert len(events) == 2
        assert events[0]["packet_id"] == events[1]["packet_id"]
        assert all(event["speed"] is None and not event["location_valid"] for event in events)
        assert sender.reconnects == 2
    finally:
        await sender.close()
        server.close()
        await server.wait_closed()
