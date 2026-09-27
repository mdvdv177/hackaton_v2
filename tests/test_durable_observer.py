import pytest

from scripts.watch_durable_load import query_text, reconcile, validate_identity

RUN_ID = "18e4c2bf-4e8d-4d20-a094-d8ca5c35d0cf"
PROJECT = "transport-load-a95a7ef7"
ENV = {"BACKEND_PORT": "8010", "DASHBOARD_PORT": "8088", "NDTP_PORT": "9211", "EMULATOR_PORT": "18080"}


def sample():
    return {"run_id": RUN_ID, "transaction_read_only": "on", "telemetry_events": 408000,
            "inbox_total": 408000, "inbox_pending": 0, "inbox_applied": 408000,
            "checkpoint_telemetry_count": 408000}


def final():
    return {"project": PROJECT, "run_id": RUN_ID, "sender": {"sent": 408000}}


def test_exact_consistent_committed_snapshot_matches_final_sender():
    result = reconcile([sample()], final(), PROJECT, RUN_ID)
    assert result["passed"]
    assert result["matching_sample_indices"] == [0]


@pytest.mark.parametrize("field,value", [("inbox_pending", 1), ("inbox_applied", 407999),
                                         ("checkpoint_telemetry_count", 407999), ("telemetry_events", 408001),
                                         ("transaction_read_only", "off"), ("run_id", "another-run")])
def test_partial_or_mismatched_proof_fails(field, value):
    bad = {**sample(), field: value}
    assert not reconcile([bad], final(), PROJECT, RUN_ID)["passed"]


def test_report_identity_cannot_be_borrowed_from_another_project():
    assert not reconcile([sample()], {**final(), "project": "transport-load-deadbeef"}, PROJECT, RUN_ID)["passed"]


def test_project_uuid_and_environment_are_strict_before_sql():
    assert validate_identity(PROJECT, RUN_ID, ENV) == ENV
    with pytest.raises(ValueError):
        validate_identity(PROJECT + ";drop", RUN_ID, ENV)
    with pytest.raises(ValueError):
        query_text("';DROP TABLE runs;--")
    with pytest.raises(ValueError):
        validate_identity(PROJECT, RUN_ID, {**ENV, "COMPOSE_FILE": "other.yaml"})
    assert "REPEATABLE READ READ ONLY" in query_text(RUN_ID)
