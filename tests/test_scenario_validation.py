"""Import boundary tests: geometry attribution and outcomes must not cross into plans."""
from copy import deepcopy
import json

import pytest

from backend.scenarios import validate_package


def package():
    return {
        "source_id": "strict-fixture", "timezone": "UTC", "device_bindings": {"123": "v"},
        "planned_visits": [
            {"tt_action_item_id": "a", "tr_id": "v", "time_begin": "2026-01-06T07:00:00Z", "geom": "POINT (37.6 55.7)", "route_id": "r", "trip_id": "trip", "stop_id": "s1"},
            {"tt_action_item_id": "b", "tr_id": "v", "time_begin": "2026-01-06T07:13:00Z", "geom": "POINT (37.61 55.71)", "route_id": "r", "trip_id": "trip", "stop_id": "s2"},
        ],
        "network": {
            "routes": [{"id": "r", "name": "Маршрут"}, {"id": "other"}],
            "trips": [{"id": "trip", "route_id": "r", "direction_id": "outbound"}],
            "stops": [{"id": "s1", "name": "Начало", "coordinates": [37.6, 55.7]}, {"id": "s2", "name": "Конец", "coordinates": [37.61, 55.71]}],
            "segments": [{"id": "road", "route_id": "r", "from_visit_id": "a", "to_visit_id": "b", "provenance": "Provided reference",
                          "geometry": {"type": "LineString", "coordinates": [[37.6, 55.7], [37.61, 55.71]]}}],
        },
    }


def test_unknown_outcomes_removed_everywhere_and_version_is_invariant():
    clean = package()
    expected = validate_package(clean)
    poisoned = deepcopy(clean)
    poisoned["time_fact_begin"] = "2099-01-01"
    poisoned["manifest"] = {"target_delay_s": 999}
    for visit in poisoned["planned_visits"]:
        visit.update(time_fact_begin="2099-01-01", target_delay_s=999)
    for rows in poisoned["network"].values():
        for row in rows:
            row.update(time_fact_begin="2099-01-01", labels={"target_delay_s": 999})
    actual = validate_package(poisoned)
    assert actual == expected
    encoded = json.dumps(actual)
    assert "time_fact_begin" not in encoded and "target_delay_s" not in encoded


@pytest.mark.parametrize("route_on_visit", [True, False])
def test_segment_must_match_declared_or_trip_route(route_on_visit):
    value = package()
    value["network"]["segments"][0]["route_id"] = "other"
    if not route_on_visit:
        for visit in value["planned_visits"]:
            visit.pop("route_id")
    with pytest.raises(ValueError, match="Segment route disagrees"):
        validate_package(value)


def test_physical_stop_must_match_explicit_visit_geometry():
    value = package()
    value["network"]["stops"][0]["coordinates"] = [38, 56]
    with pytest.raises(ValueError, match="physical stop"):
        validate_package(value)


def test_missing_visit_geometry_is_not_silently_inferred_from_stop_catalog():
    value = package()
    value["planned_visits"][0]["geom"] = ""
    value["network"]["segments"] = []
    result = validate_package(value)
    assert result["plan"][0]["geom"] == ""
    assert result["network"]["segments"] == []
    assert result["network"]["gaps"][0]["reason"] == "missing_geometry"


@pytest.mark.parametrize("shape", [None, [], "LineString", 4])
def test_malformed_geometry_is_validation_error(shape):
    value = package()
    value["network"]["segments"][0]["geometry"] = shape
    with pytest.raises(ValueError, match="GeoJSON LineString"):
        validate_package(value)


@pytest.mark.parametrize("manifest", [None, [], ["source_time"], {"source_time": {}}, {"wall_anchor": []},
    {"source_sha256": {"time_fact_begin": 5}}, {"source_sha256": "wrong"}, {"offset_s": True},
    {"offset_s": float("inf")}, {"offset_s": {"target_delay_s": 5}}, {"demonstration": {"target_delay_s": 5}}])
def test_manifest_accepts_only_typed_provenance(manifest):
    value = package()
    value["manifest"] = manifest
    with pytest.raises(ValueError, match="manifest"):
        validate_package(value)


def test_normalized_package_does_not_share_mutable_input_geometry():
    value = package()
    result = validate_package(value)
    unchanged = deepcopy(result)
    value["network"]["stops"][0]["coordinates"][0] = 38
    value["network"]["segments"][0]["geometry"]["coordinates"][0][0] = 38
    assert result == unchanged


def test_direction_binding_conflict_rejected():
    value = package()
    value["planned_visits"][0]["direction_id"] = "inbound"
    with pytest.raises(ValueError, match="direction"):
        validate_package(value)
