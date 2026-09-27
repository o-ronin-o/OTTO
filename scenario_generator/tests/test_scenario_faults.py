"""
Unit tests for scenario fault declaration schema (Stage 2, Step 2.1a).

Run with:  pytest tests/test_scenario_faults.py -v
"""

from pathlib import Path
from textwrap import dedent

import pytest
import yaml

from src.config_loader import ConfigLoader
from src.signal_generator import Scenario


CONFIG_DIR = Path(__file__).parent.parent / "config"


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------

@pytest.fixture
def loader() -> ConfigLoader:
    return ConfigLoader(CONFIG_DIR)


@pytest.fixture
def fault_registry(loader) -> dict:
    return loader.load_fault_types()


def _write_scenario(tmp_path: Path, body: dict) -> Path:
    """Write a scenario YAML from a dict and return the path."""
    p = tmp_path / "test_scenario.yaml"
    with open(p, "w") as f:
        yaml.safe_dump({"scenario": body}, f)
    return p


def _base_scenario_body(**overrides) -> dict:
    body = {
        "name": "Test Scenario",
        "duration_s": 60,
        "base_rate_hz": 100,
        "vehicle_id": "sedan_a",
        "segments": [
            {"name": "idle",    "t_start": 0,  "t_end": 10, "throttle_pct": 0,  "brake_pedal_pct": 0,  "gear": 1},
            {"name": "accel",   "t_start": 10, "t_end": 30, "throttle_pct": 40, "brake_pedal_pct": 0,  "gear": 2},
            {"name": "brake",   "t_start": 30, "t_end": 60, "throttle_pct": 0,  "brake_pedal_pct": 40, "gear": 1},
        ],
    }
    body.update(overrides)
    return body


# ----------------------------------------------------------------------
# Backward compatibility
# ----------------------------------------------------------------------

def test_scenario_without_faults_has_empty_resolved_faults(loader, tmp_path):
    path = _write_scenario(tmp_path, _base_scenario_body())
    s = Scenario.from_yaml(path, fault_registry=loader.load_fault_types())
    assert s.resolved_faults == []


def test_scenario_without_faults_works_without_registry(tmp_path):
    path = _write_scenario(tmp_path, _base_scenario_body())
    s = Scenario.from_yaml(path)   # no registry
    assert s.resolved_faults == []


def test_unnamed_segments_still_valid(loader, tmp_path):
    body = _base_scenario_body(segments=[
        {"t_start": 0, "t_end": 30, "throttle_pct": 0, "brake_pedal_pct": 0, "gear": 1},
        {"t_start": 30, "t_end": 60, "throttle_pct": 40, "brake_pedal_pct": 0, "gear": 2},
    ])
    path = _write_scenario(tmp_path, body)
    s = Scenario.from_yaml(path, fault_registry=loader.load_fault_types())
    assert len(s.segments) == 2
    assert s.resolved_faults == []


# ----------------------------------------------------------------------
# Anchor resolution
# ----------------------------------------------------------------------

def test_fault_with_scenario_start_anchor(loader, tmp_path):
    body = _base_scenario_body(faults=[{
        "id": "f1",
        "type": "brake_pad_wear",
        "onset": {"anchor": "scenario_start", "offset_s": 15},
        "end":   {"anchor": "scenario_start", "offset_s": 25},
        "params": {"severity_start": 0.0, "severity_end": 0.3,
                   "progression": "linear"},
    }])
    path = _write_scenario(tmp_path, body)
    s = Scenario.from_yaml(path, fault_registry=loader.load_fault_types())
    assert len(s.resolved_faults) == 1
    f = s.resolved_faults[0]
    assert f.onset_s == 15.0
    assert f.end_s == 25.0


def test_fault_with_segment_start_anchor(loader, tmp_path):
    body = _base_scenario_body(faults=[{
        "id": "f1",
        "type": "brake_pad_wear",
        "onset": {"anchor": "segment_start", "segment": "accel", "offset_s": 5},
        "end":   {"anchor": "segment_start", "segment": "accel", "offset_s": 15},
        "params": {"severity_start": 0.0, "severity_end": 0.3},
    }])
    path = _write_scenario(tmp_path, body)
    s = Scenario.from_yaml(path, fault_registry=loader.load_fault_types())
    f = s.resolved_faults[0]
    assert f.onset_s == 15.0   # accel starts at 10, +5
    assert f.end_s == 25.0     # accel starts at 10, +15


def test_fault_with_segment_end_negative_offset(loader, tmp_path):
    body = _base_scenario_body(faults=[{
        "id": "f1",
        "type": "brake_pad_wear",
        "onset": {"anchor": "segment_end", "segment": "brake", "offset_s": -10},
        "end":   {"anchor": "segment_end", "segment": "brake", "offset_s": 0},
        "params": {"severity_start": 0.0, "severity_end": 0.3},
    }])
    path = _write_scenario(tmp_path, body)
    s = Scenario.from_yaml(path, fault_registry=loader.load_fault_types())
    f = s.resolved_faults[0]
    assert f.onset_s == 50.0   # brake ends at 60, -10
    assert f.end_s == 60.0


def test_fault_with_null_end_runs_to_scenario_end(loader, tmp_path):
    body = _base_scenario_body(faults=[{
        "id": "f1",
        "type": "brake_pad_wear",
        "onset": {"anchor": "scenario_start", "offset_s": 10},
        "end": None,
        "params": {"severity_start": 0.0, "severity_end": 0.3},
    }])
    path = _write_scenario(tmp_path, body)
    s = Scenario.from_yaml(path, fault_registry=loader.load_fault_types())
    f = s.resolved_faults[0]
    assert f.onset_s == 10.0
    assert f.end_s == 60.0


def test_multiple_non_overlapping_faults(loader, tmp_path):
    body = _base_scenario_body(faults=[
        {
            "id": "f1", "type": "brake_pad_wear",
            "onset": {"anchor": "scenario_start", "offset_s": 10},
            "end":   {"anchor": "scenario_start", "offset_s": 20},
            "params": {"severity_start": 0.0, "severity_end": 0.3},
        },
        {
            "id": "f2", "type": "brake_pad_wear",
            "onset": {"anchor": "scenario_start", "offset_s": 30},
            "end":   {"anchor": "scenario_start", "offset_s": 40},
            "params": {"severity_start": 0.0, "severity_end": 0.3},
        },
    ])
    path = _write_scenario(tmp_path, body)
    s = Scenario.from_yaml(path, fault_registry=loader.load_fault_types())
    assert len(s.resolved_faults) == 2
    assert s.resolved_faults[0].id == "f1"
    assert s.resolved_faults[1].id == "f2"


def test_default_parameter_filled_in(loader, tmp_path):
    body = _base_scenario_body(faults=[{
        "id": "f1", "type": "brake_pad_wear",
        "onset": {"anchor": "scenario_start", "offset_s": 0},
        "end":   {"anchor": "scenario_start", "offset_s": 10},
        "params": {"severity_start": 0.0, "severity_end": 0.3},
        # progression omitted, should default to "linear"
    }])
    path = _write_scenario(tmp_path, body)
    s = Scenario.from_yaml(path, fault_registry=loader.load_fault_types())
    assert s.resolved_faults[0].params["progression"] == "linear"


# ----------------------------------------------------------------------
# Errors — unknown / malformed
# ----------------------------------------------------------------------

def test_faults_without_registry_raises(loader, tmp_path):
    body = _base_scenario_body(faults=[{
        "id": "f1", "type": "brake_pad_wear",
        "onset": {"anchor": "scenario_start", "offset_s": 0},
        "end":   {"anchor": "scenario_start", "offset_s": 10},
        "params": {"severity_start": 0.0, "severity_end": 0.3},
    }])
    path = _write_scenario(tmp_path, body)
    with pytest.raises(ValueError, match="no.*fault_registry"):
        Scenario.from_yaml(path)   # no registry provided


def test_unknown_fault_type_raises(loader, tmp_path):
    body = _base_scenario_body(faults=[{
        "id": "f1", "type": "nonexistent_fault",
        "onset": {"anchor": "scenario_start", "offset_s": 0},
        "end":   {"anchor": "scenario_start", "offset_s": 10},
        "params": {},
    }])
    path = _write_scenario(tmp_path, body)
    with pytest.raises(ValueError, match="unknown fault type"):
        Scenario.from_yaml(path, fault_registry=loader.load_fault_types())


def test_unknown_segment_raises(loader, tmp_path):
    body = _base_scenario_body(faults=[{
        "id": "f1", "type": "brake_pad_wear",
        "onset": {"anchor": "segment_start", "segment": "nonexistent", "offset_s": 0},
        "end":   {"anchor": "scenario_start", "offset_s": 10},
        "params": {"severity_start": 0.0, "severity_end": 0.3},
    }])
    path = _write_scenario(tmp_path, body)
    with pytest.raises(ValueError, match="Unknown segment"):
        Scenario.from_yaml(path, fault_registry=loader.load_fault_types())


def test_invalid_anchor_raises(loader, tmp_path):
    body = _base_scenario_body(faults=[{
        "id": "f1", "type": "brake_pad_wear",
        "onset": {"anchor": "invalid_anchor", "offset_s": 0},
        "end":   {"anchor": "scenario_start", "offset_s": 10},
        "params": {"severity_start": 0.0, "severity_end": 0.3},
    }])
    path = _write_scenario(tmp_path, body)
    with pytest.raises(ValueError, match="Invalid anchor"):
        Scenario.from_yaml(path, fault_registry=loader.load_fault_types())


def test_onset_negative_raises(loader, tmp_path):
    body = _base_scenario_body(faults=[{
        "id": "f1", "type": "brake_pad_wear",
        "onset": {"anchor": "segment_start", "segment": "idle", "offset_s": -1},
        "end":   {"anchor": "scenario_start", "offset_s": 10},
        "params": {"severity_start": 0.0, "severity_end": 0.3},
    }])
    path = _write_scenario(tmp_path, body)
    with pytest.raises(ValueError, match="resolved onset"):
        Scenario.from_yaml(path, fault_registry=loader.load_fault_types())


def test_end_past_duration_raises(loader, tmp_path):
    body = _base_scenario_body(faults=[{
        "id": "f1", "type": "brake_pad_wear",
        "onset": {"anchor": "scenario_start", "offset_s": 0},
        "end":   {"anchor": "scenario_start", "offset_s": 999},
        "params": {"severity_start": 0.0, "severity_end": 0.3},
    }])
    path = _write_scenario(tmp_path, body)
    with pytest.raises(ValueError, match="> scenario duration"):
        Scenario.from_yaml(path, fault_registry=loader.load_fault_types())


def test_onset_after_end_raises(loader, tmp_path):
    body = _base_scenario_body(faults=[{
        "id": "f1", "type": "brake_pad_wear",
        "onset": {"anchor": "scenario_start", "offset_s": 20},
        "end":   {"anchor": "scenario_start", "offset_s": 10},
        "params": {"severity_start": 0.0, "severity_end": 0.3},
    }])
    path = _write_scenario(tmp_path, body)
    with pytest.raises(ValueError, match="must be <"):
        Scenario.from_yaml(path, fault_registry=loader.load_fault_types())


def test_unknown_parameter_raises(loader, tmp_path):
    body = _base_scenario_body(faults=[{
        "id": "f1", "type": "brake_pad_wear",
        "onset": {"anchor": "scenario_start", "offset_s": 0},
        "end":   {"anchor": "scenario_start", "offset_s": 10},
        "params": {"severity_start": 0.0, "severity_end": 0.3,
                   "not_a_real_param": 5},
    }])
    path = _write_scenario(tmp_path, body)
    with pytest.raises(ValueError, match="unknown parameter"):
        Scenario.from_yaml(path, fault_registry=loader.load_fault_types())


def test_missing_required_parameter_raises(loader, tmp_path):
    body = _base_scenario_body(faults=[{
        "id": "f1", "type": "brake_pad_wear",
        "onset": {"anchor": "scenario_start", "offset_s": 0},
        "end":   {"anchor": "scenario_start", "offset_s": 10},
        "params": {"severity_start": 0.0},   # missing severity_end
    }])
    path = _write_scenario(tmp_path, body)
    with pytest.raises(ValueError, match="missing required parameter"):
        Scenario.from_yaml(path, fault_registry=loader.load_fault_types())


def test_parameter_out_of_range_raises(loader, tmp_path):
    body = _base_scenario_body(faults=[{
        "id": "f1", "type": "brake_pad_wear",
        "onset": {"anchor": "scenario_start", "offset_s": 0},
        "end":   {"anchor": "scenario_start", "offset_s": 10},
        "params": {"severity_start": 0.0, "severity_end": 1.5},  # > 1.0
    }])
    path = _write_scenario(tmp_path, body)
    with pytest.raises(ValueError, match="outside range"):
        Scenario.from_yaml(path, fault_registry=loader.load_fault_types())


def test_invalid_progression_value_raises(loader, tmp_path):
    body = _base_scenario_body(faults=[{
        "id": "f1", "type": "brake_pad_wear",
        "onset": {"anchor": "scenario_start", "offset_s": 0},
        "end":   {"anchor": "scenario_start", "offset_s": 10},
        "params": {"severity_start": 0.0, "severity_end": 0.3,
                   "progression": "sigmoid"},   # not allowed
    }])
    path = _write_scenario(tmp_path, body)
    with pytest.raises(ValueError, match="not in"):
        Scenario.from_yaml(path, fault_registry=loader.load_fault_types())

def test_overlapping_same_signal_faults_allowed(loader, tmp_path):
    """Overlapping faults on the same signal are valid — they compose."""
    body = _base_scenario_body(faults=[
        {
            "id": "f1", "type": "brake_pad_wear",
            "onset": {"anchor": "scenario_start", "offset_s": 10},
            "end":   {"anchor": "scenario_start", "offset_s": 30},
            "params": {"severity_start": 0.0, "severity_end": 0.3},
        },
        {
            "id": "f2", "type": "brake_pad_wear",
            "onset": {"anchor": "scenario_start", "offset_s": 20},
            "end":   {"anchor": "scenario_start", "offset_s": 40},
            "params": {"severity_start": 0.0, "severity_end": 0.3},
        },
    ])
    path = _write_scenario(tmp_path, body)
    s = Scenario.from_yaml(path, fault_registry=loader.load_fault_types())
    assert len(s.resolved_faults) == 2

def test_duplicate_fault_ids_raises(loader, tmp_path):
    body = _base_scenario_body(faults=[
        {
            "id": "same_id", "type": "brake_pad_wear",
            "onset": {"anchor": "scenario_start", "offset_s": 0},
            "end":   {"anchor": "scenario_start", "offset_s": 10},
            "params": {"severity_start": 0.0, "severity_end": 0.3},
        },
        {
            "id": "same_id", "type": "brake_pad_wear",
            "onset": {"anchor": "scenario_start", "offset_s": 20},
            "end":   {"anchor": "scenario_start", "offset_s": 30},
            "params": {"severity_start": 0.0, "severity_end": 0.3},
        },
    ])
    path = _write_scenario(tmp_path, body)
    with pytest.raises(ValueError, match="duplicate fault id"):
        Scenario.from_yaml(path, fault_registry=loader.load_fault_types())


# ----------------------------------------------------------------------
# config_loader.validate_scenario
# ----------------------------------------------------------------------

def test_validate_scenario_passes_on_good(loader, tmp_path):
    path = _write_scenario(tmp_path, _base_scenario_body())
    errors = loader.validate_scenario(path)
    assert errors == []


def test_validate_scenario_catches_bad_fault(loader, tmp_path):
    body = _base_scenario_body(faults=[{
        "id": "f1", "type": "nonexistent",
        "onset": {"anchor": "scenario_start", "offset_s": 0},
        "end":   {"anchor": "scenario_start", "offset_s": 10},
        "params": {},
    }])
    path = _write_scenario(tmp_path, body)
    errors = loader.validate_scenario(path)
    assert len(errors) == 1
    assert "unknown fault type" in errors[0]


# ----------------------------------------------------------------------
# The seed scenario
# ----------------------------------------------------------------------

def test_seed_scenario_resolves(loader):
    """The actual seed scenario must load and resolve."""
    path = CONFIG_DIR / "scenarios" / "highway_brake_wear_gradual.yaml"
    s = Scenario.from_yaml(path, fault_registry=loader.load_fault_types())

    assert s.name == "Highway Cruise with Brake Wear"
    assert s.duration_s == 90.0
    assert s.vehicle_id == "suv_b"
    assert len(s.resolved_faults) == 2

    wear = next(f for f in s.resolved_faults if f.id == "wear_1")
    tamper = next(f for f in s.resolved_faults if f.id == "tamper_1")

    # wear_1: cruise starts at 18, +20 → 38; exit_2 ends at 82 → 82
    assert wear.onset_s == 38.0
    assert wear.end_s == 82.0
    assert wear.params["progression"] == "linear"

    # tamper_1: stop starts at 82, +3 → 85; stop ends at 90 → 90
    assert tamper.onset_s == 85.0
    assert tamper.end_s == 90.0
    assert tamper.params["progression"] == "step"


def test_seed_scenario_does_not_overlap(loader):
    """wear_1 [38,82] and tamper_1 [88,90] do not overlap."""
    path = CONFIG_DIR / "scenarios" / "highway_brake_wear_gradual.yaml"
    s = Scenario.from_yaml(path, fault_registry=loader.load_fault_types())
    wear = next(f for f in s.resolved_faults if f.id == "wear_1")
    tamper = next(f for f in s.resolved_faults if f.id == "tamper_1")
    assert wear.end_s <= tamper.onset_s