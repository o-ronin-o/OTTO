"""
Unit tests for EvolutionBuilder (Stage 2, Step 2.5).

Run with:  pytest tests/test_evolution_builder.py -v
"""

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.config_loader import ConfigLoader
from src.evolution_builder import EvolutionBuilder, EvolutionResult


CONFIG_DIR = Path(__file__).parent.parent / "config"
SCENARIOS_DIR = CONFIG_DIR / "scenarios"
EVOLUTIONS_DIR = CONFIG_DIR / "evolutions"


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------

@pytest.fixture
def loader() -> ConfigLoader:
    return ConfigLoader(CONFIG_DIR)


@pytest.fixture
def builder() -> EvolutionBuilder:
    return EvolutionBuilder(config_dir=CONFIG_DIR)


@pytest.fixture
def seed_evolution_path() -> Path:
    return EVOLUTIONS_DIR / "brake_wear_95days.yaml"


def _write_evolution(tmp_path: Path, body: dict) -> Path:
    """Write a minimal evolution YAML to tmp_path."""
    p = tmp_path / "test_evolution.yaml"
    with open(p, "w") as f:
        yaml.safe_dump({"evolution": body}, f)
    return p


def _base_evolution_body(**overrides) -> dict:
    body = {
        "name": "Test Evolution",
        "vehicle_id": "suv_b",
        "base_scenario": "highway_brake_wear_gradual",
        "recordings": [
            {"day": 1,  "severity": 0.0},
            {"day": 15, "severity": 0.1},
            {"day": 30, "severity": 0.2},
        ],
    }
    body.update(overrides)
    return body


# ----------------------------------------------------------------------
# Schema validation — positive cases
# ----------------------------------------------------------------------

def test_seed_evolution_loads(loader, seed_evolution_path):
    evo = loader.load_evolution(seed_evolution_path)
    assert evo["name"] == "Brake Wear Over 95 Days with Tampering"
    assert evo["vehicle_id"] == "suv_b"
    assert evo["base_scenario"] == "highway_brake_wear_gradual"
    assert len(evo["recordings"]) == 7


def test_seed_evolution_days_increasing(loader, seed_evolution_path):
    evo = loader.load_evolution(seed_evolution_path)
    days = [r["day"] for r in evo["recordings"]]
    assert days == sorted(days)
    assert len(set(days)) == len(days)


def test_seed_evolution_severities_valid(loader, seed_evolution_path):
    evo = loader.load_evolution(seed_evolution_path)
    for rec in evo["recordings"]:
        assert 0.0 <= rec["severity"] <= 1.0


# ----------------------------------------------------------------------
# Schema validation — negative cases
# ----------------------------------------------------------------------

def test_missing_name_raises(loader, tmp_path):
    body = _base_evolution_body()
    del body["name"]
    p = _write_evolution(tmp_path, body)
    with pytest.raises(ValueError, match="missing required field 'name'"):
        loader.load_evolution(p)


def test_missing_recordings_raises(loader, tmp_path):
    body = _base_evolution_body()
    del body["recordings"]
    p = _write_evolution(tmp_path, body)
    with pytest.raises(ValueError, match="missing required field 'recordings'"):
        loader.load_evolution(p)


def test_unknown_vehicle_raises(loader, tmp_path):
    body = _base_evolution_body(vehicle_id="unknown_vehicle")
    p = _write_evolution(tmp_path, body)
    with pytest.raises(ValueError, match="unknown vehicle_id"):
        loader.load_evolution(p)


def test_unknown_base_scenario_raises(loader, tmp_path):
    body = _base_evolution_body(base_scenario="nonexistent_scenario")
    p = _write_evolution(tmp_path, body)
    with pytest.raises(ValueError, match="not found"):
        loader.load_evolution(p)


def test_duplicate_recording_id_raises(loader, tmp_path):
    body = _base_evolution_body(recordings=[
        {"day": 1, "recording_id": "same_id", "severity": 0.0},
        {"day": 2, "recording_id": "same_id", "severity": 0.1},
    ])
    p = _write_evolution(tmp_path, body)
    with pytest.raises(ValueError, match="duplicate recording_id"):
        loader.load_evolution(p)


def test_severity_out_of_range_raises(loader, tmp_path):
    body = _base_evolution_body(recordings=[
        {"day": 1, "severity": 1.5},
    ])
    p = _write_evolution(tmp_path, body)
    with pytest.raises(ValueError, match="severity must be in"):
        loader.load_evolution(p)


def test_non_monotonic_days_raises(loader, tmp_path):
    body = _base_evolution_body(recordings=[
        {"day": 10, "severity": 0.0},
        {"day": 5,  "severity": 0.1},
    ])
    p = _write_evolution(tmp_path, body)
    with pytest.raises(ValueError, match="strictly increasing"):
        loader.load_evolution(p)


def test_base_scenario_without_faults_raises(loader, tmp_path):
    body = _base_evolution_body(base_scenario="city_drive")  # no faults
    p = _write_evolution(tmp_path, body)
    with pytest.raises(ValueError, match="does not declare any faults"):
        loader.load_evolution(p)


def test_invalid_onset_end_raises(loader, tmp_path):
    body = _base_evolution_body(recordings=[
        {"day": 1, "severity": 0.1, "onset_s": 50, "end_s": 30},
    ])
    p = _write_evolution(tmp_path, body)
    with pytest.raises(ValueError, match="end_s"):
        loader.load_evolution(p)


# ----------------------------------------------------------------------
# Full build — end-to-end
# ----------------------------------------------------------------------

def test_build_creates_expected_files(builder, seed_evolution_path, tmp_path):
    result = builder.build(seed_evolution_path, output_dir=tmp_path)

    evo_dir = result.output_dir
    assert evo_dir.exists()
    assert (evo_dir / "evolution.yaml").exists()
    assert (evo_dir / "metadata.json").exists()
    assert (evo_dir / "timeline.csv").exists()


def test_build_produces_one_recording_per_entry(
    builder, seed_evolution_path, tmp_path
):
    result = builder.build(seed_evolution_path, output_dir=tmp_path)
    assert len(result.recordings) == 7


def test_build_recording_files_exist(builder, seed_evolution_path, tmp_path):
    result = builder.build(seed_evolution_path, output_dir=tmp_path)
    for rec_result in result.recordings:
        assert rec_result.frames_path.exists()
        assert rec_result.signals_path.exists()
        assert rec_result.faults_path is not None
        assert rec_result.labels_path is not None
        assert rec_result.faults_path.exists()
        assert rec_result.labels_path.exists()


def test_recording_files_in_evolution_dir(builder, seed_evolution_path, tmp_path):
    """All recording files should live directly in the evolution directory."""
    result = builder.build(seed_evolution_path, output_dir=tmp_path)
    for rec_result in result.recordings:
        assert rec_result.frames_path.parent == result.output_dir
        assert rec_result.signals_path.parent == result.output_dir
        assert rec_result.faults_path.parent == result.output_dir
        assert rec_result.labels_path.parent == result.output_dir


def test_timeline_row_count(builder, seed_evolution_path, tmp_path):
    result = builder.build(seed_evolution_path, output_dir=tmp_path)
    timeline = pd.read_csv(result.timeline_path)
    assert len(timeline) == 7


def test_timeline_columns(builder, seed_evolution_path, tmp_path):
    result = builder.build(seed_evolution_path, output_dir=tmp_path)
    timeline = pd.read_csv(result.timeline_path)
    expected = {
        "recording_id", "day", "severity", "progression",
        "n_frames", "n_signal_rows", "fault_id", "fault_type",
    }
    assert set(timeline.columns) == expected


def test_metadata_structure(builder, seed_evolution_path, tmp_path):
    result = builder.build(seed_evolution_path, output_dir=tmp_path)
    with open(result.metadata_path) as f:
        meta = json.load(f)

    assert meta["evolution_name"] == "Brake Wear Over 95 Days with Tampering"
    assert meta["vehicle_id"] == "suv_b"
    assert meta["base_scenario"] == "highway_brake_wear_gradual"
    assert meta["n_recordings"] == 7
    assert len(meta["recording_ids"]) == 7
    assert "severity_trajectory" in meta
    assert len(meta["severity_trajectory"]) == 7


def test_severity_trajectory_matches_spec(builder, seed_evolution_path, tmp_path):
    result = builder.build(seed_evolution_path, output_dir=tmp_path)
    with open(result.metadata_path) as f:
        meta = json.load(f)

    traj = meta["severity_trajectory"]
    assert traj["day_01"] == 0.00
    assert traj["day_15"] == 0.05
    assert traj["day_30"] == 0.10
    assert traj["day_60"] == 0.20
    assert traj["day_90"] == 0.35
    assert traj["day_91"] == 0.60
    assert traj["day_95"] == 0.05


def test_severity_zero_recording_has_labels(builder, seed_evolution_path, tmp_path):
    """Even with severity 0, labels.csv should be written (uniform treatment)."""
    result = builder.build(seed_evolution_path, output_dir=tmp_path)
    day_01 = next(r for r in result.recordings
                  if r.applied_faults and r.applied_faults[0].fault_id == "day_01")
    assert day_01.labels_path is not None
    assert day_01.labels_path.exists()

    labels = pd.read_csv(day_01.labels_path)
    assert (labels["is_fault"] == 0).all()  # no active fault


def test_sub_window_recording(builder, seed_evolution_path, tmp_path):
    """Day 91 has onset=45, end=90. Verify the fault is active only then."""
    result = builder.build(seed_evolution_path, output_dir=tmp_path)
    day_91 = next(r for r in result.recordings
                  if r.applied_faults and r.applied_faults[0].fault_id == "day_91")

    labels = pd.read_csv(day_91.labels_path)
    labels["t_s"] = labels["timestamp_ms"] / 1000.0

    before = labels[labels["t_s"] < 45.0]
    after = labels[labels["t_s"] >= 45.0]

    assert (before["is_fault"] == 0).all()
    assert (after["is_fault"] == 1).all()


def test_evolution_yaml_is_copied(builder, seed_evolution_path, tmp_path):
    result = builder.build(seed_evolution_path, output_dir=tmp_path)
    copied = result.output_dir / "evolution.yaml"
    assert copied.exists()

    src = seed_evolution_path.read_text()
    dst = copied.read_text()
    assert src == dst


# ----------------------------------------------------------------------
# Cross-recording consistency
# ----------------------------------------------------------------------

def test_all_recordings_use_same_vehicle(builder, seed_evolution_path, tmp_path):
    result = builder.build(seed_evolution_path, output_dir=tmp_path)
    vehicles = {r.scenario.vehicle_id for r in result.recordings}
    assert vehicles == {"suv_b"}


def test_all_recordings_have_same_duration(builder, seed_evolution_path, tmp_path):
    result = builder.build(seed_evolution_path, output_dir=tmp_path)
    durations = {r.scenario.duration_s for r in result.recordings}
    assert len(durations) == 1


def test_fault_id_equals_recording_id(builder, seed_evolution_path, tmp_path):
    result = builder.build(seed_evolution_path, output_dir=tmp_path)
    for rec_result in result.recordings:
        assert len(rec_result.applied_faults) == 1
        applied = rec_result.applied_faults[0]
        expected_id = rec_result.frames_path.stem.split(".")[0]
        assert applied.fault_id == expected_id


def test_severity_reflects_in_labels(builder, seed_evolution_path, tmp_path):
    """The recorded severity should match what appears in each labels file."""
    result = builder.build(seed_evolution_path, output_dir=tmp_path)
    for rec_result in result.recordings:
        applied = rec_result.applied_faults[0]
        # Read the fault event file to confirm severity
        faults_df = pd.read_csv(rec_result.faults_path)
        assert len(faults_df) == 1
        assert abs(faults_df["severity_start"].iloc[0] - applied.severity_start) < 1e-9
        assert abs(faults_df["severity_end"].iloc[0] - applied.severity_end) < 1e-9


def test_increasing_severity_produces_more_faulty_frames(
    builder, seed_evolution_path, tmp_path
):
    """
    Higher severity should not reduce the fault coverage.
    (Full-window recordings: severity does not change is_fault count,
    but corrupted signals should be more divergent.)
    """
    result = builder.build(seed_evolution_path, output_dir=tmp_path)
    # For full-window recordings, all frames are faulty regardless of severity.
    # Just confirm the count is stable across the increasing-severity days.
    full_window_ids = ["day_01", "day_15", "day_30", "day_60", "day_90", "day_95"]
    for rec_result in result.recordings:
        rec_id = rec_result.frames_path.stem.split(".")[0]
        if rec_id not in full_window_ids:
            continue
        # Skip severity=0 (day_01) — it has no faults but is still full-window
        if rec_id == "day_01":
            continue
        labels = pd.read_csv(rec_result.labels_path)
        assert (labels["is_fault"] == 1).all()


# ----------------------------------------------------------------------
# Determinism
# ----------------------------------------------------------------------

def test_build_is_deterministic(builder, seed_evolution_path, tmp_path):
    out1 = tmp_path / "run1"
    out2 = tmp_path / "run2"
    b1 = EvolutionBuilder(config_dir=CONFIG_DIR)
    b2 = EvolutionBuilder(config_dir=CONFIG_DIR)

    r1 = b1.build(seed_evolution_path, output_dir=out1)
    r2 = b2.build(seed_evolution_path, output_dir=out2)

    # Same recording IDs
    ids1 = [r.scenario.name for r in r1.recordings]
    ids2 = [r.scenario.name for r in r2.recordings]
    assert ids1 == ids2

    # Same frame counts
    frames1 = [r.n_frames for r in r1.recordings]
    frames2 = [r.n_frames for r in r2.recordings]
    assert frames1 == frames2

    # Same metadata severity trajectory
    with open(r1.metadata_path) as f:
        m1 = json.load(f)
    with open(r2.metadata_path) as f:
        m2 = json.load(f)
    assert m1["severity_trajectory"] == m2["severity_trajectory"]


# ----------------------------------------------------------------------
# New evolutions: torque_drops_90days, crank_sensor_freeze_90days
# ----------------------------------------------------------------------

def test_torque_drops_evolution_loads(loader):
    path = CONFIG_DIR / "evolutions" / "torque_drops_90days.yaml"
    evo = loader.load_evolution(path)
    assert evo["name"] == "Intermittent Torque Drops Over 90 Days"
    assert evo["base_scenario"] == "highway_cruise_torque_drop"
    assert len(evo["recordings"]) == 6


def test_crank_sensor_freeze_evolution_loads(loader):
    path = CONFIG_DIR / "evolutions" / "crank_sensor_freeze_90days.yaml"
    evo = loader.load_evolution(path)
    assert evo["name"] == "Crank Sensor Freeze Over 90 Days"
    assert evo["base_scenario"] == "highway_cruise_crank_sensor_freeze"
    assert len(evo["recordings"]) == 6


def test_torque_drops_evolution_builds(builder):
    path = CONFIG_DIR / "evolutions" / "torque_drops_90days.yaml"
    result = builder.build(path, output_dir=Path("data_test"))
    assert len(result.recordings) == 6
    # Verify that severity maps to drop_magnitude
    for rec_result in result.recordings:
        if rec_result.applied_faults:
            a = rec_result.applied_faults[0]
            assert a.fault_type == "crankshaft_torque_drop"
    # Cleanup
    import shutil
    shutil.rmtree("data_test", ignore_errors=True)


def test_crank_sensor_freeze_evolution_builds(builder):
    path = CONFIG_DIR / "evolutions" / "crank_sensor_freeze_90days.yaml"
    result = builder.build(path, output_dir=Path("data_test"))
    assert len(result.recordings) == 6
    for rec_result in result.recordings:
        if rec_result.applied_faults:
            a = rec_result.applied_faults[0]
            assert a.fault_type == "crank_sensor_failure"
    import shutil
    shutil.rmtree("data_test", ignore_errors=True)