"""
Unit tests for Pipeline.

Run with:  pytest tests/test_pipeline.py -v
"""

from pathlib import Path

import pandas as pd
import pytest

from src.pipeline import Pipeline, PipelineResult


CONFIG_DIR = Path(__file__).parent.parent / "config"
SCENARIOS_DIR = CONFIG_DIR / "scenarios"


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------

@pytest.fixture
def pipeline() -> Pipeline:
    return Pipeline(config_dir=CONFIG_DIR)


@pytest.fixture
def city_drive_path() -> Path:
    return SCENARIOS_DIR / "city_drive.yaml"


@pytest.fixture
def highway_path() -> Path:
    return SCENARIOS_DIR / "highway_cruise.yaml"


# ----------------------------------------------------------------------
# Basic run
# ----------------------------------------------------------------------

def test_run_creates_both_files(pipeline, city_drive_path, tmp_path):
    result = pipeline.run(city_drive_path, output_dir=tmp_path)
    assert isinstance(result, PipelineResult)
    assert result.frames_path.exists()
    assert result.signals_path.exists()


def test_output_directory_structure(pipeline, city_drive_path, tmp_path):
    result = pipeline.run(city_drive_path, output_dir=tmp_path)
    # Should be under {tmp_path}/sedan_a/
    assert result.frames_path.parent.name == "sedan_a"
    assert result.signals_path.parent.name == "sedan_a"


def test_output_filenames_are_slugified(pipeline, city_drive_path, tmp_path):
    result = pipeline.run(city_drive_path, output_dir=tmp_path)
    # "City Drive" → "city_drive"
    assert result.frames_path.name == "city_drive.frames.csv"
    assert result.signals_path.name == "city_drive.signals.csv"


# ----------------------------------------------------------------------
# Frames CSV
# ----------------------------------------------------------------------

def test_frames_csv_has_correct_columns(pipeline, city_drive_path, tmp_path):
    result = pipeline.run(city_drive_path, output_dir=tmp_path)
    df = pd.read_csv(result.frames_path)
    assert list(df.columns) == ["timestamp_ms", "can_id", "payload_hex"]


def test_frames_csv_row_count(pipeline, city_drive_path, tmp_path):
    """City Drive is 60 s at 100 Hz. Expected: 6000 engine + 3000 brake + 1200 dynamics = 10200."""
    result = pipeline.run(city_drive_path, output_dir=tmp_path)
    assert result.n_frames == 10200


def test_frames_csv_is_sorted(pipeline, city_drive_path, tmp_path):
    result = pipeline.run(city_drive_path, output_dir=tmp_path)
    df = pd.read_csv(result.frames_path)
    assert df["timestamp_ms"].is_monotonic_increasing


def test_frames_csv_payload_format(pipeline, city_drive_path, tmp_path):
    result = pipeline.run(city_drive_path, output_dir=tmp_path)
    df = pd.read_csv(result.frames_path)
    # Every payload should be 16 hex chars (8 bytes)
    assert df["payload_hex"].str.len().eq(16).all()
    # Every can_id should start with 0x
    assert df["can_id"].str.startswith("0x").all()


# ----------------------------------------------------------------------
# Signals CSV
# ----------------------------------------------------------------------

def test_signals_csv_has_correct_columns(pipeline, city_drive_path, tmp_path):
    result = pipeline.run(city_drive_path, output_dir=tmp_path)
    df = pd.read_csv(result.signals_path)
    assert "time_s" in df.columns
    assert "timestamp_ms" in df.columns
    # Should have all 8 mapped signals
    for sig in ["engine_rpm", "throttle_position", "coolant_temperature",
                "brake_pressure", "brake_pedal", "wheel_speed",
                "vehicle_speed", "longitudinal_acceleration"]:
        assert sig in df.columns


def test_signals_csv_row_count(pipeline, city_drive_path, tmp_path):
    """60 s at 100 Hz = 6000 rows."""
    result = pipeline.run(city_drive_path, output_dir=tmp_path)
    assert result.n_signal_rows == 6000


def test_signals_csv_time_matches_timestamp(pipeline, city_drive_path, tmp_path):
    result = pipeline.run(city_drive_path, output_dir=tmp_path)
    df = pd.read_csv(result.signals_path)
    # time_s * 1000 should equal timestamp_ms (within rounding)
    assert (df["timestamp_ms"] - (df["time_s"] * 1000).round().astype(int)).abs().max() <= 1


# ----------------------------------------------------------------------
# Round-trip verification (frames CSV → signals)
# ----------------------------------------------------------------------

def test_frames_decode_matches_signals(pipeline, city_drive_path, tmp_path):
    """Decode the frames CSV at t=30000 ms and compare to signals CSV."""
    from src.can_encoder import CANEncoder
    result = pipeline.run(city_drive_path, output_dir=tmp_path)

    # Load frames and signals
    frames_df = pd.read_csv(result.frames_path)
    signals_df = pd.read_csv(result.signals_path)

    # Find the vehicle's encoder
    vehicle = pipeline.vehicles["sedan_a"]
    encoder = CANEncoder(vehicle, pipeline.signals_cfg)

    # Get all frames at t=30000 ms
    window = frames_df[frames_df["timestamp_ms"] == 30000]

    # Decode them
    decoded = {}
    for _, row in window.iterrows():
        can_id = int(row["can_id"], 16)
        payload = bytes.fromhex(row["payload_hex"])
        decoded.update(encoder.decode(can_id, payload))

    # Find the corresponding signals row
    signals_row = signals_df[signals_df["timestamp_ms"] == 30000].iloc[0]

    # Compare (allowing for quantization: 1 unit of resolution)
    for sig, value in decoded.items():
        expected = signals_row[sig]
        resolution = pipeline.signals_cfg[sig]["resolution"]
        assert abs(value - expected) <= resolution, (
            f"{sig}: decoded={value}, signals_csv={expected}"
        )


# ----------------------------------------------------------------------
# Multiple scenarios
# ----------------------------------------------------------------------

def test_two_scenarios_different_files(pipeline, city_drive_path, highway_path, tmp_path):
    r1 = pipeline.run(city_drive_path, output_dir=tmp_path)
    r2 = pipeline.run(highway_path, output_dir=tmp_path)

    assert r1.frames_path != r2.frames_path
    assert r1.signals_path != r2.signals_path
    # Different vehicles
    assert r1.frames_path.parent.name == "sedan_a"
    assert r2.frames_path.parent.name == "suv_b"


# ----------------------------------------------------------------------
# Error handling
# ----------------------------------------------------------------------

def test_unknown_vehicle_raises(pipeline, tmp_path):
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text(
        "scenario:\n"
        "  name: 'Bad'\n"
        "  duration_s: 10\n"
        "  base_rate_hz: 100\n"
        "  vehicle_id: nonexistent_vehicle\n"
        "  segments:\n"
        "    - {t_start: 0, t_end: 10, throttle_pct: 0, brake_pedal_pct: 0, gear: 1}\n"
    )
    with pytest.raises(KeyError, match="nonexistent_vehicle"):
        pipeline.run(bad_yaml, output_dir=tmp_path)


def test_missing_scenario_file_raises(pipeline, tmp_path):
    with pytest.raises(FileNotFoundError):
        pipeline.run(tmp_path / "does_not_exist.yaml", output_dir=tmp_path)


# ----------------------------------------------------------------------
# Determinism
# ----------------------------------------------------------------------

def test_pipeline_is_deterministic(pipeline, city_drive_path, tmp_path):
    out1 = tmp_path / "run1"
    out2 = tmp_path / "run2"

    r1 = pipeline.run(city_drive_path, output_dir=out1)
    r2 = pipeline.run(city_drive_path, output_dir=out2)

    frames1 = r1.frames_path.read_text()
    frames2 = r2.frames_path.read_text()
    assert frames1 == frames2

    signals1 = r1.signals_path.read_text()
    signals2 = r2.signals_path.read_text()
    assert signals1 == signals2