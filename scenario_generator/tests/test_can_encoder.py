"""
Unit tests for CANEncoder.

Run with:  pytest tests/test_can_encoder.py -v
"""

from pathlib import Path

import pandas as pd
import pytest

from src.can_encoder import CANEncoder
from src.config_loader import ConfigLoader


CONFIG_DIR = Path(__file__).parent.parent / "config"


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------

@pytest.fixture
def loader() -> ConfigLoader:
    return ConfigLoader(CONFIG_DIR)


@pytest.fixture
def encoder_sedan(loader) -> CANEncoder:
    vehicles = loader.load_vehicles()
    signals = loader.load_signals()
    return CANEncoder(vehicles["sedan_a"], signals)


@pytest.fixture
def encoder_suv(loader) -> CANEncoder:
    vehicles = loader.load_vehicles()
    signals = loader.load_signals()
    return CANEncoder(vehicles["suv_b"], signals)


@pytest.fixture
def sample_signals() -> dict:
    """One row of realistic signal values (sedan_a)."""
    return {
        "engine_rpm": 2500.0,
        "throttle_position": 35.0,
        "coolant_temperature": 80.0,
        "brake_pressure": 20.0,
        "brake_pedal": 10.0,
        "wheel_speed": 50.0,
        "vehicle_speed": 50.0,
        "longitudinal_acceleration": -1.5,
    }


# ----------------------------------------------------------------------
# Initialization
# ----------------------------------------------------------------------

def test_encoder_initializes(encoder_sedan):
    assert encoder_sedan is not None
    assert 0x100 in encoder_sedan.mapped_can_ids()
    assert 0x1A0 in encoder_sedan.mapped_can_ids()
    assert 0x120 in encoder_sedan.mapped_can_ids()


def test_missing_can_mapping_raises(loader):
    signals = loader.load_signals()
    with pytest.raises(ValueError, match="can_mapping"):
        CANEncoder({}, signals)


def test_mapped_signals_match_config(encoder_sedan):
    mapped = set(encoder_sedan.mapped_signal_names())
    expected = {
        "engine_rpm", "throttle_position", "coolant_temperature",
        "brake_pressure", "brake_pedal", "wheel_speed",
        "vehicle_speed", "longitudinal_acceleration",
    }
    assert mapped == expected


# ----------------------------------------------------------------------
# Basic encoding
# ----------------------------------------------------------------------

def test_encode_returns_three_can_ids(encoder_sedan, sample_signals):
    frames = encoder_sedan.encode(sample_signals)
    assert set(frames.keys()) == {0x100, 0x1A0, 0x120}


def test_encode_payloads_are_8_bytes(encoder_sedan, sample_signals):
    frames = encoder_sedan.encode(sample_signals)
    for can_id, payload in frames.items():
        assert isinstance(payload, bytes)
        assert len(payload) == 8, f"CAN 0x{can_id:X} payload not 8 bytes"


def test_encode_engine_rpm_hand_computed(encoder_sedan, sample_signals):
    """engine_rpm = 2500, scale = 0.25 → raw = 10000 → bytes 0x27, 0x10"""
    frames = encoder_sedan.encode(sample_signals)
    payload = frames[0x100]
    assert payload[0:2] == bytes([0x27, 0x10])


def test_encode_throttle_hand_computed(encoder_sedan, sample_signals):
    """throttle_position = 35, scale = 1 → raw = 35 → byte 0x23"""
    frames = encoder_sedan.encode(sample_signals)
    payload = frames[0x100]
    assert payload[2] == 0x23


def test_encode_coolant_hand_computed(encoder_sedan, sample_signals):
    """coolant_temperature = 80, scale = 1, offset = -40 → raw = 120 → 0x78"""
    frames = encoder_sedan.encode(sample_signals)
    payload = frames[0x100]
    assert payload[3] == 0x78


def test_encode_negative_accel(encoder_sedan, sample_signals):
    """longitudinal_accel = -1.5, scale = 0.01, offset = -10 → raw = 850"""
    frames = encoder_sedan.encode(sample_signals)
    payload = frames[0x120]
    raw = int.from_bytes(payload[2:4], "big")
    assert raw == 850


# ----------------------------------------------------------------------
# Round-trip
# ----------------------------------------------------------------------

def test_round_trip_all_signals(encoder_sedan, sample_signals):
    """Encode then decode — values must match within resolution."""
    frames = encoder_sedan.encode(sample_signals)
    decoded = {}
    for can_id, payload in frames.items():
        decoded.update(encoder_sedan.decode(can_id, payload))

    for sig, original in sample_signals.items():
        recovered = decoded[sig]
        # Tolerance = signal resolution (scale)
        scale = encoder_sedan.can_mapping[sig]["scale"]
        assert abs(recovered - original) <= scale / 2 + 1e-9, (
            f"{sig}: {original} → {recovered}"
        )


def test_round_trip_suv(encoder_suv, sample_signals):
    frames = encoder_suv.encode(sample_signals)
    for can_id, payload in frames.items():
        assert len(payload) == 8
    # Spot-check one value
    rpm_payload = frames[0x200]
    raw = int.from_bytes(rpm_payload[0:2], "big")
    assert raw == int(round(2500 / 0.25))


# ----------------------------------------------------------------------
# Boundary values
# ----------------------------------------------------------------------

def test_encode_zero_values(encoder_sedan):
    zeros = {
        "engine_rpm": 0, "throttle_position": 0, "coolant_temperature": -40,
        "brake_pressure": 0, "brake_pedal": 0, "wheel_speed": 0,
        "vehicle_speed": 0, "longitudinal_acceleration": -10,
    }
    frames = encoder_sedan.encode(zeros)
    # All payloads should be 8 bytes
    for payload in frames.values():
        assert len(payload) == 8
    # Speed bytes should be zero (raw = 0)
    assert frames[0x120][0:2] == b"\x00\x00"


def test_encode_max_values(encoder_sedan):
    maxed = {
        "engine_rpm": 8000, "throttle_position": 100, "coolant_temperature": 150,
        "brake_pressure": 200, "brake_pedal": 100, "wheel_speed": 250,
        "vehicle_speed": 250, "longitudinal_acceleration": 5,
    }
    frames = encoder_sedan.encode(maxed)
    for payload in frames.values():
        assert len(payload) == 8


def test_out_of_range_is_clamped(encoder_sedan):
    """Values beyond declared range are clamped, not raised."""
    extreme = {
        "engine_rpm": 99999,           # above max
        "throttle_position": -50,      # below min
        "coolant_temperature": 500,    # above max
        "brake_pressure": 0,
        "brake_pedal": 0,
        "wheel_speed": 0,
        "vehicle_speed": 0,
        "longitudinal_acceleration": 0,
    }
    frames = encoder_sedan.encode(extreme)   # Should not raise
    # Clamping: RPM at max 8000 → raw = 32000 = 0x7D00
    assert frames[0x100][0:2] == (32000).to_bytes(2, "big")
    # Throttle at min 0 → raw = 0
    assert frames[0x100][2] == 0


# ----------------------------------------------------------------------
# Error handling
# ----------------------------------------------------------------------

def test_encode_missing_signal_raises(encoder_sedan):
    incomplete = {
        "engine_rpm": 2500,
        # Missing everything else
    }
    with pytest.raises(KeyError, match="Missing signals"):
        encoder_sedan.encode(incomplete)


# ----------------------------------------------------------------------
# DataFrame encoding
# ----------------------------------------------------------------------

def test_encode_dataframe(encoder_sedan, sample_signals):
    df = pd.DataFrame([sample_signals, sample_signals])
    frames_list = encoder_sedan.encode_dataframe(df)
    assert len(frames_list) == 2
    for frames in frames_list:
        assert set(frames.keys()) == {0x100, 0x1A0, 0x120}
        for payload in frames.values():
            assert len(payload) == 8


def test_encode_dataframe_ignores_extra_columns(encoder_sedan, sample_signals):
    """Columns not in can_mapping should be silently ignored."""
    data = dict(sample_signals)
    data["time_s"] = 1.23           # not a signal
    data["gear"] = 3                 # not mapped to CAN
    df = pd.DataFrame([data])
    frames_list = encoder_sedan.encode_dataframe(df)
    assert len(frames_list) == 1
    assert set(frames_list[0].keys()) == {0x100, 0x1A0, 0x120}


# ----------------------------------------------------------------------
# Round-trip via DataFrame (full pipeline smoke test)
# ----------------------------------------------------------------------

def test_dataframe_round_trip(encoder_sedan):
    """Encode a DataFrame, then decode back and compare."""
    df = pd.DataFrame([
        {
            "engine_rpm": 1200.0, "throttle_position": 10.0,
            "coolant_temperature": 25.0,
            "brake_pressure": 0.0, "brake_pedal": 0.0, "wheel_speed": 5.0,
            "vehicle_speed": 5.0, "longitudinal_acceleration": 0.5,
        },
        {
            "engine_rpm": 3500.0, "throttle_position": 60.0,
            "coolant_temperature": 70.0,
            "brake_pressure": 50.0, "brake_pedal": 25.0, "wheel_speed": 80.0,
            "vehicle_speed": 80.0, "longitudinal_acceleration": -2.0,
        },
    ])
    frames_list = encoder_sedan.encode_dataframe(df)

    # Decode each frame back
    for i, frames in enumerate(frames_list):
        decoded = {}
        for can_id, payload in frames.items():
            decoded.update(encoder_sedan.decode(can_id, payload))

        for sig in encoder_sedan.mapped_signal_names():
            scale = encoder_sedan.can_mapping[sig]["scale"]
            original = df.iloc[i][sig]
            recovered = decoded[sig]
            assert abs(recovered - original) <= scale / 2 + 1e-9, (
                f"Row {i}, signal {sig}: {original} → {recovered}"
            )