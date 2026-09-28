"""
Unit tests for SignalGenerator.

Run with:  pytest tests/test_signal_generator.py -v
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config_loader import ConfigLoader
from src.signal_generator import Scenario, SignalGenerator


CONFIG_DIR = Path(__file__).parent.parent / "config"


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------

@pytest.fixture
def loader() -> ConfigLoader:
    return ConfigLoader(CONFIG_DIR)


@pytest.fixture
def generator(loader) -> SignalGenerator:
    return SignalGenerator(loader)


@pytest.fixture
def simple_scenario() -> Scenario:
    """20 s scenario: idle → accelerate → brake."""
    return Scenario(
        name="Simple Test",
        duration_s=20.0,
        base_rate_hz=100.0,
        vehicle_id="sedan_a",
        segments=[
            {"t_start": 0,  "t_end": 5,  "throttle_pct": 0,  "brake_pedal_pct": 0,  "gear": 1},
            {"t_start": 5,  "t_end": 15, "throttle_pct": 40, "brake_pedal_pct": 0,  "gear": 2},
            {"t_start": 15, "t_end": 20, "throttle_pct": 0,  "brake_pedal_pct": 50, "gear": 2},
        ],
    )


# ----------------------------------------------------------------------
# Scenario parsing
# ----------------------------------------------------------------------

def test_scenario_from_yaml(tmp_path: Path):
    p = tmp_path / "scn.yaml"
    p.write_text(
        "scenario:\n"
        "  name: 'test'\n"
        "  duration_s: 10\n"
        "  base_rate_hz: 100\n"
        "  vehicle_id: sedan_a\n"
        "  segments:\n"
        "    - {t_start: 0, t_end: 5, throttle_pct: 20, brake_pedal_pct: 0, gear: 1}\n"
    )
    s = Scenario.from_yaml(p)
    assert s.name == "test"
    assert s.duration_s == 10.0
    assert len(s.segments) == 1


def test_scenario_missing_field_raises(tmp_path: Path):
    p = tmp_path / "bad.yaml"
    p.write_text("scenario:\n  name: 'x'\n")
    with pytest.raises(ValueError, match="missing required field"):
        Scenario.from_yaml(p)


# ----------------------------------------------------------------------
# Shape and structure
# ----------------------------------------------------------------------

def test_output_shape(generator, simple_scenario):
    df = generator.generate(simple_scenario)
    assert len(df) == 2000
    assert "time_s" in df.columns
    for name in generator.signals:
        assert name in df.columns


def test_time_starts_at_zero(generator, simple_scenario):
    df = generator.generate(simple_scenario)
    assert df["time_s"].iloc[0] == pytest.approx(0.0)


def test_time_step_matches_rate(generator, simple_scenario):
    df = generator.generate(simple_scenario)
    dt = df["time_s"].diff().dropna().values
    assert np.allclose(dt, 0.01)


# ----------------------------------------------------------------------
# Physical plausibility
# ----------------------------------------------------------------------

def test_speed_increases_with_throttle(generator, simple_scenario):
    df = generator.generate(simple_scenario)
    v_early = df["vehicle_speed"].iloc[400]      # ~4 s
    v_late = df["vehicle_speed"].iloc[1400]      # ~14 s
    assert v_late > v_early + 5.0


def test_speed_decreases_with_braking(generator, simple_scenario):
    df = generator.generate(simple_scenario)
    v_before = df["vehicle_speed"].iloc[1400]    # ~14 s (still accelerating)
    v_after = df["vehicle_speed"].iloc[-1]       # ~19.99 s (after braking)
    assert v_after < v_before


def test_vehicle_speed_never_negative(generator, simple_scenario):
    df = generator.generate(simple_scenario)
    assert (df["vehicle_speed"] >= 0).all()


def test_brake_pressure_responds_to_pedal(generator, simple_scenario):
    df = generator.generate(simple_scenario)
    braking = df[(df["time_s"] >= 16) & (df["time_s"] <= 19.9)]
    assert (braking["brake_pressure"] > 0).all()


def test_no_brake_pressure_when_pedal_zero(generator, simple_scenario):
    df = generator.generate(simple_scenario)
    accel_window = df[(df["time_s"] >= 8) & (df["time_s"] <= 14)]
    assert (accel_window["brake_pressure"] < 1.0).all()


def test_engine_rpm_above_idle(generator, simple_scenario):
    df = generator.generate(simple_scenario)
    assert (df["engine_rpm"] >= 800.0 - 1.0).all()


def test_coolant_warms_up(generator, simple_scenario):
    df = generator.generate(simple_scenario)
    t_early = df["coolant_temperature"].iloc[0]
    t_late = df["coolant_temperature"].iloc[-1]
    assert t_early == pytest.approx(20.0, abs=0.5)
    assert t_late > t_early


# ----------------------------------------------------------------------
# Range clamping
# ----------------------------------------------------------------------

def test_all_signals_within_range(generator, simple_scenario):
    df = generator.generate(simple_scenario)
    for name, spec in generator.signals.items():
        lo, hi = spec["range"]
        assert (df[name] >= lo).all(), f"{name} below range"
        assert (df[name] <= hi).all(), f"{name} above range"


# ----------------------------------------------------------------------
# Determinism
# ----------------------------------------------------------------------

def test_output_is_deterministic(generator, simple_scenario):
    df1 = generator.generate(simple_scenario)
    df2 = generator.generate(simple_scenario)
    pd.testing.assert_frame_equal(df1, df2)


# ----------------------------------------------------------------------
# Engine RPM behavior
# ----------------------------------------------------------------------

def test_idle_equilibrium(generator):
    """With zero throttle input, RPM should stay near idle."""
    scenario = Scenario(
        name="Idle test",
        duration_s=10.0,
        base_rate_hz=100.0,
        vehicle_id="sedan_a",
        segments=[
            {"t_start": 0, "t_end": 10, "throttle_pct": 0,
             "brake_pedal_pct": 0, "gear": 1},
        ],
    )
    df = generator.generate(scenario)
    assert (df["engine_rpm"] >= 800.0 - 1.0).all()
    assert (df["engine_rpm"] < 1500.0).all()


def test_throttle_increases_rpm(generator):
    """Applying throttle should raise RPM above idle."""
    scenario = Scenario(
        name="Rev test",
        duration_s=5.0,
        base_rate_hz=100.0,
        vehicle_id="sedan_a",
        segments=[
            {"t_start": 0, "t_end": 1, "throttle_pct": 0,
             "brake_pedal_pct": 0, "gear": 1},
            {"t_start": 1, "t_end": 5, "throttle_pct": 60,
             "brake_pedal_pct": 0, "gear": 1},
        ],
    )
    df = generator.generate(scenario)
    idle_window = df[(df["time_s"] > 0.5) & (df["time_s"] < 1.0)]
    rev_window = df[df["time_s"] > 4.0]
    assert rev_window["engine_rpm"].mean() > idle_window["engine_rpm"].mean() + 500


def test_rpm_decays_after_throttle_release(generator):
    """After the throttle is released, RPM should decay toward idle."""
    scenario = Scenario(
        name="Decel test",
        duration_s=10.0,
        base_rate_hz=100.0,
        vehicle_id="sedan_a",
        segments=[
            {"t_start": 0, "t_end": 3, "throttle_pct": 60,
             "brake_pedal_pct": 0, "gear": 1},
            {"t_start": 3, "t_end": 10, "throttle_pct": 0,
             "brake_pedal_pct": 0, "gear": 1},
        ],
    )
    df = generator.generate(scenario)
    at_3s = df["engine_rpm"].iloc[300]
    at_9s = df["engine_rpm"].iloc[900]
    assert at_9s < at_3s



# ----------------------------------------------------------------------
# Crank position — range, rate, and wrap
# ----------------------------------------------------------------------

def test_crank_position_in_range(generator, simple_scenario):
    df = generator.generate(simple_scenario)
    assert (df["crank_position"] >= 0).all()
    assert (df["crank_position"] < 360).all()


def test_crank_position_rate_matches_rpm(generator, simple_scenario):
    """
    d(crank_position)/dt should be 6 × engine_rpm (deg/s).

    RPM × 360 / 60 = RPM × 6.
    """
    df = generator.generate(simple_scenario)
    dt = df["time_s"].diff().iloc[1]

    # Handle wraps: if the diff is negative, add 360 to unwrap
    deltas = df["crank_position"].diff()
    deltas = deltas.where(deltas >= 0, deltas + 360.0)

    rates = deltas / dt
    expected = df["engine_rpm"] * 6.0

    mask = rates.notna() & (rates > 0)

    np.testing.assert_allclose(
        rates[mask].values,
        expected[mask].values,
        rtol=0.02,
        atol=5.0,
    )


def test_crank_position_wraps_at_360(generator):
    """A high-RPM scenario should produce at least one wrap."""
    scenario = Scenario(
        name="Long rev",
        duration_s=10.0,
        base_rate_hz=100.0,
        vehicle_id="sedan_a",
        segments=[
            {"t_start": 0, "t_end": 10, "throttle_pct": 80,
             "brake_pedal_pct": 0, "gear": 1},
        ],
    )
    df = generator.generate(scenario)
    diffs = df["crank_position"].diff()
    assert (diffs < -100).any()


# ----------------------------------------------------------------------
# Engine torque multiplier fault input
# ----------------------------------------------------------------------

def test_engine_torque_multiplier_reduces_speed(generator):
    """A torque multiplier < 1 should reduce vehicle speed."""
    scenario = Scenario(
        name="Torque drop",
        duration_s=10.0,
        base_rate_hz=100.0,
        vehicle_id="sedan_a",
        segments=[
            {"t_start": 0, "t_end": 10, "throttle_pct": 60,
             "brake_pedal_pct": 0, "gear": 2},
        ],
    )
    clean = generator.generate(scenario)

    # Apply a 50% torque multiplier over the whole scenario
    n = len(clean)
    multiplier = np.full(n, 0.5)
    corrupted = generator.generate(
        scenario, faults={"engine_torque_multiplier": multiplier}
    )

    # Corrupted should be slower at the end
    v_clean = clean["vehicle_speed"].iloc[-1]
    v_corrupted = corrupted["vehicle_speed"].iloc[-1]
    assert v_corrupted < v_clean