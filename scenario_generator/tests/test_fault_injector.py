"""
Unit tests for SignalFaultInjector (Stage 2, Step 2.1b).

Run with:  pytest tests/test_fault_injector.py -v
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config_loader import ConfigLoader
from src.fault_injector import (
    SignalFaultInjector,
    AppliedFault,
    _compute_severity_array,
    _progression_value,
)
from src.signal_generator import ResolvedFault, Scenario, SignalGenerator


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
def injector(loader, generator) -> SignalFaultInjector:
    return SignalFaultInjector(loader, generator)


@pytest.fixture
def simple_scenario() -> Scenario:
    """A 20-second scenario with an acceleration phase and a braking phase."""
    return Scenario(
        name="Simple",
        duration_s=20.0,
        base_rate_hz=100.0,
        vehicle_id="sedan_a",
        segments=[
            {"t_start": 0,  "t_end": 5,  "throttle_pct": 40, "brake_pedal_pct": 0,  "gear": 1},
            {"t_start": 5,  "t_end": 15, "throttle_pct": 0,  "brake_pedal_pct": 30, "gear": 2},
            {"t_start": 15, "t_end": 20, "throttle_pct": 0,  "brake_pedal_pct": 0,  "gear": 1},
        ],
    )


def _make_resolved_fault(
    fault_id="f1",
    onset_s=5.0,
    end_s=15.0,
    severity_start=0.0,
    severity_end=0.5,
    progression="linear",
) -> ResolvedFault:
    return ResolvedFault(
        id=fault_id,
        type="brake_pad_wear",
        onset_s=onset_s,
        end_s=end_s,
        params={
            "severity_start": severity_start,
            "severity_end": severity_end,
            "progression": progression,
        },
        physical_category="mechanical",
        observable_effect="drift",
        layer="signal",
        targets={"signals": ["longitudinal_acceleration"], "messages": []},
    )


# ----------------------------------------------------------------------
# Progression math
# ----------------------------------------------------------------------

def test_progression_constant_is_zero():
    t = np.linspace(0, 10, 11)
    p = _progression_value("constant", t, onset_s=3, end_s=7)
    assert (p == 0).all()


def test_progression_linear_ramps():
    t = np.array([0.0, 2.5, 5.0, 7.5, 10.0])
    p = _progression_value("linear", t, onset_s=0, end_s=10)
    assert p[0] == pytest.approx(0.0)
    assert p[2] == pytest.approx(0.5)
    assert p[4] == pytest.approx(1.0)


def test_progression_exponential_is_below_linear():
    t = np.array([0.0, 5.0, 10.0])
    p_lin = _progression_value("linear", t, onset_s=0, end_s=10)
    p_exp = _progression_value("exponential", t, onset_s=0, end_s=10)
    assert p_exp[1] < p_lin[1]   # 0.25 < 0.5


def test_progression_step_jumps_after_onset():
    t = np.array([0.0, 4.99, 5.0, 5.01, 10.0])
    p = _progression_value("step", t, onset_s=5, end_s=10)
    assert p[0] == 0.0
    assert p[1] == 0.0
    assert p[2] == 0.0   # at exactly onset, still 0 (pos == 0)
    assert p[3] == 1.0
    assert p[4] == 1.0


def test_progression_unknown_raises():
    t = np.array([0.0, 1.0])
    with pytest.raises(ValueError, match="Unknown progression"):
        _progression_value("sigmoid", t, onset_s=0, end_s=1)


def test_severity_array_linear():
    f = _make_resolved_fault(severity_start=0.2, severity_end=0.6, progression="linear")
    t = np.array([0.0, 5.0, 10.0, 14.99, 15.0, 20.0])
    s = _compute_severity_array(f, t)
    assert s[0] == pytest.approx(0.0)     # before onset → 0
    assert s[1] == pytest.approx(0.2)     # at onset → severity_start
    assert s[2] == pytest.approx(0.4)     # midpoint → average
    assert s[3] == pytest.approx(0.6, abs=1e-3)  # just before end → near severity_end
    assert s[4] == pytest.approx(0.0)     # at end → outside window (exclusive)
    assert s[5] == pytest.approx(0.0)     # after end → outside window

# ----------------------------------------------------------------------
# Injector: no faults
# ----------------------------------------------------------------------

def test_no_faults_returns_clean(injector, simple_scenario):
    corrupted, applied = injector.apply(simple_scenario)
    assert applied == []
    clean = injector.generator.generate(simple_scenario)
    pd.testing.assert_frame_equal(corrupted, clean)


# ----------------------------------------------------------------------
# Injector: single fault
# ----------------------------------------------------------------------

def test_single_fault_no_change_before_onset(injector, simple_scenario, generator):
    simple_scenario.resolved_faults = [_make_resolved_fault(onset_s=5.0, end_s=15.0)]
    corrupted, _ = injector.apply(simple_scenario)
    clean = generator.generate(simple_scenario)

    before = corrupted["time_s"] < 5.0
    pd.testing.assert_series_equal(
        corrupted.loc[before, "longitudinal_acceleration"].reset_index(drop=True),
        clean.loc[before, "longitudinal_acceleration"].reset_index(drop=True),
    )


def test_single_fault_keeps_speed_higher_in_window(injector, simple_scenario, generator):
    simple_scenario.resolved_faults = [
        _make_resolved_fault(onset_s=5.0, end_s=15.0, severity_end=0.5)
    ]
    corrupted, _ = injector.apply(simple_scenario)
    clean = generator.generate(simple_scenario)

    in_window = (corrupted["time_s"] >= 6.0) & (corrupted["time_s"] <= 14.0)
    assert (
        corrupted.loc[in_window, "vehicle_speed"].values
        >= clean.loc[in_window, "vehicle_speed"].values - 1e-9
    ).all()

def test_single_fault_changes_acceleration_in_window(injector, simple_scenario, generator):
    simple_scenario.resolved_faults = [
        _make_resolved_fault(onset_s=5.0, end_s=15.0, severity_end=0.5)
    ]
    corrupted, _ = injector.apply(simple_scenario)
    clean = generator.generate(simple_scenario)

    # Narrow window where neither vehicle has stopped yet
    in_window = (corrupted["time_s"] >= 6.0) & (corrupted["time_s"] <= 9.0)
    assert (
        corrupted.loc[in_window, "longitudinal_acceleration"].values
        >= clean.loc[in_window, "longitudinal_acceleration"].values - 1e-9
    ).all()

def test_single_fault_cascades_to_speed(injector, simple_scenario, generator):
    simple_scenario.resolved_faults = [
        _make_resolved_fault(onset_s=5.0, end_s=15.0, severity_end=0.5)
    ]
    corrupted, _ = injector.apply(simple_scenario)
    clean = generator.generate(simple_scenario)

    after = corrupted["time_s"] >= 15.0
    assert (
        corrupted.loc[after, "vehicle_speed"].values
        >= clean.loc[after, "vehicle_speed"].values - 1e-9
    ).all()


def test_single_fault_cascades_to_wheel_speed(injector, simple_scenario):
    simple_scenario.resolved_faults = [_make_resolved_fault()]
    corrupted, _ = injector.apply(simple_scenario)
    # wheel_speed should track vehicle_speed exactly (gain 1, no lag)
    np.testing.assert_allclose(
        corrupted["wheel_speed"].values,
        corrupted["vehicle_speed"].values,
    )


def test_single_fault_cascades_to_engine_rpm(injector, simple_scenario, generator):
    simple_scenario.resolved_faults = [
        _make_resolved_fault(onset_s=5.0, end_s=15.0, severity_end=0.5)
    ]
    corrupted, _ = injector.apply(simple_scenario)
    clean = generator.generate(simple_scenario)

    # During the braking window, RPM should differ (corrupted has higher speed → higher RPM)
    in_window = (corrupted["time_s"] >= 6.0) & (corrupted["time_s"] <= 9.0)
    assert not np.allclose(
        corrupted.loc[in_window, "engine_rpm"].values,
        clean.loc[in_window, "engine_rpm"].values,
    )


# ----------------------------------------------------------------------
# Unaffected signals
# ----------------------------------------------------------------------

def test_driver_inputs_unchanged(injector, simple_scenario, generator):
    simple_scenario.resolved_faults = [_make_resolved_fault()]
    corrupted, _ = injector.apply(simple_scenario)
    clean = generator.generate(simple_scenario)
    for col in ["throttle_position", "brake_pedal"]:
        pd.testing.assert_series_equal(
            corrupted[col].reset_index(drop=True),
            clean[col].reset_index(drop=True),
            check_names=False,
        )


def test_brake_pressure_unchanged(injector, simple_scenario, generator):
    """Pad wear is a friction problem, not hydraulic. Pressure is unaffected."""
    simple_scenario.resolved_faults = [_make_resolved_fault()]
    corrupted, _ = injector.apply(simple_scenario)
    clean = generator.generate(simple_scenario)
    pd.testing.assert_series_equal(
        corrupted["brake_pressure"].reset_index(drop=True),
        clean["brake_pressure"].reset_index(drop=True),
        check_names=False,
    )


def test_coolant_unchanged(injector, simple_scenario, generator):
    simple_scenario.resolved_faults = [_make_resolved_fault()]
    corrupted, _ = injector.apply(simple_scenario)
    clean = generator.generate(simple_scenario)
    pd.testing.assert_series_equal(
        corrupted["coolant_temperature"].reset_index(drop=True),
        clean["coolant_temperature"].reset_index(drop=True),
        check_names=False,
    )


# ----------------------------------------------------------------------
# Severity behavior
# ----------------------------------------------------------------------

def test_zero_severity_is_noop(injector, simple_scenario, generator):
    simple_scenario.resolved_faults = [
        _make_resolved_fault(severity_start=0.0, severity_end=0.0)
    ]
    corrupted, _ = injector.apply(simple_scenario)
    clean = generator.generate(simple_scenario)
    pd.testing.assert_frame_equal(corrupted, clean)


def test_full_severity_removes_braking(injector, simple_scenario, generator):
    """severity_end = 1.0 → braking force is zero → vehicle coasts through."""
    simple_scenario.resolved_faults = [
        _make_resolved_fault(onset_s=5.0, end_s=15.0, severity_end=1.0)
    ]
    corrupted, _ = injector.apply(simple_scenario)
    clean = generator.generate(simple_scenario)

    at_15_corrupted = corrupted.loc[corrupted["time_s"] == 15.0, "vehicle_speed"].values[0]
    at_15_clean = clean.loc[clean["time_s"] == 15.0, "vehicle_speed"].values[0]
    assert at_15_corrupted > at_15_clean + 5.0


def test_step_progression_differs_immediately_after_onset(injector, simple_scenario, generator):
    simple_scenario.resolved_faults = [
        _make_resolved_fault(
            onset_s=5.0, end_s=15.0,
            severity_start=0.0, severity_end=0.5,
            progression="step",
        )
    ]
    corrupted, _ = injector.apply(simple_scenario)
    clean = generator.generate(simple_scenario)

    # Just before onset: identical
    t_before = corrupted["time_s"] == 4.99
    pd.testing.assert_series_equal(
        corrupted.loc[t_before, "longitudinal_acceleration"].reset_index(drop=True),
        clean.loc[clean["time_s"] == 4.99, "longitudinal_acceleration"].reset_index(drop=True),
        check_names=False,
    )

    # Just after onset: differ
    t_after_corrupted = corrupted.loc[corrupted["time_s"] == 5.01, "longitudinal_acceleration"].values[0]
    t_after_clean = clean.loc[clean["time_s"] == 5.01, "longitudinal_acceleration"].values[0]
    assert not np.isclose(t_after_corrupted, t_after_clean)


# ----------------------------------------------------------------------
# Overlapping faults
# ----------------------------------------------------------------------

def test_overlapping_faults_compose_additively(injector, simple_scenario):
    """Two 0.3-severity faults → effective 1 - 0.7*0.7 = 0.51."""
    f1 = _make_resolved_fault("f1", onset_s=5.0, end_s=15.0,
                              severity_start=0.3, severity_end=0.3,
                              progression="constant")
    f2 = _make_resolved_fault("f2", onset_s=5.0, end_s=15.0,
                              severity_start=0.3, severity_end=0.3,
                              progression="constant")
    simple_scenario.resolved_faults = [f1, f2]
    corrupted, applied = injector.apply(simple_scenario)
    assert len(applied) == 2

    # Compare against a single fault with severity 0.51
    f_combined = _make_resolved_fault("f_combined", onset_s=5.0, end_s=15.0,
                                      severity_start=0.51, severity_end=0.51,
                                      progression="constant")
    simple_scenario.resolved_faults = [f_combined]
    reference, _ = injector.apply(simple_scenario)

    pd.testing.assert_frame_equal(corrupted, reference)


def test_applied_faults_records_metadata(injector, simple_scenario):
    simple_scenario.resolved_faults = [
        _make_resolved_fault("f1", onset_s=5.0, end_s=15.0, severity_end=0.4)
    ]
    _, applied = injector.apply(simple_scenario)
    assert len(applied) == 1
    a = applied[0]
    assert a.fault_id == "f1"
    assert a.fault_type == "brake_pad_wear"
    assert a.onset_s == 5.0
    assert a.end_s == 15.0
    assert a.severity_end == 0.4
    assert a.physical_category == "mechanical"
    assert a.observable_effect == "drift"
    assert a.affected_signals == ["longitudinal_acceleration"]


# ----------------------------------------------------------------------
# Determinism
# ----------------------------------------------------------------------

def test_injector_is_deterministic(injector, simple_scenario):
    simple_scenario.resolved_faults = [_make_resolved_fault()]
    c1, a1 = injector.apply(simple_scenario)
    c2, a2 = injector.apply(simple_scenario)
    pd.testing.assert_frame_equal(c1, c2)
    assert a1 == a2


# ----------------------------------------------------------------------
# Backward compatibility — generate() without faults
# ----------------------------------------------------------------------

def test_generate_without_faults_unchanged(generator, simple_scenario):
    clean_a = generator.generate(simple_scenario)
    clean_b = generator.generate(simple_scenario, faults=None)
    pd.testing.assert_frame_equal(clean_a, clean_b)


def test_generate_with_empty_faults_dict_unchanged(generator, simple_scenario):
    clean = generator.generate(simple_scenario)
    same = generator.generate(simple_scenario, faults={})
    pd.testing.assert_frame_equal(clean, same)


def test_multiplier_wrong_length_raises(generator, simple_scenario):
    bad = np.ones(10)   # wrong length
    with pytest.raises(ValueError, match="does not match simulation length"):
        generator.generate(simple_scenario, faults={"brake_force_multiplier": bad})


# ----------------------------------------------------------------------
# Physical plausibility
# ----------------------------------------------------------------------

def test_speed_never_negative_with_fault(injector, simple_scenario):
    simple_scenario.resolved_faults = [_make_resolved_fault(severity_end=0.5)]
    corrupted, _ = injector.apply(simple_scenario)
    assert (corrupted["vehicle_speed"] >= 0).all()
    assert (corrupted["wheel_speed"] >= 0).all()


def test_all_signals_within_range_with_fault(injector, simple_scenario):
    simple_scenario.resolved_faults = [_make_resolved_fault(severity_end=0.5)]
    corrupted, _ = injector.apply(simple_scenario)
    for name, spec in injector.cfg.load_signals().items():
        if name not in corrupted.columns:
            continue
        lo, hi = spec["range"]
        assert (corrupted[name] >= lo).all(), f"{name} below range"
        assert (corrupted[name] <= hi).all(), f"{name} above range"