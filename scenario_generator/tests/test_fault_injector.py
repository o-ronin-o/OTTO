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
    """
    20-second scenario:
      - 0–10 s:  accelerate at 60% throttle in gear 2
      - 10–20 s: brake at 20% pedal in gear 2 (mild deceleration)
    """
    return Scenario(
        name="Simple",
        duration_s=20.0,
        base_rate_hz=100.0,
        vehicle_id="sedan_a",
        segments=[
            {"t_start": 0,  "t_end": 10, "throttle_pct": 60, "brake_pedal_pct": 0,  "gear": 2},
            {"t_start": 10, "t_end": 20, "throttle_pct": 0,  "brake_pedal_pct": 20, "gear": 2},
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

    # Check that the RPM trajectories differ somewhere in the simulation
    diff = np.abs(corrupted["engine_rpm"].values - clean["engine_rpm"].values)
    assert diff.max() > 50.0, (
        f"RPM should differ under fault; max diff was {diff.max():.2f}"
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
    """severity_end = 1.0 → braking disabled → corrupted retains more speed."""
    simple_scenario.resolved_faults = [
        _make_resolved_fault(onset_s=10.0, end_s=20.0, severity_end=1.0)
    ]
    corrupted, _ = injector.apply(simple_scenario)
    clean = generator.generate(simple_scenario)

    # Both vehicles should be moving at the end of the acceleration phase (t=10)
    v_10_corrupted = corrupted["vehicle_speed"].iloc[1000]
    v_10_clean = clean["vehicle_speed"].iloc[1000]
    assert v_10_corrupted > 5.0
    assert v_10_clean > 5.0

    # At the end of the scenario (t≈20), clean has been braking the whole
    # time while corrupted had no braking. Corrupted should be much faster.
    v_end_corrupted = corrupted["vehicle_speed"].iloc[-1]
    v_end_clean = clean["vehicle_speed"].iloc[-1]
    assert v_end_corrupted > v_end_clean + 5.0, (
        f"Corrupted should be faster. "
        f"corrupted={v_end_corrupted:.2f}, clean={v_end_clean:.2f}"
    )

def test_step_progression_differs_immediately_after_onset(injector, simple_scenario, generator):
    simple_scenario.resolved_faults = [
        _make_resolved_fault(
            onset_s=10.0, end_s=20.0,
            severity_start=0.0, severity_end=0.5,
            progression="step",
        )
    ]
    corrupted, _ = injector.apply(simple_scenario)
    clean = generator.generate(simple_scenario)

    # Just before onset: identical
    t_before = corrupted["time_s"] == 9.99
    pd.testing.assert_series_equal(
        corrupted.loc[t_before, "longitudinal_acceleration"].reset_index(drop=True),
        clean.loc[clean["time_s"] == 9.99, "longitudinal_acceleration"].reset_index(drop=True),
        check_names=False,
    )

    # Just after onset: differ
    t_after_corrupted = corrupted.loc[corrupted["time_s"] == 10.01, "longitudinal_acceleration"].values[0]
    t_after_clean = clean.loc[clean["time_s"] == 10.01, "longitudinal_acceleration"].values[0]
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

# ----------------------------------------------------------------------
# Fixtures for crank faults
# ----------------------------------------------------------------------

@pytest.fixture
def torque_drop_scenario(loader) -> Scenario:
    """A simple scenario with a torque-drop fault during cruise."""
    path = CONFIG_DIR / "scenarios" / "highway_cruise_torque_drop.yaml"
    fault_registry = loader.load_fault_types()
    return Scenario.from_yaml(path, fault_registry=fault_registry)


@pytest.fixture
def crank_sensor_scenario(loader) -> Scenario:
    """A simple scenario with a crank sensor freeze fault."""
    path = CONFIG_DIR / "scenarios" / "highway_cruise_crank_sensor_freeze.yaml"
    fault_registry = loader.load_fault_types()
    return Scenario.from_yaml(path, fault_registry=fault_registry)


# ----------------------------------------------------------------------
# crankshaft_torque_drop — physics-layer behavior
# ----------------------------------------------------------------------

def test_torque_drop_scenario_resolves(loader, torque_drop_scenario):
    """The scenario loads with exactly one fault."""
    assert len(torque_drop_scenario.resolved_faults) == 1
    f = torque_drop_scenario.resolved_faults[0]
    assert f.type == "crankshaft_torque_drop"
    assert f.params["drop_magnitude"] == 0.7
    assert f.params["drop_frequency_hz"] == 0.4


def test_torque_drop_lowers_rpm(injector, torque_drop_scenario, generator):
    """During fault window, engine_rpm should dip below clean."""
    corrupted, applied = injector.apply(torque_drop_scenario)
    clean = generator.generate(torque_drop_scenario)

    # Same duration
    assert len(corrupted) == len(clean)

    # During the fault window, corrupted RPM should be lower at some point
    onset = applied[0].onset_s
    end = applied[0].end_s
    mask = (corrupted["time_s"] >= onset) & (corrupted["time_s"] < end)
    rpm_diff = clean.loc[mask, "engine_rpm"].values - corrupted.loc[mask, "engine_rpm"].values
    assert rpm_diff.max() > 100.0, (
        f"Expected RPM dip > 100 during fault; got max {rpm_diff.max():.2f}"
    )


def test_torque_drop_affects_vehicle_speed(injector, torque_drop_scenario, generator):
    """Torque drops should propagate to vehicle speed (less drive force)."""
    corrupted, _ = injector.apply(torque_drop_scenario)
    clean = generator.generate(torque_drop_scenario)

    # At the end, corrupted should be at most as fast as clean
    v_clean = clean["vehicle_speed"].iloc[-1]
    v_corrupt = corrupted["vehicle_speed"].iloc[-1]
    assert v_corrupt <= v_clean + 1e-6


def test_torque_drop_does_not_change_driver_inputs(injector, torque_drop_scenario, generator):
    """Throttle, brake_pedal must be identical between clean and corrupted."""
    corrupted, _ = injector.apply(torque_drop_scenario)
    clean = generator.generate(torque_drop_scenario)

    for col in ["throttle_position", "brake_pedal"]:
        pd.testing.assert_series_equal(
            corrupted[col].reset_index(drop=True),
            clean[col].reset_index(drop=True),
            check_names=False,
        )


def test_torque_drop_does_not_change_coolant(injector, torque_drop_scenario, generator):
    """Coolant temperature is a slow thermal process — unaffected."""
    corrupted, _ = injector.apply(torque_drop_scenario)
    clean = generator.generate(torque_drop_scenario)

    pd.testing.assert_series_equal(
        corrupted["coolant_temperature"].reset_index(drop=True),
        clean["coolant_temperature"].reset_index(drop=True),
        check_names=False,
    )


def test_torque_drop_produces_applied_fault_record(injector, torque_drop_scenario):
    _, applied = injector.apply(torque_drop_scenario)
    assert len(applied) == 1
    a = applied[0]
    assert a.fault_type == "crankshaft_torque_drop"
    assert a.physical_category == "mechanical"
    assert a.observable_effect == "dropout"


def test_torque_drop_deterministic(injector, torque_drop_scenario):
    """Same scenario → same output."""
    c1, _ = injector.apply(torque_drop_scenario)
    c2, _ = injector.apply(torque_drop_scenario)
    pd.testing.assert_frame_equal(c1, c2)


def test_torque_drop_multiplier_drops_below_one(injector, torque_drop_scenario):
    """Direct test of the multiplier: it must dip below 1.0 during the window."""
    f = torque_drop_scenario.resolved_faults[0]
    n = int(torque_drop_scenario.duration_s * torque_drop_scenario.base_rate_hz)
    dt = 1.0 / torque_drop_scenario.base_rate_hz
    t_s = np.arange(n) * dt

    mult = injector._compute_torque_drop_multiplier(f, t_s)

    # Between onset and end, at least one sample should be < 1.0
    mask = (t_s >= f.onset_s) & (t_s < f.end_s)
    assert (mult[mask] < 1.0).any()

    # Outside the window, multiplier is exactly 1.0
    mask_outside = (t_s < f.onset_s) | (t_s >= f.end_s)
    assert np.allclose(mult[mask_outside], 1.0)


# ----------------------------------------------------------------------
# crank_sensor_failure — post-physics behavior
# ----------------------------------------------------------------------

def test_crank_sensor_scenario_resolves(loader, crank_sensor_scenario):
    assert len(crank_sensor_scenario.resolved_faults) == 1
    f = crank_sensor_scenario.resolved_faults[0]
    assert f.type == "crank_sensor_failure"
    assert f.params["freeze_duration_s"] == 20.0
    assert f.params["freeze_scope"] == "per_signal"


def test_crank_sensor_freezes_rpm(injector, crank_sensor_scenario, generator):
    """In the freeze window, engine_rpm should be constant."""
    corrupted, applied = injector.apply(crank_sensor_scenario)
    onset = applied[0].onset_s
    duration = crank_sensor_scenario.resolved_faults[0].params["freeze_duration_s"]
    end = min(onset + duration, applied[0].end_s)

    mask = (corrupted["time_s"] >= onset) & (corrupted["time_s"] < end)
    frozen_slice = corrupted.loc[mask, "engine_rpm"]
    assert frozen_slice.nunique() == 1, (
        f"engine_rpm should be frozen during the fault; "
        f"found {frozen_slice.nunique()} distinct values"
    )


def test_crank_sensor_freezes_crank_position(injector, crank_sensor_scenario):
    """In the freeze window, crank_position should be constant."""
    corrupted, applied = injector.apply(crank_sensor_scenario)
    onset = applied[0].onset_s
    duration = crank_sensor_scenario.resolved_faults[0].params["freeze_duration_s"]
    end = min(onset + duration, applied[0].end_s)

    mask = (corrupted["time_s"] >= onset) & (corrupted["time_s"] < end)
    frozen_slice = corrupted.loc[mask, "crank_position"]
    assert frozen_slice.nunique() == 1, (
        f"crank_position should be frozen; found {frozen_slice.nunique()} distinct values"
    )


def test_crank_sensor_does_not_affect_vehicle_speed(injector, crank_sensor_scenario, generator):
    """
    Cross-check: the engine is fine, so vehicle_speed should be UNCHANGED
    by the sensor fault.
    """
    corrupted, _ = injector.apply(crank_sensor_scenario)
    clean = generator.generate(crank_sensor_scenario)

    pd.testing.assert_series_equal(
        corrupted["vehicle_speed"].reset_index(drop=True),
        clean["vehicle_speed"].reset_index(drop=True),
        check_names=False,
    )


def test_crank_sensor_does_not_affect_acceleration(injector, crank_sensor_scenario, generator):
    """longitudinal_acceleration is computed from physics, not the sensor."""
    corrupted, _ = injector.apply(crank_sensor_scenario)
    clean = generator.generate(crank_sensor_scenario)

    pd.testing.assert_series_equal(
        corrupted["longitudinal_acceleration"].reset_index(drop=True),
        clean["longitudinal_acceleration"].reset_index(drop=True),
        check_names=False,
    )


def test_crank_sensor_does_not_affect_wheel_speed(injector, crank_sensor_scenario, generator):
    """wheel_speed comes from a different sensor (not the crank sensor)."""
    corrupted, _ = injector.apply(crank_sensor_scenario)
    clean = generator.generate(crank_sensor_scenario)

    pd.testing.assert_series_equal(
        corrupted["wheel_speed"].reset_index(drop=True),
        clean["wheel_speed"].reset_index(drop=True),
        check_names=False,
    )


def test_crank_sensor_does_not_affect_brake_or_throttle(injector, crank_sensor_scenario, generator):
    """Driver inputs come from their own sensors."""
    corrupted, _ = injector.apply(crank_sensor_scenario)
    clean = generator.generate(crank_sensor_scenario)

    for col in ["throttle_position", "brake_pedal", "brake_pressure"]:
        pd.testing.assert_series_equal(
            corrupted[col].reset_index(drop=True),
            clean[col].reset_index(drop=True),
            check_names=False,
        )


def test_crank_sensor_before_onset_identical(injector, crank_sensor_scenario, generator):
    """Before onset, engine_rpm and crank_position match clean exactly."""
    corrupted, applied = injector.apply(crank_sensor_scenario)
    clean = generator.generate(crank_sensor_scenario)

    onset = applied[0].onset_s
    mask = corrupted["time_s"] < onset

    for col in ["engine_rpm", "crank_position"]:
        pd.testing.assert_series_equal(
            corrupted.loc[mask, col].reset_index(drop=True),
            clean.loc[mask, col].reset_index(drop=True),
            check_names=False,
        )


def test_crank_sensor_after_freeze_ends_identical(injector, crank_sensor_scenario, generator):
    """After the freeze window ends, engine_rpm and crank_position are back to clean."""
    corrupted, applied = injector.apply(crank_sensor_scenario)
    clean = generator.generate(crank_sensor_scenario)

    onset = applied[0].onset_s
    duration = crank_sensor_scenario.resolved_faults[0].params["freeze_duration_s"]
    freeze_end = onset + duration

    mask = corrupted["time_s"] >= freeze_end

    for col in ["engine_rpm", "crank_position"]:
        pd.testing.assert_series_equal(
            corrupted.loc[mask, col].reset_index(drop=True),
            clean.loc[mask, col].reset_index(drop=True),
            check_names=False,
        )


def test_crank_sensor_applied_fault_record(injector, crank_sensor_scenario):
    _, applied = injector.apply(crank_sensor_scenario)
    assert len(applied) == 1
    a = applied[0]
    assert a.fault_type == "crank_sensor_failure"
    assert a.physical_category == "sensor"
    assert a.observable_effect == "freeze"
    assert set(a.affected_signals) == {"engine_rpm", "crank_position"}


# ----------------------------------------------------------------------
# Injector dispatch — injection_point
# ----------------------------------------------------------------------

def test_injector_dispatches_physics_fault(injector, torque_drop_scenario):
    """crankshaft_torque_drop should trigger re-simulation (physics path)."""
    # Just check it runs without error and produces a fault record
    corrupted, applied = injector.apply(torque_drop_scenario)
    assert len(applied) == 1


def test_injector_dispatches_post_physics_fault(injector, crank_sensor_scenario):
    """crank_sensor_failure should trigger the post-physics path."""
    corrupted, applied = injector.apply(crank_sensor_scenario)
    assert len(applied) == 1


def test_injection_point_defaults_to_physics(loader):
    """brake_pad_wear doesn't declare injection_point → defaults to physics."""
    registry = loader.load_fault_types()
    f = registry["brake_pad_wear"]
    assert f.get("injection_point", "physics") == "physics"