"""
Signal Fault Injector — applies signal-layer faults by re-simulating the
physics engine with perturbed inputs.

Design (Step 2.1b):
    - Re-simulation approach: the injector does NOT modify signals directly.
      Instead, it computes a fault multiplier array and asks the
      SignalGenerator to re-run the physics with that multiplier.
    - Single source of truth: the physics lives in SignalGenerator. The
      injector is a thin adapter that translates fault declarations into
      physics-input perturbations.
    - Cascading effects are automatic: modifying the braking force propagates
      through longitudinal_acceleration → vehicle_speed → wheel_speed →
      engine_rpm without explicit cascade rules.

Overlap composition:
    effective_severity(t) = 1 - Π(1 - severity_i(t))
    for all faults active on the same target signal at time t.

Idempotency:
    The injector is a pure function. Pipeline always feeds clean scenarios.

Frame-layer faults:
    Currently skipped (deferred to Step 2.5).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config_loader import ConfigLoader
from src.signal_generator import ResolvedFault, Scenario, SignalGenerator


# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------

class AppliedFault:
    """
    Record of one fault's application.

    Consumed by Step 2.2 (FaultLabelBuilder). The `fault_id` here is the
    primary label — downstream code can look it up in the fault registry
    (config/faults/<fault_type>.yaml) to retrieve its description,
    injection_rule, and any other metadata.
    """

    __slots__ = (
        "fault_id", "fault_type",
        "onset_s", "end_s",
        "severity_start", "severity_end", "progression",
        "physical_category", "observable_effect",
        "affected_signals",
    )

    def __init__(
        self,
        fault_id: str,
        fault_type: str,
        onset_s: float,
        end_s: float,
        severity_start: float,
        severity_end: float,
        progression: str,
        physical_category: str,
        observable_effect: str,
        affected_signals: list[str],
    ):
        self.fault_id = fault_id
        self.fault_type = fault_type
        self.onset_s = onset_s
        self.end_s = end_s
        self.severity_start = severity_start
        self.severity_end = severity_end
        self.progression = progression
        self.physical_category = physical_category
        self.observable_effect = observable_effect
        self.affected_signals = affected_signals

    def __repr__(self) -> str:
        return (
            f"AppliedFault(id={self.fault_id!r}, type={self.fault_type!r}, "
            f"[{self.onset_s:.2f}, {self.end_s:.2f}] s)"
        )

    def __eq__(self, other) -> bool:
        if not isinstance(other, AppliedFault):
            return NotImplemented
        return (
            self.fault_id == other.fault_id
            and self.fault_type == other.fault_type
            and self.onset_s == other.onset_s
            and self.end_s == other.end_s
            and self.severity_start == other.severity_start
            and self.severity_end == other.severity_end
            and self.progression == other.progression
            and self.physical_category == other.physical_category
            and self.observable_effect == other.observable_effect
            and self.affected_signals == other.affected_signals
        )


# ----------------------------------------------------------------------
# Severity computation
# ----------------------------------------------------------------------

def _progression_value(
    progression: str,
    t_s: np.ndarray,
    onset_s: float,
    end_s: float,
) -> np.ndarray:
    """
    Compute the progression scalar in [0, 1] for each time in `t_s`.

    For t < onset: 0
    For t >= end: 1
    For t in [onset, end): depends on progression type
    """
    duration = end_s - onset_s
    if duration <= 0:
        return np.ones_like(t_s)

    # normalized position within the fault window, clamped to [0, 1]
    pos = np.clip((t_s - onset_s) / duration, 0.0, 1.0)

    if progression == "constant":
        return np.zeros_like(t_s)         # stays at severity_start
    if progression == "linear":
        return pos
    if progression == "exponential":
        return pos * pos
    if progression == "step":
        return (pos > 0).astype(float)    # jumps to 1 for any t > onset
    raise ValueError(f"Unknown progression: {progression}")


def _compute_severity_array(
    fault: ResolvedFault,
    t_s: np.ndarray,
) -> np.ndarray:
    """
    Compute the per-timestep severity of a single fault.

    Zero outside [onset_s, end_s); the ramp value inside.
    """
    progress = _progression_value(
        fault.params.get("progression", "linear"),
        t_s,
        fault.onset_s,
        fault.end_s,
    )
    s_start = float(fault.params.get("severity_start", 0.0))
    s_end = float(fault.params.get("severity_end", 0.0))
    ramp = s_start + (s_end - s_start) * progress

    # Zero outside the fault window
    in_window = (t_s >= fault.onset_s) & (t_s < fault.end_s)
    return np.where(in_window, ramp, 0.0)

def _compute_severity_at(
    fault: ResolvedFault,
    t_s: float,
) -> float:
    """
    Scalar variant of _compute_severity_array.

    Returns the fault's severity at one specific time.
    Zero outside the fault window [onset_s, end_s).
    """
    if not (fault.onset_s <= t_s < fault.end_s):
        return 0.0

    progress = _progression_value(
        fault.params.get("progression", "linear"),
        np.array([t_s]),
        fault.onset_s,
        fault.end_s,
    )[0]
    s_start = float(fault.params.get("severity_start", 0.0))
    s_end = float(fault.params.get("severity_end", 0.0))
    return s_start + (s_end - s_start) * progress


# ----------------------------------------------------------------------
# SignalFaultInjector
# ----------------------------------------------------------------------

class SignalFaultInjector:
    """
    Applies signal-layer faults to a scenario by re-running the physics
    with perturbed inputs.

    The injector is stateless and deterministic.
    """

    # Which physics input each fault type perturbs
    FAULT_TYPE_TO_PHYSICS_INPUT = {
        "brake_pad_wear": "brake_force_multiplier",
        # Future fault types go here:
        # "throttle_sensor_drift": "throttle_multiplier",
        # "engine_misfire": "engine_force_multiplier",
    }

    def __init__(
        self,
        config_loader: ConfigLoader,
        generator: SignalGenerator,
    ):
        self.cfg = config_loader
        self.generator = generator
        self.fault_registry = config_loader.load_fault_types()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply(
        self,
        scenario: Scenario,
    ) -> tuple[pd.DataFrame, list[AppliedFault]]:
        """
        Apply all signal-layer faults declared on the scenario.

        Returns
        -------
        corrupted_df : pd.DataFrame
            Same shape as a clean generate() call, with faults applied.
        applied : list[AppliedFault]
            Records of every fault that was actually applied.
        """
        if not scenario.resolved_faults:
            # No faults → return clean generate()
            clean = self.generator.generate(scenario)
            return clean, []

        # --- Split faults by layer ---
        signal_faults = [
            f for f in scenario.resolved_faults if f.layer == "signal"
        ]
        # Frame-layer faults are deferred to Step 2.5.
        # They are ignored here.

        # --- Compute physics-input perturbations ---
        n_steps = int(round(scenario.duration_s * scenario.base_rate_hz))
        dt = 1.0 / scenario.base_rate_hz
        t_s = np.arange(n_steps) * dt

        # Determine which physics inputs are perturbed by at least one fault
        physics_inputs_to_perturb: set[str] = set()
        for f in signal_faults:
            inp = self.FAULT_TYPE_TO_PHYSICS_INPUT.get(f.type)
            if inp is not None:
                physics_inputs_to_perturb.add(inp)

        perturbations: dict[str, np.ndarray] = {}
        for physics_input in physics_inputs_to_perturb:
            relevant = [
                f for f in signal_faults
                if self.FAULT_TYPE_TO_PHYSICS_INPUT.get(f.type) == physics_input
            ]
            perturbations[physics_input] = self._compose_multiplier(relevant, t_s)

        # --- Re-run the physics with perturbations ---
        corrupted_df = self.generator.generate(scenario, faults=perturbations)

        # --- Build applied-fault records ---
        applied: list[AppliedFault] = []
        for f in signal_faults:
            if self.FAULT_TYPE_TO_PHYSICS_INPUT.get(f.type) is None:
                continue  # unknown fault type → skip silently for now
            applied.append(AppliedFault(
                fault_id=f.id,
                fault_type=f.type,
                onset_s=f.onset_s,
                end_s=f.end_s,
                severity_start=float(f.params.get("severity_start", 0.0)),
                severity_end=float(f.params.get("severity_end", 0.0)),
                progression=f.params.get("progression", "linear"),
                physical_category=f.physical_category,
                observable_effect=f.observable_effect,
                affected_signals=list(f.targets.get("signals", [])),
            ))

        return corrupted_df, applied

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _compose_multiplier(
        self,
        faults: list[ResolvedFault],
        t_s: np.ndarray,
    ) -> np.ndarray:
        """
        Compose the effective multiplier from a set of overlapping faults.

        Composition rule: effective_severity = 1 - Π(1 - severity_i)
        The multiplier is the complement: multiplier = Π(1 - severity_i)
        """
        multiplier = np.ones_like(t_s)
        for fault in faults:
            severity = _compute_severity_array(fault, t_s)
            severity = np.clip(severity, 0.0, 1.0)
            multiplier *= (1.0 - severity)
        return multiplier