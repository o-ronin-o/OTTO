"""
Signal Fault Injector — applies signal-layer faults by re-simulating the
physics engine with perturbed inputs.

Design (Step 2.1b):
    - Re-simulation approach for physics faults: the injector computes a
      fault multiplier array and asks the SignalGenerator to re-run the
      physics with that multiplier.
    - Post-physics faults: the injector modifies signals after physics has
      run. Used for sensor failures that don't affect the engine's actual
      behaviour.
    - Cascading effects are automatic: modifying the braking force propagates
      through longitudinal_acceleration → vehicle_speed → wheel_speed →
      engine_rpm without explicit cascade rules.

Design (Phase B):
    - Faults declare an `injection_point`:
        physics       → applied by re-simulation
        post_physics  → applied after physics has run
    - `crankshaft_torque_drop` uses the `engine_torque_multiplier` input.
    - `crank_sensor_failure` is post-physics: it freezes engine_rpm and
      crank_position, leaving all other signals unchanged.

Overlap composition (physics faults):
    effective_severity(t) = 1 - Π(1 - severity_i(t))
    for all faults active on the same target signal at time t.

Idempotency:
    The injector is a pure function. Pipeline always feeds clean scenarios.

Frame-layer faults:
    Not handled here (deferred).
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

    Consumed by FaultLabelBuilder. The `fault_id` here is the primary
    label — downstream code can look it up in the fault registry
    (config/faults/<fault_type>.yaml) to retrieve its description,
    injection_rule, and any other metadata.

    `severity` is a normalized measure of the fault's strength at each
    time step. For brake_pad_wear it's a value in [0, 1]. For
    crankshaft_torque_drop it's `drop_magnitude`. For
    crank_sensor_failure it's a normalized freeze strength. The exact
    semantics depend on the fault family, but it's always in [0, 1]
    so the label builder can consistently decide whether a frame is
    "faulty."
    """

    __slots__ = (
        "fault_id", "fault_type",
        "onset_s", "end_s",
        "severity",                # ← NEW: normalized [0, 1] strength
        "severity_start", "severity_end", "progression",  # legacy for brake_pad_wear
        "physical_category", "observable_effect",
        "affected_signals",
    )

    def __init__(
        self,
        fault_id: str,
        fault_type: str,
        onset_s: float,
        end_s: float,
        severity: float,
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
        self.severity = severity
        self.severity_start = severity_start
        self.severity_end = severity_end
        self.progression = progression
        self.physical_category = physical_category
        self.observable_effect = observable_effect
        self.affected_signals = affected_signals

    def __repr__(self) -> str:
        return (
            f"AppliedFault(id={self.fault_id!r}, type={self.fault_type!r}, "
            f"[{self.onset_s:.2f}, {self.end_s:.2f}] s, severity={self.severity:.3f})"
        )

    def __eq__(self, other) -> bool:
        if not isinstance(other, AppliedFault):
            return NotImplemented
        return (
            self.fault_id == other.fault_id
            and self.fault_type == other.fault_type
            and self.onset_s == other.onset_s
            and self.end_s == other.end_s
            and self.severity == other.severity
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

    pos = np.clip((t_s - onset_s) / duration, 0.0, 1.0)

    if progression == "constant":
        return np.zeros_like(t_s)
    if progression == "linear":
        return pos
    if progression == "exponential":
        return pos * pos
    if progression == "step":
        return (pos > 0).astype(float)
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

    in_window = (t_s >= fault.onset_s) & (t_s < fault.end_s)
    return np.where(in_window, ramp, 0.0)


def _compute_severity_at(
    fault: ResolvedFault,
    t_s: float,
) -> float:
    """Scalar variant of _compute_severity_array."""
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
    Applies signal-layer faults to a scenario. Dispatches by the fault's
    declared injection_point (physics or post_physics).

    The injector is stateless and deterministic.
    """

    # Which physics input each physics-layer fault perturbs
    FAULT_TYPE_TO_PHYSICS_INPUT = {
        "brake_pad_wear": "brake_force_multiplier",
        "crankshaft_torque_drop": "engine_torque_multiplier",
        # crank_sensor_failure is post_physics — not in this map
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
            clean = self.generator.generate(scenario)
            return clean, []

        # --- Split faults by layer and injection point ---
        signal_faults = [f for f in scenario.resolved_faults if f.layer == "signal"]

        physics_faults = []
        post_physics_faults = []
        for f in signal_faults:
            point = self._injection_point_for(f.type)
            if point == "physics":
                physics_faults.append(f)
            elif point == "post_physics":
                post_physics_faults.append(f)

        # --- Compute physics perturbations ---
        n_steps = int(round(scenario.duration_s * scenario.base_rate_hz))
        dt = 1.0 / scenario.base_rate_hz
        t_s = np.arange(n_steps) * dt

        perturbations: dict[str, np.ndarray] = {}

        # Brake force (existing)
        brake_relevant = [
            f for f in physics_faults
            if self.FAULT_TYPE_TO_PHYSICS_INPUT.get(f.type) == "brake_force_multiplier"
        ]
        if brake_relevant:
            perturbations["brake_force_multiplier"] = self._compose_multiplier(
                brake_relevant, t_s
            )

        # Engine torque (new)
        torque_relevant = [
            f for f in physics_faults
            if self.FAULT_TYPE_TO_PHYSICS_INPUT.get(f.type) == "engine_torque_multiplier"
        ]
        if torque_relevant:
            torque_multiplier = np.ones(n_steps)
            for f in torque_relevant:
                torque_multiplier *= self._compute_torque_drop_multiplier(f, t_s)
            perturbations["engine_torque_multiplier"] = torque_multiplier

        # --- Run physics with perturbations ---
        corrupted_df = self.generator.generate(scenario, faults=perturbations)

        # --- Apply post-physics faults ---
        for f in post_physics_faults:
            corrupted_df = self._apply_post_physics_fault(corrupted_df, f)

        # --- Build applied-fault records ---
        applied: list[AppliedFault] = []
        for f in physics_faults + post_physics_faults:
            # Compute a normalized severity in [0, 1] from the fault's
            # parameters. Different fault families expose their strength
            # through different parameter names.
            severity = self._compute_normalized_severity(f)
            applied.append(AppliedFault(
                fault_id=f.id,
                fault_type=f.type,
                onset_s=f.onset_s,
                end_s=f.end_s,
                severity=severity,
                severity_start=float(f.params.get("severity_start", 0.0)),
                severity_end=float(f.params.get("severity_end", 0.0)),
                progression=f.params.get("progression", "constant"),
                physical_category=f.physical_category,
                observable_effect=f.observable_effect,
                affected_signals=list(f.targets.get("signals", [])),
            ))
        return corrupted_df, applied

    def _compute_normalized_severity(self, fault: ResolvedFault) -> float:
        """
        Return a fault's strength as a normalized [0, 1] value for the
        label builder. This lets labels mark frames as faulty regardless
        of which parameters the fault uses.
        """
        p = fault.params
        ftype = fault.type

        # brake_pad_wear: severity_end is already in [0, 1]
        if ftype == "brake_pad_wear":
            return float(p.get("severity_end", p.get("severity_start", 0.0)))

        # crankshaft_torque_drop: drop_magnitude is in [0, 1]
        if ftype == "crankshaft_torque_drop":
            return float(p.get("drop_magnitude", 0.0))

        # crank_sensor_failure: freeze_duration_s in seconds → normalized
        # against a reference max (60 s). Any positive duration means the
        # sensor is failing.
        if ftype == "crank_sensor_failure":
            duration = float(p.get("freeze_duration_s", 0.0))
            if duration <= 0.0:
                return 0.0
            return min(1.0, duration / 60.0)

        # Fallback: try severity_end, then 1.0 if a fault is declared
        # with any parameters at all.
        return float(p.get("severity_end", 1.0 if p else 0.0))
    

    # ------------------------------------------------------------------
    # Internals — dispatch
    # ------------------------------------------------------------------

    def _injection_point_for(self, fault_type: str) -> str:
        """Return the injection_point declared in the fault registry."""
        fdef = self.fault_registry.get(fault_type, {})
        return fdef.get("injection_point", "physics")

    # ------------------------------------------------------------------
    # Internals — physics fault handlers
    # ------------------------------------------------------------------

    def _compose_multiplier(
        self,
        faults: list[ResolvedFault],
        t_s: np.ndarray,
    ) -> np.ndarray:
        """
        Compose the effective multiplier from overlapping physics faults.

        Composition rule: effective_severity = 1 - Π(1 - severity_i)
        The multiplier is the complement: multiplier = Π(1 - severity_i)
        """
        multiplier = np.ones_like(t_s)
        for fault in faults:
            severity = _compute_severity_array(fault, t_s)
            severity = np.clip(severity, 0.0, 1.0)
            multiplier *= (1.0 - severity)
        return multiplier

    def _compute_torque_drop_multiplier(
        self,
        fault: ResolvedFault,
        t_s: np.ndarray,
    ) -> np.ndarray:
        """
        Build an engine torque multiplier array from a
        crankshaft_torque_drop fault.

        Drops are spaced 1/drop_frequency_hz apart within the fault
        window. Each drop has magnitude drop_magnitude and duration
        drop_duration_s. The shape of each drop is controlled by
        drop_progression.
        """
        magnitude = float(fault.params["drop_magnitude"])
        duration = float(fault.params["drop_duration_s"])
        frequency = float(fault.params["drop_frequency_hz"])
        progression = fault.params.get("drop_progression", "pulse")
        seed = int(fault.params.get("seed", 42))

        multiplier = np.ones_like(t_s)
        rng = np.random.default_rng(seed)

        period = 1.0 / frequency
        first_onset = fault.onset_s + rng.uniform(0, period)

        onset = first_onset
        while onset < fault.end_s:
            drop_end = min(onset + duration, fault.end_s)
            in_drop = (t_s >= onset) & (t_s < drop_end)
            if in_drop.any():
                local_t = t_s[in_drop] - onset
                frac = np.clip(local_t / duration, 0.0, 1.0)

                if progression == "step":
                    shape = np.ones_like(frac)
                elif progression == "linear":
                    shape = 1.0 - frac
                elif progression == "pulse":
                    shape = np.exp(-frac * 3.0)
                else:
                    raise ValueError(f"Unknown drop_progression: {progression}")

                multiplier[in_drop] = 1.0 - magnitude * shape

            onset += period

        return multiplier

    # ------------------------------------------------------------------
    # Internals — post-physics fault handlers
    # ------------------------------------------------------------------

    def _apply_post_physics_fault(
        self,
        df: pd.DataFrame,
        fault: ResolvedFault,
    ) -> pd.DataFrame:
        """Dispatch to the appropriate post-physics handler."""
        if fault.type == "crank_sensor_failure":
            return self._apply_crank_sensor_freeze(df, fault)
        raise NotImplementedError(
            f"Post-physics fault type '{fault.type}' not implemented"
        )

    def _apply_crank_sensor_freeze(
        self,
        df: pd.DataFrame,
        fault: ResolvedFault,
    ) -> pd.DataFrame:
        """
        Freeze engine_rpm and crank_position at their onset values for
        the freeze duration.
        """
        onset = fault.onset_s
        duration = float(fault.params.get("freeze_duration_s", 10.0))
        scope = fault.params.get("freeze_scope", "per_signal")

        end = min(onset + duration, fault.end_s)
        mask = (df["time_s"] >= onset) & (df["time_s"] < end)
        if not mask.any():
            return df

        # Index of first row at or after onset
        onset_idx = int(np.searchsorted(df["time_s"].values, onset, side="left"))
        # Clamp to valid range
        onset_idx = max(0, min(onset_idx, len(df) - 1))

        if scope == "per_signal":
            rpm_frozen = df["engine_rpm"].iloc[onset_idx]
            crank_frozen = df["crank_position"].iloc[onset_idx]
            df.loc[mask, "engine_rpm"] = rpm_frozen
            df.loc[mask, "crank_position"] = crank_frozen
        elif scope == "common":
            common_value = df["engine_rpm"].iloc[onset_idx]
            df.loc[mask, "engine_rpm"] = common_value
            df.loc[mask, "crank_position"] = common_value
        else:
            raise ValueError(f"Unknown freeze_scope: {scope}")

        return df