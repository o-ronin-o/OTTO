"""
Signal Generator — converts a scenario spec into physically coherent
signal values over time.

[Stage 1 docstring unchanged]

Step 2.1a addition:
    Scenario now carries `resolved_faults`: the fault declarations from
    the YAML, with relative anchors (scenario_start, segment_start,
    segment_end) resolved to absolute times.

Step 2.1b addition:
    SignalGenerator.generate() accepts an optional `faults` dict of
    physics-input multipliers:
      - "brake_force_multiplier": scales the braking force
      - "engine_torque_multiplier": scales the engine's drive torque
        (used for torque-drop and misfire faults)

Step 2.6 / Phase A addition:
    A new signal `crank_position` exposes the crankshaft's accumulated
    rotation angle (degrees, [0, 360)).

    The engine RPM is modelled kinematically: it tracks wheel speed when
    the clutch is engaged, and slips to a throttle-driven RPM below the
    engagement threshold. This replaces a previous torque-balance model
    that proved unstable at low speeds (the crankshaft's small inertia
    caused it to free-rev to redline in <1 s).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.config_loader import ConfigLoader


# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------

@dataclass
class ResolvedFault:
    """A fault declaration with its relative anchors resolved to absolute time."""
    id: str
    type: str
    onset_s: float
    end_s: float
    params: dict
    # Registry metadata (for convenience downstream)
    physical_category: str
    observable_effect: str
    layer: str
    targets: dict

    @property
    def duration_s(self) -> float:
        return self.end_s - self.onset_s

    def contains(self, t_s: float) -> bool:
        """True if time t_s falls inside [onset_s, end_s)."""
        return self.onset_s <= t_s < self.end_s


@dataclass
class Scenario:
    name: str
    duration_s: float
    base_rate_hz: float
    vehicle_id: str
    segments: list[dict]
    resolved_faults: list[ResolvedFault] = field(default_factory=list)

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        fault_registry: dict[str, dict] | None = None,
    ) -> "Scenario":
        """
        Load a scenario YAML. If `fault_registry` is provided and the
        scenario declares `faults:`, the anchors are resolved to absolute
        times using the segment timeline.

        If `fault_registry` is None and faults are present, a ValueError
        is raised — the caller must supply the registry to enable fault
        resolution.
        """
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        if not isinstance(raw, dict) or "scenario" not in raw:
            raise ValueError(f"{path} must have a top-level 'scenario' key")

        s = raw["scenario"]
        for key in ("name", "duration_s", "base_rate_hz", "vehicle_id", "segments"):
            if key not in s:
                raise ValueError(f"Scenario missing required field: '{key}'")

        segments = list(s["segments"])
        scenario = cls(
            name=s["name"],
            duration_s=float(s["duration_s"]),
            base_rate_hz=float(s["base_rate_hz"]),
            vehicle_id=s["vehicle_id"],
            segments=segments,
            resolved_faults=[],
        )

        # Resolve faults if present
        fault_decls = s.get("faults") or []
        if fault_decls:
            if fault_registry is None:
                raise ValueError(
                    f"Scenario '{scenario.name}' declares faults but no "
                    f"fault_registry was provided to Scenario.from_yaml()"
                )
            scenario.resolved_faults = _resolve_faults(
                fault_decls, segments, scenario.duration_s, fault_registry
            )

        return scenario


# ----------------------------------------------------------------------
# Fault resolution helpers
# ----------------------------------------------------------------------

_VALID_ANCHORS = {"scenario_start", "segment_start", "segment_end"}


def _index_segments(segments: list[dict]) -> dict[str, dict]:
    """Build {segment_name: segment_dict} for named segments only."""
    index: dict[str, dict] = {}
    for seg in segments:
        name = seg.get("name")
        if name is None:
            continue
        if not isinstance(name, str) or not name:
            raise ValueError(f"Segment name must be a non-empty string, got: {name!r}")
        if name in index:
            raise ValueError(f"Duplicate segment name: '{name}'")
        index[name] = seg
    return index


def _anchor_time(
    anchor_spec: dict,
    segment_index: dict[str, dict],
    scenario_duration_s: float,
) -> float:
    """Resolve a single anchor specification to an absolute time."""
    if not isinstance(anchor_spec, dict):
        raise ValueError(f"Anchor must be a mapping, got: {anchor_spec!r}")

    anchor = anchor_spec.get("anchor")
    if anchor not in _VALID_ANCHORS:
        raise ValueError(
            f"Invalid anchor '{anchor}'. Valid: {sorted(_VALID_ANCHORS)}"
        )

    offset = float(anchor_spec.get("offset_s", 0.0))

    if anchor == "scenario_start":
        return 0.0 + offset

    segment_name = anchor_spec.get("segment")
    if not segment_name:
        raise ValueError(
            f"Anchor '{anchor}' requires a 'segment' field"
        )
    if segment_name not in segment_index:
        raise ValueError(f"Unknown segment: '{segment_name}'")

    seg = segment_index[segment_name]
    if anchor == "segment_start":
        return float(seg["t_start"]) + offset
    if anchor == "segment_end":
        return float(seg["t_end"]) + offset

    raise ValueError(f"Unhandled anchor: {anchor}")  # pragma: no cover


def _resolve_faults(
    fault_decls: list[dict],
    segments: list[dict],
    duration_s: float,
    fault_registry: dict[str, dict],
) -> list[ResolvedFault]:
    """Resolve a list of fault declarations into ResolvedFault objects."""
    segment_index = _index_segments(segments)
    resolved: list[ResolvedFault] = []
    seen_ids: set[str] = set()

    for i, decl in enumerate(fault_decls):
        if not isinstance(decl, dict):
            raise ValueError(f"Fault declaration #{i} must be a mapping")
        prefix = f"Fault #{i}"

        # --- id ---
        fault_id = decl.get("id")
        if not isinstance(fault_id, str) or not fault_id:
            raise ValueError(f"{prefix}: 'id' must be a non-empty string")
        if fault_id in seen_ids:
            raise ValueError(f"{prefix}: duplicate fault id '{fault_id}'")
        seen_ids.add(fault_id)

        # --- type ---
        fault_type = decl.get("type")
        if not isinstance(fault_type, str):
            raise ValueError(f"{prefix} ('{fault_id}'): 'type' must be a string")
        if fault_type not in fault_registry:
            raise ValueError(
                f"{prefix} ('{fault_id}'): unknown fault type '{fault_type}'. "
                f"Registered: {sorted(fault_registry.keys())}"
            )
        type_def = fault_registry[fault_type]

        # --- onset ---
        if "onset" not in decl:
            raise ValueError(f"{prefix} ('{fault_id}'): missing 'onset'")
        onset_s = _anchor_time(decl["onset"], segment_index, duration_s)

        # --- end (may be null) ---
        end_spec = decl.get("end", None)
        if end_spec is None:
            end_s = duration_s
        else:
            end_s = _anchor_time(end_spec, segment_index, duration_s)

        # --- time bounds ---
        if onset_s < 0:
            raise ValueError(
                f"{prefix} ('{fault_id}'): resolved onset {onset_s} is < 0"
            )
        if end_s > duration_s:
            raise ValueError(
                f"{prefix} ('{fault_id}'): resolved end {end_s} > "
                f"scenario duration {duration_s}"
            )
        if onset_s >= end_s:
            raise ValueError(
                f"{prefix} ('{fault_id}'): onset ({onset_s}) must be < "
                f"end ({end_s})"
            )

        # --- params ---
        params = decl.get("params", {})
        if not isinstance(params, dict):
            raise ValueError(f"{prefix} ('{fault_id}'): 'params' must be a mapping")

        declared_params = {p["name"]: p for p in type_def.get("parameters", [])}

        # unknown params
        for k in params:
            if k not in declared_params:
                raise ValueError(
                    f"{prefix} ('{fault_id}'): unknown parameter '{k}'. "
                    f"Declared: {sorted(declared_params.keys())}"
                )

        # missing required params (no default)
        for name, p in declared_params.items():
            if name not in params and "default" not in p:
                raise ValueError(
                    f"{prefix} ('{fault_id}'): missing required parameter '{name}'"
                )

        # fill defaults
        resolved_params = {}
        for name, p in declared_params.items():
            if name in params:
                value = params[name]
            else:
                value = p["default"]
            resolved_params[name] = _validate_and_coerce_param(
                value, p, prefix, fault_id
            )

        resolved.append(ResolvedFault(
            id=fault_id,
            type=fault_type,
            onset_s=onset_s,
            end_s=end_s,
            params=resolved_params,
            physical_category=type_def["physical_category"],
            observable_effect=type_def["observable_effect"],
            layer=type_def["layer"],
            targets=type_def.get("targets", {}),
        ))

    return resolved


def _validate_and_coerce_param(
    value: Any,
    param_spec: dict,
    prefix: str,
    fault_id: str,
) -> Any:
    """Validate and coerce one parameter value against its spec."""
    ptype = param_spec["type"]
    pname = param_spec["name"]
    ctx = f"{prefix} ('{fault_id}') parameter '{pname}'"

    if ptype == "float":
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"{ctx}: expected float, got {type(value).__name__}")
        value = float(value)
        rng = param_spec.get("range")
        if rng and not (rng[0] <= value <= rng[1]):
            raise ValueError(f"{ctx}: {value} outside range {rng}")
        return value

    if ptype == "int":
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"{ctx}: expected int, got {type(value).__name__}")
        rng = param_spec.get("range")
        if rng and not (rng[0] <= value <= rng[1]):
            raise ValueError(f"{ctx}: {value} outside range {rng}")
        return value

    if ptype == "enum":
        if value not in param_spec["values"]:
            raise ValueError(
                f"{ctx}: '{value}' not in {param_spec['values']}"
            )
        return value

    if ptype == "bool":
        if not isinstance(value, bool):
            raise ValueError(f"{ctx}: expected bool, got {type(value).__name__}")
        return value

    if ptype == "string":
        if not isinstance(value, str):
            raise ValueError(f"{ctx}: expected string, got {type(value).__name__}")
        return value

    raise ValueError(f"{ctx}: unknown parameter type '{ptype}'")  # pragma: no cover


# ----------------------------------------------------------------------
# Signal Generator
# ----------------------------------------------------------------------

class SignalGenerator:
    """
    Generates a DataFrame of signal values over time from a Scenario.

    Three phases:
        1. Build raw driver-input arrays from piecewise segments
        2. Longitudinal dynamics + kinematic engine RPM + crank position
        3. Enforce declarative relationships (brake_pressure, wheel_speed,
           coolant_temperature)

    Fault injection (Step 2.1b):
        generate() accepts an optional `faults` dict of physics-input
        multipliers. Supported keys:
          - "brake_force_multiplier": scales braking force
          - "engine_torque_multiplier": scales engine drive torque

    Engine RPM model (Phase A — kinematic):
        When the clutch is engaged (v >= CLUTCH_ENGAGE_KMH), engine RPM is
        locked to wheel speed via the gearbox. Below that threshold, a
        sigmoid blend transitions to a throttle-driven slip RPM. A first-
        order lag filter models engine rotational inertia.

    Crank position:
        Derived from engine RPM: d(position)/dt = RPM × 6 (deg/s).
        Wrapped to [0, 360).
    """

    # Physical constants
    RHO_AIR = 1.225            # kg/m^3
    GRAVITY = 9.81             # m/s^2

    # Clutch engagement threshold — below this speed the clutch slips
    # and the engine RPM is decoupled from wheel speed.
    CLUTCH_ENGAGE_KMH = 5.0

    # Engine RPM first-order lag time constant (s)
    TAU_ENGINE_S = 0.08

    # Slip RPM gain: slip_rpm = idle + throttle × SLIP_RPM_GAIN
    SLIP_RPM_GAIN = 2500.0

    def __init__(self, config_loader: ConfigLoader):
        self.cfg = config_loader
        self.signals = config_loader.load_signals()
        self.relationships = config_loader.load_relationships()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        scenario: Scenario,
        *,
        faults: dict[str, np.ndarray] | None = None,
    ) -> pd.DataFrame:
        """
        Run the full pipeline for one scenario.

        Parameters
        ----------
        scenario : Scenario
            The scenario to generate. Faults declared on the scenario are
            NOT applied here — pass them via the `faults` parameter.

        faults : dict[str, np.ndarray] | None
            Optional fault multipliers, keyed by physics input name.
            Supported keys:
              - "brake_force_multiplier": np.ndarray in [0, 1]
              - "engine_torque_multiplier": np.ndarray in [0, 1]
            None or missing key → no fault on that input.

        Returns
        -------
        pd.DataFrame
            The (possibly corrupted) signal DataFrame.
        """
        vehicle = self.cfg.load_vehicles()[scenario.vehicle_id]

        dt = 1.0 / scenario.base_rate_hz
        n_steps = int(round(scenario.duration_s * scenario.base_rate_hz))
        time_s = np.arange(n_steps) * dt

        driver = self._build_driver_inputs(scenario, n_steps, dt)
        dynamics = self._compute_dynamics(driver, vehicle, dt, faults=faults)
        signal_df = self._apply_relationships(driver, dynamics, dt)
        signal_df = self._clamp_ranges(signal_df)

        signal_df.insert(0, "time_s", time_s)
        return signal_df

    # ------------------------------------------------------------------
    # Phase 1 — driver inputs
    # ------------------------------------------------------------------

    def _build_driver_inputs(
        self, scenario: Scenario, n_steps: int, dt: float
    ) -> dict[str, np.ndarray]:
        """Convert piecewise segments into per-step arrays."""
        throttle = np.zeros(n_steps)
        brake = np.zeros(n_steps)
        gear = np.ones(n_steps)

        for seg in scenario.segments:
            t0 = float(seg["t_start"])
            t1 = float(seg["t_end"])
            i0 = max(0, min(int(round(t0 / dt)), n_steps))
            i1 = max(0, min(int(round(t1 / dt)), n_steps))
            if i1 <= i0:
                continue
            throttle[i0:i1] = float(seg.get("throttle_pct", 0))
            brake[i0:i1] = float(seg.get("brake_pedal_pct", 0))
            gear[i0:i1] = float(seg.get("gear", 1))

        return {
            "throttle_position": throttle,
            "brake_pedal": brake,
            "gear": gear,
        }

    # ------------------------------------------------------------------
    # Phase 2 — longitudinal dynamics + kinematic engine RPM
    # ------------------------------------------------------------------

    def _compute_dynamics(
        self,
        driver: dict[str, np.ndarray],
        vehicle: dict,
        dt: float,
        *,
        faults: dict[str, np.ndarray] | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Physics engine. Computes:
            - longitudinal_acceleration
            - vehicle_speed
            - engine_rpm (kinematically coupled to wheel speed)
            - crank_position (integrated from engine RPM)

        Engine RPM model:
            The clutch transitions from slipping (below CLUTCH_ENGAGE_KMH)
            to locked (above). A sigmoid blend smooths this transition.
            When slipping, target RPM is driven by throttle (slip_rpm).
            When locked, target RPM = wheel_rpm × gear_ratio × final_drive.
            A first-order lag filter models engine rotational inertia.

        Crank position:
            Integrated from engine RPM: rate = RPM × 6 deg/s.
            Wrapped to [0, 360).
        """
        throttle = driver["throttle_position"] / 100.0   # 0–1
        brake = driver["brake_pedal"] / 100.0            # 0–1
        gear = driver["gear"]

        # Vehicle parameters
        mass = float(vehicle["mass_kg"])
        max_brake = float(vehicle["max_brake_force_n"])
        max_torque = float(vehicle["max_engine_torque_nm"])
        cd = float(vehicle["drag_coefficient"])
        area = float(vehicle["frontal_area_m2"])
        crr = float(vehicle["rolling_resistance"])
        gear_ratios = vehicle["gear_ratios"]
        final_drive = float(vehicle["final_drive"])
        wheel_radius = float(vehicle["wheel_radius_m"])
        eff = float(vehicle["drivetrain_efficiency"])
        idle_rpm = float(vehicle["idle_rpm"])
        redline_rpm = float(vehicle["redline_rpm"])
        idle_throttle_frac = float(vehicle["idle_throttle_pct"]) / 100.0

        n = len(throttle)

        # --- Resolve fault multipliers ---
        if faults and "brake_force_multiplier" in faults:
            brake_force_multiplier = faults["brake_force_multiplier"]
            if len(brake_force_multiplier) != n:
                raise ValueError(
                    f"brake_force_multiplier length {len(brake_force_multiplier)} "
                    f"does not match simulation length {n}"
                )
        else:
            brake_force_multiplier = np.ones(n)

        if faults and "engine_torque_multiplier" in faults:
            engine_torque_multiplier = faults["engine_torque_multiplier"]
            if len(engine_torque_multiplier) != n:
                raise ValueError(
                    f"engine_torque_multiplier length {len(engine_torque_multiplier)} "
                    f"does not match simulation length {n}"
                )
        else:
            engine_torque_multiplier = np.ones(n)

        # --- State arrays ---
        speed_ms = np.zeros(n)
        accel = np.zeros(n)
        rpm = np.zeros(n)
        crank_pos = np.zeros(n)

        rpm[0] = idle_rpm
        crank_pos[0] = 0.0

        alpha_rpm = dt / (self.TAU_ENGINE_S + dt)

        for i in range(n):
            v = speed_ms[i - 1] if i > 0 else 0.0
            g = int(round(gear[i]))
            gear_ratio = float(gear_ratios.get(g, gear_ratios[max(gear_ratios)]))

            # --- Effective throttle (smooth idle floor) ---
            # Idle throttle applies at rest and fades smoothly to zero
            # as the vehicle starts moving. A hard threshold at a fixed
            # speed causes numerical chatter.
            idle_ramp_speed_ms = 2.0 / 3.6   # 2 km/h in m/s
            if v < idle_ramp_speed_ms and throttle[i] < idle_throttle_frac:
                idle_blend = 1.0 - (v / idle_ramp_speed_ms)
                idle_contribution = idle_throttle_frac * idle_blend
                eff_throttle = max(throttle[i], idle_contribution)
            else:
                eff_throttle = throttle[i]
            # --- Engine torque ---
            T_engine = eff_throttle * max_torque * engine_torque_multiplier[i]

            # --- Resistive forces ---
            f_brake = brake[i] * max_brake * brake_force_multiplier[i]
            f_drag = 0.5 * self.RHO_AIR * cd * area * v * v
            f_roll = crr * mass * self.GRAVITY

            # --- Tractive force at wheels ---
            wheel_torque = T_engine * gear_ratio * final_drive * eff
            f_engine = wheel_torque / wheel_radius

            # --- Vehicle dynamics ---
            f_net = f_engine - f_brake - f_drag - f_roll

            # At rest, if the engine can't overcome static resistance,
            # the vehicle stays put. Otherwise, normal dynamics apply.
            if v <= 1e-5 and f_engine <= (f_brake + f_roll):
                f_net = 0.0
                a = 0.0
            else:
                if v <= 0.0 and f_net < 0:
                    f_net = 0.0
                a = f_net / mass

            accel[i] = a
            if i > 0:
                speed_ms[i] = max(0.0, v + a * dt)
            else:
                speed_ms[i] = 0.0
            # --- Kinematic engine RPM ---
            v_new = speed_ms[i]
            speed_kmh = v_new * 3.6
            wheel_rpm = (v_new * 60.0) / (2.0 * np.pi * wheel_radius)
            locked_rpm = wheel_rpm * gear_ratio * final_drive

            # Slip RPM: throttle-driven flare when clutch is disengaged
            slip_rpm = idle_rpm + eff_throttle * self.SLIP_RPM_GAIN

            # Sigmoid blend between slipping and locked
            x = (speed_kmh - self.CLUTCH_ENGAGE_KMH) / (self.CLUTCH_ENGAGE_KMH * 0.5)
            x_clamped = max(-10.0, min(10.0, x))
            blend = 1.0 / (1.0 + np.exp(-x_clamped))

            target_rpm = (1.0 - blend) * slip_rpm + blend * max(idle_rpm, locked_rpm)
            target_rpm = min(target_rpm, redline_rpm)

            # --- Torque multiplier effect on RPM ---
            # A reduction in engine torque prevents RPM from rising to
            # the target. Scale the deviation-from-idle by the torque
            # multiplier. Safe because multiplier ∈ [0, 1]: target_rpm
            # remains bounded between idle_rpm and the kinematic target.
            torque_mult = engine_torque_multiplier[i]
            target_rpm = idle_rpm + (target_rpm - idle_rpm) * torque_mult

            # Reapply redline cap in case future faults use multiplier > 1
            target_rpm = min(target_rpm, redline_rpm)

            # First-order lag
            if i > 0:
                rpm[i] = rpm[i - 1] + alpha_rpm * (target_rpm - rpm[i - 1])
            else:
                rpm[i] = idle_rpm
            # --- Crank position integration ---
            # deg/s = RPM × 360 / 60 = RPM × 6
            angle_delta = (rpm[i] * 6.0) * dt
            prev_angle = crank_pos[i - 1] if i > 0 else 0.0
            crank_pos[i] = (prev_angle + angle_delta) % 360.0

        return {
            "longitudinal_acceleration": accel,
            "vehicle_speed": speed_ms * 3.6,
            "engine_rpm": rpm,
            "crank_position": crank_pos,
        }

    # ------------------------------------------------------------------
    # Phase 3 — declarative relationships
    # ------------------------------------------------------------------

    def _apply_relationships(
        self,
        driver: dict[str, np.ndarray],
        dynamics: dict[str, np.ndarray],
        dt: float,
    ) -> pd.DataFrame:
        """
        Apply declarative relationships with gain, lag, offset, and combine.

        Combine semantics:
            replace  → target = computed value
            additive → target = existing target + computed value

        Resolution: multi-pass until no progress.
        """
        signals: dict[str, np.ndarray] = {}
        signals.update(driver)
        signals.update(dynamics)

        base_len = len(next(iter(signals.values())))
        for name in self.signals:
            if name not in signals:
                signals[name] = np.zeros(base_len)

        from collections import defaultdict
        by_target: dict[str, list[dict]] = defaultdict(list)
        for rel in self.relationships:
            by_target[rel["target"]].append(rel)

        n = base_len
        max_passes = 10
        for _ in range(max_passes):
            progress = False
            for target, rels in by_target.items():
                if not all(r["source"] in signals for r in rels):
                    continue

                target_values = np.zeros(n)
                for rel in rels:
                    computed = self._compute_relationship(rel, signals, n, dt)
                    mode = rel.get("combine", "replace")
                    if mode == "additive":
                        target_values = target_values + computed
                    else:
                        target_values = computed

                signals[target] = target_values
                progress = True

            if not progress:
                break

        return pd.DataFrame(signals)

    def _compute_relationship(
        self,
        rel: dict,
        signals: dict[str, np.ndarray],
        n: int,
        dt: float,
    ) -> np.ndarray:
        """Apply one relationship's math to produce target values."""
        src = signals[rel["source"]]
        lag_steps = int(round(rel.get("lag_ms", 0) / 1000.0 / dt))

        if lag_steps > 0:
            src_lagged = np.concatenate([np.zeros(lag_steps), src[:-lag_steps]])
        else:
            src_lagged = src

        rtype = rel["type"]

        if rtype == "positive":
            return rel.get("offset", 0.0) + rel.get("gain", 1.0) * src_lagged
        if rtype == "negative":
            return rel.get("offset", 0.0) - rel.get("gain", 1.0) * src_lagged
        if rtype == "integral":
            increments = src_lagged * dt
            out = np.zeros(n)
            for i in range(1, n):
                out[i] = max(0.0, out[i - 1] + increments[i])
            return out
        if rtype == "warmup":
            tau = rel.get("tau_s", 300.0)
            t = np.arange(n) * dt
            t_amb = rel.get("ambient_temp", 20.0)
            t_tgt = rel.get("target_temp", 90.0)
            return t_amb + (t_tgt - t_amb) * (1.0 - np.exp(-t / tau))

        raise ValueError(f"Unknown relationship type: {rtype}")

    # ------------------------------------------------------------------
    # Phase 4 — clamping
    # ------------------------------------------------------------------

    def _clamp_ranges(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clamp each signal to its declared range."""
        for name, spec in self.signals.items():
            if name not in df.columns:
                continue
            lo, hi = spec["range"]
            df[name] = df[name].clip(lo, hi)
        return df