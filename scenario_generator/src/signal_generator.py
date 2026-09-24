"""
Signal Generator — converts a scenario spec into physically coherent
signal values over time.

Design (Step 1, decisions A1/B1/C2/D1, refactored after validation):
    A1 — fixed internal rate, scenario provides raw driver inputs
    B1 — additive combination for same-target relationships
    C2 — piecewise segments as scenario spec
    D1 — coolant warms up monotonically (special `warmup` type)

Refactor notes:
    - engine_rpm, longitudinal_acceleration, vehicle_speed are computed
      in Phase 2 (dynamics) using vehicle mass, gear ratios, final drive,
      and wheel radius.
    - The old `throttle_position -> engine_rpm` and
      `brake_pressure -> longitudinal_acceleration` relationships were
      removed: they ignored mass and gearing and produced physically
      incorrect behavior.
    - vehicle_speed is computed ONLY in Phase 2. It must not be
      re-derived by a declarative `integral` relationship, because that
      would restart the integration from zero and destroy accumulated
      velocity.
    - Engine RPM uses a first-order lag filter (tau ≈ 150 ms) to model
      engine rotational inertia, plus a sigmoid clutch blend to smooth
      the slip-to-locked transition.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.config_loader import ConfigLoader


# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------

@dataclass
class Scenario:
    name: str
    duration_s: float
    base_rate_hz: float
    vehicle_id: str
    segments: list[dict]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Scenario":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        if "scenario" not in raw:
            raise ValueError(f"{path} must have a top-level 'scenario' key")
        s = raw["scenario"]
        for key in ("name", "duration_s", "base_rate_hz", "vehicle_id", "segments"):
            if key not in s:
                raise ValueError(f"Scenario missing required field: '{key}'")
        return cls(
            name=s["name"],
            duration_s=float(s["duration_s"]),
            base_rate_hz=float(s["base_rate_hz"]),
            vehicle_id=s["vehicle_id"],
            segments=s["segments"],
        )


# ----------------------------------------------------------------------
# Signal Generator
# ----------------------------------------------------------------------

class SignalGenerator:
    """
    Generates a DataFrame of signal values over time from a Scenario.

    Three phases:
        1. Build raw driver-input arrays from piecewise segments
        2. Longitudinal dynamics: acceleration, speed, RPM
        3. Enforce declarative relationships (brake_pressure, wheel_speed,
           coolant_temperature)
    """

    # Physical constants
    RHO_AIR = 1.225            # kg/m^3
    GRAVITY = 9.81             # m/s^2

    # Clutch engagement threshold — below this speed the clutch slips
    # and the engine can rev independently of wheel speed.
    CLUTCH_ENGAGE_KMH = 5.0

    # Throttle blip scale during clutch slip (rpm per percent throttle)
    SLIP_RPM_PER_PCT = 30.0

    # Engine rotational inertia — time constant for RPM response
    TAU_ENGINE_S = 0.15

    def __init__(self, config_loader: ConfigLoader):
        self.cfg = config_loader
        self.signals = config_loader.load_signals()
        self.relationships = config_loader.load_relationships()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, scenario: Scenario) -> pd.DataFrame:
        """Run the full pipeline for one scenario."""
        vehicle = self.cfg.load_vehicles()[scenario.vehicle_id]

        dt = 1.0 / scenario.base_rate_hz
        n_steps = int(round(scenario.duration_s * scenario.base_rate_hz))
        time_s = np.arange(n_steps) * dt

        # Phase 1 — driver inputs
        driver = self._build_driver_inputs(scenario, n_steps, dt)

        # Phase 2 — dynamics (accel, speed, RPM)
        dynamics = self._compute_dynamics(driver, vehicle, dt)

        # Phase 3 — declarative relationships
        signal_df = self._apply_relationships(driver, dynamics, dt)

        # Phase 4 — clamp to declared ranges
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
    # Phase 2 — longitudinal dynamics
    # ------------------------------------------------------------------

    def _compute_dynamics(
        self,
        driver: dict[str, np.ndarray],
        vehicle: dict,
        dt: float,
    ) -> dict[str, np.ndarray]:
        """
        Physics engine. Computes:
            - longitudinal_acceleration
            - vehicle_speed
            - engine_rpm

        Engine RPM is filtered by a first-order lag (tau = 150 ms) to
        model engine rotational inertia. This eliminates non-physical
        instantaneous RPM step changes at gear shifts. A sigmoid clutch
        blend smooths the slip-to-locked transition at low speed.
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

        n = len(throttle)
        speed_ms = np.zeros(n)
        accel = np.zeros(n)
        rpm = np.zeros(n)
        rpm[0] = idle_rpm

        alpha_rpm = dt / (self.TAU_ENGINE_S + dt)

        for i in range(n):
            v = speed_ms[i - 1] if i > 0 else 0.0

            # Gear ratio lookup
            g = int(round(gear[i]))
            gear_ratio = float(gear_ratios.get(g, gear_ratios[max(gear_ratios)]))

            # Tractive force (linear torque model B1)
            engine_torque = throttle[i] * max_torque
            wheel_torque = engine_torque * gear_ratio * final_drive * eff
            f_engine = wheel_torque / wheel_radius

            # Resistive forces
            f_brake = brake[i] * max_brake
            f_drag = 0.5 * self.RHO_AIR * cd * area * v * v
            f_roll = crr * mass * self.GRAVITY

            f_net = f_engine - f_brake - f_drag - f_roll
            if v <= 0.0 and f_net < 0:
                f_net = 0.0

            a = f_net / mass
            accel[i] = a

            if i > 0:
                speed_ms[i] = max(0.0, v + a * dt)
            else:
                speed_ms[i] = 0.0

            # --- Engine RPM (sigmoid clutch blend + first-order lag) ---
            speed_kmh = speed_ms[i] * 3.6
            wheel_rpm = (speed_ms[i] * 60.0) / (2.0 * np.pi * wheel_radius)
            locked_rpm = wheel_rpm * gear_ratio * final_drive
            slip_rpm = idle_rpm + throttle[i] * self.SLIP_RPM_PER_PCT * 100.0

            # Sigmoid blend: 0 → pure slip, 1 → pure locked
            x = (speed_kmh - self.CLUTCH_ENGAGE_KMH) / (self.CLUTCH_ENGAGE_KMH * 0.5)
            blend = 1.0 / (1.0 + np.exp(-x))
            target_rpm = (1.0 - blend) * slip_rpm + blend * max(idle_rpm, locked_rpm)

            # First-order lag filter (models engine rotational inertia)
            if i == 0:
                rpm[i] = target_rpm
            else:
                rpm[i] = rpm[i - 1] + alpha_rpm * (target_rpm - rpm[i - 1])

        return {
            "longitudinal_acceleration": accel,
            "vehicle_speed": speed_ms * 3.6,   # convert to km/h
            "engine_rpm": rpm,
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

        Combine semantics (B1):
            replace  → target = computed value
            additive → target = existing target + computed value

        Resolution: multi-pass until no progress. Handles acyclic
        dependency graphs automatically.
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