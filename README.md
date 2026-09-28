# Synthetic CAN Data Generation & Fault Injection Framework
## Project Status — Stages 1 & 2

**Last updated:** 2025-09-28
**Status:** Stage 1 complete, Stage 2 (fault injection) complete through Phase B.
**Fault types implemented:** 3 (brake_pad_wear, crankshaft_torque_drop, crank_sensor_failure)

---

## Executive Summary

This document records the state of the project through:
- **Stage 1** — Complete synthetic CAN data generation pipeline
- **Stage 2** — Fault injection, labeling, and evolution timelines
  - Step 2.0–2.2: fault registry, scenario schema, injector, labels
  - Step 2.5: evolution timeline builder
  - Phase A: kinematic engine model + `crank_position` signal
  - Phase B: crank-related faults

The framework produces physically-grounded multivariate time series from
declarative scenario specifications, injects controlled faults via
re-simulation of the physics engine (for physics faults) or post-processing
(for sensor faults), and emits per-frame labels suitable for downstream
ML training.

An ML baseline showed point-in-time classification is insufficient for
relational faults, motivating temporally-aware detection methods in later
stages.

---

## Part 1 — Stage 1: Synthetic CAN Data Generation

### 1.1 Architecture

Stage 1 is a five-module pipeline:

```
scenario.yaml  →  SignalGenerator  →  CANEncoder  →  FrameScheduler  →  CSV output
```

| Module | File | Responsibility |
|--------|------|----------------|
| Config Loader | `src/config_loader.py` | Load and validate YAML configs |
| Signal Generator | `src/signal_generator.py` | Physics engine producing semantic signals |
| CAN Encoder | `src/can_encoder.py` | Encode signals into CAN frame payloads |
| Frame Scheduler | `src/frame_scheduler.py` | Interleave messages by cycle time |
| Pipeline | `src/pipeline.py` | End-to-end orchestration |
| Dataset Builder | `src/dataset_builder.py` | Batch scenario generation with manifest |

### 1.2 Declarative Configuration

- **`config/signals.yaml`** — 9 semantic signals
- **`config/systems.yaml`** — signal groupings by subsystem
- **`config/relationships.yaml`** — cross-signal physics
- **`config/vehicles.yaml`** — vehicle profiles + CAN mappings + crankshaft params
- **`config/messages.yaml`** — CAN message definitions
- **`config/faults/*.yaml`** — one file per fault type
- **`config/scenarios/*.yaml`** — one file per scenario
- **`config/evolutions/*.yaml`** — one file per evolution timeline

### 1.3 Signals

| Signal | Unit | Range | System |
|--------|------|-------|--------|
| `engine_rpm` | rpm | [0, 8000] | powertrain |
| `throttle_position` | percent | [0, 100] | powertrain |
| `coolant_temperature` | celsius | [−40, 150] | powertrain |
| `crank_position` | degrees | [0, 360] | powertrain |
| `brake_pressure` | bar | [0, 200] | braking |
| `brake_pedal` | percent | [0, 100] | braking |
| `wheel_speed` | km/h | [0, 250] | braking |
| `vehicle_speed` | km/h | [0, 250] | dynamics |
| `longitudinal_acceleration` | m/s² | [−10, 5] | dynamics |

### 1.4 Physics Model

**Phase 1 — Driver inputs.** Converts segments into per-step arrays.

**Phase 2 — Longitudinal dynamics + kinematic engine model.**

Vehicle dynamics use a force balance:
```
F_net = F_engine − F_brake − F_drag − F_roll
a = F_net / mass
```

Engine RPM is derived from wheel speed when the clutch is engaged:
```
locked_rpm = wheel_rpm × gear_ratio × final_drive
```
Below the engagement threshold (5 km/h), a sigmoid blend transitions to
a throttle-driven slip RPM. A first-order lag filter (τ = 80 ms) models
engine rotational inertia. RPM is capped at the redline (6500).

**Torque multiplier effect on RPM:** when `engine_torque_multiplier < 1`
(torque drop faults), the RPM target is scaled toward idle:
```
target_rpm = idle_rpm + (kinematic_target − idle_rpm) × torque_multiplier
```
This keeps the RPM bounded — the multiplier can never push RPM above
its kinematic target. (See Phase B note below.)

**Idle throttle ramp:** idle throttle fades smoothly to zero by 2 km/h,
avoiding the numerical chatter a hard threshold would cause.

**`crank_position`:** integrated from engine RPM:
```
d(crank_position)/dt = RPM × 6 deg/s
crank_position = (crank_position + RPM × 6 × dt) mod 360
```

**Phase 3 — Declarative relationships.** Applies cross-signal
relationships (positive, negative, integral, warmup) via a multi-pass
dependency resolver.

#### Note on the abandoned torque-balance model

An earlier Phase A version modeled the crankshaft as a state variable
with inertia `I_crank`, integrated by torque balance. This was
numerically unstable: the small rotational inertia caused the engine to
free-rev to redline in <1 second while the vehicle was still at rest. We
reverted to the kinematic model per an external LLM's recommendation.
Fields `moment_of_inertia_kg_m2`, `engine_friction_coeff`,
`clutch_slip_drag_coeff` remain in `vehicles.yaml` for future reference
but are unused.

### 1.5 Vehicles

| Vehicle | Mass (kg) | Gear Ratios | Final Drive | Wheel Radius (m) | Redline | Message Class |
|---------|-----------|-------------|-------------|------------------|---------|---------------|
| `sedan_a` | 1500 | {1:3.8, 2:2.1, 3:1.4, 4:1.0, 5:0.8} | 3.5 | 0.31 | 6500 | `sedan_class` |
| `suv_b` | 2200 | {1:4.2, 2:2.4, 3:1.5, 4:1.0, 5:0.75} | 3.7 | 0.35 | 6500 | `suv_class` |

### 1.6 CAN Message Layout

| Message | sedan_a ID | suv_b ID | Cycle (ms) | Signals |
|---------|------------|----------|------------|---------|
| engine_data | 0x100 | 0x200 | 10 | engine_rpm, throttle_position, coolant_temperature, crank_position |
| brake_data | 0x1A0 | 0x2B0 | 20 | brake_pressure, brake_pedal, wheel_speed |
| vehicle_dynamics | 0x120 | 0x220 | 50 | vehicle_speed, longitudinal_acceleration |

### 1.7 Scenarios

- `city_drive.yaml` — 60 s, sedan_a, clean
- `highway_cruise.yaml` — 90 s, suv_b, clean
- `highway_brake_wear_gradual.yaml` — 90 s, suv_b, brake pad wear
- `highway_cruise_torque_drop.yaml` — 90 s, suv_b, intermittent torque drops
- `highway_cruise_crank_sensor_freeze.yaml` — 90 s, suv_b, crank sensor freeze

### 1.8 Output Files

- **`.frames.csv`** — raw CAN frames
- **`.signals.csv`** — synchronous decoded signals (100 Hz)
- **`.faults.csv`** — one row per applied fault (only for fault-bearing scenarios)
- **`.labels.csv`** — one row per frame (only for fault-bearing scenarios)
- Plus dataset-level **`manifest.csv`** and **`metadata.json`**

### 1.9 Validation

Stage 1 was validated via ~93 tests and external review. Multiple
refactors: RPM coupling, speed re-integration, gear-shift smoothing,
clutch blend, acceleration chatter, kinematic engine model.

---

## Part 2 — Stage 2: Fault Injection

### 2.1 Architecture

```
scenario.yaml (with faults)
    ↓
SignalGenerator → SignalFaultInjector → CANEncoder → FrameScheduler
    ↓                                              ↓
[fault registry]                          FaultLabelBuilder
    ↓                                              ↓
config/faults/*.yaml               .faults.csv + .labels.csv
```

The injector dispatches by the fault's `injection_point`:
- `physics` — perturb a physics input and re-run
- `post_physics` — override signal values after physics has run

### 2.2 Fault Registry Schema

Each fault declares:
- `id`, `physical_category`, `observable_effect`, `layer`
- `injection_point` (`physics` | `post_physics`, default `physics`)
- `severity_maps_to` (parameter name that evolution severity maps to)
- `description`, `parameters`, `targets`, `injection_rule`

### 2.3 Implemented Fault Types

#### Fault 1 — `brake_pad_wear`

- Physical category: mechanical
- Observable effect: drift
- Injection point: physics
- Target: `longitudinal_acceleration` (cascades to speed, wheel speed, RPM)
- Parameters: `severity_start`, `severity_end`, `progression`

**Cascade verification (verified in notebook 06):**
| Signal | Changed? |
|--------|----------|
| `longitudinal_acceleration` | ✅ Yes |
| `vehicle_speed` | ✅ Yes |
| `wheel_speed` | ✅ Yes |
| `engine_rpm` | ✅ Yes |
| `throttle_position` | ❌ No |
| `brake_pedal` | ❌ No |
| `brake_pressure` | ❌ No |
| `coolant_temperature` | ❌ No |

#### Fault 2 — `crankshaft_torque_drop`

- Physical category: mechanical
- Observable effect: dropout (transient)
- Injection point: physics
- Target: `engine_rpm`, `crank_position`
- Parameters: `drop_magnitude`, `drop_duration_s`, `drop_frequency_hz`,
  `drop_progression` (step | linear | pulse), `seed`
- `severity_maps_to`: `drop_magnitude`

**Behavior:** multiple drops per recording spaced by `drop_frequency_hz`.
Each drop reduces engine torque by `drop_magnitude`. RPM dips
transiently and recovers via the first-order lag. Verified in notebook 12.

**RPM scaling:** the torque multiplier scales the deviation-from-idle
of the RPM target. This keeps RPM bounded by construction.

#### Fault 3 — `crank_sensor_failure`

- Physical category: sensor
- Observable effect: freeze
- Injection point: post_physics
- Target: `engine_rpm`, `crank_position`
- Parameters: `freeze_duration_s`, `freeze_scope` (per_signal | common)
- `severity_maps_to`: `freeze_duration_s` (normalized against 60 s)
- Modes supported: **freeze only** (drift/dropout/noise deferred)

**Cross-signal signature** (verified in notebook 12):
| Signal | Affected? | Reason |
|--------|-----------|--------|
| `engine_rpm` | ✅ Frozen | Crank sensor reading |
| `crank_position` | ✅ Frozen | Same sensor |
| `vehicle_speed` | ❌ No | Different sensor |
| `wheel_speed` | ❌ No | Different sensor |
| `longitudinal_acceleration` | ❌ No | Physics, not sensor |
| `throttle_position` | ❌ No | Different sensor |
| `brake_pressure` | ❌ No | Different sensor |
| `coolant_temperature` | ❌ No | Different sensor |

**Discriminating pattern:** The engine is physically fine, but the
reported RPM is frozen. A model that correlates RPM with
`throttle_position` and `longitudinal_acceleration` can detect the
discrepancy.

### 2.4 Fault Injection Method — Re-Simulation

For `physics`-layer faults:
1. Compute the fault multiplier array from the resolved faults
2. Pass to `SignalGenerator.generate(scenario, faults={...})`
3. Physics engine re-runs with the multipliers
4. Cascade propagates automatically

For `post_physics`-layer faults:
1. Physics engine runs cleanly
2. Post-physics handler overrides the target signal values

Overlap composition: `effective_severity = 1 − Π(1 − severity_i)`

### 2.5 Seed Scenarios

**`highway_brake_wear_gradual.yaml`:**
| Fault | Window (s) | Severity | Progression |
|-------|------------|----------|-------------|
| `wear_1` | 38.0 → 82.0 | 0.0 → 0.4 | linear |
| `tamper_1` | 85.0 → 90.0 | 0.0 → 0.6 | step |

**`highway_cruise_torque_drop.yaml`:**
| Fault | Window (s) | Params |
|-------|------------|--------|
| `misfire_train` | 28.0 → 65.0 | magnitude 0.7, duration 0.15 s, frequency 0.4 Hz, progression pulse |

**`highway_cruise_crank_sensor_freeze.yaml`:**
| Fault | Window (s) | Params |
|-------|------------|--------|
| `sensor_freeze` | 38.0 → 65.0 | duration 20 s, scope per_signal |

### 2.6 Output Files (Fault-Bearing Scenarios)

**`.faults.csv`** — one row per applied fault:
```
fault_id,fault_type,onset_s,end_s,severity,severity_start,severity_end,progression,physical_category,observable_effect,affected_signals
wear_1,brake_pad_wear,38.0,82.0,0.4,0.0,0.4,linear,mechanical,drift,longitudinal_acceleration
misfire_train,crankshaft_torque_drop,28.0,65.0,0.7,0.0,0.0,constant,mechanical,dropout,crank_position;engine_rpm
sensor_freeze,crank_sensor_failure,38.0,65.0,0.333,0.0,0.0,constant,sensor,freeze,crank_position;engine_rpm
```

The `severity` column is the normalized strength used by the label
builder. `severity_start` / `severity_end` are legacy fields used only
by `brake_pad_wear`.

**`.labels.csv`** — one row per frame, aligned with `.frames.csv`:
```
timestamp_ms,can_id,is_fault,fault_count,severity_max,fault_ids,fault_types,physical_categories,observable_effects
```

Row-aligned with `frames.csv` by `(timestamp_ms, can_id)`.

### 2.7 Label Statistics

| Scenario | Faulty frames | Total | % |
|----------|--------------|-------|---|
| highway_brake_wear_gradual | 8,330 | 15,300 | 54.44% |
| highway_cruise_torque_drop | 6,290 | 15,300 | 41.11% |
| highway_cruise_crank_sensor_freeze | 4,590 | 15,300 | 30.00% |

### 2.8 Evolution Timelines

The `EvolutionBuilder` composes N recordings of the same base scenario at
different simulated days, each with a different fault severity.

Three evolutions implemented:
- `brake_wear_95days.yaml` — 7 recordings
- `torque_drops_90days.yaml` — 6 recordings
- `crank_sensor_freeze_90days.yaml` — 6 recordings

Each evolution produces:
- Per-recording `.frames.csv`, `.signals.csv`, `.faults.csv`, `.labels.csv`
- Evolution-level `evolution.yaml` (source copy), `timeline.csv`, `metadata.json`

### 2.9 Fault Injection Dispatch

The injector reads `injection_point` from the fault registry for each
resolved fault and routes it to:
- The physics path (via `_compose_multiplier` and `_compute_torque_drop_multiplier`)
- The post-physics path (via `_apply_post_physics_fault`)

---

## Part 3 — ML Baseline (Validation Slice)

A deliberately naive point-in-time classifier was trained on `brake_pad_wear`.

**Results — imbalanced test set (91.67% faulty):**
| Metric | Value |
|--------|-------|
| AUC-PR | 0.9507 |
| AUC-ROC | 0.6656 |
| Random baseline | 0.9167 |
| Improvement factor | 1.04× |

**Results — balanced test set:**
| Metric | Value |
|--------|-------|
| AUC-PR | 0.7181 |
| AUC-ROC | 0.6783 |

**The model never correctly classified a clean frame.** The high
imbalanced AUC-PR was an artifact of class imbalance.

**Key finding:** the fault is relational and temporal. A point-in-time
classifier cannot detect the broken relationship between
`brake_pressure` and `longitudinal_acceleration`.

**Implication:** temporally-aware detection (window-based, residual-based,
change-point detection) is required. This is a Stage 4 concern.

---

## Part 4 — Test Suite

| Module | Tests |
|--------|-------|
| Config loader | ~21 |
| Signal generator | ~26 |
| CAN encoder | ~19 |
| Frame scheduler | ~19 |
| Pipeline | ~13 |
| Dataset builder | ~11 |
| Fault registry | ~19 |
| Scenario faults | ~28 |
| Fault injector | ~22 |
| Fault labels | ~20 |
| Evolution builder | ~22 |
| **Total** | **~240** |

All tests pass.

---

## Part 5 — Key Design Decisions

- **Message sets per vehicle class** — same semantic structure, different CAN IDs
- **Signal ownership** — Phase 2 owns dynamics + crank_position; Phase 3 owns relationships
- **Re-simulation fault injection** for physics faults
- **Post-physics signal override** for sensor faults
- **Relative time anchors** for fault declarations
- **Overlap composition** via `1 − Π(1 − sᵢ)`
- **Deterministic output** — no randomness except fault drop seeds
- **Kinematic engine model** — stable, physically defensible
- **Smooth idle ramp** — avoids low-speed chatter

---

## Part 6 — External Validation

Reviews by external LLMs at multiple points:
1. Signal plausibility (Stage 1)
2. Fault timing (Stage 2.1b)
3. Scenario design (Stage 2.1b)
4. Crankshaft physics (Phase A) — recommended kinematic model
5. Acceleration chatter (Phase A)
6. Crank position aliasing (visualization issue)
7. Phase B crank faults (final review — "physically correct and highly distinguishable")

---

## Part 7 — Known Limitations

**Data limitations:**
- 5 scenarios, 3 evolution timelines, 3 fault types
- Class imbalance in labels (30–54% faulty frames; real faults <1%)
- No noise in synthetic signals
- No real vehicle validation yet

**Model limitations:**
- Linear torque model (no torque curve)
- No gear-shift torque interruption
- No brake fade or tire slip
- Static thermal model
- Kinematic engine model (no dynamic crankshaft transients)
- Crank sensor failure supports only `freeze` mode (drift/dropout/noise deferred)

**ML validation limitation:**
- Point-in-time classifier inadequate (proven empirically)
- Temporal detection untested
- Cross-vehicle generalization untested

---

## Part 8 — Planned Next Steps

**Batch 2 — Remaining engine-side faults:**
- `throttle_sensor_gain_fault`
- `engine_misfire` (or is `crankshaft_torque_drop` sufficient?)
- `coolant_sensor_drift`

**Batch 3 — Vehicle dynamics faults:**
- `rolling_resistance_increase`
- `aerodynamic_drag_increase`

**Frame-layer faults:**
- `can_bus_frame_drop`
- `can_bus_frame_duplicate`
- `can_bus_jitter`

**Signal expansion:**
- `mass_air_flow`, `steering_angle`, `battery_voltage`, etc.

**Longer-term:**
- Stage 3 (LLM scenario planner)
- Stage 4 (full ML pipeline)
- Stage 5 (real vehicle validation)

---

## Part 9 — Paper Contributions (Emerging)

1. A declarative, vehicle-agnostic synthetic CAN data generation framework
2. A hybrid fault injection architecture (re-simulation for physics, override for sensor)
3. A two-layer fault taxonomy (physical category + observable effect)
4. A reproducible labeling pipeline for ML training
5. An empirical finding that point-in-time classifiers fail on relational faults
6. A kinematic engine model that is physically defensible and numerically stable
7. Open-source release of configs, code, scenarios, and generated datasets

---

## Appendix A — Directory Structure

```
scenario_generator/
├── config/
│   ├── signals.yaml
│   ├── systems.yaml
│   ├── relationships.yaml
│   ├── vehicles.yaml
│   ├── messages.yaml
│   ├── faults/
│   │   ├── brake_pad_wear.yaml
│   │   ├── crankshaft_torque_drop.yaml
│   │   └── crank_sensor_failure.yaml
│   ├── scenarios/
│   │   ├── city_drive.yaml
│   │   ├── highway_cruise.yaml
│   │   ├── highway_brake_wear_gradual.yaml
│   │   ├── highway_cruise_torque_drop.yaml
│   │   └── highway_cruise_crank_sensor_freeze.yaml
│   └── evolutions/
│       ├── brake_wear_95days.yaml
│       ├── torque_drops_90days.yaml
│       └── crank_sensor_freeze_90days.yaml
├── src/
│   ├── config_loader.py
│   ├── signal_generator.py
│   ├── can_encoder.py
│   ├── frame_scheduler.py
│   ├── pipeline.py
│   ├── dataset_builder.py
│   ├── fault_injector.py
│   ├── fault_labels.py
│   └── evolution_builder.py
├── tests/
│   ├── test_config_loader.py
│   ├── test_signal_generator.py
│   ├── test_can_encoder.py
│   ├── test_frame_scheduler.py
│   ├── test_pipeline.py
│   ├── test_dataset_builder.py
│   ├── test_fault_registry.py
│   ├── test_scenario_faults.py
│   ├── test_fault_injector.py
│   ├── test_fault_labels.py
│   └── test_evolution_builder.py
├── notebooks/
│   ├── 01–11 (existing validation notebooks)
│   └── 12_crank_faults.ipynb
├── scripts/
│   └── build_all_data.py
├── data/                          ← gitignored
│   ├── manifest.csv
│   ├── metadata.json
│   ├── sedan_a/
│   ├── suv_b/
│   └── ...
├── docs/
│   ├── fault_catalog.html
│   ├── verification_log.md
│   └── project_status.md          ← this document
└── .gitignore
```

---

## Appendix B — Reproducibility

```bash
pip install -r requirements.txt
pip install -e .

python -m src.config_loader config/
pytest tests/ -v
python -m scripts.build_all_data
jupyter notebook notebooks/
```

---

*End of document.*
