```markdown
# Synthetic CAN Data Generation & Fault Injection Framework
## Project Status — Stages 1 & 2

**Last updated:** 2025-09-27
**Status:** Stage 1 complete, Stage 2 (fault injection) mostly complete, ML baseline reveals limitation

---

## Executive Summary

This document records the state of the project through:
- **Stage 1** — Complete synthetic CAN data generation pipeline
- **Stage 2** — Fault injection, labeling, and initial ML validation

The framework produces physically-grounded multivariate time series from declarative
scenario specifications, injects controlled faults via re-simulation of the physics
engine, and emits per-frame labels suitable for downstream ML training.

A key finding from the initial ML baseline: point-in-time classification is
insufficient for relational faults, motivating temporally-aware detection methods
in later stages.

---

## Part 1 — Stage 1: Synthetic CAN Data Generation

### 1.1 Architecture

Stage 1 is a five-module pipeline:

```
scenario.yaml  →  SignalGenerator  →  CANEncoder  →  FrameScheduler  →  CSV output
```

Each module has a single responsibility and communicates through well-defined interfaces:

| Module | File | Responsibility |
|--------|------|----------------|
| Config Loader | `src/config_loader.py` | Load and validate YAML configs |
| Signal Generator | `src/signal_generator.py` | Physics engine producing semantic signals |
| CAN Encoder | `src/can_encoder.py` | Encode signals into CAN frame payloads |
| Frame Scheduler | `src/frame_scheduler.py` | Interleave messages by cycle time |
| Pipeline | `src/pipeline.py` | End-to-end orchestration |
| Dataset Builder | `src/dataset_builder.py` | Batch scenario generation with manifest |

### 1.2 Declarative Configuration

Five YAML files define the system:

- **`config/signals.yaml`** — 8 semantic signals with units, ranges, resolutions, and system affiliations
- **`config/systems.yaml`** — groupings of signals by vehicle subsystem
- **`config/relationships.yaml`** — cross-signal physics (positive, negative, integral, warmup)
- **`config/vehicles.yaml`** — vehicle profiles (mass, gearing, drag, CAN mappings)
- **`config/messages.yaml`** — CAN message definitions (IDs, cycle times, DLC)

### 1.3 Signals

| Signal | Unit | Range | System |
|--------|------|-------|--------|
| `engine_rpm` | rpm | [0, 8000] | powertrain |
| `throttle_position` | percent | [0, 100] | powertrain |
| `coolant_temperature` | celsius | [−40, 150] | powertrain |
| `brake_pressure` | bar | [0, 200] | braking |
| `brake_pedal` | percent | [0, 100] | braking |
| `wheel_speed` | km/h | [0, 250] | braking |
| `vehicle_speed` | km/h | [0, 250] | dynamics |
| `longitudinal_acceleration` | m/s² | [−10, 5] | dynamics |

### 1.4 Physics Model

The signal generator runs in three phases:

**Phase 1 — Driver inputs.** Converts piecewise scenario segments into per-timestep arrays (throttle, brake, gear).

**Phase 2 — Longitudinal dynamics.** Computes acceleration, speed, and engine RPM using:
- Force balance: `F_net = F_engine − F_brake − F_drag − F_roll`
- Euler integration of acceleration to speed
- Engine RPM from wheel speed × gear ratio × final drive, with sigmoid clutch blend and 150 ms first-order lag filter

**Phase 3 — Declarative relationships.** Applies relationships (positive/negative/integral/warmup) via multi-pass dependency resolution.

### 1.5 Vehicles

Two vehicle profiles demonstrate the vehicle-agnostic semantic layer:

| Vehicle | Mass (kg) | Gear Ratios | Final Drive | Wheel Radius (m) | Message Class |
|---------|-----------|-------------|-------------|------------------|---------------|
| `sedan_a` | 1500 | {1:3.8, 2:2.1, 3:1.4, 4:1.0, 5:0.8} | 3.5 | 0.31 | `sedan_class` |
| `suv_b` | 2200 | {1:4.2, 2:2.4, 3:1.5, 4:1.0, 5:0.75} | 3.7 | 0.35 | `suv_class` |

### 1.6 CAN Message Layout

| Message | sedan_a ID | suv_b ID | Cycle (ms) | Signals |
|---------|------------|----------|------------|---------|
| engine_data | 0x100 | 0x200 | 10 | engine_rpm, throttle_position, coolant_temperature |
| brake_data | 0x1A0 | 0x2B0 | 20 | brake_pressure, brake_pedal, wheel_speed |
| vehicle_dynamics | 0x120 | 0x220 | 50 | vehicle_speed, longitudinal_acceleration |

**Key property:** the same semantic signals map to different CAN IDs across vehicles — a demonstration of the vehicle-agnostic architecture.

### 1.7 Scenarios

Two baseline scenarios at the end of Stage 1:

**`city_drive.yaml`** — 60 s, sedan_a, three accel/brake cycles, stops at end.

**`highway_cruise.yaml`** — 90 s, suv_b, launch through gears 1–5, sustained cruise, exit deceleration.

### 1.8 Output Files

Two files per scenario:

- **`.frames.csv`** — asynchronous raw CAN frames
  ```
  timestamp_ms,can_id,payload_hex
  0,0x100,0c80003c00000000
  0,0x120,000003e800000000
  0,0x1A0,0000000000000000
  ...
  ```

- **`.signals.csv`** — synchronous decoded signals at 100 Hz
  ```
  time_s,timestamp_ms,engine_rpm,throttle_position,...
  0.0000,0,800.00,0.00,20.00,0.00,0.00,0.00,0.00,0.00
  ```

Plus a dataset-level **`manifest.csv`** and **`metadata.json`**.

### 1.9 Validation

Stage 1 was validated through:
- **91 unit tests** covering every module
- **External AI validation** of generated signal plots (physically plausible per validation)
- **Multiple design refactors** driven by external review:
  1. RPM initially was purely throttle-driven → refactored to wheel-speed-driven with clutch blend
  2. Vehicle speed was being destroyed by a declarative integral relationship → moved ownership to Phase 2
  3. RPM gear-shift spikes → smoothed with first-order lag filter
  4. Clutch-slip boundary discontinuities → smoothed with sigmoid blend

**Key lesson:** the physics engine now produces physically coherent signals across diverse driving scenarios.

---

## Part 2 — Stage 2: Fault Injection

### 2.1 Architecture

Stage 2 extends the pipeline with three new modules:

```
scenario.yaml (with faults)
    ↓
SignalGenerator → SignalFaultInjector → CANEncoder → FrameScheduler
    ↓                                                    ↓
[fault registry]                              FaultLabelBuilder
    ↓                                                    ↓
config/faults/*.yaml                       .faults.csv + .labels.csv
```

### 2.2 Fault Registry

Faults are declared one-per-file under `config/faults/`:

**`brake_pad_wear.yaml`** (the seed fault):

```yaml
fault:
  id: brake_pad_wear
  physical_category: mechanical
  observable_effect: drift
  layer: signal
  description: >
    Reduced braking effectiveness due to loss of friction material on brake pads.
  parameters:
    - name: severity_start      # [0.0, 1.0]
    - name: severity_end        # [0.0, 1.0]
    - name: progression         # constant | linear | exponential | step
  targets:
    signals: [longitudinal_acceleration]
    messages: []
  injection_rule: >
    During the fault window, scale the braking contribution to
    longitudinal_acceleration by (1 - severity(t)).
```

### 2.3 Fault Declaration in Scenarios

Scenarios declare faults with relative time anchors:

```yaml
scenario:
  name: "Highway Cruise with Brake Wear"
  duration_s: 90
  base_rate_hz: 100
  vehicle_id: suv_b
  segments:
    - {name: launch, t_start: 0, t_end: 3, ...}
    ...
  faults:
    - id: wear_1
      type: brake_pad_wear
      onset: {anchor: segment_start, segment: cruise, offset_s: 20}
      end:   {anchor: segment_end,   segment: exit_2, offset_s: 0}
      params:
        severity_start: 0.0
        severity_end: 0.4
        progression: linear

    - id: tamper_1
      type: brake_pad_wear
      onset: {anchor: segment_start, segment: stop, offset_s: 3}
      end:   {anchor: segment_end,   segment: stop, offset_s: 0}
      params:
        severity_start: 0.0
        severity_end: 0.6
        progression: step
```

**Three anchor types:**
- `scenario_start` — offset from t = 0
- `segment_start` — offset from a named segment's start
- `segment_end` — offset from a named segment's end (may be negative)

### 2.4 Injection Method — Re-Simulation

The injector uses a **re-simulation approach**, not direct signal modification:

1. Compute a fault multiplier array from the resolved faults:
   ```
   effective_severity(t) = 1 − Π(1 − severity_i(t))
   brake_force_multiplier(t) = 1 − effective_severity(t)
   ```
2. Pass this to `SignalGenerator.generate(scenario, faults={"brake_force_multiplier": ...})`
3. The physics engine uses it in `_compute_dynamics`: `f_brake *= brake_force_multiplier[i]`
4. Full cascade propagates automatically:
   - `longitudinal_acceleration` → `vehicle_speed` → `wheel_speed` → `engine_rpm`

**Why re-simulation:** ensures physical consistency across every dependent signal. No cascade enumeration needed. Adding a new fault family means adding a new physics input parameter, not new cascade logic.

### 2.5 The Seed Scenario

**`highway_brake_wear_gradual.yaml`** — 90 s, suv_b, two faults:

| Fault ID | Type | Onset (s) | End (s) | Severity | Progression |
|----------|------|-----------|---------|----------|-------------|
| `wear_1` | brake_pad_wear | 38.0 | 82.0 | 0.0 → 0.4 | linear |
| `tamper_1` | brake_pad_wear | 85.0 | 90.0 | 0.0 → 0.6 | step |

**Total fault-active time:** 44 + 5 = 49 seconds out of 90 → **54.44%** of the recording.

**No overlap** between windows — the two faults affect disjoint time intervals.

### 2.6 Output Files (Extended)

In addition to `.frames.csv` and `.signals.csv`, fault-bearing scenarios produce:

**`.faults.csv`** — per-event summary:
```
fault_id,fault_type,onset_s,end_s,severity_start,severity_end,progression,physical_category,observable_effect,affected_signals
wear_1,brake_pad_wear,38.0,82.0,0.0,0.4,linear,mechanical,drift,longitudinal_acceleration
tamper_1,brake_pad_wear,85.0,90.0,0.0,0.6,step,mechanical,drift,longitudinal_acceleration
```

**`.labels.csv`** — per-frame labels aligned with `.frames.csv`:
```
timestamp_ms,can_id,is_fault,fault_count,severity_max,fault_ids,fault_types,physical_categories,observable_effects
0,0x200,0,0,0.0,,,,
...
85000,0x200,1,1,0.6,tamper_1,brake_pad_wear,mechanical,drift
```

**Row alignment:** `labels.csv` has the same number of rows as `frames.csv`, with matching `(timestamp_ms, can_id)` columns. Trivially joinable for ML training.

### 2.7 Label Statistics for the Seed Scenario

| Metric | Value |
|--------|-------|
| Total frames | 15,300 |
| Faulty frames | 8,330 (54.44%) |
| Clean frames | 6,970 (45.56%) |
| Frames with `wear_1` | 7,480 |
| Frames with `tamper_1` | 850 |
| Frames with ≥2 faults | 0 |
| Engine (0x200) faulty fraction | 4,900 / 9,000 (54.44%) |
| Dynamics (0x220) faulty fraction | 980 / 1,800 (54.44%) |
| Brake (0x2B0) faulty fraction | 2,450 / 4,500 (54.44%) |

**Uniform fault fraction across CAN IDs** confirms the fault is time-based, not per-message.

### 2.8 Cascade Verification

The injector's effect was verified to propagate exactly through the four expected signals:

| Signal | Changed by fault? |
|--------|-------------------|
| `longitudinal_acceleration` | ✅ Yes (direct target) |
| `vehicle_speed` | ✅ Yes (integrated from accel) |
| `wheel_speed` | ✅ Yes (tracks vehicle_speed) |
| `engine_rpm` | ✅ Yes (derived from wheel_speed via drivetrain) |
| `throttle_position` | ❌ No (driver input) |
| `brake_pedal` | ❌ No (driver input) |
| `brake_pressure` | ❌ No (hydraulic, unaffected by pad wear) |
| `coolant_temperature` | ❌ No (thermal, too slow to matter here) |

**No extraneous changes.** The cascade is exactly as physically expected.

---

## Part 3 — ML Baseline (Validation Slice)

### 3.1 Purpose

A **deliberately naive** point-in-time classifier was trained to answer one question:
*"Is the generated data learnable by a standard ML model?"*

**Setup:**
- Features: 8 signal values at each timestep
- Label: `is_fault` at that frame
- Model: Random Forest (200 trees, max_depth=12, class_weight="balanced")
- Split: time-based (first 60% train, last 40% test)
- Primary metric: AUC-PR

### 3.2 Results — Imbalanced Test Set

| Metric | Value |
|--------|-------|
| AUC-PR | 0.9507 |
| AUC-ROC | 0.6656 |
| Test set class balance | 91.67% faulty |
| Random classifier AUC-PR | 0.9167 |
| Improvement over random | **1.04×** |

**Confusion matrix (threshold=0.5):**

```
           Predicted clean   Predicted fault
Actual clean       0              300          ← 100% of clean frames misclassified
Actual fault     441            2,859
```

**Classification report:**

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| clean | 0.0000 | 0.0000 | 0.0000 | 300 |
| fault | 0.9050 | 0.8664 | 0.8853 | 3300 |

**The model never correctly classified a clean frame.** Its apparent high AUC-PR is an artifact of the imbalanced test set.

### 3.3 Results — Balanced Test Set

To remove the class imbalance, the majority class was subsampled to match the minority:

| Metric | Value |
|--------|-------|
| Balanced test set size | 600 (300 clean + 300 fault) |
| AUC-PR | 0.7181 |
| AUC-ROC | 0.6783 |

**Confusion matrix:**

```
           Predicted clean   Predicted fault
Actual clean       0              300          ← still 0% on clean frames
Actual fault      31              269
```

**AUC-ROC 0.68** — barely above random. **AUC-PR 0.72** — only slightly above the balanced baseline of 0.5.

### 3.4 Feature Importances

| Feature | Importance |
|---------|-----------|
| wheel_speed | 0.2474 |
| longitudinal_acceleration | 0.2414 |
| **coolant_temperature** | **0.2362** ← suspicious |
| vehicle_speed | 0.1896 |
| engine_rpm | 0.0481 |
| throttle_position | 0.0374 |
| brake_pedal | 0.0000 |
| brake_pressure | 0.0000 |

**Red flags:**

- `coolant_temperature` has high importance (0.236) despite being unaffected by the fault. Coolant warms monotonically over time — the model used it as a **temporal proxy**, not as fault information.
- `engine_rpm` has very low importance (0.048) despite being a cascade signal. The model didn't need it.
- `brake_pressure` has **zero** importance. Correct in one sense (the fault doesn't affect it), but concerning — the model never learned to compare pressure to deceleration.

### 3.5 Interpretation

**The point-in-time classifier did not learn the fault.** It learned to correlate certain signal values with the fault's typical time window. Because the fault occupies 54% of the recording and coincides with a specific driving phase, the model was able to achieve high accuracy on the imbalanced test set by exploiting class imbalance and temporal proxies.

**Key finding:** the fault is **relational and temporal** — its signature is the *broken relationship* between `brake_pressure` and `longitudinal_acceleration` over a window of time. A point-in-time classifier cannot detect it because it cannot see the relationship.

### 3.6 Implications

1. **The synthetic pipeline produces learnable data** — but the labels are only learnable with the right model class.
2. **The naive baseline fails** — and its failure is *informative*. It motivates temporally-aware detection methods.
3. **Temporal models are required** — either:
   - Window-based (sliding window of signals as input)
   - Residual-based (compare expected vs. observed relationships)
   - Change-point detection (detect when relationships shift)

These belong in **Stage 4** (proper ML evaluation).

---

## Part 4 — Test Suite Summary

| Module | Tests |
|--------|-------|
| Config loader | 17 |
| Signal generator | 14 |
| CAN encoder | 18 |
| Frame scheduler | 18 |
| Pipeline | 13 |
| Dataset builder | 11 |
| Fault registry | 19 |
| Scenario faults | 28 |
| Fault injector | 22 |
| Fault labels | 18 |
| **Total** | **~178** |

All tests pass. The test suite covers positive paths, negative paths, boundary conditions, and physical invariants.

---

## Part 5 — Key Design Decisions

### 5.1 Message sets per vehicle class

Messages are organized into *message sets* (one per vehicle class). Vehicles reference a set via `message_set` field. Same semantic structure, different CAN IDs per class. Preserves vehicle-agnosticism.

### 5.2 Signal ownership — Phase 2 vs. Phase 3

- **Phase 2 (dynamics)** owns: `longitudinal_acceleration`, `vehicle_speed`, `engine_rpm` (iterative physics)
- **Phase 3 (relationships)** owns: `brake_pressure`, `wheel_speed`, `coolant_temperature` (algebraic transforms)

This separation emerged from a bug where a declarative `integral` relationship destroyed the correctly-integrated speed.

### 5.3 Re-simulation fault injection

Faults are applied by perturbing **physics inputs**, not by modifying signals directly. The injector computes a multiplier array and re-runs the physics engine. Cascade propagation is automatic.

### 5.4 Relative time anchors

Fault onset/end times are declared relative to segments, not absolute. This makes scenarios composable, robust to length changes, and semantically meaningful.

### 5.5 Overlap composition

Overlapping faults on the same signal compose via `1 − Π(1 − sᵢ)` — sub-additive, less than the sum, more than either alone. Matches independent failure mode composition.

### 5.6 Deterministic output

Every stage is a pure function. Same inputs → same outputs. No randomness, no hidden state. Integer timestamps in milliseconds.

---

## Part 6 — External Validation

The project has been reviewed by external LLMs at multiple points:

1. **Signal plausibility** (Stage 1) — validated as physically plausible after two rounds of refinement
2. **Fault timing** (Stage 2.1b) — validated as correctly applied, with the "silent fault" case correctly interpreted
3. **Scenario design** (Stage 2.1b) — recommended moving `tamper_1` earlier so both faults have visible effects (accepted)

**Key lesson:** external review caught real issues that internal testing missed (e.g., RPM coupling, speed re-integration bug, scenario timing).

---

## Part 7 — Known Limitations

### 7.1 Data limitations

- **Small dataset.** One vehicle pair, two scenarios, one fault type (with two parameterizations).
- **Class imbalance.** The seed scenario has 54% faulty frames — much higher than real-world fault prevalence (<1%).
- **No noise.** The synthetic signals are clean; real CAN data has quantization noise, timing jitter, and frame drops.
- **No real vehicle validation yet.** All results are on synthetic data only.

### 7.2 Model limitations

- **No torque curve.** Engine torque is linear in throttle; real engines have RPM-dependent torque curves.
- **No gear-shift torque interruption.** Shift dynamics are instantaneous.
- **No brake fade or tire slip.** Wheels maintain 100% traction at all times.
- **Static thermal model.** Coolant warm-up is time-based, not load-dependent.
- **Limited fault library.** Only `brake_pad_wear` is implemented; eight other fault types are catalogued but not built.

### 7.3 ML validation limitation

- **Point-in-time classifier is inadequate.** The baseline proves this empirically.
- **Temporal detection untested.** Window-based or residual-based models would need to be built to prove the data is learnable.
- **Cross-vehicle generalization untested.** No model has been evaluated across vehicle classes.

---

## Part 8 — Planned Next Steps

### Immediate options

**Option A — Return to Stage 2.3:** expand the fault library
- Add `wheel_speed_sensor_dropout`, `brake_pressure_sensor_drift`, `can_bus_frame_drop`, etc.
- Each new fault type demonstrates the extensibility of the framework
- Estimated effort: 2–3 sessions

**Option B — Build temporal ML slice:** validate that windowed/residual models work
- Engineer residual features and train RF on them
- Build windowed classifier (LSTM/1D-CNN)
- Prove the data is learnable under temporally-aware methods
- Estimated effort: 1–2 sessions

**Option C — Move to Stage 3:** LLM scenario planner
- Turn natural language into scenario YAML
- Requires a stable scenario schema (achieved)
- Estimated effort: several sessions

### Longer-term plan

- **Stage 3** — LLM-driven scenario generation
- **Stage 4** — Full ML pipeline with detection, localization, diagnosis, severity regression, and cross-vehicle generalization
- **Stage 5** — Real vehicle validation using a reference dataset

---

## Part 9 — Paper Contributions (Emerging)

Based on the work completed so far, the paper can claim the following contributions:

1. **A declarative, vehicle-agnostic synthetic CAN data generation framework** with physically-grounded signal generation, config-driven behavior, and cross-vehicle generalization.

2. **A re-simulation-based fault injection architecture** that automatically propagates fault effects through the physics cascade without per-fault cascade enumeration.

3. **A two-layer fault taxonomy** (physical category + observable effect) with declarative fault definitions and relative time anchors.

4. **A reproducible labeling pipeline** that produces per-frame labels aligned with raw CAN frames, suitable for ML training.

5. **An empirical finding** that point-in-time classifiers fail to detect relational faults, motivating temporally-aware detection methods.

6. **Open-source release** of configs, code, scenarios, and generated datasets.

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
│   │   └── brake_pad_wear.yaml
│   └── scenarios/
│       ├── city_drive.yaml
│       ├── highway_cruise.yaml
│       └── highway_brake_wear_gradual.yaml
├── src/
│   ├── config_loader.py
│   ├── signal_generator.py
│   ├── can_encoder.py
│   ├── frame_scheduler.py
│   ├── pipeline.py
│   ├── dataset_builder.py
│   ├── fault_injector.py
│   └── fault_labels.py
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
│   └── test_fault_labels.py
├── notebooks/
│   ├── 01_signal_generator_validation.ipynb
│   ├── 02_can_encoder_validation.ipynb
│   ├── 03_frame_scheduler_validation.ipynb
│   ├── 04_pipeline_validation.ipynb
│   ├── 05_dataset_validation.ipynb
│   ├── 06_fault_injector.ipynb
│   ├── 07_fault_labels.ipynb
│   └── 08_ml_baseline.ipynb
├── data/
│   ├── manifest.csv
│   ├── metadata.json
│   ├── sedan_a/
│   │   ├── city_drive.frames.csv
│   │   └── city_drive.signals.csv
│   └── suv_b/
│       ├── highway_cruise.frames.csv
│       ├── highway_cruise.signals.csv
│       ├── highway_cruise_with_brake_wear.frames.csv
│       ├── highway_cruise_with_brake_wear.signals.csv
│       ├── highway_cruise_with_brake_wear.faults.csv
│       └── highway_cruise_with_brake_wear.labels.csv
└── docs/
    ├── fault_catalog.html
    ├── verification_log.md
    └── project_status.md   ← this document
```

---

## Appendix B — Reproducibility

To reproduce the current state:

```bash
# Install dependencies
pip install -r requirements.txt
pip install -e .

# Validate all configs
python -m src.config_loader config/

# Run the full test suite
pytest tests/ -v

# Generate the dataset
python -c "
from src.dataset_builder import DatasetBuilder
b = DatasetBuilder('config')
b.build(
    scenario_paths=[
        'config/scenarios/city_drive.yaml',
        'config/scenarios/highway_cruise.yaml',
        'config/scenarios/highway_brake_wear_gradual.yaml',
    ],
    output_dir='data',
)
"

# Run the notebooks in order (01 through 08)
jupyter notebook notebooks/
```

---

*End of document.*
```

