# Mermaid Diagrams — Full Project Documentation

Below is a set of Mermaid diagrams covering the entire project, from macro architecture down to micro-level details. Each is styled with a white background for readability.

---

## Diagram 1 — Master Overview (Macro)

The full system, from scenario YAML to consumed output.

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'background': '#ffffff', 'primaryColor': '#e0f2fe', 'primaryTextColor': '#0f172a', 'primaryBorderColor': '#0284c7', 'lineColor': '#475569', 'secondaryColor': '#f1f5f9', 'tertiaryColor': '#fef3c7', 'fontFamily': 'Segoe UI, Arial' }}}%%
flowchart TB
    subgraph CONFIG["Declarative Config (YAML)"]
        direction LR
        S["signals.yaml<br/>9 signals"]
        SY["systems.yaml<br/>subsystems"]
        R["relationships.yaml<br/>cross-signal physics"]
        V["vehicles.yaml<br/>2 vehicle profiles"]
        M["messages.yaml<br/>CAN message sets"]
        F["faults/*.yaml<br/>3 fault types"]
        SC["scenarios/*.yaml<br/>5 scenarios"]
        EV["evolutions/*.yaml<br/>3 timelines"]
    end

    subgraph STAGE1["Stage 1 — Clean Data Generation"]
        SG["SignalGenerator<br/>physics engine"]
        ENC["CANEncoder<br/>signal → bytes"]
        SCH["FrameScheduler<br/>asynchronous frames"]
    end

    subgraph STAGE2["Stage 2 — Fault Injection"]
        FI["SignalFaultInjector<br/>physics + post-physics"]
        FL["FaultLabelBuilder<br/>per-frame labels"]
    end

    subgraph ORCH["Orchestration"]
        P["Pipeline<br/>single scenario"]
        DB["DatasetBuilder<br/>batch scenarios"]
        EB["EvolutionBuilder<br/>multi-day timelines"]
    end

    subgraph OUTPUT["Output Files"]
        O1[".frames.csv<br/>raw CAN wire"]
        O2[".signals.csv<br/>semantic values"]
        O3[".faults.csv<br/>event summary"]
        O4[".labels.csv<br/>per-frame labels"]
        O5["manifest.csv + metadata.json"]
        O6["evolution/*<br/>timeline + metadata"]
    end

    subgraph CONSUMERS["Downstream Consumers"]
        ML["Stage 4 — ML<br/>detection & diagnosis"]
        PAPER["Stage 2 — Paper<br/>benchmark & results"]
        REAL["Stage 5 — Real vehicle<br/>validation"]
    end

    CONFIG --> ORCH
    ORCH --> STAGE1
    STAGE1 --> STAGE2
    STAGE2 --> OUTPUT
    OUTPUT --> CONSUMERS

    style CONFIG fill:#e0f2fe,stroke:#0284c7,color:#0f172a
    style STAGE1 fill:#dcfce7,stroke:#16a34a,color:#0f172a
    style STAGE2 fill:#fee2e2,stroke:#dc2626,color:#0f172a
    style ORCH fill:#fef3c7,stroke:#d97706,color:#0f172a
    style OUTPUT fill:#e9d5ff,stroke:#9333ea,color:#0f172a
    style CONSUMERS fill:#f1f5f9,stroke:#475569,color:#0f172a
```

---

## Diagram 2 — Physical Cascade (Micro)

How a single fault propagates through the physics engine.

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'background': '#ffffff', 'primaryColor': '#e0f2fe', 'primaryTextColor': '#0f172a', 'primaryBorderColor': '#0284c7', 'lineColor': '#475569', 'fontFamily': 'Segoe UI, Arial' }}}%%
flowchart TD
    F["Fault declaration<br/>(type, params, window)"]

    F --> P{"injection_point?"}

    P -->|physics| PH["Physics input perturbation<br/>brake_force_multiplier<br/>engine_torque_multiplier"]
    P -->|post_physics| PO["Signal override<br/>after physics"]

    PH --> SIM["Re-run SignalGenerator<br/>with perturbation"]

    SIM --> P1["Phase 1<br/>driver inputs"]
    P1 --> P2["Phase 2<br/>vehicle dynamics + RPM"]
    P2 --> P3["Phase 3<br/>relationships"]
    P3 --> P4["Phase 4<br/>range clamping"]

    P4 --> CASCADE["Automatic cascade"]
    CASCADE --> C1["longitudinal_acceleration"]
    C1 --> C2["vehicle_speed"]
    C2 --> C3["wheel_speed"]
    C3 --> C4["engine_rpm"]
    C4 --> C5["crank_position"]

    PO --> CROSS["Cross-signal effect<br/>target signals only"]
    CROSS --> T1["engine_rpm frozen"]
    CROSS --> T2["crank_position frozen"]

    P4 --> OUT1["Corrupted signals"]
    CROSS --> OUT1
    OUT1 --> LABELS["Fault labels<br/>per-frame + per-event"]

    style F fill:#fef3c7,stroke:#d97706,color:#0f172a
    style P fill:#e0f2fe,stroke:#0284c7,color:#0f172a
    style PH fill:#dcfce7,stroke:#16a34a,color:#0f172a
    style PO fill:#fee2e2,stroke:#dc2626,color:#0f172a
    style CASCADE fill:#f1f5f9,stroke:#475569,color:#0f172a
    style CROSS fill:#f1f5f9,stroke:#475569,color:#0f172a
    style LABELS fill:#e9d5ff,stroke:#9333ea,color:#0f172a
```

---

## Diagram 3 — Three Fault Types, Side by Side

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'background': '#ffffff', 'primaryColor': '#e0f2fe', 'primaryTextColor': '#0f172a', 'primaryBorderColor': '#0284c7', 'lineColor': '#475569', 'fontFamily': 'Segoe UI, Arial' }}}%%
flowchart LR
    subgraph FD1["Fault 1<br/>brake_pad_wear"]
        direction TB
        A1["mechanical · drift · physics"]
        A2["Reduced braking force"]
        A3["Cascade: accel → speed → wheel → RPM"]
    end

    subgraph FD2["Fault 2<br/>crankshaft_torque_drop"]
        direction TB
        B1["mechanical · dropout · physics"]
        B2["Repeated brief torque losses"]
        B3["engine_rpm transient dips"]
    end

    subgraph FD3["Fault 3<br/>crank_sensor_failure"]
        direction TB
        C1["sensor · freeze · post_physics"]
        C2["Reported RPM stuck"]
        C3["engine_rpm + crank_position frozen<br/>everything else normal"]
    end

    style FD1 fill:#dcfce7,stroke:#16a34a,color:#0f172a
    style FD2 fill:#fef3c7,stroke:#d97706,color:#0f172a
    style FD3 fill:#fee2e2,stroke:#dc2626,color:#0f172a
```

---

## Diagram 4 — Signal Flow Inside SignalGenerator (Micro)

How the physics engine computes each signal.

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'background': '#ffffff', 'primaryColor': '#e0f2fe', 'primaryTextColor': '#0f172a', 'primaryBorderColor': '#0284c7', 'lineColor': '#475569', 'fontFamily': 'Segoe UI, Arial' }}}%%
flowchart TB
    IN["Scenario segments<br/>(throttle, brake, gear over time)"]

    IN --> PH1["Phase 1<br/>Driver inputs"]

    PH1 --> T["throttle_position[]"]
    PH1 --> B["brake_pedal[]"]
    PH1 --> G["gear[]"]

    T --> PH2
    B --> PH2
    G --> PH2

    subgraph PH2["Phase 2 — Longitudinal dynamics + Kinematic RPM"]
        direction TB
        F_BAL["Force balance<br/>F_net = F_engine − F_brake − F_drag − F_roll"]
        INT["Integrate accel → speed"]
        RPM_TGT["Kinematic RPM target<br/>locked_rpm vs slip_rpm<br/>sigmoid blend"]
        TQ["Torque multiplier effect<br/>scale deviation from idle"]
        LAG["First-order lag<br/>τ = 80 ms"]
        CRANK["Integrate crank_position<br/>rate = RPM × 6 deg/s"]

        F_BAL --> INT
        INT --> RPM_TGT
        RPM_TGT --> TQ
        TQ --> LAG
        LAG --> CRANK
    end

    PH2 --> OUT2["longitudinal_acceleration<br/>vehicle_speed<br/>engine_rpm<br/>crank_position"]

    OUT2 --> PH3["Phase 3 — relationships"]
    PH3 --> REL["brake_pressure (from pedal)<br/>wheel_speed (from speed)<br/>coolant_temperature (warmup)"]

    REL --> PH4["Phase 4 — range clamp"]
    PH4 --> DF["DataFrame (9 signals × n timesteps)"]

    style IN fill:#e0f2fe,stroke:#0284c7,color:#0f172a
    style PH1 fill:#f1f5f9,stroke:#475569,color:#0f172a
    style PH2 fill:#dcfce7,stroke:#16a34a,color:#0f172a
    style PH3 fill:#f1f5f9,stroke:#475569,color:#0f172a
    style PH4 fill:#f1f5f9,stroke:#475569,color:#0f172a
    style DF fill:#e9d5ff,stroke:#9333ea,color:#0f172a
```

---

## Diagram 5 — Config Layer (Entity Relationships)

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'background': '#ffffff', 'primaryColor': '#e0f2fe', 'primaryTextColor': '#0f172a', 'primaryBorderColor': '#0284c7', 'lineColor': '#475569', 'fontFamily': 'Segoe UI, Arial' }}}%%
erDiagram
    SIGNALS ||--o{ SYSTEMS : "grouped into"
    SIGNALS ||--o{ RELATIONSHIPS : "source or target"
    SIGNALS ||--o{ VEHICLES : "mapped via can_mapping"
    SIGNALS ||--o{ MESSAGES : "packed into"
    SIGNALS ||--o{ FAULTS : "targeted by"
    VEHICLES ||--|| MESSAGES : "references message_set"
    FAULTS ||--o{ SCENARIOS : "referenced by"
    SCENARIOS ||--o{ EVOLUTIONS : "base for"

    SIGNALS {
        string name PK
        string unit
        list range
        float resolution
        string system
    }
    SYSTEMS {
        string name PK
        list signals
    }
    RELATIONSHIPS {
        string source FK
        string target FK
        string type
        float gain
        int lag_ms
    }
    VEHICLES {
        string vehicle_id PK
        string message_set FK
        float mass_kg
        dict can_mapping
    }
    MESSAGES {
        string message_set PK
        int id
        int cycle_ms
        list signals FK
    }
    FAULTS {
        string fault_id PK
        string physical_category
        string observable_effect
        string layer
        string injection_point
    }
    SCENARIOS {
        string name PK
        float duration_s
        string vehicle_id FK
        list segments
        list faults
    }
    EVOLUTIONS {
        string name PK
        string base_scenario FK
        list recordings
    }
```

---

## Diagram 6 — Fault Injection Sequence (Micro)

Time-ordered sequence of one fault-bearing pipeline run.

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'background': '#ffffff', 'primaryColor': '#e0f2fe', 'primaryTextColor': '#0f172a', 'primaryBorderColor': '#0284c7', 'lineColor': '#475569', 'fontFamily': 'Segoe UI, Arial' }}}%%
sequenceDiagram
    participant U as Pipeline
    participant L as ConfigLoader
    participant S as Scenario
    participant I as SignalFaultInjector
    participant G as SignalGenerator
    participant E as CANEncoder
    participant F as FrameScheduler
    participant LB as FaultLabelBuilder

    U->>L: load_fault_types()
    L-->>U: fault_registry
    U->>S: from_yaml(scenario_path)
    S-->>U: scenario with resolved_faults

    U->>I: apply(scenario)
    I->>I: split faults by injection_point

    Note over I: Physics faults
    I->>I: compute multiplier arrays
    I->>G: generate(scenario, faults={...})
    G-->>I: corrupted_df

    Note over I: Post-physics faults
    I->>I: apply_signal_override(corrupted_df)

    I-->>U: (corrupted_df, applied_faults)

    U->>E: encode_dataframe(corrupted_df)
    E-->>U: frames_by_id
    U->>F: schedule(frames_by_id)
    F-->>U: scheduled_frames

    U->>LB: build(scheduled_frames, applied_faults)
    LB-->>U: event_rows, frame_rows

    U->>U: write 4 CSVs
```

---

## Diagram 7 — Stage 2 Module Map (Macro)

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'background': '#ffffff', 'primaryColor': '#e0f2fe', 'primaryTextColor': '#0f172a', 'primaryBorderColor': '#0284c7', 'lineColor': '#475569', 'fontFamily': 'Segoe UI, Arial' }}}%%
flowchart TB
    subgraph LOADER["ConfigLoader"]
        L1["load_signals / systems / relationships"]
        L2["load_vehicles / message_sets"]
        L3["load_fault_types"]
        L4["load_evolution"]
        L5["validate / validate_scenario / validate_all"]
    end

    subgraph GEN["SignalGenerator"]
        G1["generate(scenario, faults)"]
        G2["_build_driver_inputs"]
        G3["_compute_dynamics"]
        G4["_apply_relationships"]
        G5["_clamp_ranges"]
    end

    subgraph INJ["SignalFaultInjector"]
        I1["apply(scenario)"]
        I2["_compute_torque_drop_multiplier"]
        I3["_apply_post_physics_fault"]
        I4["_apply_crank_sensor_freeze"]
        I5["_compose_multiplier"]
    end

    subgraph ENC["CANEncoder"]
        E1["encode(signal_values)"]
        E2["decode(can_id, payload)"]
        E3["encode_dataframe(df)"]
    end

    subgraph SCH["FrameScheduler"]
        S1["schedule(frames_by_id)"]
        S2["Sort by (timestamp_ms, can_id)"]
    end

    subgraph LBL["FaultLabelBuilder"]
        LBL1["build(scheduled, applied_faults)"]
        LBL2["_build_event_rows"]
        LBL3["_build_frame_rows"]
    end

    subgraph PIPE["Pipeline"]
        P1["run(scenario_path)"]
        P2["run_with_fault_severity(...)"]
        P3["_run_scenario_to_directory(...)"]
    end

    subgraph BUILD["Builders"]
        DB["DatasetBuilder.build(scenarios)"]
        EB["EvolutionBuilder.build(evolution)"]
    end

    LOADER --> GEN
    LOADER --> INJ
    INJ --> GEN
    GEN --> ENC
    ENC --> SCH
    INJ --> LBL
    SCH --> LBL
    GEN --> PIPE
    ENC --> PIPE
    SCH --> PIPE
    LBL --> PIPE
    PIPE --> DB
    PIPE --> EB

    style LOADER fill:#e0f2fe,stroke:#0284c7,color:#0f172a
    style GEN fill:#dcfce7,stroke:#16a34a,color:#0f172a
    style INJ fill:#fee2e2,stroke:#dc2626,color:#0f172a
    style ENC fill:#fef3c7,stroke:#d97706,color:#0f172a
    style SCH fill:#e9d5ff,stroke:#9333ea,color:#0f172a
    style LBL fill:#fce7f3,stroke:#be185d,color:#0f172a
    style PIPE fill:#f1f5f9,stroke:#475569,color:#0f172a
    style BUILD fill:#f1f5f9,stroke:#475569,color:#0f172a
```

---

## Diagram 8 — Evolution Timeline Composition (Macro)

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'background': '#ffffff', 'primaryColor': '#e0f2fe', 'primaryTextColor': '#0f172a', 'primaryBorderColor': '#0284c7', 'lineColor': '#475569', 'fontFamily': 'Segoe UI, Arial' }}}%%
flowchart LR
    subgraph EVO["Evolution: brake_wear_95days"]
        direction TB
        E1["day_01<br/>severity 0.00"]
        E2["day_15<br/>severity 0.05"]
        E3["day_30<br/>severity 0.10"]
        E4["day_60<br/>severity 0.20"]
        E5["day_90<br/>severity 0.35"]
        E6["day_91<br/>severity 0.60"]
        E7["day_95<br/>severity 0.05"]
    end

    E1 --> R1["4 files per recording:<br/>frames, signals, faults, labels"]
    E2 --> R1
    E3 --> R1
    E4 --> R1
    E5 --> R1
    E6 --> R1
    E7 --> R1

    R1 --> SUM["Evolution summary"]
    SUM --> T["timeline.csv<br/>per-recording summary"]
    SUM --> M["metadata.json<br/>trajectory + manifest"]
    SUM --> PROV["evolution.yaml<br/>provenance copy"]

    style EVO fill:#fef3c7,stroke:#d97706,color:#0f172a
    style SUM fill:#e9d5ff,stroke:#9333ea,color:#0f172a
```

---

## Diagram 9 — Data Pipeline: From Scenario to ML Training

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'background': '#ffffff', 'primaryColor': '#e0f2fe', 'primaryTextColor': '#0f172a', 'primaryBorderColor': '#0284c7', 'lineColor': '#475569', 'fontFamily': 'Segoe UI, Arial' }}}%%
flowchart LR
    A["Scenario YAML<br/>segments + faults"] --> B["SignalGenerator<br/>physics"]

    B --> C["Clean signals_df<br/>9 signals × n_steps"]
    C --> D["SignalFaultInjector<br/>apply faults"]
    D --> E["Corrupted signals_df"]

    E --> F["CANEncoder<br/>signal → bytes"]
    F --> G["frames_by_id"]

    G --> H["FrameScheduler<br/>asynchronous"]

    H --> I[".frames.csv"]
    E --> J[".signals.csv"]

    H --> K["FaultLabelBuilder"]
    D --> K
    K --> L[".faults.csv"]
    K --> M[".labels.csv"]

    I --> N["ML features<br/>raw wire format"]
    J --> O["ML features<br/>semantic values"]
    M --> P["ML labels"]

    N --> Q["Stage 4 ML pipeline"]
    O --> Q
    P --> Q

    style A fill:#fef3c7,stroke:#d97706,color:#0f172a
    style B fill:#dcfce7,stroke:#16a34a,color:#0f172a
    style D fill:#fee2e2,stroke:#dc2626,color:#0f172a
    style H fill:#e0f2fe,stroke:#0284c7,color:#0f172a
    style K fill:#fce7f3,stroke:#be185d,color:#0f172a
    style Q fill:#e9d5ff,stroke:#9333ea,color:#0f172a
```

---

## Diagram 10 — Fault Lifecycle (Micro)

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'background': '#ffffff', 'primaryColor': '#e0f2fe', 'primaryTextColor': '#0f172a', 'primaryBorderColor': '#0284c7', 'lineColor': '#475569', 'fontFamily': 'Segoe UI, Arial' }}}%%
stateDiagram-v2
    [*] --> Declared: scenario YAML
    Declared --> Resolved: anchors resolved to absolute times
    Resolved --> Validated: params checked against registry
    Validated --> Dispatched: injection_point routing
    Dispatched --> Applied_physics: physics fault
    Dispatched --> Applied_post: post-physics fault

    Applied_physics --> Multipliers: compute perturbation array
    Multipliers --> ReSim: re-run physics
    ReSim --> Corrupted: corrupted signals

    Applied_post --> Override: override signal values
    Override --> Corrupted

    Corrupted --> Labeled: fault label builder
    Labeled --> Enriched: event + frame labels
    Enriched --> [*]

    note right of Applied_physics
        brake_pad_wear
        crankshaft_torque_drop
    end note

    note right of Applied_post
        crank_sensor_failure
    end note
```

---

## Diagram 11 — MACRO View: The Five Stages of the Project

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'background': '#ffffff', 'primaryColor': '#e0f2fe', 'primaryTextColor': '#0f172a', 'primaryBorderColor': '#0284c7', 'lineColor': '#475569', 'fontFamily': 'Segoe UI, Arial' }}}%%
flowchart LR
    S1["Stage 1<br/>Clean generation<br/>DONE"]
    S2["Stage 2<br/>Fault injection<br/>DONE through Phase B"]
    S3["Stage 3<br/>LLM scenario planner<br/>not started"]
    S4["Stage 4<br/>ML detection & diagnosis<br/>parallel track"]
    S5["Stage 5<br/>Real vehicle validation<br/>future"]

    S1 --> S2 --> S3 --> S4 --> S5

    style S1 fill:#dcfce7,stroke:#16a34a,color:#0f172a
    style S2 fill:#dcfce7,stroke:#16a34a,color:#0f172a
    style S3 fill:#fef3c7,stroke:#d97706,color:#0f172a
    style S4 fill:#e0f2fe,stroke:#0284c7,color:#0f172a
    style S5 fill:#f1f5f9,stroke:#475569,color:#0f172a
```

---

## Diagram 12 — Key Architectural Decisions (Macro)

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'background': '#ffffff', 'primaryColor': '#e0f2fe', 'primaryTextColor': '#0f172a', 'primaryBorderColor': '#0284c7', 'lineColor': '#475569', 'fontFamily': 'Segoe UI, Arial' }}}%%
mindmap
    root((Synthetic CAN<br/>Framework))
        Declarative Layer
            signals.yaml
            systems.yaml
            relationships.yaml
            vehicles.yaml
            messages.yaml
            faults/*.yaml
            scenarios/*.yaml
            evolutions/*.yaml
        Physics Engine
            Longitudinal dynamics
            Kinematic engine RPM
            Idle throttle ramp
            Crank position integration
        Fault Injection
            physics: re-simulation
            post_physics: signal override
            relative time anchors
            overlap composition
        Outputs
            frames.csv (wire)
            signals.csv (semantic)
            faults.csv (events)
            labels.csv (per-frame)
            timeline.csv + metadata.json
        Consumers
            Stage 4 ML
            Stage 2 paper
            Stage 5 real data
```

---

## Diagram 13 — Testing Hierarchy (Micro)

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'background': '#ffffff', 'primaryColor': '#e0f2fe', 'primaryTextColor': '#0f172a', 'primaryBorderColor': '#0284c7', 'lineColor': '#475569', 'fontFamily': 'Segoe UI, Arial' }}}%%
flowchart TB
    subgraph UNIT["Unit Tests (~200)"]
        U1["test_config_loader"]
        U2["test_signal_generator"]
        U3["test_can_encoder"]
        U4["test_frame_scheduler"]
        U5["test_fault_registry"]
        U6["test_scenario_faults"]
        U7["test_fault_injector"]
        U8["test_fault_labels"]
    end

    subgraph INTEGRATION["Integration Tests"]
        I1["test_pipeline"]
        I2["test_dataset_builder"]
        I3["test_evolution_builder"]
    end

    subgraph VALIDATION["Validation Artifacts"]
        V1["notebooks 01-12"]
        V2["external validator reviews"]
        V3["diagnostic scripts"]
    end

    UNIT --> INTEGRATION
    INTEGRATION --> VALIDATION

    style UNIT fill:#dcfce7,stroke:#16a34a,color:#0f172a
    style INTEGRATION fill:#fef3c7,stroke:#d97706,color:#0f172a
    style VALIDATION fill:#e0f2fe,stroke:#0284c7,color:#0f172a
```

---

## Diagram 14 — Data Volume Summary (Macro)

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'background': '#ffffff', 'primaryColor': '#e0f2fe', 'primaryTextColor': '#0f172a', 'primaryBorderColor': '#0284c7', 'lineColor': '#475569', 'fontFamily': 'Segoe UI, Arial' }}}%%
pie showData
    title Data volume by scenario type
    "Single scenarios (5)" : 5
    "Evolution: brake_wear (7)" : 7
    "Evolution: torque_drops (6)" : 6
    "Evolution: crank_sensor (6)" : 6
```

---

## Diagram 15 — Canonical Time-Series: One Fault, One Window (Micro)

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'background': '#ffffff', 'primaryColor': '#e0f2fe', 'primaryTextColor': '#0f172a', 'primaryBorderColor': '#0284c7', 'lineColor': '#475569', 'fontFamily': 'Segoe UI, Arial' }}}%%
gantt
    title Crank sensor freeze: fault window [38, 65] with freeze [38, 58]
    dateFormat X
    axisFormat %s
    
    section Vehicle state
    Launch & cruise clean    :a1, 0, 38
    Cruise (engine still running normally) :a2, 38, 58
    Cruise (sensor recovered) :a3, 58, 65
    Exit braking              :a4, 65, 90

    section engine_rpm (reported)
    Normal                   :b1, 0, 38
    FROZEN                   :crit, b2, 38, 58
    Normal                   :b3, 58, 90

    section crank_position (reported)
    Normal                   :c1, 0, 38
    FROZEN                   :crit, c2, 38, 58
    Normal                   :c3, 58, 90

    section vehicle_speed
    Normal throughout        :d1, 0, 90
```

---

## How to Use These Diagrams

**For the presentation:**

1. **Start with Diagram 1** (Master Overview) to give the big picture
2. **Show Diagram 11** (Five Stages) to position where the project is
3. **Use Diagram 3** (Three Fault Types) to introduce the fault library
4. **Show Diagram 6** (Sequence) or **Diagram 7** (Module Map) for architecture
5. **Show Diagram 15** (Gantt) as a concrete example of a sensor freeze
6. **Close with Diagram 12** (mind map) as a summary

**For the paper's method section:**

- Diagram 1 (Master Overview) — Figure 1
- Diagram 4 (Signal Flow) — Figure 2
- Diagram 6 (Sequence) — Figure 3
- Diagram 13 (Testing Hierarchy) — Figure 4 (appendix)

**For the ML LLM:**

- Diagram 2 (Fault Cascade) — explains how physics faults propagate
- Diagram 3 (Three Fault Types) — enumerates what's available
- Diagram 9 (Data Pipeline) — shows what inputs are available at each stage

---

## Rendering Notes

All diagrams use:
- **White background** (`background: '#ffffff'`)
- **Light palette** with semantic colors (green=complete, yellow=in progress, red=fault, blue=infrastructure, purple=output)
- **Mermaid `base` theme** for consistency

To render:
- **GitHub/GitLab:** paste into a `.md` file — rendered automatically
- **Mermaid Live Editor:** https://mermaid.live — paste and export SVG/PNG
- **VS Code:** Mermaid Preview extension
- **Documentation sites:** MkDocs, Docusaurus with Mermaid plugin

If you want a **single-page PDF** combining all diagrams with section headers, I can also produce a Markdown file that's ready for `pandoc` to convert. Just say the word.