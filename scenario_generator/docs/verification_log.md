# Verification Log

A step is complete only when all four phases are checked.

## Stage 1 — Complete

| Step | Module | Tests | Status |
|------|--------|-------|--------|
| 0 | Config Loader | 17 | ✅ |
| 1 | Signal Generator | 14 | ✅ |
| 2 | CAN Encoder | 18 | ✅ |
| 3 | Frame Scheduler | 18 | ✅ |
| 4 | Pipeline | 13 | ✅ |
| 5 | Dataset Builder | 11 | ✅ |

**Stage 1 test total: 91 passing**

---

## Stage 2 — In Progress

## Step 2.0 — Fault Registry & Schema
- [x] Phase 1: Design 
- [x] Phase 2: Implementated
- [x] Phase 3: Tests pass (110/110)

## Step 2.1a — Scenario Fault Declaration Schema
- [x] Phase 1: Design 
- [x] Phase 2: Implementated
- [x] Phase 3: Tests pass

## Step 2.1b — SignalFaultInjector
- [x] Phase 1: Design 
- [x] Phase 2: Implementated
- [x] Phase 3: Tests pass (163/163)


## Step 2.2 — FaultLabelBuilder
- [x] Phase 1: Design 
- [x] Phase 2: Implemented
- [x] Phase 3: Tests pass (163/163)

## Step 2.5 — Evolution Timeline Builder
- [x] Phase 1: Design 
- [x] Phase 2: Implementation 
- [x] Phase 3: Tests pass
)

Zero-severity faults are treated as non-faulty. A fault declaration with severity_start = severity_end = 0 produces a faults.csv row (for trajectory completeness) but does not set is_fault = 1 in the per-frame labels. The label reflects the fault's effect, not its scheduling window.