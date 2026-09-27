"""
Fault Label Builder — produces per-event and per-frame fault labels
for a fault-bearing scenario.

Design (Step 2.2):
    - Per-event table  ({scenario}.faults.csv):
        One row per applied fault. Human-readable summary.
    - Per-frame table  ({scenario}.labels.csv):
        One row per CAN frame. Aligned with frames.csv by
        (timestamp_ms, can_id). ML-ready.

Design choices:
    - Half-open fault window: onset_s <= t < end_s
    - Semicolon-joined lists, alphabetically sorted
    - Empty string when no fault is active at a frame
    - Scenarios without faults produce neither file
      (pipeline skips label generation entirely)
"""

from __future__ import annotations

from src.config_loader import ConfigLoader
from src.fault_injector import AppliedFault, _compute_severity_at
from src.frame_scheduler import ScheduledFrame
import numpy as np
from src.fault_injector import _progression_value


class FaultLabelBuilder:
    """
    Produces per-event and per-frame label rows from applied faults.

    Stateless. Deterministic. No I/O — the pipeline writes the rows.
    """

    def __init__(self, config_loader: ConfigLoader):
        self.cfg = config_loader
        # Load the fault registry once for metadata lookups
        self.fault_registry = config_loader.load_fault_types()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        scheduled: list[ScheduledFrame],
        applied_faults: list[AppliedFault],
    ) -> tuple[list[dict], list[dict]]:
        """
        Produce label rows for a fault-bearing scenario.

        Parameters
        ----------
        scheduled : list[ScheduledFrame]
            The chronological frame stream from FrameScheduler. Each frame
            has timestamp_ms, can_id, payload.
        applied_faults : list[AppliedFault]
            The applied fault records from SignalFaultInjector.

        Returns
        -------
        event_rows : list[dict]
            One dict per fault, for writing to faults.csv.
        frame_rows : list[dict]
            One dict per frame, for writing to labels.csv.
        """
        event_rows = self._build_event_rows(applied_faults)
        frame_rows = self._build_frame_rows(scheduled, applied_faults)
        return event_rows, frame_rows

    # ------------------------------------------------------------------
    # Per-event table
    # ------------------------------------------------------------------

    def _build_event_rows(
        self,
        applied_faults: list[AppliedFault],
    ) -> list[dict]:
        """One row per applied fault."""
        rows: list[dict] = []
        for a in applied_faults:
            rows.append({
                "fault_id": a.fault_id,
                "fault_type": a.fault_type,
                "onset_s": a.onset_s,
                "end_s": a.end_s,
                "severity_start": a.severity_start,
                "severity_end": a.severity_end,
                "progression": a.progression,
                "physical_category": a.physical_category,
                "observable_effect": a.observable_effect,
                "affected_signals": ";".join(sorted(a.affected_signals)),
            })
        return rows

    # ------------------------------------------------------------------
    # Per-frame table
    # ------------------------------------------------------------------

    def _build_frame_rows(
        self,
        scheduled: list[ScheduledFrame],
        applied_faults: list[AppliedFault],
    ) -> list[dict]:
        """
        One row per frame.

        For each frame:
            - Determine which faults are active at this frame's time
            - Compute fault_count, is_fault, fault_ids, severity_max,
              fault_types, physical_categories, observable_effects
        """
        rows: list[dict] = []
        for frame in scheduled:
            t_s = frame.timestamp_ms / 1000.0

            # Active faults at this timestamp
            active = [
                (f, _compute_severity_at_from_applied(f, t_s))
                for f in applied_faults
                if f.onset_s <= t_s < f.end_s
            ]

            if not active:
                rows.append({
                    "timestamp_ms": frame.timestamp_ms,
                    "can_id": f"0x{frame.can_id:03X}",
                    "is_fault": 0,
                    "fault_count": 0,
                    "severity_max": 0.0,
                    "fault_ids": "",
                    "fault_types": "",
                    "physical_categories": "",
                    "observable_effects": "",
                })
                continue

            # Sort active faults by id for deterministic output
            active.sort(key=lambda pair: pair[0].fault_id)

            fault_ids = [f.fault_id for f, _ in active]
            fault_types = [f.fault_type for f, _ in active]
            physical_categories = [f.physical_category for f, _ in active]
            observable_effects = [f.observable_effect for f, _ in active]
            severities = [s for _, s in active]

            rows.append({
                "timestamp_ms": frame.timestamp_ms,
                "can_id": f"0x{frame.can_id:03X}",
                "is_fault": 1,
                "fault_count": len(active),
                "severity_max": float(max(severities)),
                "fault_ids": ";".join(fault_ids),
                "fault_types": ";".join(fault_types),
                "physical_categories": ";".join(physical_categories),
                "observable_effects": ";".join(observable_effects),
            })

        return rows


# ----------------------------------------------------------------------
# Helper: severity from an AppliedFault record (not a ResolvedFault)
# ----------------------------------------------------------------------

def _compute_severity_at_from_applied(
    fault: AppliedFault,
    t_s: float,
) -> float:
    """
    Scalar severity for an AppliedFault record.

    AppliedFault carries the same params as the source ResolvedFault,
    but in flattened form. We reconstruct the same formula.
    """
   
    if not (fault.onset_s <= t_s < fault.end_s):
        return 0.0

    progress = _progression_value(
        fault.progression,
        np.array([t_s]),
        fault.onset_s,
        fault.end_s,
    )[0]
    s_start = fault.severity_start
    s_end = fault.severity_end
    return s_start + (s_end - s_start) * progress