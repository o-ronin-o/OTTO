"""
Frame Scheduler — takes encoded CAN payloads and interleaves them
into a single timestamped frame stream according to each message's
cycle time.

Design (Step 3):
    - Input:  {can_id: [payload_0, payload_1, ...]} from CANEncoder,
              plus message definitions (cycle_ms per message)
    - Output: list[{timestamp_ms, can_id, payload}], chronologically sorted

Decisions:
    - Timestamps in integer milliseconds (exact, no FP drift)
    - Base simulation rate determines the index mapping:
        idx = timestamp_ms // base_step_ms
      where base_step_ms = 1000 / base_rate_hz.
    - Frames with cycle_ms not aligned to base_step_ms are NOT supported
      (validation error).
    - Every message defined in the message set is emitted; no skipping.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ScheduledFrame:
    """One CAN frame with a timestamp."""
    timestamp_ms: int
    can_id: int
    payload: bytes

    def as_dict(self) -> dict:
        return {
            "timestamp_ms": self.timestamp_ms,
            "can_id": self.can_id,
            "payload": self.payload,
        }


class FrameScheduler:
    """Interleaves encoded CAN payloads into a timestamped stream."""

    def __init__(
        self,
        messages: dict[str, dict],
        base_rate_hz: float,
        duration_s: float,
    ):
        """
        Parameters
        ----------
        messages : dict[str, dict]
            The message set for the vehicle (one entry from messages.yaml).
            Each message must have: id, cycle_ms, dlc, signals.
        base_rate_hz : float
            The simulation rate used by the Signal Generator (e.g., 100).
        duration_s : float
            Total duration of the scenario in seconds.
        """
        self.messages = messages
        self.base_rate_hz = float(base_rate_hz)
        self.duration_s = float(duration_s)

        # Base simulation step in milliseconds
        self.base_step_ms = 1000.0 / self.base_rate_hz

        # Total number of simulation steps
        self.n_steps = int(round(self.duration_s * self.base_rate_hz))

        # Validate that every cycle_ms is an integer multiple of base_step_ms
        self._validate_cycle_alignment()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_cycle_alignment(self) -> None:
        """Ensure every message's cycle time is an integer multiple of base_step_ms."""
        for name, msg in self.messages.items():
            cycle_ms = msg["cycle_ms"]
            ratio = cycle_ms / self.base_step_ms
            if abs(ratio - round(ratio)) > 1e-9:
                raise ValueError(
                    f"Message '{name}' has cycle_ms={cycle_ms} which is not an "
                    f"integer multiple of base simulation step "
                    f"{self.base_step_ms:.4f} ms (base_rate_hz={self.base_rate_hz}). "
                    f"Cannot schedule."
                )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def schedule(
        self,
        frames_by_id: dict[int, list[bytes]],
    ) -> list[ScheduledFrame]:
        """
        Interleave encoded payloads into a chronologically sorted frame stream.

        Parameters
        ----------
        frames_by_id : dict[int, list[bytes]]
            {can_id: [payload_0, payload_1, ...]} where each payload
            corresponds to one simulation step (index i = time i * base_step_ms).

        Returns
        -------
        list[ScheduledFrame]
            Chronologically sorted. Ties (multiple messages on the same
            millisecond) are broken by ascending CAN ID.
        """
        duration_ms = int(round(self.duration_s * 1000.0))
        scheduled: list[ScheduledFrame] = []

        for msg_name, msg in self.messages.items():
            can_id = msg["id"]
            cycle_ms = int(msg["cycle_ms"])

            if can_id not in frames_by_id:
                raise KeyError(
                    f"Message '{msg_name}' (0x{can_id:X}) not found in "
                    f"frames_by_id. Encoder must produce payloads for "
                    f"every message in the message set."
                )

            payloads = frames_by_id[can_id]

            # Steps between consecutive emissions of this message
            step_interval = cycle_ms // int(self.base_step_ms)
            if step_interval <= 0:
                raise ValueError(
                    f"Message '{msg_name}' cycle_ms={cycle_ms} is shorter than "
                    f"the base step {self.base_step_ms} ms. Cannot schedule."
                )

            # Emit at t = 0, cycle_ms, 2*cycle_ms, ...
            t_ms = 0
            step_idx = 0
            while t_ms < duration_ms:
                if step_idx >= len(payloads):
                    break  # ran out of payloads (shouldn't happen with valid input)
                scheduled.append(ScheduledFrame(
                    timestamp_ms=t_ms,
                    can_id=can_id,
                    payload=payloads[step_idx],
                ))
                t_ms += cycle_ms
                step_idx += step_interval

        # Sort by timestamp ascending, ties broken by CAN ID
        scheduled.sort(key=lambda f: (f.timestamp_ms, f.can_id))
        return scheduled

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def expected_frame_count(self) -> int:
        """Return the total number of frames the scheduler will emit."""
        total = 0
        for msg in self.messages.values():
            cycle_ms = int(msg["cycle_ms"])
            total += (int(self.duration_s * 1000.0) + cycle_ms - 1) // cycle_ms
        return total

    def expected_frame_count_per_message(self) -> dict[int, int]:
        """Return the expected frame count per CAN ID."""
        duration_ms = int(self.duration_s * 1000.0)
        counts: dict[int, int] = {}
        for msg in self.messages.values():
            cycle_ms = int(msg["cycle_ms"])
            counts[msg["id"]] = (duration_ms + cycle_ms - 1) // cycle_ms
        return counts