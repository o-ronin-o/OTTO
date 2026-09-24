"""
CAN Encoder — converts semantic signal values into CAN frames.

Design (Step 2):
    - Input:  {signal_name: float}
    - Output: {can_id: bytes}   (payload only; timestamping is Step 3)

Decisions:
    - Missing mapping   → raise KeyError (fail-fast)
    - Out-of-range      → clamp to signals.yaml range
    - Byte ordering     → big-endian (CAN standard)
    - Payload type      → bytes (immutable, serializable)

Encoding math per signal:
    raw_int = round((value - offset) / scale)
    raw_int = clamp(raw_int, 0, 2**(8 * length) - 1)
    bytes   = raw_int.to_bytes(length, "big")
    payload[start_byte : start_byte + length] = bytes
"""

from __future__ import annotations

from typing import Any

import pandas as pd


class CANEncoder:
    """Encodes semantic signal values into CAN frames."""

    def __init__(self, vehicle_profile: dict, signals_config: dict):
        """
        Parameters
        ----------
        vehicle_profile : dict
            One entry from vehicles.yaml. Must contain `can_mapping`.
        signals_config : dict
            The full signals dict (for range clamping).
        """
        if "can_mapping" not in vehicle_profile:
            raise ValueError("Vehicle profile is missing 'can_mapping'")

        self.vehicle = vehicle_profile
        self.can_mapping = vehicle_profile["can_mapping"]
        self.signals_config = signals_config

        # Precompute payload length per can_id (in bytes)
        self._payload_lengths: dict[int, int] = {}
        for sig_name, mapping in self.can_mapping.items():
            can_id = mapping["id"]
            end_byte = mapping["start_byte"] + mapping["length"]
            self._payload_lengths[can_id] = max(
                self._payload_lengths.get(can_id, 0), end_byte
            )

        # Every CAN ID is padded to exactly 8 bytes (standard DLC)
        self._dlc = 8

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(self, signal_values: dict[str, float]) -> dict[int, bytes]:
        """
        Encode one row of signal values into a dict of CAN payloads.

        Parameters
        ----------
        signal_values : dict[str, float]
            One row from the Signal Generator's DataFrame. Must contain
            every signal declared in this vehicle's `can_mapping`.

        Returns
        -------
        dict[int, bytes]
            {can_id: 8-byte payload}

        Raises
        ------
        KeyError
            If a signal in `signal_values` has no entry in the vehicle's
            `can_mapping`, or if a required signal is missing.
        """
        # Validate: every mapped signal must be present in the input
        missing = [
            sig for sig in self.can_mapping.keys() if sig not in signal_values
        ]
        if missing:
            raise KeyError(
                f"Missing signals in input: {missing}. "
                f"Vehicle has mappings for: {list(self.can_mapping.keys())}"
            )

        # Initialize payloads for all known CAN IDs
        payloads: dict[int, bytearray] = {
            can_id: bytearray(self._dlc) for can_id in self._payload_lengths
        }

        # Encode each signal into its message
        for sig_name, mapping in self.can_mapping.items():
            value = float(signal_values[sig_name])

            # --- Range clamp ---
            if sig_name in self.signals_config:
                lo, hi = self.signals_config[sig_name]["range"]
                value = max(lo, min(hi, value))

            # --- Convert to raw integer ---
            raw = int(round((value - mapping["offset"]) / mapping["scale"]))

            # --- Clamp raw integer to fit in length bytes ---
            max_raw = 2 ** (8 * mapping["length"]) - 1
            raw = max(0, min(max_raw, raw))

            # --- Pack into payload (big-endian) ---
            encoded = raw.to_bytes(mapping["length"], "big")
            start = mapping["start_byte"]
            end = start + mapping["length"]
            payloads[mapping["id"]][start:end] = encoded

        # Convert bytearrays to immutable bytes
        return {can_id: bytes(payload) for can_id, payload in payloads.items()}

    def decode(self, can_id: int, payload: bytes) -> dict[str, float]:
        """
        Decode a CAN payload back into signal values.

        Inverse of `encode()`. Used for tests and debugging.

        Parameters
        ----------
        can_id : int
            The CAN arbitration ID.
        payload : bytes
            The 8-byte payload.

        Returns
        -------
        dict[str, float]
            {signal_name: value} for every signal mapped to this CAN ID.
        """
        if len(payload) != self._dlc:
            raise ValueError(
                f"Payload must be {self._dlc} bytes, got {len(payload)}"
            )

        result: dict[str, float] = {}
        for sig_name, mapping in self.can_mapping.items():
            if mapping["id"] != can_id:
                continue

            start = mapping["start_byte"]
            end = start + mapping["length"]
            raw = int.from_bytes(payload[start:end], "big")
            value = raw * mapping["scale"] + mapping["offset"]
            result[sig_name] = value

        return result

    def encode_dataframe(self, df: pd.DataFrame) -> list[dict[int, bytes]]:
        """
        Encode every row of a DataFrame into a list of CAN payload dicts.

        Parameters
        ----------
        df : pd.DataFrame
            Output of SignalGenerator.generate(). Must contain a column
            for every signal in the vehicle's can_mapping (plus extra
            columns, which are ignored).

        Returns
        -------
        list[dict[int, bytes]]
            One entry per row: {can_id: 8-byte payload}
        """
        results: list[dict[int, bytes]] = []
        for _, row in df.iterrows():
            signal_values = {
                sig: float(row[sig])
                for sig in self.can_mapping.keys()
                if sig in df.columns
            }
            results.append(self.encode(signal_values))
        return results

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def mapped_signal_names(self) -> list[str]:
        """Return the list of signal names this encoder knows how to encode."""
        return list(self.can_mapping.keys())

    def mapped_can_ids(self) -> list[int]:
        """Return the list of CAN IDs this encoder will produce."""
        return list(self._payload_lengths.keys())

    def payload_length(self, can_id: int) -> int:
        """Return the DLC (payload length in bytes) for a given CAN ID."""
        return self._dlc