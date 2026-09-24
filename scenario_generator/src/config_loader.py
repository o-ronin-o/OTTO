"""
ConfigLoader — reads and validates YAML configuration files.

This module is the foundation of Stage 1. It ensures that all config
files are internally consistent before any generation begins.

Architecture note (Option C):
    Messages are organized into *message sets* (one per vehicle class).
    Vehicles reference a message set via the `message_set` field.
    This preserves vehicle-agnosticism: same semantic message structure,
    different CAN IDs per class.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


# Required top-level keys in each config file
REQUIRED_KEYS = {
    "signals.yaml": ["signals"],
    "systems.yaml": ["systems"],
    "relationships.yaml": ["relationships"],
    "vehicles.yaml": ["vehicles"],
    "messages.yaml": ["message_sets"],
}

# Required fields for each entity type
SIGNAL_REQUIRED_FIELDS = ["unit", "range", "resolution", "description", "system"]
RELATIONSHIP_REQUIRED_FIELDS = ["source", "target", "type", "description"]
VEHICLE_REQUIRED_FIELDS = ["name", "message_set", "mass_kg", "can_mapping"]
MESSAGE_REQUIRED_FIELDS = ["id", "cycle_ms", "dlc", "signals"]
CAN_MAPPING_REQUIRED_FIELDS = ["id", "start_byte", "length", "scale", "offset"]

# Valid relationship types
VALID_RELATIONSHIP_TYPES = ["positive", "negative", "integral", "warmup"]


class ConfigLoader:
    """Loads and validates all YAML configuration files."""

    def __init__(self, config_dir: str | Path):
        self.config_dir = Path(config_dir)
        if not self.config_dir.is_dir():
            raise FileNotFoundError(f"Config directory not found: {self.config_dir}")

        # Cache loaded configs
        self._signals: dict[str, dict] | None = None
        self._systems: dict[str, dict] | None = None
        self._relationships: list[dict] | None = None
        self._vehicles: dict[str, dict] | None = None
        self._message_sets: dict[str, dict] | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_yaml(self, filename: str) -> dict:
        path = self.config_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing config file: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError(f"{filename} must contain a YAML mapping at the top level")

        for key in REQUIRED_KEYS.get(filename, []):
            if key not in data:
                raise ValueError(f"{filename} is missing required top-level key: '{key}'")

        return data

    # ------------------------------------------------------------------
    # Public loaders
    # ------------------------------------------------------------------

    def load_signals(self) -> dict[str, dict]:
        if self._signals is None:
            raw = self._load_yaml("signals.yaml")
            self._signals = raw["signals"]
        return self._signals

    def load_systems(self) -> dict[str, dict]:
        if self._systems is None:
            raw = self._load_yaml("systems.yaml")
            self._systems = raw["systems"]
        return self._systems

    def load_relationships(self) -> list[dict]:
        if self._relationships is None:
            raw = self._load_yaml("relationships.yaml")
            self._relationships = raw["relationships"]
        return self._relationships

    def load_vehicles(self) -> dict[str, dict]:
        if self._vehicles is None:
            raw = self._load_yaml("vehicles.yaml")
            self._vehicles = raw["vehicles"]
        return self._vehicles

    def load_message_sets(self) -> dict[str, dict]:
        """Returns {message_set_name: {message_name: message_def}}."""
        if self._message_sets is None:
            raw = self._load_yaml("messages.yaml")
            self._message_sets = raw["message_sets"]
        return self._message_sets

    def get_messages_for_vehicle(self, vehicle_id: str) -> dict:
        """Return the message definitions for a specific vehicle."""
        vehicles = self.load_vehicles()
        if vehicle_id not in vehicles:
            raise KeyError(f"Unknown vehicle: {vehicle_id}")
        message_set = vehicles[vehicle_id].get("message_set")
        if message_set is None:
            raise ValueError(f"Vehicle '{vehicle_id}' has no 'message_set' defined")
        message_sets = self.load_message_sets()
        if message_set not in message_sets:
            raise KeyError(f"Message set '{message_set}' not found")
        return message_sets[message_set]

    def load_all(self) -> dict[str, Any]:
        return {
            "signals": self.load_signals(),
            "systems": self.load_systems(),
            "relationships": self.load_relationships(),
            "vehicles": self.load_vehicles(),
            "message_sets": self.load_message_sets(),
        }

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> list[str]:
        errors: list[str] = []

        signals = self.load_signals()
        systems = self.load_systems()
        relationships = self.load_relationships()
        vehicles = self.load_vehicles()
        message_sets = self.load_message_sets()

        signal_names = set(signals.keys())

        # --- Rule 1: required fields on each signal ---
        for name, sig in signals.items():
            for field in SIGNAL_REQUIRED_FIELDS:
                if field not in sig:
                    errors.append(f"Signal '{name}' missing required field '{field}'")
            if "range" in sig:
                r = sig["range"]
                if not (isinstance(r, list) and len(r) == 2):
                    errors.append(f"Signal '{name}' range must be a 2-element list")
                elif r[0] >= r[1]:
                    errors.append(f"Signal '{name}' range min must be < max")

        # --- Rule 2: signals in systems.yaml must exist ---
        for sys_name, sys_def in systems.items():
            if "signals" not in sys_def:
                errors.append(f"System '{sys_name}' missing 'signals' list")
                continue
            for s in sys_def["signals"]:
                if s not in signal_names:
                    errors.append(
                        f"System '{sys_name}' references unknown signal '{s}'"
                    )

        # --- Rule 3: signals in relationships.yaml must exist ---
        for i, rel in enumerate(relationships):
            for field in RELATIONSHIP_REQUIRED_FIELDS:
                if field not in rel:
                    errors.append(f"Relationship #{i} missing required field '{field}'")
            for key in ("source", "target"):
                if key in rel and rel[key] not in signal_names:
                    errors.append(
                        f"Relationship #{i} references unknown signal '{rel[key]}'"
                    )
            if "type" in rel and rel["type"] not in VALID_RELATIONSHIP_TYPES:
                errors.append(
                    f"Relationship #{i} has invalid type '{rel['type']}'. "
                    f"Valid: {VALID_RELATIONSHIP_TYPES}"
                )

        # --- Rule 4: signals in vehicles' can_mapping must exist ---
        for vname, vdef in vehicles.items():
            for field in VEHICLE_REQUIRED_FIELDS:
                if field not in vdef:
                    errors.append(f"Vehicle '{vname}' missing required field '{field}'")
            can_map = vdef.get("can_mapping", {})
            for sig_name, mapping in can_map.items():
                if sig_name not in signal_names:
                    errors.append(
                        f"Vehicle '{vname}' CAN mapping references unknown signal '{sig_name}'"
                    )
                for field in CAN_MAPPING_REQUIRED_FIELDS:
                    if field not in mapping:
                        errors.append(
                            f"Vehicle '{vname}' signal '{sig_name}' mapping "
                            f"missing field '{field}'"
                        )

        # --- Rule 5: signals in messages must exist; required fields present ---
        for set_name, msg_set in message_sets.items():
            for mname, mdef in msg_set.items():
                for field in MESSAGE_REQUIRED_FIELDS:
                    if field not in mdef:
                        errors.append(
                            f"Message '{set_name}.{mname}' missing required field '{field}'"
                        )
                for s in mdef.get("signals", []):
                    if s not in signal_names:
                        errors.append(
                            f"Message '{set_name}.{mname}' references unknown signal '{s}'"
                        )

        # --- Rule 6: vehicle's message_set IDs must match its CAN mapping ---
        for vname, vdef in vehicles.items():
            message_set_name = vdef.get("message_set")
            if message_set_name is None:
                continue  # already reported in Rule 4
            if message_set_name not in message_sets:
                errors.append(
                    f"Vehicle '{vname}' references unknown message_set '{message_set_name}'"
                )
                continue

            expected_ids = {m["id"] for m in message_sets[message_set_name].values()
                            if "id" in m}
            actual_ids = {m["id"] for m in vdef.get("can_mapping", {}).values()
                          if "id" in m}

            missing = expected_ids - actual_ids
            extra = actual_ids - expected_ids
            if missing:
                errors.append(
                    f"Vehicle '{vname}' CAN mapping is missing IDs from "
                    f"message_set '{message_set_name}': "
                    f"{sorted(hex(i) for i in missing)}"
                )
            if extra:
                errors.append(
                    f"Vehicle '{vname}' CAN mapping contains IDs not in "
                    f"message_set '{message_set_name}': "
                    f"{sorted(hex(i) for i in extra)}"
                )

        # --- Rule 7: messages in a set must cover the signals the vehicle maps ---
        # Every signal carried by a message must appear in the vehicle's can_mapping.
        for vname, vdef in vehicles.items():
            message_set_name = vdef.get("message_set")
            if not message_set_name or message_set_name not in message_sets:
                continue
            vehicle_signals = set(vdef.get("can_mapping", {}).keys())
            for mname, mdef in message_sets[message_set_name].items():
                for s in mdef.get("signals", []):
                    if s not in vehicle_signals:
                        errors.append(
                            f"Vehicle '{vname}': message '{message_set_name}.{mname}' "
                            f"carries signal '{s}' but vehicle has no CAN mapping for it"
                        )

        return errors


# ----------------------------------------------------------------------
# CLI convenience: `python -m src.config_loader config/`
# ----------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    config_dir = sys.argv[1] if len(sys.argv) > 1 else "config"
    loader = ConfigLoader(config_dir)
    errs = loader.validate()
    if errs:
        print(f"❌ Validation failed with {len(errs)} error(s):")
        for e in errs:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("✅ All config files are valid.")