"""
ConfigLoader — reads and validates YAML configuration files.

This module is the foundation of Stage 1 and Stage 2. It ensures that all
config files are internally consistent before any generation begins.

Architecture note (Option C):
    Messages are organized into *message sets* (one per vehicle class).
    Vehicles reference a message set via the `message_set` field.

Fault registry (Stage 2):
    Fault types are stored one-per-file under `config/faults/*.yaml`.
    Each file declares a single `fault:` entry with a unique `id`.
    The loader scans the directory and builds `{fault_id: fault_def}`.
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

# Required fields for each fault definition
FAULT_REQUIRED_FIELDS = [
    "id", "physical_category", "observable_effect", "layer",
    "description", "parameters", "targets", "injection_rule",
]

# Evolution constants (Step 2.5)
EVOLUTION_REQUIRED_FIELDS = ["name", "vehicle_id", "base_scenario", "recordings"]
RECORDING_REQUIRED_FIELDS = ["day", "severity"]
# Valid relationship types
VALID_RELATIONSHIP_TYPES = ["positive", "negative", "integral", "warmup"]

# ----------------------------------------------------------------------
# Fault registry constants
# ----------------------------------------------------------------------

VALID_PHYSICAL_CATEGORIES = [
    "mechanical", "electrical", "communication", "sensor",
    "power", "software", "cyber",
]

VALID_OBSERVABLE_EFFECTS = [
    "drift", "freeze", "dropout", "noise", "jitter", "drop",
    "duplicate", "flood",
]

VALID_FAULT_LAYERS = ["signal", "frame"]

VALID_PARAMETER_TYPES = ["float", "int", "enum", "bool", "string"]

VALID_PROGRESSION_VALUES = [
    "constant", "linear", "exponential", "step", "oscillating",
]

FAULT_OPTIONAL_FIELDS = ["injection_point", "severity_maps_to"]

VALID_INJECTION_POINTS = ["physics", "post_physics"]
DEFAULT_INJECTION_POINT = "physics"
DEFAULT_SEVERITY_MAPS_TO = "severity_end"

# Maximum allowed target signals per fault (sanity guard)
MAX_TARGET_SIGNALS = 5


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
        self._fault_types: dict[str, dict] | None = None

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

    def load_fault_types(self) -> dict[str, dict]:
        """
        Scan config/faults/*.yaml and return {fault_id: fault_def}.

        Empty or missing directory → returns {} (Stage 1 compatibility).
        """
        if self._fault_types is not None:
            return self._fault_types

        faults_dir = self.config_dir / "faults"
        result: dict[str, dict] = {}

        if not faults_dir.is_dir():
            self._fault_types = result
            return result

        for path in sorted(faults_dir.glob("*.yaml")):
            with open(path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f)

            if not isinstance(raw, dict) or "fault" not in raw:
                raise ValueError(
                    f"{path.name}: must contain a top-level 'fault' key"
                )

            fault_def = raw["fault"]
            if not isinstance(fault_def, dict):
                raise ValueError(
                    f"{path.name}: 'fault' must be a mapping"
                )

            fault_id = fault_def.get("id")
            if not isinstance(fault_id, str) or not fault_id:
                raise ValueError(
                    f"{path.name}: 'fault.id' must be a non-empty string"
                )

            # Rule F9: filename must match fault.id
            if path.stem != fault_id:
                raise ValueError(
                    f"{path.name}: filename must match fault id "
                    f"('{path.stem}' != '{fault_id}')"
                )

            # Rule F8: duplicate id across files
            if fault_id in result:
                raise ValueError(
                    f"{path.name}: duplicate fault id '{fault_id}'"
                )

            result[fault_id] = fault_def

        self._fault_types = result
        return result

    def load_all(self) -> dict[str, Any]:
        return {
            "signals": self.load_signals(),
            "systems": self.load_systems(),
            "relationships": self.load_relationships(),
            "vehicles": self.load_vehicles(),
            "message_sets": self.load_message_sets(),
            "fault_types": self.load_fault_types(),
        }

    def load_evolution(self, evolution_path: str | Path) -> dict:
        """
        Load and validate a single evolution YAML file.

        Returns the parsed `evolution` mapping. Raises ValueError on
        schema violations.
        """
        path = Path(evolution_path)
        if not path.exists():
            raise FileNotFoundError(f"Evolution file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        if not isinstance(raw, dict) or "evolution" not in raw:
            raise ValueError(f"{path.name}: missing top-level 'evolution' key")

        evo = raw["evolution"]
        if not isinstance(evo, dict):
            raise ValueError(f"{path.name}: 'evolution' must be a mapping")

        for field in EVOLUTION_REQUIRED_FIELDS:
            if field not in evo:
                raise ValueError(f"{path.name}: missing required field '{field}'")

        # Validate vehicle
        vehicles = self.load_vehicles()
        if evo["vehicle_id"] not in vehicles:
            raise ValueError(
                f"{path.name}: unknown vehicle_id '{evo['vehicle_id']}'. "
                f"Available: {sorted(vehicles.keys())}"
            )

        # Validate base scenario exists and declares at least one fault
        scenarios_dir = self.config_dir / "scenarios"
        base_scenario_path = scenarios_dir / f"{evo['base_scenario']}.yaml"
        if not base_scenario_path.exists():
            raise ValueError(
                f"{path.name}: base_scenario '{evo['base_scenario']}' not found "
                f"at {base_scenario_path}"
            )

        fault_registry = self.load_fault_types()
        try:
            from src.signal_generator import Scenario
            base = Scenario.from_yaml(base_scenario_path, fault_registry=fault_registry)
        except (ValueError, FileNotFoundError) as exc:
            raise ValueError(
                f"{path.name}: failed to load base_scenario "
                f"'{evo['base_scenario']}': {exc}"
            ) from exc

        if not base.resolved_faults:
            raise ValueError(
                f"{path.name}: base_scenario '{evo['base_scenario']}' "
                f"does not declare any faults"
            )

        base_fault_type = base.resolved_faults[0].type

        # Validate recordings
        recordings = evo["recordings"]
        if not isinstance(recordings, list) or not recordings:
            raise ValueError(f"{path.name}: 'recordings' must be a non-empty list")

        seen_ids: set[str] = set()
        seen_days: list[int] = []

        for i, rec in enumerate(recordings):
            ctx = f"{path.name} recording #{i}"

            if not isinstance(rec, dict):
                raise ValueError(f"{ctx}: must be a mapping")

            for field in RECORDING_REQUIRED_FIELDS:
                if field not in rec:
                    raise ValueError(f"{ctx}: missing '{field}'")

            # day
            day = rec["day"]
            if not isinstance(day, int) or day < 0:
                raise ValueError(f"{ctx}: 'day' must be a non-negative integer")
            seen_days.append(day)

            # recording_id
            rid = rec.get("recording_id", f"day_{day:02d}")
            if not isinstance(rid, str) or not rid:
                raise ValueError(f"{ctx}: 'recording_id' must be a non-empty string")
            if rid in seen_ids:
                raise ValueError(f"{ctx}: duplicate recording_id '{rid}'")
            seen_ids.add(rid)

            # severity
            sev = rec["severity"]
            if not isinstance(sev, (int, float)) or isinstance(sev, bool):
                raise ValueError(f"{ctx}: 'severity' must be a number")
            if not (0.0 <= sev <= 1.0):
                raise ValueError(f"{ctx}: severity must be in [0, 1], got {sev}")

            # progression (optional)
            if "progression" in rec:
                if not isinstance(rec["progression"], str):
                    raise ValueError(
                        f"{ctx}: 'progression' must be a string"
                    )

            # onset_s / end_s (optional)
            onset = rec.get("onset_s", None)
            end = rec.get("end_s", None)
            if (onset is None) != (end is None):
                raise ValueError(
                    f"{ctx}: 'onset_s' and 'end_s' must be provided together"
                )
            if onset is not None:
                if onset < 0 or onset >= base.duration_s:
                    raise ValueError(
                        f"{ctx}: onset_s={onset} out of [0, {base.duration_s})"
                    )
                if end <= onset or end > base.duration_s:
                    raise ValueError(
                        f"{ctx}: end_s={end} out of ({onset}, {base.duration_s}]"
                    )

        # Days must be strictly increasing
        for i in range(1, len(seen_days)):
            if seen_days[i] <= seen_days[i - 1]:
                raise ValueError(
                    f"{path.name}: recording days must be strictly increasing "
                    f"(day {seen_days[i]} comes after day {seen_days[i-1]})"
                )

        return evo

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
                continue
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

        # --- Rule 7: message signals must be mapped for the vehicle ---
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

        # --- Rule 8: fault registry validation ---
        try:
            fault_types = self.load_fault_types()
        except (ValueError, FileNotFoundError) as exc:
            errors.append(f"Fault registry load error: {exc}")
            return errors

        errors.extend(self._validate_fault_types(fault_types, signal_names, vehicles))

        return errors

    # ------------------------------------------------------------------
    # Fault registry validation
    # ------------------------------------------------------------------

    def _validate_fault_types(
        self,
        fault_types: dict[str, dict],
        signal_names: set[str],
        vehicles: dict[str, dict],
    ) -> list[str]:
        """Run validation rules F1–F11 on each fault definition."""
        errors: list[str] = []

        # All CAN IDs known across all vehicles
        all_can_ids: set[int] = set()
        for vdef in vehicles.values():
            for mapping in vdef.get("can_mapping", {}).values():
                if "id" in mapping:
                    all_can_ids.add(mapping["id"])

        for fault_id, fdef in fault_types.items():
            prefix = f"Fault '{fault_id}'"

            # F1 — required fields
            for field in FAULT_REQUIRED_FIELDS:
                if field not in fdef:
                    errors.append(f"{prefix} missing required field '{field}'")

            # F2 — enum values
            if "physical_category" in fdef:
                if fdef["physical_category"] not in VALID_PHYSICAL_CATEGORIES:
                    errors.append(
                        f"{prefix} invalid physical_category "
                        f"'{fdef['physical_category']}'. "
                        f"Valid: {VALID_PHYSICAL_CATEGORIES}"
                    )
            if "observable_effect" in fdef:
                if fdef["observable_effect"] not in VALID_OBSERVABLE_EFFECTS:
                    errors.append(
                        f"{prefix} invalid observable_effect "
                        f"'{fdef['observable_effect']}'. "
                        f"Valid: {VALID_OBSERVABLE_EFFECTS}"
                    )
            if "layer" in fdef:
                if fdef["layer"] not in VALID_FAULT_LAYERS:
                    errors.append(
                        f"{prefix} invalid layer '{fdef['layer']}'. "
                        f"Valid: {VALID_FAULT_LAYERS}"
                    )

            # F3 — layer-target consistency
            targets = fdef.get("targets", {})
            sig_targets = targets.get("signals", []) if isinstance(targets, dict) else []
            msg_targets = targets.get("messages", []) if isinstance(targets, dict) else []
            layer = fdef.get("layer")

            if layer == "signal":
                if not sig_targets:
                    errors.append(
                        f"{prefix} layer='signal' but targets.signals is empty"
                    )
                if msg_targets:
                    errors.append(
                        f"{prefix} layer='signal' but targets.messages is non-empty"
                    )
            elif layer == "frame":
                if not msg_targets:
                    errors.append(
                        f"{prefix} layer='frame' but targets.messages is empty"
                    )
                if sig_targets:
                    errors.append(
                        f"{prefix} layer='frame' but targets.signals is non-empty"
                    )

            # F4 — signal targets exist
            for sig in sig_targets:
                if sig not in signal_names:
                    errors.append(
                        f"{prefix} targets unknown signal '{sig}'"
                    )

            # F5 — message targets exist in at least one vehicle
            for mid in msg_targets:
                if mid not in all_can_ids:
                    errors.append(
                        f"{prefix} targets CAN ID 0x{mid:03X} not present in any vehicle"
                    )

            # Sanity guard: max target signals
            if len(sig_targets) > MAX_TARGET_SIGNALS:
                errors.append(
                    f"{prefix} declares {len(sig_targets)} target signals "
                    f"(max {MAX_TARGET_SIGNALS})"
                )

            # F6 — parameter specs valid
            params = fdef.get("parameters", [])
            if not isinstance(params, list):
                errors.append(f"{prefix} parameters must be a list")
            else:
                seen_names: set[str] = set()
                for i, p in enumerate(params):
                    pp = f"{prefix} parameter #{i}"
                    if not isinstance(p, dict):
                        errors.append(f"{pp} is not a mapping")
                        continue
                    if "name" not in p:
                        errors.append(f"{pp} missing 'name'")
                        continue
                    name = p["name"]
                    if name in seen_names:
                        errors.append(f"{prefix} duplicate parameter name '{name}'")
                    seen_names.add(name)

                    if "type" not in p:
                        errors.append(f"{pp} ('{name}') missing 'type'")
                        continue
                    if p["type"] not in VALID_PARAMETER_TYPES:
                        errors.append(
                            f"{pp} ('{name}') invalid type '{p['type']}'. "
                            f"Valid: {VALID_PARAMETER_TYPES}"
                        )

                    if p["type"] in ("float", "int"):
                        rng = p.get("range")
                        if not (isinstance(rng, list) and len(rng) == 2):
                            errors.append(
                                f"{pp} ('{name}') needs a 2-element 'range'"
                            )
                        elif rng[0] >= rng[1]:
                            errors.append(
                                f"{pp} ('{name}') range min must be < max"
                            )

                    if p["type"] == "enum":
                        vals = p.get("values")
                        if not (isinstance(vals, list) and len(vals) > 0):
                            errors.append(
                                f"{pp} ('{name}') needs a non-empty 'values' list"
                            )

                    if "description" not in p:
                        errors.append(f"{pp} ('{name}') missing 'description'")

                    # F7 — progression-specific rule
                    if name == "progression" and p.get("type") == "enum":
                        vals = p.get("values", [])
                        for v in vals:
                            if v not in VALID_PROGRESSION_VALUES:
                                errors.append(
                                    f"{pp} ('{name}') invalid progression value "
                                    f"'{v}'. Valid: {VALID_PROGRESSION_VALUES}"
                                )

            # F10 — valid injection_point
            if "injection_point" in fdef:
                if fdef["injection_point"] not in VALID_INJECTION_POINTS:
                    errors.append(
                        f"{prefix} invalid injection_point "
                        f"'{fdef['injection_point']}'. "
                        f"Valid: {VALID_INJECTION_POINTS}"
                    )

            # F11 — severity_maps_to references a declared parameter
            if "severity_maps_to" in fdef:
                declared_param_names = {p["name"] for p in params if isinstance(p, dict)}
                if fdef["severity_maps_to"] not in declared_param_names:
                    errors.append(
                        f"{prefix} severity_maps_to='{fdef['severity_maps_to']}' "
                        f"is not a declared parameter. "
                        f"Declared: {sorted(declared_param_names)}"
                    )

        return errors

    # ------------------------------------------------------------------
    # Scenario validation (Stage 2)
    # ------------------------------------------------------------------

    def validate_scenario(self, scenario_path: str | Path) -> list[str]:
        """
        Validate a scenario YAML including any fault declarations.

        Returns a list of error strings (empty if valid).
        """
        from src.signal_generator import Scenario

        errors: list[str] = []
        path = Path(scenario_path)

        if not path.exists():
            return [f"Scenario file not found: {path}"]

        try:
            fault_registry = self.load_fault_types()
        except (ValueError, FileNotFoundError) as exc:
            return [f"Fault registry load error: {exc}"]

        try:
            Scenario.from_yaml(path, fault_registry=fault_registry)
        except (ValueError, KeyError, FileNotFoundError) as exc:
            errors.append(f"{path.name}: {exc}")

        return errors

    def validate_all_scenarios(self) -> list[str]:
        """Validate every scenario YAML under config/scenarios/."""
        scenarios_dir = self.config_dir / "scenarios"
        if not scenarios_dir.is_dir():
            return []
        errors: list[str] = []
        for path in sorted(scenarios_dir.glob("*.yaml")):
            errors.extend(self.validate_scenario(path))
        return errors

    def validate_all_evolutions(self) -> list[str]:
        """Validate every evolution YAML under config/evolutions/."""
        evolutions_dir = self.config_dir / "evolutions"
        if not evolutions_dir.is_dir():
            return []
        errors: list[str] = []
        for path in sorted(evolutions_dir.glob("*.yaml")):
            try:
                self.load_evolution(path)
            except (ValueError, FileNotFoundError) as exc:
                errors.append(str(exc))
        return errors

    def validate_all(self) -> list[str]:
        """Run all config validation, including scenario validation."""
        errors = self.validate()
        errors.extend(self.validate_all_scenarios())
        errors.extend(self.validate_all_evolutions())
        return errors


# ----------------------------------------------------------------------
# CLI convenience: `python -m src.config_loader config/`
# ----------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    config_dir = sys.argv[1] if len(sys.argv) > 1 else "config"
    loader = ConfigLoader(config_dir)
    errs = loader.validate_all()
    if errs:
        print(f"❌ Validation failed with {len(errs)} error(s):")
        for e in errs:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("✅ All config files are valid.")