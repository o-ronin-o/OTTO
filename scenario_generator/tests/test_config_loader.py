"""
Unit tests for ConfigLoader (Option C — message sets).

Run with:  pytest tests/ -v
"""

from pathlib import Path

import pytest

from src.config_loader import ConfigLoader


CONFIG_DIR = Path(__file__).parent.parent / "config"


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------

@pytest.fixture
def loader() -> ConfigLoader:
    return ConfigLoader(CONFIG_DIR)


@pytest.fixture
def bad_config_dir(tmp_path: Path) -> Path:
    """Broken config for negative tests: unknown signals + mismatched IDs."""
    d = tmp_path / "bad_config"
    d.mkdir()

    (d / "signals.yaml").write_text(
        "signals:\n"
        "  engine_rpm:\n"
        "    unit: rpm\n"
        "    range: [0, 8000]\n"
        "    resolution: 0.25\n"
        "    description: 'Engine RPM'\n"
        "    system: powertrain\n"
    )
    (d / "systems.yaml").write_text(
        "systems:\n"
        "  powertrain:\n"
        "    description: 'Engine'\n"
        "    signals: [engine_rpm, nonexistent_signal]\n"
    )
    (d / "relationships.yaml").write_text("relationships: []\n")
    (d / "vehicles.yaml").write_text(
        "vehicles:\n"
        "  test_car:\n"
        "    name: 'Test'\n"
        "    message_set: test_class\n"
        "    mass_kg: 1000\n"
        "    can_mapping:\n"
        "      engine_rpm:\n"
        "        id: 0x100\n"
        "        start_byte: 0\n"
        "        length: 2\n"
        "        scale: 0.25\n"
        "        offset: 0\n"
        "      ghost_signal:\n"
        "        id: 0x100\n"
        "        start_byte: 2\n"
        "        length: 1\n"
        "        scale: 1\n"
        "        offset: 0\n"
    )
    (d / "messages.yaml").write_text(
        "message_sets:\n"
        "  test_class:\n"
        "    engine_data:\n"
        "      id: 0x100\n"
        "      cycle_ms: 10\n"
        "      dlc: 8\n"
        "      signals: [engine_rpm, ghost_signal]\n"
    )
    return d


# ----------------------------------------------------------------------
# Positive tests
# ----------------------------------------------------------------------

def test_load_signals(loader):
    signals = loader.load_signals()
    assert "engine_rpm" in signals
    assert signals["engine_rpm"]["unit"] == "rpm"
    assert signals["engine_rpm"]["range"] == [0, 8000]


def test_load_systems(loader):
    systems = loader.load_systems()
    assert "powertrain" in systems
    assert "engine_rpm" in systems["powertrain"]["signals"]


def test_load_relationships(loader):
    rels = loader.load_relationships()
    assert isinstance(rels, list)
    assert len(rels) > 0
    assert all("source" in r and "target" in r for r in rels)


def test_load_vehicles(loader):
    vehicles = loader.load_vehicles()
    assert "sedan_a" in vehicles
    assert "suv_b" in vehicles
    assert vehicles["sedan_a"]["message_set"] == "sedan_class"
    assert vehicles["suv_b"]["message_set"] == "suv_class"


def test_load_message_sets(loader):
    sets = loader.load_message_sets()
    assert "sedan_class" in sets
    assert "suv_class" in sets
    assert "engine_data" in sets["sedan_class"]
    assert sets["sedan_class"]["engine_data"]["cycle_ms"] == 10


def test_get_messages_for_vehicle(loader):
    sedan_msgs = loader.get_messages_for_vehicle("sedan_a")
    suv_msgs = loader.get_messages_for_vehicle("suv_b")
    assert sedan_msgs["engine_data"]["id"] == 0x100
    assert suv_msgs["engine_data"]["id"] == 0x200


def test_load_all_keys(loader):
    all_cfg = loader.load_all()
    assert set(all_cfg.keys()) == {
        "signals", "systems", "relationships", "vehicles",
        "message_sets", "fault_types",
    }

def test_validation_passes_on_good_config(loader):
    errors = loader.validate()
    assert errors == [], f"Expected no errors, got: {errors}"


# ----------------------------------------------------------------------
# Negative tests
# ----------------------------------------------------------------------

def test_validation_catches_unknown_signal_in_systems(bad_config_dir):
    loader = ConfigLoader(bad_config_dir)
    errors = loader.validate()
    assert any("nonexistent_signal" in e for e in errors)


def test_validation_catches_unknown_signal_in_vehicles(bad_config_dir):
    loader = ConfigLoader(bad_config_dir)
    errors = loader.validate()
    assert any("ghost_signal" in e for e in errors)


def test_validation_catches_unknown_signal_in_messages(bad_config_dir):
    loader = ConfigLoader(bad_config_dir)
    errors = loader.validate()
    assert any("ghost_signal" in e for e in errors)


def test_validation_catches_missing_required_field(tmp_path: Path):
    d = tmp_path / "missing_field"
    d.mkdir()
    (d / "signals.yaml").write_text(
        "signals:\n"
        "  engine_rpm:\n"
        "    unit: rpm\n"
        "    range: [0, 8000]\n"
    )
    (d / "systems.yaml").write_text("systems: {}\n")
    (d / "relationships.yaml").write_text("relationships: []\n")
    (d / "vehicles.yaml").write_text("vehicles: {}\n")
    (d / "messages.yaml").write_text("message_sets: {}\n")

    loader = ConfigLoader(d)
    errors = loader.validate()
    assert any("missing required field" in e for e in errors)


def test_validation_catches_id_mismatch(tmp_path: Path):
    """Vehicle's CAN IDs do not match its message_set IDs."""
    d = tmp_path / "id_mismatch"
    d.mkdir()
    (d / "signals.yaml").write_text(
        "signals:\n"
        "  engine_rpm:\n"
        "    unit: rpm\n"
        "    range: [0, 8000]\n"
        "    resolution: 0.25\n"
        "    description: 'RPM'\n"
        "    system: powertrain\n"
    )
    (d / "systems.yaml").write_text("systems: {}\n")
    (d / "relationships.yaml").write_text("relationships: []\n")
    (d / "vehicles.yaml").write_text(
        "vehicles:\n"
        "  car:\n"
        "    name: 'Car'\n"
        "    message_set: cls\n"
        "    mass_kg: 1000\n"
        "    can_mapping:\n"
        "      engine_rpm:\n"
        "        id: 0x999\n"          # ← wrong ID
        "        start_byte: 0\n"
        "        length: 2\n"
        "        scale: 0.25\n"
        "        offset: 0\n"
    )
    (d / "messages.yaml").write_text(
        "message_sets:\n"
        "  cls:\n"
        "    engine_data:\n"
        "      id: 0x100\n"           # ← expected ID
        "      cycle_ms: 10\n"
        "      dlc: 8\n"
        "      signals: [engine_rpm]\n"
    )
    loader = ConfigLoader(d)
    errors = loader.validate()
    assert any("missing IDs" in e for e in errors)
    assert any("contains IDs not in" in e for e in errors)


def test_validation_catches_unknown_message_set(tmp_path: Path):
    d = tmp_path / "unknown_ms"
    d.mkdir()
    (d / "signals.yaml").write_text("signals: {}\n")
    (d / "systems.yaml").write_text("systems: {}\n")
    (d / "relationships.yaml").write_text("relationships: []\n")
    (d / "vehicles.yaml").write_text(
        "vehicles:\n"
        "  car:\n"
        "    name: 'Car'\n"
        "    message_set: does_not_exist\n"
        "    mass_kg: 1000\n"
        "    can_mapping: {}\n"
    )
    (d / "messages.yaml").write_text("message_sets: {}\n")
    loader = ConfigLoader(d)
    errors = loader.validate()
    assert any("unknown message_set" in e for e in errors)


def test_missing_config_file_raises(tmp_path: Path):
    loader = ConfigLoader(tmp_path)
    with pytest.raises(FileNotFoundError):
        loader.load_signals()


def test_invalid_config_dir_raises():
    with pytest.raises(FileNotFoundError):
        ConfigLoader("/path/that/does/not/exist")


# ----------------------------------------------------------------------
# Acceptance test (mirrors the spec)
# ----------------------------------------------------------------------

def test_validate_all_includes_scenarios(loader):
    """validate_all() runs both config and scenario validation."""
    errors = loader.validate_all()
    assert errors == [], f"Expected no errors, got: {errors}"