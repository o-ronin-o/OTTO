"""
Unit tests for DatasetBuilder.

Run with:  pytest tests/test_dataset_builder.py -v
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from src.dataset_builder import DatasetBuilder


CONFIG_DIR = Path(__file__).parent.parent / "config"
SCENARIOS_DIR = CONFIG_DIR / "scenarios"


@pytest.fixture
def builder() -> DatasetBuilder:
    return DatasetBuilder(config_dir=CONFIG_DIR)


@pytest.fixture
def all_scenarios() -> list[Path]:
    return sorted(SCENARIOS_DIR.glob("*.yaml"))


# ----------------------------------------------------------------------
# Full batch run
# ----------------------------------------------------------------------

def test_build_creates_output_dir(builder, all_scenarios, tmp_path):
    dataset = builder.build(all_scenarios, output_dir=tmp_path)
    assert dataset.output_dir == tmp_path
    assert tmp_path.exists()


def test_build_processes_all_scenarios(builder, all_scenarios, tmp_path):
    dataset = builder.build(all_scenarios, output_dir=tmp_path)
    assert len(dataset.results) == len(all_scenarios)


def test_build_creates_expected_files(builder, all_scenarios, tmp_path):
    builder.build(all_scenarios, output_dir=tmp_path)
    # N scenarios × 2 files each (frames + signals) = 2N files
    # PLUS 2 extra files (faults + labels) for each fault-bearing scenario
    # PLUS manifest.csv
    all_files = list(tmp_path.rglob("*"))
    csvs = [f for f in all_files if f.suffix == ".csv"]
    jsons = [f for f in all_files if f.suffix == ".json"]

    # Count fault-bearing scenarios
    n_fault_scenarios = 0
    for path in all_scenarios:
        with open(path) as f:
            import yaml
            raw = yaml.safe_load(f)
        if raw.get("scenario", {}).get("faults"):
            n_fault_scenarios += 1

    expected = 2 * len(all_scenarios) + 2 * n_fault_scenarios + 1
    assert len(csvs) == expected
    assert len(jsons) == 1

# ----------------------------------------------------------------------
# Manifest
# ----------------------------------------------------------------------

def test_manifest_exists(builder, all_scenarios, tmp_path):
    dataset = builder.build(all_scenarios, output_dir=tmp_path)
    assert dataset.manifest_path is not None
    assert dataset.manifest_path.exists()
    assert dataset.manifest_path.name == "manifest.csv"


def test_manifest_has_correct_columns(builder, all_scenarios, tmp_path):
    dataset = builder.build(all_scenarios, output_dir=tmp_path)
    df = pd.read_csv(dataset.manifest_path)
    expected_cols = {
        "vehicle_id", "scenario_name", "scenario_slug",
        "file_type", "relative_path", "n_rows", "size_bytes",
    }
    assert set(df.columns) == expected_cols


def test_manifest_row_count(builder, all_scenarios, tmp_path):
    dataset = builder.build(all_scenarios, output_dir=tmp_path)
    df = pd.read_csv(dataset.manifest_path)
    # N scenarios × 2 files each = 2N rows
    assert len(df) == 2 * len(all_scenarios)


def test_manifest_covers_both_file_types(builder, all_scenarios, tmp_path):
    dataset = builder.build(all_scenarios, output_dir=tmp_path)
    df = pd.read_csv(dataset.manifest_path)
    assert set(df["file_type"].unique()) == {"frames", "signals"}


def test_manifest_paths_are_relative(builder, all_scenarios, tmp_path):
    dataset = builder.build(all_scenarios, output_dir=tmp_path)
    df = pd.read_csv(dataset.manifest_path)
    for rel_path in df["relative_path"]:
        assert not Path(rel_path).is_absolute()


def test_manifest_paths_resolve_to_real_files(builder, all_scenarios, tmp_path):
    dataset = builder.build(all_scenarios, output_dir=tmp_path)
    df = pd.read_csv(dataset.manifest_path)
    for rel_path in df["relative_path"]:
        full_path = tmp_path / rel_path
        assert full_path.exists(), f"Missing: {full_path}"


def test_manifest_includes_both_vehicles(builder, all_scenarios, tmp_path):
    dataset = builder.build(all_scenarios, output_dir=tmp_path)
    df = pd.read_csv(dataset.manifest_path)
    assert set(df["vehicle_id"].unique()) == {"sedan_a", "suv_b"}


# ----------------------------------------------------------------------
# Metadata
# ----------------------------------------------------------------------

def test_metadata_exists(builder, all_scenarios, tmp_path):
    dataset = builder.build(all_scenarios, output_dir=tmp_path)
    assert dataset.metadata_path is not None
    assert dataset.metadata_path.exists()
    assert dataset.metadata_path.name == "metadata.json"


def test_metadata_structure(builder, all_scenarios, tmp_path):
    dataset = builder.build(all_scenarios, output_dir=tmp_path)
    with open(dataset.metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    for key in [
        "generated_at", "config_dir", "vehicles", "n_scenarios",
        "n_files", "total_frames", "total_signal_rows",
        "total_size_bytes", "total_size_mb", "scenarios",
    ]:
        assert key in meta


def test_metadata_counts_match_manifest(builder, all_scenarios, tmp_path):
    dataset = builder.build(all_scenarios, output_dir=tmp_path)
    with open(dataset.metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    df = pd.read_csv(dataset.manifest_path)

    assert meta["n_scenarios"] == len(all_scenarios)
    assert meta["n_files"] == len(df)
    assert meta["total_frames"] == sum(
        r.n_frames for r in dataset.results
    )
    assert meta["total_signal_rows"] == sum(
        r.n_signal_rows for r in dataset.results
    )


def test_metadata_scenarios_entries(builder, all_scenarios, tmp_path):
    dataset = builder.build(all_scenarios, output_dir=tmp_path)
    with open(dataset.metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # Every scenario file must appear in the metadata
    names = {s["name"] for s in meta["scenarios"]}
    assert "City Drive" in names
    assert "Highway Cruise" in names
    assert "Highway Cruise with Brake Wear" in names

    # All vehicles present
    vehicles = {s["vehicle_id"] for s in meta["scenarios"]}
    assert vehicles == {"sedan_a", "suv_b"}

# ----------------------------------------------------------------------
# Determinism
# ----------------------------------------------------------------------

def test_build_is_deterministic(builder, all_scenarios, tmp_path):
    out1 = tmp_path / "run1"
    out2 = tmp_path / "run2"
    builder.build(all_scenarios, output_dir=out1)
    builder.build(all_scenarios, output_dir=out2)

    # Only check combinations that actually exist
    existing = [
        ("sedan_a", "city_drive"),
        ("suv_b", "highway_cruise"),
    ]
    for vehicle, scenario_slug in existing:
        for kind in ["frames", "signals"]:
            f1 = (out1 / vehicle / f"{scenario_slug}.{kind}.csv").read_text()
            f2 = (out2 / vehicle / f"{scenario_slug}.{kind}.csv").read_text()
            assert f1 == f2


# ----------------------------------------------------------------------
# Empty / edge cases
# ----------------------------------------------------------------------

def test_build_with_empty_list(builder, tmp_path):
    dataset = builder.build([], output_dir=tmp_path)
    assert len(dataset.results) == 0
    assert dataset.manifest_path is not None
    assert dataset.manifest_path.exists()

    df = pd.read_csv(dataset.manifest_path)
    assert len(df) == 0
    # Header should still be present
    assert "vehicle_id" in df.columns
    assert "n_rows" in df.columns


# ----------------------------------------------------------------------
# Cross-vehicle semantic consistency
# ----------------------------------------------------------------------

def test_same_scenario_different_vehicles_different_can_ids(builder, all_scenarios, tmp_path):
    """Same scenario name across two vehicles: same signals, different CAN IDs."""
    dataset = builder.build(all_scenarios, output_dir=tmp_path)

    # Find city_drive results
    sedan_result = next(
        r for r in dataset.results
        if r.scenario.name == "City Drive" and r.scenario.vehicle_id == "sedan_a"
    )
    # Note: city_drive is declared as sedan_a in YAML, so we can't get a
    # suv_b result from the same YAML. Instead, check that the two
    # *different* scenarios have their own CAN IDs.
    sedan_frames = pd.read_csv(sedan_result.frames_path)
    sedan_ids = set(sedan_frames["can_id"].unique())
    assert sedan_ids == {"0x100", "0x1A0", "0x120"}

    # Highway Cruise uses suv_b → different IDs
    suv_result = next(
        r for r in dataset.results
        if r.scenario.name == "Highway Cruise" and r.scenario.vehicle_id == "suv_b"
    )
    suv_frames = pd.read_csv(suv_result.frames_path)
    suv_ids = set(suv_frames["can_id"].unique())
    assert suv_ids == {"0x200", "0x2B0", "0x220"}