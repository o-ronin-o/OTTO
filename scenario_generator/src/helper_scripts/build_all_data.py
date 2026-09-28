"""
Rebuild the entire dataset from configs.

Usage (from project root):
    python -m scripts.build_all_data
    # or:
    python scripts/build_all_data.py
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset_builder import DatasetBuilder
from src.evolution_builder import EvolutionBuilder


def main():
    config_dir = PROJECT_ROOT / "config"
    data_dir = PROJECT_ROOT / "data"

    scenarios = [
        config_dir / "scenarios" / "city_drive.yaml",
        config_dir / "scenarios" / "highway_cruise.yaml",
        config_dir / "scenarios" / "highway_brake_wear_gradual.yaml",
        config_dir / "scenarios" / "highway_cruise_torque_drop.yaml",
        config_dir / "scenarios" / "highway_cruise_crank_sensor_freeze.yaml",
    ]

    evolutions = [
        config_dir / "evolutions" / "brake_wear_95days.yaml",
        config_dir / "evolutions" / "torque_drops_90days.yaml",
        config_dir / "evolutions" / "crank_sensor_freeze_90days.yaml",
    ]

    print("=" * 60)
    print("BUILDING SINGLE SCENARIOS")
    print("=" * 60)
    b = DatasetBuilder(str(config_dir))
    b.build([str(s) for s in scenarios], output_dir=str(data_dir))

    print()
    print("=" * 60)
    print("BUILDING EVOLUTIONS")
    print("=" * 60)
    e = EvolutionBuilder(str(config_dir))
    for evo in evolutions:
        print()
        print(f"--- {evo.name} ---")
        e.build(str(evo), output_dir=str(data_dir))

    print()
    print("=" * 60)
    print("REGENERATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()