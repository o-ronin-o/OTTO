"""
Dataset Builder — batches the Pipeline across multiple scenarios and vehicles,
then produces a manifest and metadata summary.

Design (Step 5):
    - Inputs: a config dir, a list of scenario YAML paths, an output dir
    - For each (scenario, vehicle implied by scenario) pair:
        - Run Pipeline.run() to produce frames.csv + signals.csv
    - After all runs:
        - Write manifest.csv  (one row per generated file)
        - Write metadata.json (dataset-level summary)

Decisions:
    - One scenario YAML = one (scenario, vehicle) pair. The vehicle is
      declared inside the scenario, not passed separately.
    - Manifest is flat (one row per output file), not per-dataset.
    - metadata.json is human-readable (indent=2).
    - Reproducible: same inputs → same outputs, no randomness.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.pipeline import Pipeline, PipelineResult
from src.signal_generator import Scenario


# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------

@dataclass
class BuiltFile:
    """One generated output file, for the manifest."""
    vehicle_id: str
    scenario_name: str
    scenario_slug: str
    file_type: str          # "frames" or "signals"
    path: Path
    n_rows: int
    size_bytes: int


@dataclass
class BuiltDataset:
    """Summary of a complete dataset generation run."""
    output_dir: Path
    results: list[PipelineResult] = field(default_factory=list)
    files: list[BuiltFile] = field(default_factory=list)
    manifest_path: Path | None = None
    metadata_path: Path | None = None


# ----------------------------------------------------------------------
# Dataset Builder
# ----------------------------------------------------------------------

class DatasetBuilder:
    """Batch-generates the full dataset and writes manifest + metadata."""

    def __init__(self, config_dir: str | Path = "config"):
        self.config_dir = Path(config_dir)
        self.pipeline = Pipeline(config_dir=self.config_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        scenario_paths: list[str | Path],
        output_dir: str | Path = "data",
    ) -> BuiltDataset:
        """
        Run every scenario through the pipeline and produce a dataset.

        Parameters
        ----------
        scenario_paths : list of YAML paths
        output_dir : root output directory

        Returns
        -------
        BuiltDataset with paths to manifest.csv and metadata.json
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results: list[PipelineResult] = []

        for scenario_path in scenario_paths:
            scenario_path = Path(scenario_path)
            result = self.pipeline.run(scenario_path, output_dir=output_dir)
            results.append(result)
            print(
                f"  ✓ {result.scenario.vehicle_id:10s} / "
                f"{result.scenario.name:20s} → "
                f"{result.n_frames:>6d} frames, "
                f"{result.n_signal_rows:>5d} signal rows"
            )

        # Build manifest rows from results
        built_files = self._collect_files(results)

        # Write manifest + metadata
        manifest_path = output_dir / "manifest.csv"
        metadata_path = output_dir / "metadata.json"
        self._write_manifest(manifest_path, built_files)
        self._write_metadata(metadata_path, results, built_files)

        return BuiltDataset(
            output_dir=output_dir,
            results=results,
            files=built_files,
            manifest_path=manifest_path,
            metadata_path=metadata_path,
        )

    # ------------------------------------------------------------------
    # Internal helpers — file collection
    # ------------------------------------------------------------------

    def _collect_files(self, results: list[PipelineResult]) -> list[BuiltFile]:
        """Build a BuiltFile entry for every output file from every result."""
        files: list[BuiltFile] = []
        for result in results:
            for file_type, path in (
                ("frames", result.frames_path),
                ("signals", result.signals_path),
            ):
                n_rows = self._count_rows(path)
                files.append(BuiltFile(
                    vehicle_id=result.scenario.vehicle_id,
                    scenario_name=result.scenario.name,
                    scenario_slug=result.frames_path.stem.split(".")[0],
                    file_type=file_type,
                    path=path,
                    n_rows=n_rows,
                    size_bytes=path.stat().st_size,
                ))
        return files

    @staticmethod
    def _count_rows(csv_path: Path) -> int:
        """Count data rows (excluding header) in a CSV."""
        with open(csv_path, "r", encoding="utf-8") as f:
            return sum(1 for _ in f) - 1  # subtract header

    # ------------------------------------------------------------------
    # Internal helpers — manifest
    # ------------------------------------------------------------------

    def _write_manifest(self, path: Path, files: list[BuiltFile]) -> None:
        """Write manifest.csv with one row per generated file."""
        columns = [
            "vehicle_id", "scenario_name", "scenario_slug",
            "file_type", "relative_path", "n_rows", "size_bytes",
        ]
        rows = []
        for bf in files:
            rows.append({
                "vehicle_id": bf.vehicle_id,
                "scenario_name": bf.scenario_name,
                "scenario_slug": bf.scenario_slug,
                "file_type": bf.file_type,
                "relative_path": str(bf.path.relative_to(path.parent)),
                "n_rows": bf.n_rows,
                "size_bytes": bf.size_bytes,
            })
        # Write header even if rows is empty
        df = pd.DataFrame(rows, columns=columns)
        df.to_csv(path, index=False, encoding="utf-8")

    # ------------------------------------------------------------------
    # Internal helpers — metadata
    # ------------------------------------------------------------------

    def _write_metadata(
        self,
        path: Path,
        results: list[PipelineResult],
        files: list[BuiltFile],
    ) -> None:
        """Write metadata.json with a summary of the whole dataset."""
        # Load scenario YAMLs to capture their configuration
        scenarios_meta = []
        for result in results:
            scenarios_meta.append({
                "name": result.scenario.name,
                "vehicle_id": result.scenario.vehicle_id,
                "duration_s": result.scenario.duration_s,
                "base_rate_hz": result.scenario.base_rate_hz,
                "n_segments": len(result.scenario.segments),
                "n_frames": result.n_frames,
                "n_signal_rows": result.n_signal_rows,
            })

        # Vehicles present
        vehicles = sorted({r.scenario.vehicle_id for r in results})

        # Totals
        total_frames = sum(r.n_frames for r in results)
        total_signal_rows = sum(r.n_signal_rows for r in results)
        total_bytes = sum(bf.size_bytes for bf in files)

        metadata = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "config_dir": str(self.config_dir.resolve()),
            "vehicles": vehicles,
            "n_scenarios": len(results),
            "n_files": len(files),
            "total_frames": total_frames,
            "total_signal_rows": total_signal_rows,
            "total_size_bytes": total_bytes,
            "total_size_mb": round(total_bytes / (1024 * 1024), 2),
            "scenarios": scenarios_meta,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)