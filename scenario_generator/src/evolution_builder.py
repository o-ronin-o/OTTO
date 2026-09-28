"""
Evolution Timeline Builder — composes multiple recordings of the same
base scenario at different simulated days, each with a different fault
severity, producing a vehicle health trajectory.

Design (Step 2.5):
    - Reads an evolution YAML from config/evolutions/
    - For each recording, runs the base scenario with the specified
      severity (constant progression by default)
    - Writes each recording into a dedicated directory with a
      recording-specific filename prefix
    - Produces a timeline.csv and metadata.json at the evolution level
    - Copies the source evolution YAML into the output for provenance

Output structure:
    data/{vehicle_id}/evolutions/{evolution_slug}/
        evolution.yaml           (copy of the source)
        metadata.json
        timeline.csv
        day_01.frames.csv
        day_01.signals.csv
        day_01.faults.csv
        day_01.labels.csv
        day_15....
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config_loader import ConfigLoader
from src.pipeline import Pipeline, PipelineResult


@dataclass
class EvolutionResult:
    evolution_name: str
    vehicle_id: str
    output_dir: Path
    recordings: list[PipelineResult] = field(default_factory=list)
    timeline_path: Path | None = None
    metadata_path: Path | None = None


class EvolutionBuilder:
    """Composes recordings into a single evolution."""

    def __init__(self, config_dir: str | Path = "config"):
        self.config_dir = Path(config_dir)
        self.loader = ConfigLoader(self.config_dir)
        self.pipeline = Pipeline(config_dir=self.config_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        evolution_path: str | Path,
        output_dir: str | Path = "data",
    ) -> EvolutionResult:
        """
        Run every recording in the evolution and write the summary files.
        """
        evolution_path = Path(evolution_path)
        evo = self.loader.load_evolution(evolution_path)

        vehicle_id = evo["vehicle_id"]
        base_scenario_path = (
            self.config_dir / "scenarios" / f"{evo['base_scenario']}.yaml"
        )

        evo_slug = self._slugify(evo["name"])

        evolution_dir = (
            Path(output_dir) / vehicle_id / "evolutions" / evo_slug
        )
        evolution_dir.mkdir(parents=True, exist_ok=True)

        # Copy the source YAML for provenance
        shutil.copy2(evolution_path, evolution_dir / "evolution.yaml")

        # Run each recording
        results: list[PipelineResult] = []
        for rec in evo["recordings"]:
            result = self._run_recording(
                base_scenario_path=base_scenario_path,
                recording=rec,
                output_dir=evolution_dir,
            )
            results.append(result)

            rec_id = rec.get("recording_id", f"day_{rec['day']:02d}")
            print(
                f"  ✓ {rec_id:10s}  "
                f"day={rec['day']:>3}  "
                f"severity={rec['severity']:.2f}  → "
                f"{result.n_frames:>6d} frames"
            )
        # Write summary files
        timeline_path = evolution_dir / "timeline.csv"
        metadata_path = evolution_dir / "metadata.json"
        self._write_timeline(timeline_path, results, evo)
        self._write_metadata(metadata_path, evo, results, evolution_dir)

        return EvolutionResult(
            evolution_name=evo["name"],
            vehicle_id=vehicle_id,
            output_dir=evolution_dir,
            recordings=results,
            timeline_path=timeline_path,
            metadata_path=metadata_path,
        )

    # ------------------------------------------------------------------
    # Internals — run a single recording
    # ------------------------------------------------------------------

    def _run_recording(
        self,
        base_scenario_path: Path,
        recording: dict,
        output_dir: Path,
    ) -> PipelineResult:
        rec_id = recording.get("recording_id", f"day_{recording['day']:02d}")
        progression = recording.get("progression", None)   # ← was "constant"
        onset = recording.get("onset_s", None)
        end = recording.get("end_s", None)

        return self.pipeline.run_with_fault_severity(
            base_scenario_path=base_scenario_path,
            fault_severity=recording["severity"],
            fault_progression=progression,                 # may be None
            output_dir=output_dir,
            recording_id=rec_id,
            onset_s=onset,
            end_s=end,
            create_vehicle_subdir=False,
        )

    # ------------------------------------------------------------------
    # Internals — summary files
    # ------------------------------------------------------------------

    def _write_timeline(
        self,
        path: Path,
        results: list[PipelineResult],
        evo: dict,
    ) -> None:
        """One row per recording."""
        rows: list[dict] = []
        for rec, result in zip(evo["recordings"], results):
            rec_id = rec.get("recording_id", f"day_{rec['day']:02d}")
            applied = result.applied_faults
            fault_id = applied[0].fault_id if applied else ""
            fault_type = applied[0].fault_type if applied else ""
            rows.append({
                "recording_id": rec_id,
                "day": rec["day"],
                "severity": rec["severity"],
                "progression": rec.get("progression", "constant"),
                "n_frames": result.n_frames,
                "n_signal_rows": result.n_signal_rows,
                "fault_id": fault_id,
                "fault_type": fault_type,
            })
        pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8")

    def _write_metadata(
        self,
        path: Path,
        evo: dict,
        results: list[PipelineResult],
        evolution_dir: Path,
    ) -> None:
        """Dataset-level summary of the evolution."""
        recording_ids = [
            rec.get("recording_id", f"day_{rec['day']:02d}")
            for rec in evo["recordings"]
        ]

        severity_trajectory = {
            rec.get("recording_id", f"day_{rec['day']:02d}"): rec["severity"]
            for rec in evo["recordings"]
        }

        manifest = []
        for rec, result in zip(evo["recordings"], results):
            rec_id = rec.get("recording_id", f"day_{rec['day']:02d}")
            files = []
            for p in (result.frames_path, result.signals_path,
                      result.faults_path, result.labels_path):
                if p is not None and p.exists():
                    files.append(p.name)
            manifest.append({"recording_id": rec_id, "files": files})

        metadata = {
            "evolution_name": evo["name"],
            "vehicle_id": evo["vehicle_id"],
            "base_scenario": evo["base_scenario"],
            "description": evo.get("description", ""),
            "n_recordings": len(results),
            "recording_ids": recording_ids,
            "severity_trajectory": severity_trajectory,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "manifest": manifest,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _slugify(name: str) -> str:
        return (
            name.lower()
            .replace(" ", "_")
            .replace("/", "_")
            .replace("\\", "_")
        )