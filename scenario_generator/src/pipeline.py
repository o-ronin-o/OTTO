"""
Pipeline — orchestrates the full synthetic CAN data generation workflow.

Chain:
    Scenario → SignalGenerator → [SignalFaultInjector] → CANEncoder
             → FrameScheduler → [FaultLabelBuilder] → CSV output

Design (Step 4):
    - Two files per scenario:
        {vehicle_id}/{scenario_name}.frames.csv   (raw CAN frames)
        {vehicle_id}/{scenario_name}.signals.csv  (decoded signals, 100 Hz)
    - Vehicle ID comes from the scenario YAML (no override).
    - Directory structure: data/{vehicle_id}/{scenario_name}.*.csv
    - CSV encoding: UTF-8, newline='', deterministic column order.

Design (Step 2.1b):
    - If the scenario declares signal-layer faults, the injector is
      applied between SignalGenerator and CANEncoder.
    - The list of applied faults is carried on PipelineResult for
      downstream label generation.

Design (Step 2.2):
    - If applied_faults is non-empty, two additional files are written:
        {vehicle_id}/{scenario_name}.faults.csv   (per-event table)
        {vehicle_id}/{scenario_name}.labels.csv   (per-frame labels)
    - Scenarios without faults do NOT produce these files.

Design (Step 2.5):
    - run_with_fault_severity() runs a base scenario with a single
      synthetic fault at a specified severity. Used by EvolutionBuilder.
    - Both run() and run_with_fault_severity() share the execution path
      via _run_scenario_to_directory().
    - create_vehicle_subdir=False disables the {vehicle_id}/ subdirectory
      creation (EvolutionBuilder writes directly into an evolution dir).

Outputs:
    - frames CSV:  timestamp_ms, can_id, payload_hex
    - signals CSV: time_s, timestamp_ms, <signals...>
    - faults CSV:  one row per applied fault (event summary)
    - labels CSV:  one row per frame (aligned with frames.csv)
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.can_encoder import CANEncoder
from src.config_loader import ConfigLoader
from src.fault_labels import FaultLabelBuilder
from src.frame_scheduler import FrameScheduler, ScheduledFrame
from src.signal_generator import Scenario, SignalGenerator, _resolve_faults


@dataclass
class PipelineResult:
    """Returned by Pipeline.run() — paths to the output files."""
    frames_path: Path
    signals_path: Path
    n_frames: int
    n_signal_rows: int
    scenario: Scenario
    applied_faults: list = field(default_factory=list)
    faults_path: Path | None = None
    labels_path: Path | None = None


class Pipeline:
    """
    End-to-end orchestrator for synthetic CAN data generation.

    Parameters
    ----------
    config_dir : str | Path
        Path to the config directory containing YAML files.
    """

    def __init__(self, config_dir: str | Path = "config"):
        self.config_dir = Path(config_dir)
        self.loader = ConfigLoader(self.config_dir)
        self.generator = SignalGenerator(self.loader)
        self.signals_cfg = self.loader.load_signals()
        self.vehicles = self.loader.load_vehicles()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        scenario_path: str | Path,
        output_dir: str | Path = "data",
    ) -> PipelineResult:
        """
        Run the full pipeline for one scenario.

        Parameters
        ----------
        scenario_path : str | Path
            Path to a scenario YAML file.
        output_dir : str | Path
            Root output directory.

        Returns
        -------
        PipelineResult
        """
        scenario_path = Path(scenario_path)
        fault_registry = self.loader.load_fault_types()
        scenario = Scenario.from_yaml(scenario_path, fault_registry=fault_registry)

        safe_name = self._slugify(scenario.name)
        return self._run_scenario_to_directory(
            scenario,
            output_dir=Path(output_dir),
            filename_prefix=safe_name,
            create_vehicle_subdir=True,
        )

    def run_with_fault_severity(
        self,
        base_scenario_path: str | Path,
        fault_severity: float,
        fault_progression: str,
        output_dir: str | Path,
        recording_id: str,
        *,
        onset_s: float | None = None,
        end_s: float | None = None,
        create_vehicle_subdir: bool = False,
    ) -> PipelineResult:
        """
        Run a base scenario with a single fault instance at a specific severity.

        The severity maps to the parameter declared in the fault registry's
        `severity_maps_to` field (default `severity_end`).

        Used by EvolutionBuilder to compose trajectories. Ignores any faults
        declared in the base scenario and instead injects one fault with the
        given severity.
        """
        base_scenario_path = Path(base_scenario_path)
        fault_registry = self.loader.load_fault_types()
        base_scenario = Scenario.from_yaml(
            base_scenario_path, fault_registry=fault_registry
        )

        if not base_scenario.resolved_faults:
            raise ValueError(
                f"Base scenario '{base_scenario.name}' has no faults to override"
            )
        base_fault_type = base_scenario.resolved_faults[0].type
        fault_def = fault_registry.get(base_fault_type, {})
        severity_maps_to = fault_def.get("severity_maps_to", "severity_end")

        duration = base_scenario.duration_s
        effective_onset = 0.0 if onset_s is None else float(onset_s)
        effective_end = duration if end_s is None else float(end_s)

        # --- Build params for the synthetic fault declaration ---
        synthetic_params: dict = {}

        if severity_maps_to in ("severity_start", "severity_end"):
            # Legacy path: severity maps to severity_start / severity_end
            synthetic_params["severity_start"] = float(fault_severity)
            synthetic_params["severity_end"] = float(fault_severity)
            if fault_progression and self._fault_has_param(fault_def, "progression"):
                synthetic_params["progression"] = fault_progression
        else:
            # New path: severity maps to a custom parameter
            synthetic_params[severity_maps_to] = float(fault_severity)
            # Progression handling depends on which progression parameter
            # the fault declares
            if fault_progression:
                if self._fault_has_param(fault_def, "drop_progression"):
                    synthetic_params["drop_progression"] = fault_progression
                elif self._fault_has_param(fault_def, "progression"):
                    synthetic_params["progression"] = fault_progression

        # Fill in defaults for any other declared parameters
        for p in fault_def.get("parameters", []):
            if p["name"] not in synthetic_params and "default" in p:
                synthetic_params[p["name"]] = p["default"]

        synthetic_fault_decl = {
            "id": recording_id,
            "type": base_fault_type,
            "onset": {"anchor": "scenario_start", "offset_s": effective_onset},
            "end":   {"anchor": "scenario_start", "offset_s": effective_end},
            "params": synthetic_params,
        }

        resolved_faults = _resolve_faults(
            [synthetic_fault_decl],
            base_scenario.segments,
            duration,
            fault_registry,
        )

        modified_scenario = Scenario(
            name=base_scenario.name,
            duration_s=duration,
            base_rate_hz=base_scenario.base_rate_hz,
            vehicle_id=base_scenario.vehicle_id,
            segments=base_scenario.segments,
            resolved_faults=resolved_faults,
        )

        return self._run_scenario_to_directory(
            modified_scenario,
            output_dir=Path(output_dir),
            filename_prefix=recording_id,
            create_vehicle_subdir=create_vehicle_subdir,
        )

    @staticmethod
    def _fault_has_param(fault_def: dict, param_name: str) -> bool:
        """Check whether the fault declares a parameter of the given name."""
        return any(p["name"] == param_name for p in fault_def.get("parameters", []))
    # ------------------------------------------------------------------
    # Internal execution — shared by run() and run_with_fault_severity()
    # ------------------------------------------------------------------

    def _run_scenario_to_directory(
        self,
        scenario: Scenario,
        output_dir: Path,
        filename_prefix: str,
        *,
        create_vehicle_subdir: bool = True,
    ) -> PipelineResult:
        """
        Run the full pipeline for a Scenario object and write files with a
        custom filename prefix.
        """
        # --- Validate vehicle exists ---
        if scenario.vehicle_id not in self.vehicles:
            raise KeyError(
                f"Scenario '{scenario.name}' references unknown vehicle "
                f"'{scenario.vehicle_id}'. Available: {list(self.vehicles.keys())}"
            )

        vehicle = self.vehicles[scenario.vehicle_id]

        # --- Stage 1: signals (with optional signal-layer fault injection) ---
        if scenario.resolved_faults:
            # Local import to avoid a circular dependency at module load time
            from src.fault_injector import SignalFaultInjector

            injector = SignalFaultInjector(self.loader, self.generator)
            signals_df, applied_faults = injector.apply(scenario)
        else:
            signals_df = self.generator.generate(scenario)
            applied_faults = []

        # --- Stage 2: encode ---
        encoder = CANEncoder(vehicle, self.signals_cfg)
        frames_by_id = self._encode_to_frames_by_id(encoder, signals_df)

        # --- Stage 3: schedule ---
        messages = self.loader.get_messages_for_vehicle(scenario.vehicle_id)
        scheduler = FrameScheduler(
            messages=messages,
            base_rate_hz=scenario.base_rate_hz,
            duration_s=scenario.duration_s,
        )
        scheduled = scheduler.schedule(frames_by_id)

        # --- Stage 4: write signals + frames CSVs ---
        output_dir.mkdir(parents=True, exist_ok=True)
        if create_vehicle_subdir:
            vehicle_dir = output_dir / scenario.vehicle_id
        else:
            vehicle_dir = output_dir
        vehicle_dir.mkdir(parents=True, exist_ok=True)

        frames_path = vehicle_dir / f"{filename_prefix}.frames.csv"
        signals_path = vehicle_dir / f"{filename_prefix}.signals.csv"

        self._write_frames_csv(frames_path, scheduled)
        self._write_signals_csv(
            signals_path,
            signals_df,
            scenario,
            encoder=encoder,
        )

        # --- Stage 5 (conditional): write fault label files ---
        faults_path: Path | None = None
        labels_path: Path | None = None

        if applied_faults:
            label_builder = FaultLabelBuilder(self.loader)
            event_rows, frame_rows = label_builder.build(scheduled, applied_faults)

            faults_path = vehicle_dir / f"{filename_prefix}.faults.csv"
            labels_path = vehicle_dir / f"{filename_prefix}.labels.csv"

            self._write_dict_rows_csv(faults_path, event_rows)
            self._write_dict_rows_csv(labels_path, frame_rows)

        return PipelineResult(
            frames_path=frames_path,
            signals_path=signals_path,
            n_frames=len(scheduled),
            n_signal_rows=len(signals_df),
            scenario=scenario,
            applied_faults=applied_faults,
            faults_path=faults_path,
            labels_path=labels_path,
        )

    # ------------------------------------------------------------------
    # Internal helpers — stage transitions
    # ------------------------------------------------------------------

    def _encode_to_frames_by_id(
        self,
        encoder: CANEncoder,
        df: pd.DataFrame,
    ) -> dict[int, list[bytes]]:
        """
        Encode a DataFrame into {can_id: [payload_0, payload_1, ...]}.
        """
        frames_by_id: dict[int, list[bytes]] = {}
        for step_frames in encoder.encode_dataframe(df):
            for can_id, payload in step_frames.items():
                frames_by_id.setdefault(can_id, []).append(payload)
        return frames_by_id

    # ------------------------------------------------------------------
    # Internal helpers — CSV writing
    # ------------------------------------------------------------------

    def _write_frames_csv(
        self,
        path: Path,
        scheduled: list[ScheduledFrame],
    ) -> None:
        """Write the asynchronous frame stream to CSV."""
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp_ms", "can_id", "payload_hex"])
            for frame in scheduled:
                writer.writerow([
                    frame.timestamp_ms,
                    f"0x{frame.can_id:03X}",
                    frame.payload.hex(),
                ])

    def _write_signals_csv(
        self,
        path: Path,
        df: pd.DataFrame,
        scenario: Scenario,
        encoder: CANEncoder,
    ) -> None:
        """
        Write the decoded synchronous signals to CSV.

        Columns:
            time_s, timestamp_ms, <each mapped signal>
        """
        signal_columns = [s for s in encoder.mapped_signal_names() if s in df.columns]

        out = pd.DataFrame()
        out["time_s"] = df["time_s"].round(4)
        out["timestamp_ms"] = (df["time_s"] * 1000).round().astype(int)
        for col in signal_columns:
            out[col] = df[col]

        out.to_csv(path, index=False, encoding="utf-8")

    @staticmethod
    def _write_dict_rows_csv(path: Path, rows: list[dict]) -> None:
        """
        Write a list of dicts to CSV.

        Column order comes from the first row's keys. If `rows` is empty,
        nothing is written (no headers).
        """
        if not rows:
            return
        columns = list(rows[0].keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    @staticmethod
    def _slugify(name: str) -> str:
        """Convert a scenario name into a safe filename."""
        return (
            name.lower()
            .replace(" ", "_")
            .replace("/", "_")
            .replace("\\", "_")
        )