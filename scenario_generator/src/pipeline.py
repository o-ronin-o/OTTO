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
      applied between SignalGenerator and CANEncoder. Frame-layer faults
      are handled later (Step 2.5).
    - The list of applied faults is carried on PipelineResult for
      downstream label generation (Step 2.2).

Design (Step 2.2):
    - If applied_faults is non-empty, two additional files are written:
        {vehicle_id}/{scenario_name}.faults.csv   (per-event table)
        {vehicle_id}/{scenario_name}.labels.csv   (per-frame labels)
    - Scenarios without faults do NOT produce these files.

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
from src.signal_generator import Scenario, SignalGenerator


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
            Root output directory. Files written to:
                {output_dir}/{vehicle_id}/{scenario_name}.frames.csv
                {output_dir}/{vehicle_id}/{scenario_name}.signals.csv
                {output_dir}/{vehicle_id}/{scenario_name}.faults.csv   (if faults)
                {output_dir}/{vehicle_id}/{scenario_name}.labels.csv   (if faults)

        Returns
        -------
        PipelineResult
            Paths and metadata for the generated files.
        """
        scenario_path = Path(scenario_path)
        fault_registry = self.loader.load_fault_types()
        scenario = Scenario.from_yaml(scenario_path, fault_registry=fault_registry)

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
        output_dir = Path(output_dir)
        vehicle_dir = output_dir / scenario.vehicle_id
        vehicle_dir.mkdir(parents=True, exist_ok=True)

        safe_name = self._slugify(scenario.name)
        frames_path = vehicle_dir / f"{safe_name}.frames.csv"
        signals_path = vehicle_dir / f"{safe_name}.signals.csv"

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

            faults_path = vehicle_dir / f"{safe_name}.faults.csv"
            labels_path = vehicle_dir / f"{safe_name}.labels.csv"

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
        # Determine which signals to write (only those with CAN mappings)
        signal_columns = [s for s in encoder.mapped_signal_names() if s in df.columns]

        # Build a clean output DataFrame
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