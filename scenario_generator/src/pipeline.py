"""
Pipeline — orchestrates the full synthetic CAN data generation workflow.

Chain:
    Scenario → SignalGenerator → CANEncoder → FrameScheduler → CSV output

Design (Step 4):
    - Two files per scenario:
        {vehicle_id}/{scenario_name}.frames.csv   (raw CAN frames)
        {vehicle_id}/{scenario_name}.signals.csv  (decoded signals, 100 Hz)
    - Vehicle ID comes from the scenario YAML (no override).
    - Directory structure: data/{vehicle_id}/{scenario_name}.*.csv
    - CSV encoding: UTF-8, newline='', deterministic column order.

Outputs:
    - frames CSV:  timestamp_ms, can_id, payload_hex
    - signals CSV: time_s, timestamp_ms, <signals...>
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.can_encoder import CANEncoder
from src.config_loader import ConfigLoader
from src.frame_scheduler import FrameScheduler, ScheduledFrame
from src.signal_generator import Scenario, SignalGenerator


@dataclass
class PipelineResult:
    """Returned by Pipeline.run() — paths to the two output files."""
    frames_path: Path
    signals_path: Path
    n_frames: int
    n_signal_rows: int
    scenario: Scenario


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

        Returns
        -------
        PipelineResult
            Paths and metadata for the generated files.
        """
        scenario_path = Path(scenario_path)
        scenario = Scenario.from_yaml(scenario_path)

        # --- Validate vehicle exists ---
        if scenario.vehicle_id not in self.vehicles:
            raise KeyError(
                f"Scenario '{scenario.name}' references unknown vehicle "
                f"'{scenario.vehicle_id}'. Available: {list(self.vehicles.keys())}"
            )

        vehicle = self.vehicles[scenario.vehicle_id]

        # --- Stage 1: signals ---
        signals_df = self.generator.generate(scenario)

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

        # --- Stage 4: write CSVs ---
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

        return PipelineResult(
            frames_path=frames_path,
            signals_path=signals_path,
            n_frames=len(scheduled),
            n_signal_rows=len(signals_df),
            scenario=scenario,
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
    def _slugify(name: str) -> str:
        """Convert a scenario name into a safe filename."""
        return (
            name.lower()
            .replace(" ", "_")
            .replace("/", "_")
            .replace("\\", "_")
        )