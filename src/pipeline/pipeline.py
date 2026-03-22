"""
pipeline.py
-----------
Top-level orchestrator.  Call run_pipeline() from your main script or notebook.

Returns a PipelineResult dataclass with all four DataFrames and basic summary stats.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import pandas as pd

from .config import MAX_NULL_FRACTION, LOG_REQUIRED_COLS
from .ingestor import LogIngestor, FlightIngestor
from .merger import DataMerger
from .validator import DataValidator


@dataclass
class PipelineResult:
    """Container for all DataFrames produced by the pipeline."""

    flight_df: pd.DataFrame        # cleaned flight reference data
    log_edited_df: pd.DataFrame    # log flight params (TIME, KTAS, OAT, LOG_AOA, PINF)
    log_sensor_df: pd.DataFrame    # normalised sensor data (5× rows per timestamp)
    merged_params_df: pd.DataFrame # log params merged with flight reference cols
    merged_sensor_df: pd.DataFrame # sensor data merged with flight reference cols
    summary: dict = field(default_factory=dict)

    def __post_init__(self):
        self.summary = {
            "flight_rows":        len(self.flight_df),
            "log_rows":           len(self.log_edited_df),
            "sensor_rows":        len(self.log_sensor_df),
            "merged_params_rows": len(self.merged_params_df),
            "merged_sensor_rows": len(self.merged_sensor_df),
            "ref_lwc_coverage":   self.merged_params_df["REF_LWC"].notna().mean()
                                  if "REF_LWC" in self.merged_params_df.columns else None,
        }

    def print_summary(self):
        print("\n" + "=" * 55)
        print("  PIPELINE RESULT SUMMARY")
        print("=" * 55)
        for k, v in self.summary.items():
            if isinstance(v, float):
                print(f"  {k:<28} {v:.1%}")
            else:
                print(f"  {k:<28} {v:,}")
        print("=" * 55)


def run_pipeline(
    flight_file: str | Path,
    log_files: List[str | Path],
    validate: bool = True,
) -> PipelineResult:
    """
    Execute the full ingestion pipeline.

    Parameters
    ----------
    flight_file : path to the flight Excel file (absolute or relative to DATA_DIR)
    log_files   : list of paths to AIP log Excel files
    validate    : if True, run data-quality checks and raise on hard errors

    Returns
    -------
    PipelineResult with all DataFrames populated
    """

    print("\n" + "─" * 55)
    print("  STEP 1 / 3  —  Ingesting log files")
    print("─" * 55)
    log_ing = LogIngestor(log_files)
    log_edited, log_sensor, log_raw = log_ing.load()

    print("\n" + "─" * 55)
    print("  STEP 2 / 3  —  Ingesting flight file")
    print("─" * 55)
    flt_ing = FlightIngestor(flight_file)
    flight_df = flt_ing.load()

    if validate:
        print("\n  Validating log data …")
        (
            DataValidator(log_edited, stage="log")
            .check_required_columns(["TIME", "KTAS", "OAT", "LOG_AOA", "PINF"])
            .check_null_fractions()
            .check_time_monotonic()
            .check_time_overlap(flight_df)
            .validate()
        )
        print("  Validating flight data …")
        (
            DataValidator(flight_df, stage="flight")
            .check_required_columns(["TIME"])
            .check_null_fractions()
            .check_time_monotonic()
            .validate()
        )

    print("\n" + "─" * 55)
    print("  STEP 3 / 3  —  Merging log ↔ flight")
    print("─" * 55)
    merger = DataMerger(log_edited, log_sensor, flight_df)
    merged_params, merged_sensor = merger.merge()

    if validate:
        print("  Validating merged data …")
        (
            DataValidator(merged_params, stage="merged")
            .check_null_fractions()
            .validate()
        )

    result = PipelineResult(
        flight_df=flight_df,
        log_edited_df=log_edited,
        log_sensor_df=log_sensor,
        merged_params_df=merged_params,
        merged_sensor_df=merged_sensor,
    )
    result.print_summary()
    return result
