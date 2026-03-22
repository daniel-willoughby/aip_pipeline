"""
merger.py
---------
Aligns and merges the log-derived DataFrames with the flight reference data
using a nearest-neighbour time join.

Both sources use seconds-since-WOW-off as their TIME axis, so no conversion
is needed — they can be aligned directly.
"""

from __future__ import annotations

import pandas as pd

from .config import MERGE_TOLERANCE_S


class DataMerger:
    """
    Nearest-neighbour time merge between log data and flight reference data.

    Parameters
    ----------
    log_edited_df   : output of LogIngestor._build_edited()
    log_sensor_df   : output of LogIngestor._build_sensor()
    flight_df       : output of FlightIngestor.load()
    tolerance_s     : max time gap (seconds) to still consider a match
    """

    def __init__(
        self,
        log_edited_df: pd.DataFrame,
        log_sensor_df: pd.DataFrame,
        flight_df: pd.DataFrame,
        tolerance_s: float = MERGE_TOLERANCE_S,
    ):
        self.log_edited = log_edited_df.copy()
        self.log_sensor = log_sensor_df.copy()
        self.flight = flight_df.copy()
        self.tolerance_s = tolerance_s

    # ── public API ────────────────────────────────────────────────────────────

    def merge(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Returns
        -------
        merged_params   : log flight-params + flight reference cols, one row per log timestamp
        merged_sensor   : sensor DataFrame with reference cols attached
        """
        print(f"\nMerging on TIME (tolerance = {self.tolerance_s} s) …")

        params = self._merge_nearest(self.log_edited, self.flight)
        print(f"  Params merge  →  {params.shape[0]:,} rows  "
              f"({params['REF_LWC'].notna().sum():,} rows with reference LWC)")

        # Attach flight reference columns to the sensor df via TIME
        ref_cols = ["TIME"] + [c for c in params.columns if c.startswith("FLT_") or c.startswith("REF_")]
        sensor_left = self.log_sensor.copy()
        sensor_left["TIME"] = sensor_left["TIME"].astype(float)
        sensor = pd.merge_asof(
            sensor_left.sort_values("TIME"),
            params[ref_cols].sort_values("TIME"),
            on="TIME",
            direction="nearest",
            tolerance=self.tolerance_s,
        )
        print(f"  Sensor merge  →  {sensor.shape[0]:,} rows")
        return params, sensor

    # ── private ───────────────────────────────────────────────────────────────

    def _merge_nearest(
        self, left: pd.DataFrame, right: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge two DataFrames on TIME using nearest-neighbour within tolerance."""
        # Ensure TIME columns share the same dtype for merge_asof
        left = left.copy()
        right = right.copy()
        left["TIME"] = left["TIME"].astype(float)
        right["TIME"] = right["TIME"].astype(float)

        merged = pd.merge_asof(
            left.sort_values("TIME"),
            right.sort_values("TIME"),
            on="TIME",
            direction="nearest",
            tolerance=self.tolerance_s,
            suffixes=("", "_FLT"),
        )
        return merged.reset_index(drop=True)
