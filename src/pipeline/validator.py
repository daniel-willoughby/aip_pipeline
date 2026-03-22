"""
validator.py
------------
Data-quality checks applied after ingestion and after merging.
Raises ValueError on hard failures; prints warnings for soft ones.
"""

from __future__ import annotations

import pandas as pd

from .config import MAX_NULL_FRACTION, SENSOR_COUNT


class DataValidator:
    """Run a suite of checks on a DataFrame and collect all issues."""

    def __init__(self, df: pd.DataFrame, stage: str = ""):
        self.df = df
        self.stage = stage
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def check_required_columns(self, required: list[str]) -> "DataValidator":
        missing = [c for c in required if c not in self.df.columns]
        if missing:
            self.errors.append(f"Missing required columns: {missing}")
        return self

    def check_null_fractions(
        self, threshold: float = MAX_NULL_FRACTION
    ) -> "DataValidator":
        null_rates = self.df.isnull().mean()
        bad = null_rates[null_rates > threshold]
        for col, rate in bad.items():
            self.warnings.append(
                f"Column '{col}' has {rate:.1%} NaN values (threshold {threshold:.0%})"
            )
        return self

    def check_time_monotonic(self, time_col: str = "TIME") -> "DataValidator":
        if time_col in self.df.columns:
            if not self.df[time_col].is_monotonic_increasing:
                diffs = self.df[time_col].diff()
                n_back = (diffs < 0).sum()
                self.warnings.append(
                    f"TIME column is not monotonically increasing ({n_back} backwards steps)"
                )
        return self

    def check_sensor_columns(self) -> "DataValidator":
        """Warn if any BL/HM sensor columns are all-zero (may indicate missing data)."""
        for s in range(1, SENSOR_COUNT + 1):
            for prefix in ("BL", "HM"):
                col = f"{prefix}_{s}"
                if col in self.df.columns and (self.df[col] == 0).all():
                    self.warnings.append(f"Sensor column '{col}' is all zeros")
        return self

    def check_time_overlap(
        self, flight_df: pd.DataFrame, log_time_col: str = "TIME"
    ) -> "DataValidator":
        """Warn if log and flight time ranges don't overlap substantially."""
        lt_min, lt_max = self.df[log_time_col].min(), self.df[log_time_col].max()
        ft_min, ft_max = flight_df["TIME"].min(), flight_df["TIME"].max()
        overlap_start = max(lt_min, ft_min)
        overlap_end = min(lt_max, ft_max)
        if overlap_end <= overlap_start:
            self.errors.append(
                f"No time overlap between log ({lt_min:.0f}–{lt_max:.0f} s) "
                f"and flight ({ft_min:.0f}–{ft_max:.0f} s)"
            )
        else:
            log_span = lt_max - lt_min
            overlap_frac = (overlap_end - overlap_start) / log_span if log_span > 0 else 0
            if overlap_frac < 0.5:
                self.warnings.append(
                    f"Only {overlap_frac:.0%} of log time range overlaps with flight data"
                )
        return self

    def validate(self, raise_on_error: bool = True) -> "DataValidator":
        label = f"[{self.stage}] " if self.stage else ""
        for w in self.warnings:
            print(f"  {label}WARNING: {w}")
        if self.errors:
            msg = "\n".join(f"  {label}ERROR: {e}" for e in self.errors)
            if raise_on_error:
                raise ValueError(f"Validation failed:\n{msg}")
            else:
                print(msg)
        else:
            print(f"  {label}Validation passed ✓  "
                  f"({len(self.warnings)} warning(s))")
        return self
