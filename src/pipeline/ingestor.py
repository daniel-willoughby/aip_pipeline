"""
ingestor.py
-----------
Handles reading and initial cleaning of flight and log Excel files.

Both LogIngestor and FlightIngestor use a multi-strategy approach to find
the correct time column, so they are robust to minor formatting changes
between flight campaigns.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

from .config import (
    DATA_DIR,
    LOG_REQUIRED_COLS,
    LOG_SENSOR_COL_TEMPLATES,
    SENSOR_COUNT,
    FLIGHT_COL_MAP,
)


# ── Classification constants ──────────────────────────────────────────────────

# Substrings that only appear in flight file headers
_FLIGHT_SIGNALS = [
    "iasp",
    "true airspeed",
    "pressure altitude",
    "static pressure true",
    "icd lwc",
    "ccp mvd",
]

# All known elapsed-time column name variants (case-insensitive)
_LOG_TIME_CANDIDATES = [
    "time s", "time_s", "time (s)", "elapsed_s", "time inc",
]

# Flight elapsed-time column candidates (checked before falling back to index 1)
_FLIGHT_TIME_CANDIDATES = [
    "time inc", "elapsed", "time s", "time_s",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _resolve(path: str | Path) -> Path:
    """Return absolute path; if relative, look inside DATA_DIR."""
    p = Path(path)
    return p if p.is_absolute() else DATA_DIR / p


def _find_col(df: pd.DataFrame, substring: str) -> str | None:
    """Return the first column whose name contains *substring* (case-insensitive)."""
    sub = substring.lower()
    for col in df.columns:
        if sub in str(col).lower():
            return col
    return None


def _col_names_lower(df: pd.DataFrame) -> list[str]:
    return [str(c).strip().lower() for c in df.columns]


def _is_flight_file(df: pd.DataFrame) -> bool:
    """
    Detect whether a DataFrame came from a flight file by scanning all
    column names and the first two rows for known flight-only signals.
    """
    # Check column headers
    header_text = " ".join(str(c).lower() for c in df.columns)
    if any(sig in header_text for sig in _FLIGHT_SIGNALS):
        return True

    # Check first two data rows (captures multi-row headers)
    for _, row in df.head(2).iterrows():
        row_text = " ".join(str(v).lower() for v in row if v is not None)
        if any(sig in row_text for sig in _FLIGHT_SIGNALS):
            return True

    return False


# ── Log file ingestion ────────────────────────────────────────────────────────

class LogIngestor:
    """
    Reads one or more AIP log Excel files, merges them on elapsed time,
    and returns tidy DataFrames with TIME, flight params, and per-sensor columns.

    Time column resolution (tried in order):
      1. Named column matching LOG_TIME_CANDIDATES (e.g. "Time S", "Time s")
      2. Compute from TIM microsecond counter: (TIM - TIM[0]) / 1,000,000
      3. Parse TFS datetime strings as a last resort
      4. Fall back to column index 2 with a warning
    """

    def __init__(self, filepaths: List[str | Path]):
        self.filepaths = [_resolve(f) for f in filepaths]

    # ── public API ─────────────────────────────────────────────────────────

    def load(self):
        frames = [self._load_single(fp) for fp in self.filepaths]
        merged = self._merge_files(frames)
        edited = self._build_edited(merged)
        sensor = self._build_sensor(merged, edited)
        return edited, sensor, merged

    # ── private ────────────────────────────────────────────────────────────

    def _load_single(self, fp: Path) -> pd.DataFrame:
        df = pd.read_excel(fp, engine="openpyxl")
        print(f"  Loaded log  '{fp.name}'  →  {df.shape[0]:,} rows × {df.shape[1]} cols")

        col_lower = _col_names_lower(df)
        time_col  = None

        # Strategy 1: named time column (case-insensitive, partial match)
        for candidate in _LOG_TIME_CANDIDATES:
            cand_lower = candidate.lower()
            match = next(
                (df.columns[i] for i, c in enumerate(col_lower) if c == cand_lower),
                None
            )
            if match:
                time_col = match
                break

        # Strategy 2: compute from TIM microsecond counter
        if time_col is None:
            tim_idx = next(
                (i for i, c in enumerate(col_lower) if c == "tim"), None
            )
            if tim_idx is not None:
                tim_col = df.columns[tim_idx]
                tim = pd.to_numeric(df[tim_col], errors="coerce")
                df["_TIME_S"] = (tim - tim.iloc[0]) / 1_000_000
                time_col = "_TIME_S"
                print(f"    INFO: computed TIME_S from '{tim_col}'")

        # Strategy 3: parse TFS datetime strings
        if time_col is None:
            tfs_idx = next(
                (i for i, c in enumerate(col_lower) if "tfs" in c), None
            )
            if tfs_idx is not None:
                tfs_col = df.columns[tfs_idx]
                ts = pd.to_datetime(df[tfs_col], errors="coerce")
                df["_TIME_S"] = (ts - ts.iloc[0]).dt.total_seconds()
                time_col = "_TIME_S"
                print(f"    INFO: computed TIME_S from datetime column '{tfs_col}'")

        # Strategy 4: fall back to column index 2
        if time_col is None:
            time_col = df.columns[2]
            print(f"    WARNING: no time column found; using col[2] '{time_col}'")

        # Coerce to numeric (handles Excel formula strings like "=(B3-$B$2)/1000000")
        df[time_col] = pd.to_numeric(df[time_col], errors="coerce")

        # Drop rows where time is NaN
        bad = df[time_col].isna().sum()
        if bad:
            print(f"    Dropping {bad} rows with non-numeric time in '{time_col}'")
            df = df.dropna(subset=[time_col]).reset_index(drop=True)

        df["_time_col"] = time_col
        return df

    def _merge_files(self, frames: List[pd.DataFrame]) -> pd.DataFrame:
        renamed = []
        for df in frames:
            tc = df["_time_col"].iloc[0]
            df = df.drop(columns=["_time_col"])
            df = df.rename(columns={tc: "TIME"})
            renamed.append(df)

        merged = pd.concat(renamed, ignore_index=True)
        merged = merged.sort_values("TIME", ascending=True).reset_index(drop=True)
        print(f"  Merged log  →  {merged.shape[0]:,} rows  |  TIME: "
              f"{merged['TIME'].min():.2f} – {merged['TIME'].max():.2f} s")
        return merged

    def _build_edited(self, df: pd.DataFrame) -> pd.DataFrame:
        edited = pd.DataFrame()
        edited["TIME"] = df["TIME"]

        for col in LOG_REQUIRED_COLS:
            match = _find_col(df, col)
            if match is None:
                raise KeyError(
                    f"Required log column '{col}' not found. "
                    f"Available: {list(df.columns)}"
                )
            edited[col] = df[match]

        edited = edited.rename(columns={"AOA": "LOG_AOA", "Pinf": "PINF"})
        return edited

    def _build_sensor(self, merged: pd.DataFrame,
                      edited: pd.DataFrame) -> pd.DataFrame:
        """Build the normalised sensor DataFrame — vectorised."""
        frames = []
        for s in range(1, SENSOR_COUNT + 1):
            s_df = pd.DataFrame()
            s_df["TIME"]   = edited["TIME"].values
            s_df["SENSOR"] = s
            for out_key, template in LOG_SENSOR_COL_TEMPLATES.items():
                if template is None:
                    s_df[out_key] = 0.0
                else:
                    col_name = template.replace("{n}", str(s))
                    match    = _find_col(merged, col_name)
                    s_df[out_key] = merged[match].values if match else 0.0
            frames.append(s_df)

        sensor_df = pd.concat(frames, ignore_index=True)
        sensor_df = sensor_df.sort_values(["TIME", "SENSOR"]).reset_index(drop=True)
        print(f"  Sensor DataFrame  →  {sensor_df.shape[0]:,} rows")
        return sensor_df


# ── Flight file ingestion ─────────────────────────────────────────────────────

class FlightIngestor:
    """
    Reads a single AIP flight Excel file.

    Flight files have two header rows: row 0 = long column names, row 1 = units.
    The units row is stripped automatically.

    Time column resolution (tried in order):
      1. Named elapsed-time column (e.g. "Time inc", "Time s")
      2. Column index 1 (always elapsed seconds in known flight files)
    """

    def __init__(self, filepath: str | Path):
        self.filepath = _resolve(filepath)

    def load(self) -> pd.DataFrame:
        df = pd.read_excel(self.filepath, engine="openpyxl", header=0)
        print(f"  Loaded flight '{self.filepath.name}'  →  "
              f"{df.shape[0]:,} rows × {df.shape[1]} cols")

        # Row 0 of the data is the units row — drop it
        df = df.iloc[1:].reset_index(drop=True)

        col_lower = _col_names_lower(df)
        out = pd.DataFrame()

        # ── Elapsed time column ───────────────────────────────────────────────
        # Strategy 1: named column
        time_col = next(
            (df.columns[i] for i, c in enumerate(col_lower)
             if c in _FLIGHT_TIME_CANDIDATES),
            None
        )
        # Strategy 2: column index 1
        if time_col is None:
            time_col = df.columns[1]
            print(f"    INFO: no named time column found; using col[1] '{time_col}'")

        out["TIME"] = pd.to_numeric(df[time_col], errors="coerce")

        # ── Reference and flight parameter columns ────────────────────────────
        for out_name, substring in FLIGHT_COL_MAP.items():
            match = _find_col(df, substring)
            if match is None:
                print(f"    WARNING: '{out_name}' ('{substring}') not found.")
                out[out_name] = float("nan")
            else:
                out[out_name] = pd.to_numeric(df[match], errors="coerce")

        out = out.dropna(subset=["TIME"]).reset_index(drop=True)
        print(f"  Flight DataFrame  →  {out.shape[0]:,} rows  |  TIME: "
              f"{out['TIME'].min():.2f} – {out['TIME'].max():.2f} s")
        return out
