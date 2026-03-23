"""
inference.py
------------
Run LWC and MVD estimation on a new flight using a pre-trained model bundle.
"""

from __future__ import annotations

import pandas as pd
from pathlib import Path

from .ingestor  import LogIngestor, FlightIngestor
from .merger    import DataMerger
from .validator import DataValidator


def run_inference(
    flight_file: str | Path,
    log_files:   list[str | Path],
    model_bundle: dict,
) -> pd.DataFrame:
    """
    Ingest a new flight + log, merge, and apply pre-trained LWC/MVD models.

    Parameters
    ----------
    flight_file   : path to the flight .xlsx file
    log_files     : list of paths to log .xlsx files
    model_bundle  : dict loaded from models/aip_model.pkl, containing
                    'lwc_estimator' and 'mvd_estimator'

    Returns
    -------
    DataFrame with columns: TIME, FLT_KTAS, FLT_OAT, FLT_PINF,
                            LWC_1..5, LWC_MEAN, MVD_EST
                            (plus REF_LWC / REF_MVD if present in the data)
    """
    lwc_est = model_bundle["lwc_estimator"]
    mvd_est = model_bundle["mvd_estimator"]

    # ── Ingest ────────────────────────────────────────────────────────────────
    log_ing = LogIngestor(log_files)
    log_edited, log_sensor, log_raw = log_ing.load()

    flt_ing = FlightIngestor(flight_file)
    flt_df  = flt_ing.load()

    # ── Validate ──────────────────────────────────────────────────────────────
    validator = DataValidator()
    validator.validate(log_edited,  label="log")
    validator.validate(flt_df,      label="flight")

    # ── Merge ─────────────────────────────────────────────────────────────────
    merger = DataMerger()
    _, sensor_df = merger.merge(log_edited, log_sensor, flt_df)
    validator.validate(sensor_df, label="merged")

    # ── Predict ───────────────────────────────────────────────────────────────
    lwc_wide = lwc_est.predict(sensor_df)
    mvd_out  = mvd_est.predict(lwc_wide)

    return mvd_out
