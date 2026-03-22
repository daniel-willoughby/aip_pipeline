"""
config.py
---------
Central configuration for the AIP data ingestion and estimation pipeline.
All tuneable constants live here — edit this file rather than the source modules.
"""

from pathlib import Path

# ── Directory layout ─────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

# ── Sensor count ─────────────────────────────────────────────────────────────
SENSOR_COUNT = 5

# ── Log file schema ──────────────────────────────────────────────────────────
# Elapsed-time column — searched in order, case-insensitive, first match wins
LOG_TIME_COL_CANDIDATES = ["time s", "time_s", "time (s)", "elapsed_s"]

LOG_REQUIRED_COLS = ["KTAS", "OAT", "AOA", "Pinf"]

# Per-sensor column templates  ({n} = 1-based sensor number)
LOG_SENSOR_COL_TEMPLATES = {
    "BL":          "BL_{n}",
    "HM":          "HM_{n}",
    "POWER_TOTAL": "Power_Total_{n}",
    "POWER_DRY":   "Power_dry_{n}",
    "POWER_WET":   "Power_wet_{n}",
}

# ── Flight file schema ───────────────────────────────────────────────────────
# Row 0 = long column names, row 1 = units  (units row is stripped automatically)
FLIGHT_TIME_COL_IDX = 1   # 0-based index of elapsed-time column

FLIGHT_COL_MAP = {
    "FLT_KTAS":  "true airspeed",
    "FLT_OAT":   "static air temperature",
    "FLT_ALT":   "pressure altitude",
    "FLT_PINF":  "static pressure true",
    "FLT_AOA":   "angle of attack",
    "FLT_ROLL":  "roll angle",
    "REF_LWC":   "icd lwc",        # reference LWC from M300 probe  (g/m3)
    "REF_MVD":   "ccp mvd",        # reference MVD from M300 probe  (um)
}

# ── Merge / alignment ────────────────────────────────────────────────────────
MERGE_TOLERANCE_S = 0.6    # max gap (s) for nearest-neighbour time join

# ── Validation ───────────────────────────────────────────────────────────────
MAX_NULL_FRACTION    = 0.10   # warn if any column exceeds this NaN fraction
MIN_CALIBRATION_ROWS = 100    # minimum rows needed for a reliable fit

# ── LWC estimation ───────────────────────────────────────────────────────────
# Heat balance:  LWC_n = P_wet_n / (KTAS_ms * k_n)
# k_n is a per-sensor calibration constant fitted against REF_LWC.

KTAS_TO_MS = 0.514444        # 1 knot -> m/s

# Only rows where REF_LWC >= this threshold are used for calibration
# (avoids fitting on near-zero / noise-dominated points)
LWC_CALIB_MIN_REF = 0.01     # g/m3

# Heater warm-up window (seconds after WOW-off).
# Power_wet readings before this point are unreliable — forced to zero.
DRY_WINDOW_END_S = 5.0

# ── MVD estimation ───────────────────────────────────────────────────────────
# Empirical regression: MVD = f(LWC_1..5, KTAS, OAT, PINF)

# Only rows where REF_MVD >= this threshold are used for calibration
MVD_CALIB_MIN_REF = 1.0      # um

# 'gradient_boosting' (recommended), 'ridge', or 'linear'
MVD_MODEL_TYPE = "gradient_boosting"
MVD_RIDGE_ALPHA = 1.0        # regularisation strength for Ridge
