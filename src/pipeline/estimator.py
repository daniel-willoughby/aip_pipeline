"""
estimator.py
------------
LWC estimation (per sensor, direct ML regression) and MVD estimation
(multi-sensor empirical regression).

LWC approach — direct ML regression
-------------------------------------
Correlation analysis across all three flights shows that POWER_TOTAL/KTAS
is the strongest single LWC signal (corr ~0.63-0.65), suggesting the heater
operates in constant-temperature mode — the controller raises total power
when water impinges rather than recording a wet/dry split.

Rather than assuming a physical formula, we fit a per-sensor Ridge regression
directly against REF_LWC using all available sensor features:

    Features per sensor:  POWER_TOTAL, POWER_DRY, POWER_WET, HM, BL,
                          HM-BL, POWER_TOTAL/KTAS, HM-BL/KTAS,
                          KTAS, OAT, PINF

    Target: REF_LWC (g/m³)

    Model:  Ridge regression with StandardScaler (per sensor)

A dry baseline window (first DRY_WINDOW_END_S seconds) is excluded from
training to avoid fitting on non-icing startup transients.

MVD
---
All five per-sensor LWC estimates plus flight parameters are used as features
in a regularised regression against REF_MVD.
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

from .config import (
    SENSOR_COUNT,
    KTAS_TO_MS,
    DRY_WINDOW_END_S,
    LWC_CALIB_MIN_REF,
    MVD_CALIB_MIN_REF,
    MVD_MODEL_TYPE,
    MVD_RIDGE_ALPHA,
    MIN_CALIBRATION_ROWS,
)

# Ridge alpha for LWC model — slightly higher than MVD for stability
LWC_RIDGE_ALPHA = 0.1


# ── Utilities ─────────────────────────────────────────────────────────────────

def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _print_metrics(label: str, y_true: np.ndarray, y_pred: np.ndarray):
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    yt, yp = y_true[mask], y_pred[mask]
    if len(yt) < 2:
        print(f"  {label}: insufficient data")
        return
    print(f"  {label:<14}  R²={r2_score(yt, yp):+.4f}  "
          f"MAE={mean_absolute_error(yt, yp):.5f}  "
          f"RMSE={_rmse(yt, yp):.5f}")


def _prefer_col(df: pd.DataFrame, primary: str, fallback: str) -> str:
    return primary if primary in df.columns else fallback


def _build_lwc_features(df: pd.DataFrame) -> np.ndarray:
    """
    Build the LWC feature matrix for a single-sensor slice of the DataFrame.
    Returns a 2D array of shape (n_rows, n_features).
    """
    ktas_col = _prefer_col(df, "KTAS", "FLT_KTAS")
    oat_col  = _prefer_col(df, "OAT",  "FLT_OAT")
    pinf_col = _prefer_col(df, "PINF", "FLT_PINF")

    ktas_ms = df[ktas_col].values.astype(float) * KTAS_TO_MS

    ptot = df["POWER_TOTAL"].values.astype(float)
    pdry = df["POWER_DRY"].values.astype(float)
    pwet = df["POWER_WET"].values.astype(float)
    hm   = df["HM"].values.astype(float)
    bl   = df["BL"].values.astype(float)

    with np.errstate(divide="ignore", invalid="ignore"):
        ptot_ktas  = np.where(ktas_ms > 0, ptot / ktas_ms, 0.0)
        hm_bl_ktas = np.where(ktas_ms > 0, (hm - bl) / ktas_ms, 0.0)
        pwet_ktas  = np.where(ktas_ms > 0, pwet / ktas_ms, 0.0)

    features = np.column_stack([
        ptot,                                    # total heater power
        pdry,                                    # dry component
        pwet,                                    # wet component
        hm,                                      # heater mat temp
        bl,                                      # baseline skin temp
        hm - bl,                                 # temperature differential
        ptot_ktas,                               # power / airspeed  ← strongest signal
        hm_bl_ktas,                              # temp diff / airspeed
        pwet_ktas,                               # wet power / airspeed
        df[ktas_col].values.astype(float),       # KTAS
        df[oat_col].values.astype(float),        # OAT
        df[pinf_col].values.astype(float) if pinf_col in df.columns else np.zeros(len(df)),
    ])
    return features


# ── LWC Estimator ─────────────────────────────────────────────────────────────

class LWCEstimator:
    """
    Per-sensor LWC estimation via direct Ridge regression on all available
    sensor and flight features.

    A separate model is fitted per sensor, allowing each sensor's unique
    thermal characteristics to be captured independently.
    """

    def __init__(self):
        self._models:   dict[int, Ridge] = {}          # sensor → model
        self._scalers:  dict[int, StandardScaler] = {} # sensor → scaler
        self._calibrated = False

    # ── public API ─────────────────────────────────────────────────────────

    def calibrate(self, sensor_df: pd.DataFrame,
                  label: str = "") -> "LWCEstimator":
        """
        Fit a Ridge regression per sensor against REF_LWC.

        Training rows: TIME > DRY_WINDOW_END_S and REF_LWC >= LWC_CALIB_MIN_REF
        """
        tag = f" [{label}]" if label else ""
        print(f"\n{'─' * 55}")
        print(f"  LWC CALIBRATION{tag}")
        print(f"{'─' * 55}")

        cal_mask = (
            (sensor_df["TIME"] > DRY_WINDOW_END_S) &
            (sensor_df["REF_LWC"] >= LWC_CALIB_MIN_REF) &
            sensor_df["POWER_TOTAL"].notna()
        )
        cal_df = sensor_df[cal_mask]
        print(f"  Calibration rows (REF_LWC >= {LWC_CALIB_MIN_REF}): {cal_mask.sum():,}")

        feature_names = [
            "POWER_TOTAL", "POWER_DRY", "POWER_WET", "HM", "BL",
            "HM-BL", "POWER_TOTAL/KTAS", "HM-BL/KTAS", "POWER_WET/KTAS",
            "KTAS", "OAT", "PINF"
        ]

        for n in range(1, SENSOR_COUNT + 1):
            s = cal_df[cal_df["SENSOR"] == n].copy()
            if len(s) < MIN_CALIBRATION_ROWS:
                warnings.warn(f"Sensor {n}: only {len(s)} cal rows.")
                continue

            X = _build_lwc_features(s)
            y = s["REF_LWC"].values

            # Drop rows with any NaN in features
            valid = np.isfinite(X).all(axis=1) & np.isfinite(y)
            if valid.sum() < MIN_CALIBRATION_ROWS:
                warnings.warn(f"Sensor {n}: only {valid.sum()} valid rows after NaN drop.")
                continue

            scaler = StandardScaler()
            X_sc   = scaler.fit_transform(X[valid])
            model  = Ridge(alpha=LWC_RIDGE_ALPHA)
            model.fit(X_sc, y[valid])

            r2 = r2_score(y[valid], np.clip(model.predict(X_sc), 0, None))
            self._models[n]  = model
            self._scalers[n] = scaler
            print(f"  Sensor {n}:  R²={r2:.4f}  (n={valid.sum():,})")

            # Print top 3 feature importances
            coef_ranked = sorted(zip(feature_names, model.coef_),
                                 key=lambda x: abs(x[1]), reverse=True)[:3]
            top = "  ".join(f"{f}={c:+.3f}" for f, c in coef_ranked)
            print(f"    Top features: {top}")

        self._calibrated = True
        return self

    def predict(self, sensor_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply calibrated LWC models. Returns wide DataFrame with
        TIME, flight params, LWC_1..5, LWC_MEAN (one row per timestep).
        """
        if not self._calibrated:
            raise RuntimeError("Call calibrate() before predict().")

        sens_arr = sensor_df["SENSOR"].values
        lwc_arrays: dict[int, np.ndarray] = {
            n: np.zeros(len(sensor_df)) for n in range(1, SENSOR_COUNT + 1)
        }

        for n in range(1, SENSOR_COUNT + 1):
            if n not in self._models:
                continue
            mask = sens_arr == n
            s_df = sensor_df[mask].copy()
            X    = _build_lwc_features(s_df)
            valid = np.isfinite(X).all(axis=1)

            preds = np.zeros(len(s_df))
            if valid.any():
                X_sc = self._scalers[n].transform(X[valid])
                preds[valid] = np.clip(self._models[n].predict(X_sc), 0, None)

            # Zero out warm-up window
            preds[s_df["TIME"].values <= DRY_WINDOW_END_S] = 0.0
            lwc_arrays[n][mask] = preds

        # ── Pivot to wide ─────────────────────────────────────────────────
        lwc_cols  = [f"LWC_{n}" for n in range(1, SENSOR_COUNT + 1)]
        drop_cols = {"SENSOR", "BL", "HM", "POWER_TOTAL", "POWER_DRY",
                     "POWER_WET", "_flight", "_split"}
        s1_mask   = sens_arr == 1
        base_cols = [c for c in sensor_df.columns
                     if c not in drop_cols and c not in lwc_cols]

        wide = sensor_df.loc[s1_mask, base_cols].copy().reset_index(drop=True)
        for n in range(1, SENSOR_COUNT + 1):
            sn_mask = sens_arr == n
            wide[f"LWC_{n}"] = lwc_arrays[n][sn_mask][:len(wide)]
        wide["LWC_MEAN"] = wide[lwc_cols].mean(axis=1)

        print(f"\n  LWC predicted  →  {len(wide):,} timesteps")
        if "REF_LWC" in wide.columns:
            ref_valid = wide["REF_LWC"] >= LWC_CALIB_MIN_REF
            if ref_valid.sum() > 0:
                print(f"  Accuracy (REF_LWC >= {LWC_CALIB_MIN_REF}, "
                      f"n={ref_valid.sum():,}):")
                yt = wide.loc[ref_valid, "REF_LWC"].values
                for n in range(1, SENSOR_COUNT + 1):
                    _print_metrics(f"    Sensor {n}", yt,
                                   wide.loc[ref_valid, f"LWC_{n}"].values)
                _print_metrics("    Mean   ", yt,
                               wide.loc[ref_valid, "LWC_MEAN"].values)
        return wide


# ── MVD Estimator ─────────────────────────────────────────────────────────────

class MVDEstimator:
    """
    Empirical MVD estimation using per-sensor LWC and flight parameters.
    Features: LWC_1..5, KTAS, OAT, PINF   Target: REF_MVD (µm)
    """

    def __init__(self):
        self._model        = None
        self._scaler       = StandardScaler()
        self._feature_cols: list[str] = []
        self._calibrated   = False
        self._use_scaler   = True

    def calibrate(self, lwc_wide_df: pd.DataFrame,
                  label: str = "") -> "MVDEstimator":
        tag = f" [{label}]" if label else ""
        print("\n" + "─" * 55)
        print(f"  MVD CALIBRATION{tag}")
        print("─" * 55)

        df, self._feature_cols = self._build_features(lwc_wide_df)
        train = df.dropna(subset=self._feature_cols + ["REF_MVD"])
        train = train[train["REF_MVD"] >= MVD_CALIB_MIN_REF]

        print(f"  Training rows (REF_MVD >= {MVD_CALIB_MIN_REF} µm): {len(train):,}")
        if len(train) < MIN_CALIBRATION_ROWS:
            warnings.warn(f"Only {len(train)} MVD calibration rows.")

        X = train[self._feature_cols].values
        y = train["REF_MVD"].values

        if MVD_MODEL_TYPE == "gradient_boosting":
            print("  Model: HistGradientBoostingRegressor")
            self._model = HistGradientBoostingRegressor(
                learning_rate=0.05,
                max_leaf_nodes=50,
                min_samples_leaf=20,
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=25,
                random_state=42,
            )
            self._model.fit(X, y)
            self._scaler = None  # HGBR does not need scaling
        else:
            print(f"  Model: Ridge (alpha={MVD_RIDGE_ALPHA})")
            X_sc = self._scaler.fit_transform(X)
            self._model = Ridge(alpha=MVD_RIDGE_ALPHA)
            self._model.fit(X_sc, y)

        self._calibrated = True
        _print_metrics("  Train ", y, self._predict_raw(X))

        if MVD_MODEL_TYPE == "gradient_boosting":
            print(f"\n  Feature importances ({len(self._feature_cols)} features):")
            for feat, imp in sorted(zip(self._feature_cols,
                                        self._model.feature_importances_ if hasattr(self._model, "feature_importances_") else np.ones(len(self._feature_cols))),
                                    key=lambda x: x[1], reverse=True):
                print(f"    {feat:<22} {imp:.4f}")
        else:
            print(f"\n  Features: {self._feature_cols}")
            print("  Standardised coefficients:")
            for feat, c in sorted(zip(self._feature_cols, self._model.coef_),
                                   key=lambda x: abs(x[1]), reverse=True):
                print(f"    {feat:<22} {c:+.4f}")
        return self

    def _predict_raw(self, X: np.ndarray) -> np.ndarray:
        if MVD_MODEL_TYPE == "gradient_boosting":
            return np.clip(self._model.predict(X), 0, None)
        X_sc = self._scaler.transform(X)
        return np.clip(self._model.predict(X_sc), 0, None)

    def predict(self, lwc_wide_df: pd.DataFrame) -> pd.DataFrame:
        if not self._calibrated:
            raise RuntimeError("Call calibrate() before predict().")

        df, _ = self._build_features(lwc_wide_df)
        X     = df[self._feature_cols].values.astype(float)
        valid = np.isfinite(X).all(axis=1)
        y_est = np.full(len(df), np.nan)
        if valid.any():
            y_est[valid] = np.clip(self._model.predict(X[valid]), 0, None)

        out = lwc_wide_df.copy()
        out["MVD_EST"] = np.nan
        out.loc[valid, "MVD_EST"] = y_est[valid]

        print(f"\n  MVD predicted  ->  {valid.sum():,} / {len(out):,} rows")
        if "REF_MVD" in out.columns:
            ref_mask = out["REF_MVD"] >= MVD_CALIB_MIN_REF
            if ref_mask.sum() > 0:
                _print_metrics("  Test  ",
                               out.loc[ref_mask, "REF_MVD"].values,
                               out.loc[ref_mask, "MVD_EST"].values)
        return out
    def _build_features(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        lwc_cols  = [f"LWC_{n}" for n in range(1, SENSOR_COUNT + 1)]
        ktas_col  = _prefer_col(df, "KTAS", "FLT_KTAS")
        oat_col   = _prefer_col(df, "OAT",  "FLT_OAT")
        pinf_col  = _prefer_col(df, "PINF", "FLT_PINF")

        # Start with a working copy to add engineered features
        out = df.copy()

        # ── Interaction: LWC_n × KTAS ─────────────────────────────────────
        # Strongest new predictors from correlation analysis
        for n in range(1, SENSOR_COUNT + 1):
            col = f"LWC_{n}_x_KTAS"
            out[col] = out[f"LWC_{n}"] * out[ktas_col]

        # ── Sensor spread features ────────────────────────────────────────
        lwc_matrix = out[lwc_cols].values
        with np.errstate(divide="ignore", invalid="ignore"):
            lwc_mean = np.nanmean(lwc_matrix, axis=1)
            lwc_std  = np.nanstd(lwc_matrix,  axis=1)
            lwc_cv   = np.where(lwc_mean > 1e-6, lwc_std / lwc_mean, 0.0)
        out["LWC_CV"]       = lwc_cv        # coefficient of variation (corr=-0.22)
        out["LWC_3-LWC_1"]  = out["LWC_3"] - out["LWC_1"]  # corr=+0.19
        out["LWC_3-LWC_2"]  = out["LWC_3"] - out["LWC_2"]  # corr=+0.17

        # ── Assemble feature list ─────────────────────────────────────────
        engineered = (
            [f"LWC_{n}_x_KTAS" for n in range(1, SENSOR_COUNT + 1)] +
            ["LWC_CV", "LWC_3-LWC_1", "LWC_3-LWC_2"]
        )
        base = lwc_cols + [ktas_col, oat_col, pinf_col]
        all_candidates = base + engineered

        seen, unique = set(), []
        for c in all_candidates:
            if c not in seen:
                unique.append(c)
                seen.add(c)

        feature_cols = [c for c in unique
                        if c in out.columns and out[c].notna().any()]

        keep = ["TIME"] + feature_cols
        if "REF_MVD" in out.columns:
            keep.append("REF_MVD")
        return out[keep].copy(), feature_cols
