"""
session.py
----------
Manages multiple PipelineResult objects across flights and lets the user
interactively assign each flight to a training or test split before running
the LWC / MVD estimators.

Because LWC and MVD ranges vary significantly between flights, training and
testing on the same flight can give misleadingly optimistic metrics.  This
module makes cross-flight evaluation straightforward.

Typical usage
-------------
    from src.pipeline import run_pipeline
    from src.pipeline.session import EstimationSession

    session = EstimationSession()
    session.add_flight("Flight 1", run_pipeline(flight1, [log3]))
    session.add_flight("Flight 2", run_pipeline(flight2, [log6, log7]))

    session.assign_splits()          # interactive prompt
    results = session.run()          # fit on train, evaluate on test

    # Access results
    results["Flight 1"].lwc_df       # wide DataFrame with LWC_1..5, LWC_MEAN
    results["Flight 1"].mvd_df       # same + MVD_EST
    results["Flight 1"].split        # 'train' or 'test'
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import pandas as pd

from .pipeline import PipelineResult
from .estimator import LWCEstimator, MVDEstimator


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class FlightResult:
    """Holds the estimation outputs for a single flight."""
    name:  str
    split: str                          # 'train' or 'test'
    pipeline: PipelineResult
    lwc_df: pd.DataFrame | None = None  # wide LWC predictions
    mvd_df: pd.DataFrame | None = None  # wide MVD predictions
    metrics: dict = field(default_factory=dict)


# ── Session ───────────────────────────────────────────────────────────────────

class EstimationSession:
    """
    Collects flights, manages train/test splits, and orchestrates estimation.
    """

    def __init__(self):
        self._flights: Dict[str, PipelineResult] = {}   # name → PipelineResult
        self._splits:  Dict[str, str] = {}              # name → 'train'/'test'
        self._lwc_est: LWCEstimator | None = None
        self._mvd_est: MVDEstimator | None = None

    # ── building the session ─────────────────────────────────────────────────

    def add_flight(self, name: str, result: PipelineResult) -> "EstimationSession":
        """Register a pipeline result under a friendly name."""
        self._flights[name] = result
        print(f"  Added flight '{name}'  "
              f"({result.summary['log_rows']:,} log rows, "
              f"{result.summary['sensor_rows']:,} sensor rows)")
        return self

    # ── split assignment ─────────────────────────────────────────────────────

    def assign_splits(self, splits: dict[str, str] | None = None) -> "EstimationSession":
        """
        Assign each flight to 'train' or 'test'.

        Parameters
        ----------
        splits : optional dict {flight_name: 'train'/'test'}.
                 If None (default), the user is prompted interactively.
        """
        if splits is not None:
            self._splits = {k: v.lower() for k, v in splits.items()}
            self._validate_splits()
            return self

        # ── Interactive prompt ───────────────────────────────────────────
        names = list(self._flights.keys())

        print("\n" + "═" * 55)
        print("  TRAIN / TEST SPLIT ASSIGNMENT")
        print("═" * 55)
        print("\nAvailable flights:")
        for i, name in enumerate(names, 1):
            r = self._flights[name]
            ref_lwc_max = r.merged_sensor_df["REF_LWC"].max() if "REF_LWC" in r.merged_sensor_df.columns else float("nan")
            ref_mvd_max = r.merged_sensor_df["REF_MVD"].max() if "REF_MVD" in r.merged_sensor_df.columns else float("nan")
            print(f"  {i}. {name:<20}  "
                  f"REF_LWC max={ref_lwc_max:.3f} g/m³  "
                  f"REF_MVD max={ref_mvd_max:.1f} µm")

        # Collect train indices
        train_indices = self._prompt_indices(
            names,
            "\nSelect TRAINING flight(s) — comma-separated numbers (e.g. 1,2): "
        )
        test_indices = self._prompt_indices(
            names,
            "Select TEST flight(s) — comma-separated numbers (e.g. 3): ",
            exclude=train_indices,
            allow_overlap=True,
        )

        for i, name in enumerate(names):
            if i in train_indices and i in test_indices:
                self._splits[name] = "both"
            elif i in train_indices:
                self._splits[name] = "train"
            elif i in test_indices:
                self._splits[name] = "test"
            else:
                self._splits[name] = "excluded"

        self._print_split_summary()
        self._validate_splits()
        return self

    # ── run estimation ────────────────────────────────────────────────────────

    def run(self) -> Dict[str, FlightResult]:
        """
        1. Concatenate all training sensor DataFrames and calibrate LWC + MVD.
        2. Predict on every flight (train and test).
        3. Return a dict of FlightResult objects.
        """
        if not self._splits:
            raise RuntimeError("Call assign_splits() before run().")

        # ── Build training set ────────────────────────────────────────────
        train_sensor_parts, train_lwc_parts = [], []

        print("\n" + "═" * 55)
        print("  COMBINING TRAINING DATA")
        print("═" * 55)
        for name, split in self._splits.items():
            if split in ("train", "both"):
                df = self._flights[name].merged_sensor_df.copy()
                df["_flight"] = name
                train_sensor_parts.append(df)
                print(f"  + {name}  ({len(df):,} sensor rows)")

        if not train_sensor_parts:
            raise RuntimeError("No flights assigned to training. "
                               "Call assign_splits() and select at least one training flight.")

        train_sensor_df = pd.concat(train_sensor_parts, ignore_index=True)
        print(f"  Total training sensor rows: {len(train_sensor_df):,}")

        # ── Calibrate LWC ────────────────────────────────────────────────
        self._lwc_est = LWCEstimator()
        self._lwc_est.calibrate(train_sensor_df, label="combined training")

        # Build wide LWC for training flights (needed for MVD calibration)
        for name, split in self._splits.items():
            if split in ("train", "both"):
                df = self._flights[name].merged_sensor_df.copy()
                lwc_wide = self._lwc_est.predict(df)
                train_lwc_parts.append(lwc_wide)

        train_lwc_df = pd.concat(train_lwc_parts, ignore_index=True)

        # ── Calibrate MVD ─────────────────────────────────────────────────
        self._mvd_est = MVDEstimator()
        self._mvd_est.calibrate(train_lwc_df, label="combined training")

        # ── Predict on all flights ────────────────────────────────────────
        print("\n" + "═" * 55)
        print("  PREDICTION — ALL FLIGHTS")
        print("═" * 55)

        results: Dict[str, FlightResult] = {}
        for name, split in self._splits.items():
            if split == "excluded":
                continue

            print(f"\n  [{name}]  split={split}")
            sensor_df = self._flights[name].merged_sensor_df
            lwc_df    = self._lwc_est.predict(sensor_df)
            mvd_df    = self._mvd_est.predict(lwc_df)

            metrics = self._compute_metrics(mvd_df, split=split)

            results[name] = FlightResult(
                name=name,
                split=split,
                pipeline=self._flights[name],
                lwc_df=lwc_df,
                mvd_df=mvd_df,
                metrics=metrics,
            )

        self._print_results_summary(results)
        return results

    # ── convenience properties ────────────────────────────────────────────────

    @property
    def lwc_estimator(self) -> LWCEstimator | None:
        return self._lwc_est

    @property
    def mvd_estimator(self) -> MVDEstimator | None:
        return self._mvd_est

    def flight_names(self) -> List[str]:
        return list(self._flights.keys())

    # ── private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _prompt_indices(
        names: list[str],
        prompt: str,
        exclude: set[int] | None = None,
        allow_overlap: bool = False,
    ) -> set[int]:
        while True:
            try:
                raw     = input(prompt)
                indices = {int(x.strip()) - 1 for x in raw.split(",") if x.strip()}
                valid   = {i for i in indices if 0 <= i < len(names)}
                if not valid:
                    print("  No valid selections — try again.")
                    continue
                if exclude and not allow_overlap:
                    overlap = valid & exclude
                    if overlap:
                        print(f"  Flights {[i+1 for i in overlap]} already selected — choose different ones.")
                        continue
                return valid
            except ValueError:
                print("  Enter numbers separated by commas.")

    def _validate_splits(self):
        train_count = sum(1 for s in self._splits.values() if s in ("train", "both"))
        if train_count == 0:
            raise ValueError("No flights assigned to training. "
                             "At least one flight must be in 'train'.")

    def _print_split_summary(self):
        print("\n  Split assignments:")
        for name, split in self._splits.items():
            print(f"    {name:<25}  →  {split}")

    @staticmethod
    def _compute_metrics(mvd_df: pd.DataFrame, split: str) -> dict:
        from sklearn.metrics import r2_score, mean_absolute_error
        metrics = {"split": split}

        lwc_cols = [f"LWC_{n}" for n in range(1, 6)]
        if "REF_LWC" in mvd_df.columns:
            from .config import LWC_CALIB_MIN_REF
            mask = mvd_df["REF_LWC"] >= LWC_CALIB_MIN_REF
            if mask.sum() > 1:
                yt = mvd_df.loc[mask, "REF_LWC"].values
                yp = mvd_df.loc[mask, "LWC_MEAN"].values
                valid = ~(np.isnan(yt) | np.isnan(yp))
                if valid.sum() > 1:
                    metrics["lwc_r2"]   = round(r2_score(yt[valid], yp[valid]), 4)
                    metrics["lwc_mae"]  = round(mean_absolute_error(yt[valid], yp[valid]), 5)

        if "REF_MVD" in mvd_df.columns and "MVD_EST" in mvd_df.columns:
            from .config import MVD_CALIB_MIN_REF
            mask = mvd_df["REF_MVD"] >= MVD_CALIB_MIN_REF
            if mask.sum() > 1:
                yt = mvd_df.loc[mask, "REF_MVD"].values
                yp = mvd_df.loc[mask, "MVD_EST"].values
                valid = ~(np.isnan(yt) | np.isnan(yp))
                if valid.sum() > 1:
                    metrics["mvd_r2"]   = round(r2_score(yt[valid], yp[valid]), 4)
                    metrics["mvd_mae"]  = round(mean_absolute_error(yt[valid], yp[valid]), 5)

        return metrics

    @staticmethod
    def _print_results_summary(results: Dict[str, FlightResult]):
        import numpy as np
        print("\n" + "═" * 55)
        print("  RESULTS SUMMARY")
        print("═" * 55)
        print(f"  {'Flight':<22} {'Split':<8} {'LWC R²':>8} {'LWC MAE':>10} {'MVD R²':>8} {'MVD MAE':>10}")
        print("  " + "─" * 53)
        for name, fr in results.items():
            m = fr.metrics
            print(f"  {name:<22} {m.get('split',''):<8} "
                  f"{m.get('lwc_r2', float('nan')):>8.4f} "
                  f"{m.get('lwc_mae', float('nan')):>10.5f} "
                  f"{m.get('mvd_r2', float('nan')):>8.4f} "
                  f"{m.get('mvd_mae', float('nan')):>10.3f}")
        print("═" * 55)

import numpy as np   # needed by _compute_metrics at module level


# ── Combined dataset split ────────────────────────────────────────────────────

class CombinedSession:
    """
    Combines sensor data from all flights into a single dataset, performs a
    randomised train/test split, calibrates on the training portion, and
    evaluates on the test portion.

    This gives a fairer picture of model performance than a per-flight split
    because both train and test sets contain samples from all flight conditions.

    Typical usage
    -------------
        session = CombinedSession(test_fraction=0.2, random_seed=42)
        session.add_flight("Flight 1", r1)
        session.add_flight("Flight 2", r2)
        results = session.run()

        results.train_lwc_df    # training set with LWC predictions
        results.test_lwc_df     # test set with LWC predictions
        results.test_mvd_df     # test set with MVD predictions
        results.metrics         # R², MAE for LWC and MVD on test set
    """

    def __init__(self, test_fraction: float = 0.2, random_seed: int = 42):
        if not 0.0 < test_fraction < 1.0:
            raise ValueError("test_fraction must be between 0 and 1.")
        self.test_fraction = test_fraction
        self.random_seed   = random_seed
        self._flights: Dict[str, PipelineResult] = {}
        self._lwc_est: LWCEstimator | None = None
        self._mvd_est: MVDEstimator | None = None

    # ── building ──────────────────────────────────────────────────────────────

    def add_flight(self, name: str, result: PipelineResult) -> "CombinedSession":
        self._flights[name] = result
        print(f"  Added flight '{name}'  "
              f"({result.summary['log_rows']:,} log rows)")
        return self

    # ── run ───────────────────────────────────────────────────────────────────

    def run(self, held_out_flight: str | None = None) -> "CombinedResult":
        """
        1. Concatenate all flights' sensor DataFrames into one dataset.
        2. Split into train / test:
             - If held_out_flight is set: that flight is test-only,
               all others use the random train/test split.
             - Otherwise: stratified random split across all flights.
        3. Calibrate LWC and MVD on training rows.
        4. Predict on the full dataset and report test-set metrics.

        Parameters
        ----------
        held_out_flight : optional flight name to exclude from training entirely.
                          All timesteps for that flight become test-only.
        """
        print(f"\n{'═' * 55}")
        print("  COMBINED DATASET SPLIT")
        if held_out_flight:
            print(f"  Leave-one-out: '{held_out_flight}' held out for testing only")
        print(f"{'═' * 55}")

        # ── Step 1: combine ───────────────────────────────────────────────
        parts = []
        for name, result in self._flights.items():
            df = result.merged_sensor_df.copy()
            df["_flight"] = name
            parts.append(df)

        combined_sensor = pd.concat(parts, ignore_index=True)

        s1 = combined_sensor[combined_sensor["SENSOR"] == 1].copy()
        n_total = len(s1)
        print(f"\n  Total timesteps : {n_total:,}")

        rng    = np.random.default_rng(self.random_seed)
        s1["_split"] = "train"

        print("\n  Split assignment per flight:")
        for name in self._flights:
            flight_mask  = s1["_flight"] == name
            flight_idxs  = s1.index[flight_mask]

            if held_out_flight and name == held_out_flight:
                # Entire flight is test-only
                s1.loc[flight_idxs, "_split"] = "test"
                print(f"    {name:<20}  total={len(flight_idxs):,}  "
                      f"*** HELD OUT (test only) ***")
            else:
                # Stratified random split
                n_test_flight = max(1, int(len(flight_idxs) * self.test_fraction))
                chosen = rng.choice(flight_idxs, size=n_test_flight, replace=False)
                s1.loc[chosen, "_split"] = "test"
                n_tr = len(flight_idxs) - n_test_flight
                print(f"    {name:<20}  total={len(flight_idxs):,}  "
                      f"test={n_test_flight:,}  train={n_tr:,}")

        n_test  = (s1["_split"] == "test").sum()
        n_train = (s1["_split"] == "train").sum()
        print(f"\n  Total: {n_train:,} train / {n_test:,} test timesteps")

        # Broadcast split to all 5 sensor rows
        time_to_split = dict(zip(s1["TIME"].values, s1["_split"].values))
        combined_sensor["_split"] = combined_sensor["TIME"].map(time_to_split)

        train_sensor = combined_sensor[combined_sensor["_split"] == "train"].copy()
        test_sensor  = combined_sensor[combined_sensor["_split"] == "test"].copy()

        # ── Flight composition of each split ──────────────────────────────
        print("\n  Flight composition:")
        for name in self._flights:
            n_tr = (train_sensor["_flight"] == name).sum() // 5
            n_te = (test_sensor["_flight"]  == name).sum() // 5
            print(f"    {name:<20}  train={n_tr:,}  test={n_te:,}  timesteps")

        # ── Step 2: calibrate LWC ─────────────────────────────────────────
        self._lwc_est = LWCEstimator()
        self._lwc_est.calibrate(train_sensor, label="combined train split")

        # ── Step 3: calibrate MVD ─────────────────────────────────────────
        train_lwc = self._lwc_est.predict(train_sensor)
        self._mvd_est = MVDEstimator()
        self._mvd_est.calibrate(train_lwc, label="combined train split")

        # ── Step 4: predict on train and test ─────────────────────────────
        print(f"\n{'─' * 55}")
        print("  PREDICTING — TRAIN SET")
        print(f"{'─' * 55}")
        train_lwc_df = self._lwc_est.predict(train_sensor)
        train_mvd_df = self._mvd_est.predict(train_lwc_df)

        print(f"\n{'─' * 55}")
        print("  PREDICTING — TEST SET")
        print(f"{'─' * 55}")
        test_lwc_df = self._lwc_est.predict(test_sensor)
        test_mvd_df = self._mvd_est.predict(test_lwc_df)

        # ── Compute metrics ───────────────────────────────────────────────
        train_metrics = _split_metrics(train_mvd_df, label="Train")
        test_metrics  = _split_metrics(test_mvd_df,  label="Test ")

        result = CombinedResult(
            train_lwc_df   = train_lwc_df,
            train_mvd_df   = train_mvd_df,
            test_lwc_df    = test_lwc_df,
            test_mvd_df    = test_mvd_df,
            train_metrics  = train_metrics,
            test_metrics   = test_metrics,
            lwc_estimator  = self._lwc_est,
            mvd_estimator  = self._mvd_est,
        )
        result.print_summary()
        return result

    @property
    def lwc_estimator(self) -> LWCEstimator | None:
        return self._lwc_est

    @property
    def mvd_estimator(self) -> MVDEstimator | None:
        return self._mvd_est


# ── CombinedResult ─────────────────────────────────────────────────────────────

@dataclass
class CombinedResult:
    """Holds train/test DataFrames and metrics from a CombinedSession run."""
    train_lwc_df:  pd.DataFrame
    train_mvd_df:  pd.DataFrame
    test_lwc_df:   pd.DataFrame
    test_mvd_df:   pd.DataFrame
    train_metrics: dict
    test_metrics:  dict
    lwc_estimator: LWCEstimator
    mvd_estimator: MVDEstimator

    def print_summary(self):
        print(f"\n{'═' * 55}")
        print("  COMBINED SPLIT — RESULTS SUMMARY")
        print(f"{'═' * 55}")
        print(f"  {'Metric':<18} {'Train':>10} {'Test':>10}")
        print(f"  {'─' * 40}")
        all_keys = sorted(set(self.train_metrics) | set(self.test_metrics))
        for k in all_keys:
            tv = self.train_metrics.get(k, float("nan"))
            ev = self.test_metrics.get(k,  float("nan"))
            fmt = ".4f" if "r2" in k else ".5f" if "lwc" in k else ".3f"
            print(f"  {k:<18} {tv:>10{fmt}} {ev:>10{fmt}}")
        print(f"{'═' * 55}")


# ── Metric helper ──────────────────────────────────────────────────────────────

def _split_metrics(mvd_df: pd.DataFrame, label: str = "") -> dict:
    from sklearn.metrics import r2_score, mean_absolute_error
    from .config import LWC_CALIB_MIN_REF, MVD_CALIB_MIN_REF

    metrics = {}

    if "REF_LWC" in mvd_df.columns and "LWC_MEAN" in mvd_df.columns:
        mask = mvd_df["REF_LWC"] >= LWC_CALIB_MIN_REF
        if mask.sum() > 1:
            yt = mvd_df.loc[mask, "REF_LWC"].values
            yp = mvd_df.loc[mask, "LWC_MEAN"].values
            v  = ~(np.isnan(yt) | np.isnan(yp))
            if v.sum() > 1:
                metrics["lwc_r2"]  = round(r2_score(yt[v], yp[v]), 4)
                metrics["lwc_mae"] = round(mean_absolute_error(yt[v], yp[v]), 5)

    if "REF_MVD" in mvd_df.columns and "MVD_EST" in mvd_df.columns:
        mask = mvd_df["REF_MVD"] >= MVD_CALIB_MIN_REF
        if mask.sum() > 1:
            yt = mvd_df.loc[mask, "REF_MVD"].values
            yp = mvd_df.loc[mask, "MVD_EST"].values
            v  = ~(np.isnan(yt) | np.isnan(yp))
            if v.sum() > 1:
                metrics["mvd_r2"]  = round(r2_score(yt[v], yp[v]), 4)
                metrics["mvd_mae"] = round(mean_absolute_error(yt[v], yp[v]), 5)

    return metrics
