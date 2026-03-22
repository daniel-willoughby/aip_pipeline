"""
visualiser.py
-------------
Plotting functions for AIP pipeline results.

Four figure types:
  1. plot_flight_overview()   — KTAS, OAT, altitude over time
  2. plot_lwc()               — LWC time series (per sensor + mean vs reference)
                                + predicted vs actual scatter per sensor
  3. plot_mvd()               — MVD time series + predicted vs actual scatter
  4. plot_summary()           — side-by-side metrics across all flights in a session

Usage
-----
    from src.pipeline.visualiser import FlightVisualiser

    vis = FlightVisualiser(results, output_dir="outputs/plots")
    vis.plot_flight_overview("Flight 1")
    vis.plot_lwc("Flight 1")
    vis.plot_mvd("Flight 1")
    vis.plot_summary()
    vis.plot_all()               # generates everything for all flights
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe for PyCharm scripts
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import AutoMinorLocator

from .session import FlightResult
from .config import SENSOR_COUNT, LWC_CALIB_MIN_REF, MVD_CALIB_MIN_REF

# ── Style constants ───────────────────────────────────────────────────────────

SENSOR_COLOURS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]
REF_COLOUR     = "#2d2d2d"
MEAN_COLOUR    = "#E84393"

plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "#F8F8F8",
    "axes.edgecolor":    "#CCCCCC",
    "axes.grid":         True,
    "grid.color":        "white",
    "grid.linewidth":    0.8,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.family":       "sans-serif",
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.titleweight":  "bold",
    "axes.labelsize":    9,
    "legend.fontsize":   8,
    "legend.framealpha": 0.9,
    "figure.dpi":        130,
})

_MAX_TIMESERIES_POINTS = 8_000   # downsample threshold for time series plots


# ── Helper utilities ──────────────────────────────────────────────────────────

def _downsample(df: pd.DataFrame, max_pts: int = _MAX_TIMESERIES_POINTS) -> pd.DataFrame:
    """Evenly subsample rows if the DataFrame is larger than max_pts."""
    if len(df) <= max_pts:
        return df
    step = max(1, len(df) // max_pts)
    return df.iloc[::step].copy()


def _col(df: pd.DataFrame, primary: str, fallback: str) -> str:
    return primary if primary in df.columns else fallback


def _save(fig: plt.Figure, path: Path, tight: bool = True):
    path.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=130)
    plt.close(fig)
    print(f"  Saved  →  {path}")


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() < 2:
        return float("nan")
    yt, yp = y_true[mask], y_pred[mask]
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - np.mean(yt)) ** 2)
    return 1.0 - ss_res / (ss_tot + 1e-12)


# ── Main class ────────────────────────────────────────────────────────────────

class FlightVisualiser:
    """
    Generates and saves diagnostic plots for all flights in a session.

    Parameters
    ----------
    results    : dict returned by EstimationSession.run()
    output_dir : folder where figures are saved  (default: 'outputs/plots')
    """

    def __init__(
        self,
        results: Dict[str, FlightResult],
        output_dir: str | Path = "outputs/plots",
    ):
        self.results    = results
        self.output_dir = Path(output_dir)

    # ── public API ────────────────────────────────────────────────────────────

    def plot_all(self):
        """Generate every plot for every flight."""
        print(f"\n{'═' * 55}")
        print("  GENERATING PLOTS")
        print(f"{'═' * 55}")
        for name in self.results:
            print(f"\n  [{name}]")
            self.plot_flight_overview(name)
            self.plot_lwc(name)
            self.plot_mvd(name)
        self.plot_summary()
        print(f"\n  All plots saved to: {self.output_dir.resolve()}")

    def plot_flight_overview(self, flight_name: str):
        """
        Three-panel time series: KTAS, OAT, pressure altitude.
        Gives context for where icing conditions occur in the flight.
        """
        fr  = self.results[flight_name]
        df  = _downsample(fr.mvd_df)
        t   = df["TIME"].values / 60.0   # convert to minutes

        ktas_col = _col(df, "KTAS", "FLT_KTAS")
        oat_col  = _col(df, "OAT",  "FLT_OAT")
        alt_col  = "FLT_ALT"

        fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
        fig.suptitle(f"{flight_name}  —  Flight Conditions Overview",
                     fontsize=13, fontweight="bold", y=1.01)

        # KTAS
        axes[0].plot(t, df[ktas_col].values, color="#4C72B0", lw=0.8)
        axes[0].set_ylabel("KTAS (kt)")
        axes[0].set_title("True Airspeed")

        # OAT
        axes[1].plot(t, df[oat_col].values, color="#DD8452", lw=0.8)
        axes[1].axhline(0, color="#888", lw=0.8, ls="--", label="0 °C")
        axes[1].set_ylabel("OAT (°C)")
        axes[1].set_title("Outside Air Temperature")
        axes[1].legend(loc="upper right")

        # Altitude
        if alt_col in df.columns:
            axes[2].plot(t, df[alt_col].values, color="#55A868", lw=0.8)
            axes[2].set_ylabel("Altitude (ft)")
            axes[2].set_title("Pressure Altitude")
        else:
            axes[2].set_visible(False)

        axes[-1].set_xlabel("Time (min)")
        for ax in axes:
            ax.xaxis.set_minor_locator(AutoMinorLocator())

        _save(fig, self.output_dir / f"{self._slug(flight_name)}_overview.png")

    def plot_lwc(self, flight_name: str):
        """
        Top row  : LWC time series — all 5 sensors + mean vs REF_LWC
        Bottom row: predicted vs actual scatter, one panel per sensor
        """
        fr  = self.results[flight_name]
        df  = fr.mvd_df
        ds  = _downsample(df)
        t   = ds["TIME"].values / 60.0

        lwc_cols = [f"LWC_{n}" for n in range(1, SENSOR_COUNT + 1)]

        fig = plt.figure(figsize=(16, 9))
        gs  = gridspec.GridSpec(
            2, SENSOR_COUNT,
            height_ratios=[2, 1.2],
            hspace=0.45, wspace=0.35,
        )
        fig.suptitle(f"{flight_name}  —  LWC Estimation",
                     fontsize=13, fontweight="bold")

        # ── Time series (spans full width) ───────────────────────────────
        ax_ts = fig.add_subplot(gs[0, :])

        ax_ts.plot(t, ds["REF_LWC"].values, color=REF_COLOUR,
                   lw=1.2, label="REF_LWC (M300)", zorder=5)
        ax_ts.plot(t, ds["LWC_MEAN"].values, color=MEAN_COLOUR,
                   lw=1.0, ls="--", label="LWC mean (AIP)", zorder=4)
        for n, col in enumerate(lwc_cols, 1):
            ax_ts.plot(t, ds[col].values, color=SENSOR_COLOURS[n - 1],
                       lw=0.5, alpha=0.55, label=f"S{n}")

        ax_ts.set_xlabel("Time (min)")
        ax_ts.set_ylabel("LWC (g/m³)")
        ax_ts.set_title("LWC Time Series")
        ax_ts.legend(ncol=7, loc="upper right", fontsize=7.5)
        ax_ts.xaxis.set_minor_locator(AutoMinorLocator())

        # ── Scatter: predicted vs actual, one per sensor ──────────────────
        icing_mask = df["REF_LWC"] >= LWC_CALIB_MIN_REF
        icing_df   = df[icing_mask]

        for n in range(1, SENSOR_COUNT + 1):
            ax = fig.add_subplot(gs[1, n - 1])
            col  = f"LWC_{n}"
            yt   = icing_df["REF_LWC"].values
            yp   = icing_df[col].values
            r2   = _r2(yt, yp)

            # Downsample scatter to keep it readable
            idx = np.random.choice(len(yt), min(3000, len(yt)), replace=False)

            ax.scatter(yt[idx], yp[idx],
                       color=SENSOR_COLOURS[n - 1],
                       s=2, alpha=0.35, rasterized=True)

            lim = max(yt.max(), yp.max()) * 1.05
            ax.plot([0, lim], [0, lim], "k--", lw=0.8, alpha=0.6)
            ax.set_xlim(0, lim)
            ax.set_ylim(0, lim)
            ax.set_xlabel("REF_LWC (g/m³)", fontsize=8)
            ax.set_ylabel("LWC Est. (g/m³)", fontsize=8)
            ax.set_title(f"Sensor {n}  (R²={r2:.3f})", fontsize=9)
            ax.tick_params(labelsize=7)

        _save(fig, self.output_dir / f"{self._slug(flight_name)}_lwc.png",
              tight=False)

    def plot_mvd(self, flight_name: str):
        """
        Left : MVD time series — MVD_EST vs REF_MVD
        Right: predicted vs actual scatter (icing rows only)
        """
        fr  = self.results[flight_name]
        df  = fr.mvd_df
        ds  = _downsample(df)
        t   = ds["TIME"].values / 60.0

        fig, (ax_ts, ax_sc) = plt.subplots(
            1, 2, figsize=(14, 5),
            gridspec_kw={"width_ratios": [2.5, 1]},
        )
        fig.suptitle(f"{flight_name}  —  MVD Estimation",
                     fontsize=13, fontweight="bold")

        # ── Time series ───────────────────────────────────────────────────
        ax_ts.plot(t, ds["REF_MVD"].values, color=REF_COLOUR,
                   lw=1.2, label="REF_MVD (M300)", zorder=5)
        ax_ts.plot(t, ds["MVD_EST"].values, color=MEAN_COLOUR,
                   lw=0.9, ls="--", label="MVD_EST (AIP)", zorder=4)
        ax_ts.set_xlabel("Time (min)")
        ax_ts.set_ylabel("MVD (µm)")
        ax_ts.set_title("MVD Time Series")
        ax_ts.legend(loc="upper right")
        ax_ts.xaxis.set_minor_locator(AutoMinorLocator())

        # ── Scatter ───────────────────────────────────────────────────────
        icing_mask = df["REF_MVD"] >= MVD_CALIB_MIN_REF
        icing_df   = df[icing_mask]

        if len(icing_df) > 0:
            yt  = icing_df["REF_MVD"].values
            yp  = icing_df["MVD_EST"].values
            r2  = _r2(yt, yp)
            idx = np.random.choice(len(yt), min(4000, len(yt)), replace=False)

            ax_sc.scatter(yt[idx], yp[idx],
                          color=MEAN_COLOUR, s=3, alpha=0.4, rasterized=True)
            lim = max(yt.max(), yp.max()) * 1.05
            ax_sc.plot([0, lim], [0, lim], "k--", lw=0.8, alpha=0.6)
            ax_sc.set_xlim(0, lim)
            ax_sc.set_ylim(0, lim)
            ax_sc.set_xlabel("REF_MVD (µm)")
            ax_sc.set_ylabel("MVD_EST (µm)")
            ax_sc.set_title(f"Predicted vs Actual\n(R²={r2:.3f},  n={len(yt):,})")
        else:
            ax_sc.text(0.5, 0.5, "No icing rows\nfor scatter",
                       ha="center", va="center", transform=ax_sc.transAxes)
            ax_sc.set_title("Predicted vs Actual")

        _save(fig, self.output_dir / f"{self._slug(flight_name)}_mvd.png")

    def plot_summary(self):
        """
        Multi-flight summary: grouped bar chart of R² and MAE for
        LWC (mean) and MVD across all flights, coloured by split.
        """
        names  = list(self.results.keys())
        splits = [self.results[n].split for n in names]
        split_colours = {"train": "#4C72B0", "test": "#DD8452", "both": "#55A868"}

        lwc_r2  = [self.results[n].metrics.get("lwc_r2",  float("nan")) for n in names]
        lwc_mae = [self.results[n].metrics.get("lwc_mae", float("nan")) for n in names]
        mvd_r2  = [self.results[n].metrics.get("mvd_r2",  float("nan")) for n in names]
        mvd_mae = [self.results[n].metrics.get("mvd_mae", float("nan")) for n in names]

        fig, axes = plt.subplots(2, 2, figsize=(12, 7))
        fig.suptitle("Multi-Flight Performance Summary",
                     fontsize=13, fontweight="bold")

        datasets = [
            (axes[0, 0], lwc_r2,  "LWC — R²",        "R²",     True),
            (axes[0, 1], lwc_mae, "LWC — MAE (g/m³)", "g/m³",   False),
            (axes[1, 0], mvd_r2,  "MVD — R²",        "R²",     True),
            (axes[1, 1], mvd_mae, "MVD — MAE (µm)",   "µm",     False),
        ]

        x = np.arange(len(names))
        for ax, values, title, ylabel, is_r2 in datasets:
            colours = [split_colours.get(s, "#AAAAAA") for s in splits]
            bars = ax.bar(x, values, color=colours, edgecolor="white",
                          linewidth=0.5, width=0.6)

            if is_r2:
                ax.axhline(0, color="#888", lw=0.8, ls="--")
                ax.axhline(1, color="#55A868", lw=0.8, ls=":", alpha=0.6,
                           label="Perfect fit")
                ax.set_ylim(
                    min(-0.1, min(v for v in values if not np.isnan(v)) - 0.05)
                    if any(not np.isnan(v) for v in values) else -0.1,
                    1.05,
                )

            # Value labels on bars
            for bar, val in zip(bars, values):
                if not np.isnan(val):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + (0.01 if val >= 0 else -0.06),
                        f"{val:.3f}",
                        ha="center", va="bottom", fontsize=7.5,
                    )

            ax.set_xticks(x)
            ax.set_xticklabels(names, rotation=15, ha="right", fontsize=8)
            ax.set_ylabel(ylabel)
            ax.set_title(title)

        # Legend for split colours
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=c, label=s.capitalize())
            for s, c in split_colours.items()
        ]
        fig.legend(handles=legend_elements, loc="lower center",
                   ncol=3, fontsize=9, framealpha=0.9,
                   bbox_to_anchor=(0.5, -0.02))

        _save(fig, self.output_dir / "summary.png")

    # ── private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _slug(name: str) -> str:
        """Convert flight name to a safe filename prefix."""
        return name.lower().replace(" ", "_").replace("/", "_")


# ── Combined split plots ──────────────────────────────────────────────────────

class CombinedVisualiser:
    """
    Generates train vs test diagnostic plots for a CombinedResult.

    Usage
    -----
        from src.pipeline.visualiser import CombinedVisualiser
        vis = CombinedVisualiser(combined_result, output_dir="outputs/plots")
        vis.plot_all()
    """

    def __init__(self, result, output_dir: str | Path = "outputs/plots"):
        self.result     = result
        self.output_dir = Path(output_dir)

    def plot_all(self):
        print(f"\n{'═' * 55}")
        print("  GENERATING COMBINED SPLIT PLOTS")
        print(f"{'═' * 55}")
        self.plot_lwc_combined()
        self.plot_mvd_combined()
        self.plot_mvd_timeseries()
        self.plot_combined_summary()
        print(f"\n  All plots saved to: {self.output_dir.resolve()}")

    def plot_mvd_timeseries(self):
        """
        MVD_EST vs REF_MVD over the full flight duration, shown separately
        for train and test rows so it is clear where the model was fitted
        and how it performs on held-out data.
        """
        # Reconstruct full chronological dataset by concatenating train + test
        # and sorting by TIME so the x-axis runs continuously
        train = self.result.train_mvd_df.copy()
        test  = self.result.test_mvd_df.copy()
        train["_split"] = "train"
        test["_split"]  = "test"
        full = pd.concat([train, test]).sort_values("TIME").reset_index(drop=True)

        t_min = full["TIME"].min() / 60.0
        t_max = full["TIME"].max() / 60.0

        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                 gridspec_kw={"hspace": 0.35})
        fig.suptitle("MVD — Predicted vs Actual  (full flight duration)",
                     fontsize=13, fontweight="bold")

        for ax, (split_label, split_df, colour) in zip(
            axes,
            [("Train", _downsample(train), "#4C72B0"),
             ("Test",  _downsample(test),  "#DD8452")]
        ):
            t = split_df["TIME"].values / 60.0

            ax.plot(t, split_df["REF_MVD"].values,
                    color=REF_COLOUR, lw=1.2, label="REF_MVD (M300)", zorder=5)
            ax.plot(t, split_df["MVD_EST"].values,
                    color=colour, lw=0.9, ls="--",
                    label=f"MVD_EST ({split_label})", zorder=4, alpha=0.85)

            # Shade the region between estimate and reference to highlight error
            ax.fill_between(t,
                            split_df["REF_MVD"].values,
                            split_df["MVD_EST"].values,
                            alpha=0.12, color=colour)

            r2  = _r2(split_df["REF_MVD"].values, split_df["MVD_EST"].values)
            ax.set_ylabel("MVD (µm)")
            ax.set_title(f"{split_label} set  (R²={r2:.3f})", fontsize=10)
            ax.legend(loc="upper right", fontsize=8)
            ax.xaxis.set_minor_locator(AutoMinorLocator())

            # Shade total x range to show where this split's points sit
            # relative to the full flight
            ax.set_xlim(t_min, t_max)

        axes[-1].set_xlabel("Time (min)")
        _save(fig, self.output_dir / "combined_mvd_timeseries.png")

    def plot_lwc_combined(self):
        """
        Two rows of scatter plots — top row train, bottom row test —
        one column per sensor.  Lets you see whether the fit generalises.
        """
        fig, axes = plt.subplots(
            2, SENSOR_COUNT,
            figsize=(18, 9),
            sharey="row",
            gridspec_kw={"hspace": 0.55, "wspace": 0.3},
        )
        fig.suptitle("LWC — Train vs Test  (per sensor)",
                     fontsize=13, fontweight="bold", y=1.01)

        for row_idx, (label, df) in enumerate(
            [("Train", self.result.train_mvd_df),
             ("Test",  self.result.test_mvd_df)]
        ):
            icing = df[df["REF_LWC"] >= LWC_CALIB_MIN_REF]
            for n in range(1, SENSOR_COUNT + 1):
                ax  = axes[row_idx, n - 1]
                col = f"LWC_{n}"
                yt  = icing["REF_LWC"].values
                yp  = icing[col].values
                r2  = _r2(yt, yp)

                idx = np.random.choice(len(yt), min(3000, len(yt)), replace=False)
                ax.scatter(yt[idx], yp[idx],
                           color=SENSOR_COLOURS[n - 1],
                           s=2, alpha=0.35, rasterized=True)

                lim = max(yt.max(), yp.max()) * 1.05
                ax.plot([0, lim], [0, lim], "k--", lw=0.8, alpha=0.6)
                ax.set_xlim(0, lim)
                ax.set_ylim(0, lim)
                ax.set_xlabel("REF_LWC (g/m³)", fontsize=8)
                if n == 1:
                    ax.set_ylabel(f"{label}\nLWC Est. (g/m³)", fontsize=8)
                ax.set_title(f"Sensor {n}  R²={r2:.3f}", fontsize=9)

                # 4 ticks max and rotated x labels to prevent overlap
                ax.xaxis.set_major_locator(plt.MaxNLocator(4, prune="both"))
                ax.yaxis.set_major_locator(plt.MaxNLocator(4, prune="both"))
                ax.tick_params(axis="x", labelsize=7, rotation=30)
                ax.tick_params(axis="y", labelsize=7)

        _save(fig, self.output_dir / "combined_lwc_scatter.png", tight=False)

    def plot_mvd_combined(self):
        """
        Side-by-side MVD scatter: train (left) vs test (right).
        """
        fig, (ax_tr, ax_te) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("MVD — Train vs Test", fontsize=13, fontweight="bold")

        for ax, label, df in [
            (ax_tr, "Train", self.result.train_mvd_df),
            (ax_te, "Test",  self.result.test_mvd_df),
        ]:
            icing = df[df["REF_MVD"] >= MVD_CALIB_MIN_REF]
            if len(icing) == 0:
                ax.text(0.5, 0.5, "No icing rows",
                        ha="center", va="center", transform=ax.transAxes)
                ax.set_title(label)
                continue

            yt  = icing["REF_MVD"].values
            yp  = icing["MVD_EST"].values
            r2  = _r2(yt, yp)
            idx = np.random.choice(len(yt), min(4000, len(yt)), replace=False)

            colour = "#4C72B0" if label == "Train" else "#DD8452"
            ax.scatter(yt[idx], yp[idx],
                       color=colour, s=3, alpha=0.4, rasterized=True)

            lim = max(yt.max(), yp.max()) * 1.05
            ax.plot([0, lim], [0, lim], "k--", lw=0.8, alpha=0.6)
            ax.set_xlim(0, lim)
            ax.set_ylim(0, lim)
            ax.set_xlabel("REF_MVD (µm)")
            ax.set_ylabel("MVD_EST (µm)")
            ax.set_title(f"{label}  (R²={r2:.3f},  n={len(yt):,})")

        _save(fig, self.output_dir / "combined_mvd_scatter.png")

    def plot_combined_summary(self):
        """
        Bar chart comparing train vs test R² and MAE for LWC and MVD.
        """
        metrics_pairs = [
            ("lwc_r2",  "LWC R²",        "R²"),
            ("lwc_mae", "LWC MAE (g/m³)", "g/m³"),
            ("mvd_r2",  "MVD R²",        "R²"),
            ("mvd_mae", "MVD MAE (µm)",   "µm"),
        ]

        fig, axes = plt.subplots(1, 4, figsize=(14, 4))
        fig.suptitle("Combined Split — Performance Summary",
                     fontsize=13, fontweight="bold")

        tr_c = "#4C72B0"
        te_c = "#DD8452"

        for ax, (key, title, ylabel) in zip(axes, metrics_pairs):
            tv = self.result.train_metrics.get(key, float("nan"))
            ev = self.result.test_metrics.get(key,  float("nan"))

            bars = ax.bar(["Train", "Test"], [tv, ev],
                          color=[tr_c, te_c], edgecolor="white",
                          linewidth=0.5, width=0.5)

            if "r2" in key:
                ax.axhline(0, color="#888", lw=0.8, ls="--")
                ax.axhline(1, color="#55A868", lw=0.8, ls=":", alpha=0.5)
                ymin = min(-0.1, min(v for v in [tv, ev] if not np.isnan(v)) - 0.05) \
                       if any(not np.isnan(v) for v in [tv, ev]) else -0.1
                ax.set_ylim(ymin, 1.05)

            for bar, val in zip(bars, [tv, ev]):
                if not np.isnan(val):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + (0.01 if val >= 0 else -0.08),
                        f"{val:.3f}",
                        ha="center", va="bottom", fontsize=9,
                    )

            ax.set_ylabel(ylabel)
            ax.set_title(title)

        _save(fig, self.output_dir / "combined_summary.png")
