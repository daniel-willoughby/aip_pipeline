"""
train.py
--------
Train the LWC and MVD models on all known flights and save to disk.

Run this locally whenever you add new flight data:
    python train.py

The trained model is saved to models/aip_model.pkl and should be
committed to your GitHub repo so Streamlit Cloud can load it.
"""

import matplotlib
matplotlib.use("Agg")

import pickle
from pathlib import Path
from src.pipeline import run_pipeline
from src.pipeline.session import CombinedSession
from src.pipeline.visualiser import CombinedVisualiser

# ── Configure your flights here ───────────────────────────────────────────────
FLIGHTS = [
    {
        "name":        "Flight 1 (1475-1)",
        "flight_file": "flight1475-AEROTEX-AIP-1_WOW0.xlsx",
        "log_files":   ["AIP_log3_WOW_0.xlsx"],
    },
    {
        "name":        "Flight 2 (1475-2)",
        "flight_file": "flight1475-AEROTEX-AIP-2_WOW0.xlsx",
        "log_files":   ["AIP_Log6_WOW_0.xlsx", "AIP_Log7_WOW_0.xlsx"],
    },
    {
        "name":        "Flight 3 (1476-1)",
        "flight_file": "Flight1476-AEROTEX-AIP-1_WOW_0.xlsx",
        "log_files":   ["AIP_log_4_WOW_0.xlsx"],
    },
    {
        "name":        "Flight 4 (1477-1)",
        "flight_file": "flight1477-AEROTEX-AIP-1_WOW_0.xlsx",
        "log_files":   ["AIP_log_7_WOW_0.xlsx"],
    },
    {
        "name":        "Flight 5 (1477-2)",
        "flight_file": "flight1477-AEROTEX-AIP-2_WOW_0.xlsx",
        "log_files":   ["AIP_log_8_WOW_0.xlsx"],
    },
]

TEST_FRACTION = 0.2
RANDOM_SEED   = 42
MODEL_PATH    = Path("models/aip_model.pkl")
PLOTS_DIR     = Path("outputs/training_plots")

# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  AIP PIPELINE — TRAINING RUN")
    print("=" * 60)

    # Ingest all flights
    results = []
    for cfg in FLIGHTS:
        print(f"\n▶ Loading {cfg['name']}…")
        r = run_pipeline(cfg["flight_file"], cfg["log_files"])
        results.append((cfg["name"], r))

    # Train combined session
    session = CombinedSession(
        test_fraction=TEST_FRACTION,
        random_seed=RANDOM_SEED,
    )
    for name, r in results:
        session.add_flight(name, r)
    combined = session.run()

    # Save plots
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    vis = CombinedVisualiser(combined, output_dir=str(PLOTS_DIR))
    vis.plot_all()
    print(f"\n  Plots saved to {PLOTS_DIR}/")

    # Save model
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model_bundle = {
        "lwc_estimator": combined.lwc_estimator,
        "mvd_estimator": combined.mvd_estimator,
        "train_metrics": combined.train_metrics,
        "test_metrics":  combined.test_metrics,
        "flights":       [name for name, _ in results],
        "test_fraction": TEST_FRACTION,
        "random_seed":   RANDOM_SEED,
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model_bundle, f)

    print(f"\n  Model saved to {MODEL_PATH}")
    print(f"\n  Test metrics:")
    for k, v in combined.test_metrics.items():
        print(f"    {k}: {v:.4f}")

    print("\n✓ Training complete. Commit models/aip_model.pkl to GitHub.")
