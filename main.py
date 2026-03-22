import matplotlib
matplotlib.use("Agg")

from src.pipeline import run_pipeline
from src.pipeline.session import CombinedSession
from src.pipeline.visualiser import CombinedVisualiser

r1 = run_pipeline(
    "flight1475-AEROTEX-AIP-1_WOW0.xlsx",
    ["AIP_log3_WOW_0.xlsx"]
)
r2 = run_pipeline(
    "flight1475-AEROTEX-AIP-2_WOW0.xlsx",
    ["AIP_Log6_WOW_0.xlsx", "AIP_Log7_WOW_0.xlsx"]
)
r3 = run_pipeline(
    "Flight1476-AEROTEX-AIP-1_WOW_0.xlsx",
    ["AIP_log_4_WOW_0.xlsx"]
)

session = CombinedSession(test_fraction=0.2, random_seed=42)
session.add_flight("Flight 1", r1)
session.add_flight("Flight 2", r2)
session.add_flight("Flight 3", r3)
combined = session.run()

vis = CombinedVisualiser(combined, output_dir="outputs/plots")
vis.plot_all()

print(combined.test_metrics)