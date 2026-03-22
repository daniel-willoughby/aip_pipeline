"""
streamlit_app.py
----------------
Streamlit web interface for the AIP data pipeline.
Run with: streamlit run streamlit_app.py

Drop all flight and log files in — the app automatically matches them.

Password protection via .streamlit/secrets.toml:
    password = "your_chosen_password"
"""

from __future__ import annotations

import hmac
import io
import sys
import tempfile
import traceback
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AIP Pipeline",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Password protection ───────────────────────────────────────────────────────

def check_password() -> bool:
    def password_entered():
        if hmac.compare_digest(
            st.session_state["password"],
            st.secrets.get("password", "")
        ):
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if st.session_state.get("password_correct", False):
        return True

    st.markdown("""
    <div style="max-width:380px; margin:80px auto 0 auto; text-align:center;">
        <div style="font-size:48px; margin-bottom:12px;">✈️</div>
        <div style="font-size:24px; font-weight:700; color:#ffffff; margin-bottom:4px;">
            AIP Pipeline</div>
        <div style="font-size:13px; color:#6b7280; margin-bottom:32px;">
            LWC & MVD Estimation</div>
    </div>
    """, unsafe_allow_html=True)

    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        st.text_input("Password", type="password",
                      on_change=password_entered, key="password",
                      placeholder="Enter password…")
        if "password_correct" in st.session_state:
            st.error("❌ Incorrect password — please try again.")
    return False

if not check_password():
    st.stop()

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .stApp { background-color: #0f1117; }
    [data-testid="stSidebar"] {
        background-color: #1a1d27;
        border-right: 1px solid #2a2d3e;
    }
    .accent-rule {
        height: 3px;
        background: linear-gradient(90deg, #4f8ef7, #7c5cbf);
        border-radius: 2px;
        margin-bottom: 24px;
    }
    .match-card {
        background: #1a1d27;
        border: 1px solid #2a2d3e;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 10px;
    }
    .match-title {
        color: #ffffff;
        font-weight: 600;
        font-size: 14px;
        margin-bottom: 6px;
    }
    .match-log {
        color: #a8b4c8;
        font-size: 12px;
        font-family: monospace;
        padding: 2px 0;
    }
    .metric-card {
        background: #1a1d27;
        border: 1px solid #2a2d3e;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .metric-value { font-size: 28px; font-weight: 700; margin: 4px 0; }
    .metric-label {
        color: #6b7280; font-size: 12px;
        text-transform: uppercase; letter-spacing: 0.05em;
    }
    .good  { color: #3ecf8e; }
    .ok    { color: #f0a500; }
    .poor  { color: #e05252; }
    #MainMenu { visibility: hidden; }
    footer    { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── File classification and time-range helpers ────────────────────────────────

# ── Robust classification constants ──────────────────────────────────────────

# Substrings found in flight file headers (any row of first 2)
_FLIGHT_SIGNALS = [
    "iasp",                 # IASP 1 - True Airspeed...
    "true airspeed",
    "pressure altitude",
    "static pressure true",
    "icd lwc",              # M300 reference probe
    "ccp mvd",
]

# Log files always have TFS col0 + TIM col1
_LOG_TIME_CANDIDATES = [
    "time s", "time_s", "time (s)", "elapsed_s", "time inc",
]


def classify_and_get_range(uploaded_file) -> tuple[str, float, float]:
    """
    Robustly classify an uploaded file as 'flight' or 'log' and return
    its elapsed time range (t_min, t_max) in seconds.

    Classification strategy (in order):
      1. Any FLIGHT_SIGNALS substring found in the first two header rows → flight
      2. TFS + TIM columns present → log
      3. KTAS without IASP → log
      4. Unknown — returns ('unknown', 0, 0)

    Time extraction strategy:
      Flight: col index 1 (skip units row at row 0)
              Falls back to 'Time inc' or any named time column
      Log:    Named time column ('Time S', 'Time s', etc.)
              Falls back to computing (TIM - TIM[0]) / 1e6
              Falls back to parsing TFS datetime strings
    """
    import openpyxl
    import pandas as pd

    data = uploaded_file.getbuffer()

    # ── Step 1: classify ───────────────────────────────────────────────────
    wb = openpyxl.load_workbook(io.BytesIO(data), read_only=True)
    ws = wb.active
    first_two_rows = list(ws.iter_rows(max_row=2, values_only=True))
    wb.close()

    all_header_text = " ".join(
        str(c).strip().lower()
        for row in first_two_rows for c in row if c is not None
    )
    row0_cells = [str(c).strip().lower() if c else ""
                  for c in first_two_rows[0]]

    has_flight_signal = any(sig in all_header_text for sig in _FLIGHT_SIGNALS)
    has_tfs = any("tfs" in c for c in row0_cells[:4])
    has_tim = any(c == "tim" for c in row0_cells[:4])
    has_ktas = "ktas" in all_header_text
    has_iasp = "iasp" in all_header_text

    if has_flight_signal:
        kind = "flight"
    elif has_tfs and has_tim:
        kind = "log"
    elif has_ktas and not has_iasp:
        kind = "log"
    else:
        return "unknown", 0.0, 0.0

    # ── Step 2: extract time range ─────────────────────────────────────────
    df = pd.read_excel(io.BytesIO(data), engine="openpyxl", header=0)
    col_names_lower = [str(c).strip().lower() for c in df.columns]

    if kind == "flight":
        # Look for a named elapsed-time column first
        elapsed_col = next(
            (df.columns[i] for i, c in enumerate(col_names_lower)
             if c in _LOG_TIME_CANDIDATES), None
        )
        if elapsed_col is None:
            # Default: col index 1 (always elapsed seconds in flight files)
            elapsed_col = df.columns[1]

        # Skip the units row (first data row in flight files)
        t = pd.to_numeric(df.iloc[1:][elapsed_col], errors="coerce").dropna()
        t = t[t >= 0]

    else:  # log
        # Strategy 1: named time column
        elapsed_col = next(
            (df.columns[i] for i, c in enumerate(col_names_lower)
             if c in _LOG_TIME_CANDIDATES), None
        )
        if elapsed_col is not None:
            t = pd.to_numeric(df[elapsed_col], errors="coerce").dropna()
            t = t[t >= 0]
        else:
            # Strategy 2: compute from TIM microsecond counter
            tim_col = next(
                (df.columns[i] for i, c in enumerate(col_names_lower)
                 if c == "tim"), None
            )
            if tim_col is not None:
                tim = pd.to_numeric(df[tim_col], errors="coerce").dropna()
                t = (tim - tim.iloc[0]) / 1_000_000
            else:
                # Strategy 3: parse TFS datetime strings
                tfs_col = next(
                    (df.columns[i] for i, c in enumerate(col_names_lower)
                     if "tfs" in c), None
                )
                if tfs_col is not None:
                    ts = pd.to_datetime(df[tfs_col], errors="coerce").dropna()
                    t  = (ts - ts.iloc[0]).dt.total_seconds()
                else:
                    return kind, 0.0, 0.0

    if len(t) == 0:
        return kind, 0.0, 0.0

    return kind, float(t.min()), float(t.max())


def match_score(f_min, f_max, l_min, l_max) -> float:
    """
    Bidirectional overlap score — minimum of:
      - fraction of log's duration covered by flight
      - fraction of flight's duration covered by log
    Penalises both under-coverage and over-coverage.
    """
    overlap     = max(0.0, min(f_max, l_max) - max(f_min, l_min))
    log_frac    = overlap / (l_max - l_min) if (l_max - l_min) > 0 else 0
    flight_frac = overlap / (f_max - f_min) if (f_max - f_min) > 0 else 0
    return min(log_frac, flight_frac)


def auto_match(flight_info: dict, log_info: dict) -> dict[str, list[str]]:
    """
    For each log, find the best matching flight using bidirectional overlap.
    Returns: {flight_name: [log_name, ...]}
    """
    groups = defaultdict(list)
    for lname, (_, lt_min, lt_max) in log_info.items():
        scores = {
            fname: match_score(ft_min, ft_max, lt_min, lt_max)
            for fname, (_, ft_min, ft_max) in flight_info.items()
        }
        best_flight = max(scores, key=scores.get)
        groups[best_flight].append(lname)

    # Ensure every flight appears even if it got no logs
    for fname in flight_info:
        if fname not in groups:
            groups[fname] = []

    return dict(groups)


# ── Helpers ───────────────────────────────────────────────────────────────────

def save_uploaded(uploaded_file, tmp_dir: Path) -> Path:
    dest = tmp_dir / uploaded_file.name
    dest.write_bytes(uploaded_file.getbuffer())
    return dest


def metric_colour(key: str, value: float) -> str:
    if "r2" in key:
        return "good" if value > 0.5 else "ok" if value > 0.2 else "poor"
    return ""


def metric_html(label: str, value, key: str = "", unit: str = "") -> str:
    if value is None:
        display, cls = "—", ""
    elif isinstance(value, float):
        cls     = metric_colour(key, value)
        display = f"{value:+.4f}" if "r2" in key else f"{value:.4f}{unit}"
    else:
        display, cls = str(value), ""
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value {cls}">{display}</div>
    </div>"""


# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.markdown("""
<div style="padding: 8px 0 16px 0;">
    <div style="font-size:20px; font-weight:700; color:#ffffff;">✈️ AIP Pipeline</div>
    <div style="font-size:12px; color:#6b7280; margin-top:2px;">LWC & MVD Estimation</div>
</div>
<div class="accent-rule"></div>
""", unsafe_allow_html=True)

st.sidebar.markdown("### Upload Files")
st.sidebar.caption(
    "Drop all flight and log `.xlsx` files here — "
    "the pipeline will automatically match them."
)

uploaded_files = st.sidebar.file_uploader(
    "Flight & log files",
    type=["xlsx"],
    accept_multiple_files=True,
    label_visibility="collapsed",
)

st.sidebar.divider()
st.sidebar.markdown("### Settings")
test_frac = st.sidebar.slider("Test split", 0.1, 0.4, 0.2, 0.05)
seed      = st.sidebar.number_input("Random seed", value=42, step=1)

st.sidebar.divider()
if st.sidebar.button("🔒 Log out", use_container_width=True):
    st.session_state["password_correct"] = False
    st.rerun()

# ── Main area ─────────────────────────────────────────────────────────────────

st.markdown("""
<h1 style="color:#ffffff; font-size:28px; font-weight:700; margin-bottom:4px;">
    AIP Data Pipeline
</h1>
<div class="accent-rule"></div>
""", unsafe_allow_html=True)

# ── File classification & matching preview ────────────────────────────────────

if uploaded_files:
    with st.spinner("Classifying files…"):
        flight_info, log_info = {}, {}
        errors = []
        for f in uploaded_files:
            try:
                kind, t_min, t_max = classify_and_get_range(f)
                if kind == "flight":
                    flight_info[f.name] = (f, t_min, t_max)
                else:
                    log_info[f.name] = (f, t_min, t_max)
            except Exception as e:
                errors.append(f"{f.name}: {e}")

    if errors:
        for e in errors:
            st.warning(f"⚠️ {e}")

    if not flight_info:
        st.error("No flight files detected. Upload at least one flight `.xlsx` file.")
        st.stop()

    if not log_info:
        st.error("No log files detected. Upload at least one log `.xlsx` file.")
        st.stop()

    # Auto-match
    groups = auto_match(flight_info, log_info)

    # Show matching preview
    st.markdown("### Detected File Matching")
    st.caption(
        "The pipeline automatically matched log files to flights by time overlap. "
        "Check this looks correct before running."
    )

    cols = st.columns(max(len(groups), 1))
    all_ok = True
    for col, (fname, lnames) in zip(cols, groups.items()):
        with col:
            f_obj, ft_min, ft_max = flight_info[fname]
            st.markdown(f"""
            <div class="match-card">
                <div class="match-title">✈ {fname}</div>
                <div class="match-log" style="color:#6b7280;">
                    {ft_min:.0f}s – {ft_max:.0f}s
                </div>
                <hr style="border-color:#2a2d3e; margin:8px 0;">
            """ + "".join(
                f'<div class="match-log">📋 {l}</div>'
                for l in lnames
            ) + ("" if lnames else
                 '<div class="match-log" style="color:#e05252;">⚠ No logs matched</div>'
            ) + "</div>", unsafe_allow_html=True)

            if not lnames:
                all_ok = False

    if not all_ok:
        st.warning(
            "One or more flights have no matching log files. "
            "Check your uploads and try again."
        )

    st.divider()

    run_clicked = st.button(
        "▶  Run Pipeline",
        type="primary",
        disabled=not all_ok,
        use_container_width=False,
    )

else:
    st.info(
        "Upload your flight and log `.xlsx` files using the sidebar. "
        "The pipeline will automatically detect which logs belong to which flight."
    )
    run_clicked = False

# ── Results tabs ──────────────────────────────────────────────────────────────

# Initialise session state
for key in ["metrics", "train_metrics", "log_text", "plots"]:
    if key not in st.session_state:
        st.session_state[key] = {} if key in ("metrics", "train_metrics", "plots") \
                                 else ""

if st.session_state.metrics or st.session_state.log_text:
    st.markdown("### Results")
    tab_results, tab_lwc, tab_mvd, tab_summary, tab_log = st.tabs([
        "📊 Metrics", "💧 LWC", "🔵 MVD", "📈 Summary", "📋 Log"
    ])

    plots = st.session_state.plots

    with tab_results:
        m  = st.session_state.metrics
        tm = st.session_state.train_metrics

        st.markdown("**Test Set**")
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.markdown(metric_html("LWC R²",  m.get("lwc_r2"),  "lwc_r2"),          unsafe_allow_html=True)
        with c2: st.markdown(metric_html("LWC MAE", m.get("lwc_mae"), "lwc_mae", " g/m³"), unsafe_allow_html=True)
        with c3: st.markdown(metric_html("MVD R²",  m.get("mvd_r2"),  "mvd_r2"),           unsafe_allow_html=True)
        with c4: st.markdown(metric_html("MVD MAE", m.get("mvd_mae"), "mvd_mae", " µm"),   unsafe_allow_html=True)

        st.divider()
        st.markdown("**Train Set**")
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.markdown(metric_html("LWC R²",  tm.get("lwc_r2"),  "lwc_r2"),          unsafe_allow_html=True)
        with c2: st.markdown(metric_html("LWC MAE", tm.get("lwc_mae"), "lwc_mae", " g/m³"), unsafe_allow_html=True)
        with c3: st.markdown(metric_html("MVD R²",  tm.get("mvd_r2"),  "mvd_r2"),           unsafe_allow_html=True)
        with c4: st.markdown(metric_html("MVD MAE", tm.get("mvd_mae"), "mvd_mae", " µm"),   unsafe_allow_html=True)

    with tab_lwc:
        if plots.get("lwc_scatter"):
            st.image(plots["lwc_scatter"], use_container_width=True)
            st.download_button("⬇ Download", plots["lwc_scatter"], "lwc_scatter.png", "image/png")
        else:
            st.info("No LWC plot available.")

    with tab_mvd:
        if plots.get("mvd_scatter"):
            st.image(plots["mvd_scatter"], use_container_width=True)
            st.download_button("⬇ Download", plots["mvd_scatter"], "mvd_scatter.png", "image/png")
        else:
            st.info("No MVD plot available.")

    with tab_summary:
        if plots.get("summary"):
            st.image(plots["summary"], use_container_width=True)
            st.download_button("⬇ Download", plots["summary"], "summary.png", "image/png")
        else:
            st.info("No summary plot available.")

    with tab_log:
        if st.session_state.log_text:
            st.code(st.session_state.log_text, language=None)
        else:
            st.info("No log available.")

# ── Pipeline execution ────────────────────────────────────────────────────────

if run_clicked:
    log_capture = io.StringIO()

    with st.spinner("Running pipeline — this may take a few minutes…"):
        try:
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                plot_dir = tmp_path / "plots"
                plot_dir.mkdir()

                old_stdout = sys.stdout
                sys.stdout = log_capture

                sys.path.insert(0, str(Path(__file__).parent))
                from src.pipeline import run_pipeline
                from src.pipeline.session import CombinedSession
                from src.pipeline.visualiser import CombinedVisualiser

                results = []
                for i, (fname, lnames) in enumerate(groups.items()):
                    f_obj = flight_info[fname][0]
                    l_objs = [log_info[ln][0] for ln in lnames]

                    flight_path = save_uploaded(f_obj, tmp_path)
                    log_paths   = [save_uploaded(l, tmp_path) for l in l_objs]

                    r = run_pipeline(
                        str(flight_path),
                        [str(p) for p in log_paths]
                    )
                    results.append((f"Flight {i + 1}", r))

                session = CombinedSession(
                    test_fraction=float(test_frac),
                    random_seed=int(seed),
                )
                for name, r in results:
                    session.add_flight(name, r)

                combined = session.run()

                vis = CombinedVisualiser(combined, output_dir=str(plot_dir))
                vis.plot_all()

                sys.stdout = old_stdout

                st.session_state.log_text     = log_capture.getvalue()
                st.session_state.metrics      = combined.test_metrics
                st.session_state.train_metrics = combined.train_metrics

                def read_plot(name: str) -> bytes | None:
                    p = plot_dir / name
                    return p.read_bytes() if p.exists() else None

                st.session_state.plots = {
                    "lwc_scatter": read_plot("combined_lwc_scatter.png"),
                    "mvd_scatter": read_plot("combined_mvd_scatter.png"),
                    "summary":     read_plot("combined_summary.png"),
                }

            st.success("✓ Pipeline complete!")
            st.rerun()

        except Exception:
            sys.stdout = old_stdout
            st.session_state.log_text = (
                log_capture.getvalue() + "\n" + traceback.format_exc()
            )
            st.error("Pipeline failed — see the **Log** tab for details.")
            st.rerun()
