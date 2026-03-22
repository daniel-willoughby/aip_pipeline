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

_FLIGHT_SIGNALS = [
    "iasp", "true airspeed", "pressure altitude",
    "static pressure true", "icd lwc", "ccp mvd",
]
_LOG_TIME_CANDIDATES = ["time s", "time_s", "time (s)", "elapsed_s", "time inc"]

# Minimum fraction of a log's duration that must fall within a flight window
# for the match to be accepted. 0.80 = 80% of the log must be inside the flight.
MATCH_THRESHOLD = 0.80


def classify_and_get_range(uploaded_file) -> tuple[str, float, float]:
    """
    Classify a file as 'flight' or 'log' and return its time range as
    seconds-past-midnight (absolute, for matching across files).

    Classification:
      1. Any flight signal in first two header rows  → flight
      2. TFS + TIM columns present                   → log
      3. KTAS without IASP                           → log
      4. Unknown                                     → ('unknown', 0, 0)

    Time extraction — all return seconds-past-midnight:
      Flight:  col 0 = absolute seconds past midnight
      Log TFS as numeric in 0–86400 range: seconds past midnight directly
      Log TFS as datetime string:          convert to seconds past midnight
      Log TFS as epoch / unreliable:       fall back to elapsed duration
                                           (matched by duration similarity)
    """
    import openpyxl
    import pandas as pd

    data = uploaded_file.getbuffer()

    # ── Classify ───────────────────────────────────────────────────────────
    wb = openpyxl.load_workbook(io.BytesIO(data), read_only=True)
    ws = wb.active
    first_two_rows = list(ws.iter_rows(max_row=2, values_only=True))
    wb.close()

    all_text = " ".join(
        str(c).strip().lower()
        for row in first_two_rows for c in row if c is not None
    )
    row0 = [str(c).strip().lower() if c else "" for c in first_two_rows[0]]

    has_flight_signal = any(sig in all_text for sig in _FLIGHT_SIGNALS)
    has_tfs  = any("tfs" in c for c in row0[:4])
    has_tim  = any(c == "tim" for c in row0[:4])
    has_ktas = "ktas" in all_text
    has_iasp = "iasp" in all_text

    if has_flight_signal:
        kind = "flight"
    elif has_tfs and has_tim:
        kind = "log"
    elif has_ktas and not has_iasp:
        kind = "log"
    else:
        return "unknown", 0.0, 0.0

    # ── Extract time range ─────────────────────────────────────────────────
    df          = pd.read_excel(io.BytesIO(data), engine="openpyxl", header=0)
    col_lower   = [str(c).strip().lower() for c in df.columns]

    if kind == "flight":
        # Col 0 = absolute seconds past midnight; skip row 0 (units row)
        abs_sec = pd.to_numeric(df.iloc[1:, 0], errors="coerce").dropna()
        abs_sec = abs_sec[abs_sec > 0]
        if len(abs_sec) == 0:
            return kind, 0.0, 0.0
        return kind, float(abs_sec.min()), float(abs_sec.max())

    # ── Log file time extraction ───────────────────────────────────────────
    tfs_idx = next((i for i, c in enumerate(col_lower) if "tfs" in c), None)
    tim_idx = next((i for i, c in enumerate(col_lower) if c == "tim"), None)

    if tfs_idx is not None:
        tfs_raw = df.iloc[:, tfs_idx]

        # Case 1: TFS is numeric in seconds-past-midnight range (0–86400)
        tfs_num = pd.to_numeric(tfs_raw, errors="coerce").dropna()
        if len(tfs_num) > 10:
            med = tfs_num.median()
            if 0 < med < 86400:
                # Valid seconds-past-midnight
                return kind, float(tfs_num.min()), float(tfs_num.max())

        # Case 2: TFS is a datetime string
        tfs_dt = pd.to_datetime(tfs_raw, errors="coerce").dropna()
        if len(tfs_dt) > 10 and tfs_dt.dt.year.median() > 2000:
            tod = (tfs_dt.dt.hour * 3600 +
                   tfs_dt.dt.minute * 60 +
                   tfs_dt.dt.second +
                   tfs_dt.dt.microsecond / 1e6)
            return kind, float(tod.min()), float(tod.max())

    # Case 3: TFS unreliable — use elapsed time for duration-based matching
    # Return kind="log_elapsed" so matcher uses duration similarity instead
    time_col = next((df.columns[i] for i, c in enumerate(col_lower)
                     if c in _LOG_TIME_CANDIDATES), None)
    if time_col is not None:
        t = pd.to_numeric(df[time_col], errors="coerce").dropna()
        t = t[t >= 0]
        if len(t) > 0:
            return "log_elapsed", float(t.min()), float(t.max())

    if tim_idx is not None:
        tim = pd.to_numeric(df.iloc[:, tim_idx], errors="coerce").dropna()
        t   = (tim - tim.iloc[0]) / 1_000_000
        return "log_elapsed", float(t.min()), float(t.max())

    return kind, 0.0, 0.0


def _containment_score(f_min, f_max, l_min, l_max) -> float:
    """Fraction of the log's duration that falls within the flight window."""
    overlap  = max(0.0, min(f_max, l_max) - max(f_min, l_min))
    log_span = l_max - l_min
    return overlap / log_span if log_span > 0 else 0.0


def _duration_score(f_min, f_max, l_min, l_max) -> float:
    """
    Similarity of durations — used for logs without absolute timestamps.
    Score of 1.0 means identical duration.
    """
    f_dur = f_max - f_min
    l_dur = l_max - l_min
    diff  = abs(f_dur - l_dur)
    return 1.0 - diff / max(f_dur, l_dur) if max(f_dur, l_dur) > 0 else 0.0


def auto_match(flight_info: dict, log_info: dict) -> tuple[dict, list, list]:
    """
    Match each log to its best flight and reject poor matches.

    Logs with absolute timestamps (kind='log') are matched by containment:
      what fraction of the log's duration falls within the flight window?

    Logs with only elapsed time (kind='log_elapsed') are matched by duration
    similarity: which flight has the most similar total duration?

    Returns: (groups, rejected, warnings)
      groups:   {flight_name: [(log_name, score), ...]}
      rejected: [(log_name, best_score, best_flight_name)]
      warnings: [human-readable warning string]
    """
    groups   = defaultdict(list)
    rejected = []
    warnings = []

    for lname, (lkind, lt_min, lt_max) in log_info.items():
        if lkind == "log_elapsed":
            # Duration-based matching for logs without absolute timestamps
            scores = {
                fname: _duration_score(fv[1], fv[2], lt_min, lt_max)
                for fname, fv in flight_info.items()
            }
            threshold = 0.95  # duration must match within 5%
        else:
            # Absolute containment matching
            scores = {
                fname: _containment_score(fv[1], fv[2], lt_min, lt_max)
                for fname, fv in flight_info.items()
            }
            threshold = MATCH_THRESHOLD

        best       = max(scores, key=scores.get)
        best_score = scores[best]

        if best_score < threshold:
            rejected.append((lname, best_score, best))
            method = "duration similarity" if lkind == "log_elapsed" else "time containment"
            warnings.append(
                f"**{lname}** could not be reliably matched to any flight "
                f"({method} {best_score:.0%} < {threshold:.0%}). "
                f"Check that the correct files were uploaded."
            )
        else:
            groups[best].append((lname, best_score))

    for fname in flight_info:
        if fname not in groups:
            groups[fname] = []

    return dict(groups), rejected, warnings


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
st.sidebar.markdown("### Leave-One-Out Testing")
st.sidebar.caption(
    "Optionally hold out one entire flight from training. "
    "This tests how well the model generalises to completely unseen conditions."
)
loo_enabled = st.sidebar.toggle("Hold out one flight", value=False)
loo_flight  = None
if loo_enabled:
    # Populated after files are uploaded and matched
    loo_flight = st.sidebar.selectbox(
        "Flight to hold out (test only)",
        options=["— upload files first —"],
        key="loo_select",
    )

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
    groups, rejected, match_warnings = auto_match(flight_info, log_info)

    # Populate LOO selectbox now that we know flight names
    flight_names_for_loo = list(groups.keys())
    if loo_enabled and flight_names_for_loo:
        loo_flight = st.sidebar.selectbox(
            "Flight to hold out (test only)",
            options=flight_names_for_loo,
            key="loo_select_real",
            help="This flight will not be used in training — only for evaluation.",
        )
    else:
        loo_flight = None

    # Show matching preview
    st.markdown("### Detected File Matching")
    st.caption(
        "Log files are automatically matched to flights using absolute timestamps. "
        "Check this looks correct before running."
    )

    # Show rejection warnings prominently
    for w in match_warnings:
        st.error(f"❌ {w}")

    # Show rejected files info
    if rejected:
        with st.expander(f"⚠️ {len(rejected)} file(s) could not be matched — click to see details"):
            for lname, score, best in rejected:
                st.write(f"• **{lname}** — best match was `{best}` with score {score:.0%}")
            st.write("These files will be excluded from the pipeline run.")

    cols = st.columns(max(len(groups), 1))
    all_ok = True
    for col, (fname, lmatches) in zip(cols, groups.items()):
        lnames = [l for l, _ in lmatches]
        scores = {l: s for l, s in lmatches}
        with col:
            f_obj, fkind, ft_min, ft_max = (*flight_info[fname],)
            # Format time range as HH:MM if in reasonable range
            def fmt_tod(s):
                if 0 < s < 86400:
                    h, rem = divmod(int(s), 3600)
                    m = rem // 60
                    return f"{h:02d}:{m:02d} UTC"
                return f"{s:.0f}s"

            st.markdown(f"""
            <div class="match-card">
                <div class="match-title">✈ {fname}</div>
                <div class="match-log" style="color:#6b7280;">
                    {fmt_tod(ft_min)} – {fmt_tod(ft_max)}
                </div>
                <hr style="border-color:#2a2d3e; margin:8px 0;">
            """ + "".join(
                f'<div class="match-log">📋 {l} <span style="color:#3ecf8e;">({scores[l]:.0%})</span></div>'
                for l in lnames
            ) + ("" if lnames else
                 '<div class="match-log" style="color:#e05252;">⚠ No logs matched</div>'
            ) + "</div>", unsafe_allow_html=True)

            if not lnames:
                all_ok = False

    if not all_ok and not match_warnings:
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

                # Build the session — if LOO is enabled, the held-out
                # flight goes into a test-only split; all others use the
                # random train/test split as normal
                session = CombinedSession(
                    test_fraction=float(test_frac),
                    random_seed=int(seed),
                )
                for name, r in results:
                    session.add_flight(name, r)

                # Apply LOO override if selected
                loo_name = st.session_state.get("loo_select_real") if loo_enabled else None
                if loo_name and loo_name in [n for n, _ in results]:
                    combined = session.run(held_out_flight=loo_name)
                else:
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
