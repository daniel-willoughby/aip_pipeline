"""
streamlit_app.py
----------------
Streamlit web interface for the AIP data pipeline.

Two modes:
  1. INFERENCE — upload a new flight+log, run against pre-trained model
  2. TRAINING  — upload all flights, train a new model from scratch

Run with: streamlit run streamlit_app.py

Password: set in .streamlit/secrets.toml  →  password = "your_password"
"""

from __future__ import annotations

import hmac
import io
import pickle
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

# ── Password ──────────────────────────────────────────────────────────────────

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
            st.error("❌ Incorrect password")
    return False

if not check_password():
    st.stop()

# ── CSS ───────────────────────────────────────────────────────────────────────

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
    .metric-card {
        background: #1a1d27; border: 1px solid #2a2d3e;
        border-radius: 8px; padding: 16px; text-align: center;
    }
    .metric-value { font-size: 28px; font-weight: 700; margin: 4px 0; }
    .metric-label {
        color: #6b7280; font-size: 12px;
        text-transform: uppercase; letter-spacing: 0.05em;
    }
    .good { color: #3ecf8e; }
    .ok   { color: #f0a500; }
    .poor { color: #e05252; }
    #MainMenu { visibility: hidden; }
    footer    { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────

MODEL_PATH = Path(__file__).parent / "models" / "aip_model.pkl"

def load_model() -> dict | None:
    if MODEL_PATH.exists():
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    return None

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

# ── Classification & matching (shared between modes) ──────────────────────────

_FLIGHT_SIGNALS   = ["iasp","true airspeed","pressure altitude",
                     "static pressure true","icd lwc","ccp mvd"]
_LOG_TIME_CANDS   = ["time s","time_s","time (s)","elapsed_s","time inc"]
MATCH_THRESHOLD   = 0.80
KNOWN_ASSOCIATIONS = {
    "AIP_log3_WOW_0.xlsx":          "flight1475-AEROTEX-AIP-1_WOW0.xlsx",
    "AIP_Log6_WOW_0.xlsx":          "flight1475-AEROTEX-AIP-2_WOW0.xlsx",
    "AIP_Log7_WOW_0.xlsx":          "flight1475-AEROTEX-AIP-2_WOW0.xlsx",
    "AIP_log_4_WOW_0.xlsx":         "Flight1476-AEROTEX-AIP-1_WOW_0.xlsx",
    "AIP_log_7_WOW_0.xlsx":         "flight1477-AEROTEX-AIP-1_WOW_0.xlsx",
    "AIP_log_8_WOW_0.xlsx":         "flight1477-AEROTEX-AIP-2_WOW_0.xlsx",
}

def classify_and_get_range(uploaded_file) -> tuple[str, float, float]:
    import openpyxl
    import pandas as pd
    data = uploaded_file.getbuffer()
    wb = openpyxl.load_workbook(io.BytesIO(data), read_only=True)
    ws = wb.active
    first_two = list(ws.iter_rows(max_row=2, values_only=True))
    wb.close()
    all_text = " ".join(str(c).strip().lower()
                        for row in first_two for c in row if c is not None)
    row0 = [str(c).strip().lower() if c else "" for c in first_two[0]]
    has_flight = any(sig in all_text for sig in _FLIGHT_SIGNALS)
    has_tfs    = any("tfs" in c for c in row0[:4])
    has_tim    = any(c == "tim" for c in row0[:4])
    has_ktas   = "ktas" in all_text
    has_iasp   = "iasp" in all_text
    if has_flight:      kind = "flight"
    elif has_tfs and has_tim: kind = "log"
    elif has_ktas and not has_iasp: kind = "log"
    else: return "unknown", 0.0, 0.0
    df = pd.read_excel(io.BytesIO(data), engine="openpyxl", header=0)
    col_lower = [str(c).strip().lower() for c in df.columns]
    if kind == "flight":
        abs_sec = pd.to_numeric(df.iloc[1:, 0], errors="coerce").dropna()
        abs_sec = abs_sec[abs_sec > 0]
        return kind, float(abs_sec.min()) if len(abs_sec) else 0.0, \
               float(abs_sec.max()) if len(abs_sec) else 0.0
    tfs_idx = next((i for i, c in enumerate(col_lower) if "tfs" in c), None)
    if tfs_idx is not None:
        tfs_raw = df.iloc[:, tfs_idx]
        tfs_num = pd.to_numeric(tfs_raw, errors="coerce").dropna()
        if len(tfs_num) > 10:
            med, span = tfs_num.median(), float(tfs_num.max() - tfs_num.min())
            if 0 < med < 86400 and span >= 60.0:
                return kind, float(tfs_num.min()), float(tfs_num.max())
        tfs_dt = pd.to_datetime(tfs_raw, errors="coerce").dropna()
        if len(tfs_dt) > 10 and tfs_dt.dt.year.median() > 2000:
            tod  = (tfs_dt.dt.hour*3600 + tfs_dt.dt.minute*60 +
                    tfs_dt.dt.second + tfs_dt.dt.microsecond/1e6)
            if float(tod.max()-tod.min()) >= 60.0:
                return kind, float(tod.min()), float(tod.max())
    time_col = next((df.columns[i] for i, c in enumerate(col_lower)
                     if c in _LOG_TIME_CANDS), None)
    if time_col is not None:
        t = pd.to_numeric(df[time_col], errors="coerce").dropna()
        t = t[t >= 0]
        if len(t) > 0:
            return "log_elapsed", float(t.min()), float(t.max())
    tim_idx = next((i for i, c in enumerate(col_lower) if c == "tim"), None)
    if tim_idx is not None:
        tim = pd.to_numeric(df.iloc[:, tim_idx], errors="coerce").dropna()
        t   = (tim - tim.iloc[0]) / 1_000_000
        return "log_elapsed", float(t.min()), float(t.max())
    return kind, 0.0, 0.0

def _containment(f_min,f_max,l_min,l_max):
    overlap  = max(0.0, min(f_max,l_max)-max(f_min,l_min))
    log_span = l_max - l_min
    return overlap/log_span if log_span > 0 else 0.0

def _tightness(f_min,f_max,l_min,l_max):
    f_dur = f_max - f_min
    l_dur = l_max - l_min
    return l_dur/f_dur if f_dur > 0 else 0.0

def _duration(f_min,f_max,l_min,l_max):
    f_dur = f_max - f_min
    l_dur = l_max - l_min
    diff  = abs(f_dur - l_dur)
    return 1.0 - diff/max(f_dur,l_dur) if max(f_dur,l_dur) > 0 else 0.0

def auto_match(flight_info, log_info):
    groups, rejected, warnings = defaultdict(list), [], []
    for lname, (lkind, lt_min, lt_max) in log_info.items():
        known = KNOWN_ASSOCIATIONS.get(lname)
        if known:
            matched = next((fn for fn in flight_info
                            if fn.lower() == known.lower()), None)
            if matched:
                groups[matched].append((lname, 1.0))
                continue
        if lkind == "log_elapsed":
            scores = {fn: _duration(fv[1],fv[2],lt_min,lt_max)
                      for fn,fv in flight_info.items()}
            thresh = 0.95
            best   = max(flight_info, key=lambda fn: (
                scores[fn], _tightness(flight_info[fn][1],flight_info[fn][2],lt_min,lt_max)))
        else:
            scores = {fn: _containment(fv[1],fv[2],lt_min,lt_max)
                      for fn,fv in flight_info.items()}
            thresh = MATCH_THRESHOLD
            best   = max(flight_info, key=lambda fn: (
                round(scores[fn],2),
                _tightness(flight_info[fn][1],flight_info[fn][2],lt_min,lt_max)))
        best_score = scores[best]
        if best_score < thresh:
            rejected.append((lname, best_score, best))
            method = "duration" if lkind == "log_elapsed" else "time containment"
            warnings.append(
                f"**{lname}** could not be matched ({method} "
                f"{best_score:.0%} < {thresh:.0%}).")
        else:
            groups[best].append((lname, best_score))
    for fn in flight_info:
        if fn not in groups:
            groups[fn] = []
    return dict(groups), rejected, warnings

# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.markdown("""
<div style="padding:8px 0 16px 0;">
    <div style="font-size:20px;font-weight:700;color:#ffffff;">✈️ AIP Pipeline</div>
    <div style="font-size:12px;color:#6b7280;margin-top:2px;">LWC & MVD Estimation</div>
</div>
<div class="accent-rule"></div>
""", unsafe_allow_html=True)

model_bundle = load_model()

# Mode selector
if model_bundle:
    mode = st.sidebar.radio(
        "Mode",
        ["🔍 Inference (use saved model)", "🏋️ Training (build new model)"],
        index=0,
    )
else:
    st.sidebar.warning("No saved model found. Run `python train.py` locally first, "
                       "or use Training mode to build one.")
    mode = "🏋️ Training (build new model)"

st.sidebar.divider()

# ── INFERENCE MODE sidebar ────────────────────────────────────────────────────

if mode == "🔍 Inference (use saved model)":
    st.sidebar.markdown("### Upload New Flight Data")
    st.sidebar.caption(
        "Upload one flight file and its log file(s). "
        "The pre-trained model will estimate LWC and MVD."
    )

    inf_flight = st.sidebar.file_uploader(
        "Flight file (.xlsx)", type=["xlsx"], key="inf_flight")
    inf_logs = st.sidebar.file_uploader(
        "Log file(s) (.xlsx)", type=["xlsx"],
        accept_multiple_files=True, key="inf_logs")

    if inf_flight:
        st.sidebar.caption(f"📄 {inf_flight.name}")
    if inf_logs:
        for lf in inf_logs:
            st.sidebar.caption(f"📋 {lf.name}")

    st.sidebar.divider()
    if model_bundle:
        st.sidebar.markdown("**Trained on:**")
        for fn in model_bundle.get("flights", []):
            st.sidebar.caption(f"• {fn}")
        st.sidebar.markdown("**Training metrics:**")
        tm = model_bundle.get("test_metrics", {})
        st.sidebar.caption(
            f"LWC R² {tm.get('lwc_r2',float('nan')):.3f} · "
            f"MVD R² {tm.get('mvd_r2',float('nan')):.3f}"
        )

    st.sidebar.divider()
    if st.sidebar.button("🔒 Log out", use_container_width=True):
        st.session_state["password_correct"] = False
        st.rerun()

    run_inference_btn = st.sidebar.button(
        "▶  Run Inference",
        type="primary",
        use_container_width=True,
        disabled=(inf_flight is None or not inf_logs or model_bundle is None),
    )

# ── TRAINING MODE sidebar ─────────────────────────────────────────────────────

else:
    st.sidebar.markdown("### Upload All Flights")
    st.sidebar.caption(
        "Upload all flight and log files. The app will auto-match them, "
        "train a new model and save it."
    )

    if "n_flights" not in st.session_state:
        st.session_state.n_flights = 1

    ca, cb = st.sidebar.columns(2)
    if ca.button("＋ Add", use_container_width=True):
        st.session_state.n_flights += 1
    if cb.button("－ Remove", use_container_width=True):
        st.session_state.n_flights = max(1, st.session_state.n_flights-1)

    uploaded_files = st.sidebar.file_uploader(
        "All flight & log files", type=["xlsx"],
        accept_multiple_files=True, key="train_files",
        label_visibility="collapsed",
    )

    st.sidebar.divider()
    st.sidebar.markdown("### Settings")
    test_frac = st.sidebar.slider("Test split", 0.1, 0.4, 0.2, 0.05)
    seed      = st.sidebar.number_input("Random seed", value=42, step=1)
    loo_enabled = st.sidebar.toggle("Hold out one flight (LOO)", value=False)

    st.sidebar.divider()
    if st.sidebar.button("🔒 Log out", use_container_width=True):
        st.session_state["password_correct"] = False
        st.rerun()

    run_training_btn = st.sidebar.button(
        "▶  Train & Save Model",
        type="primary",
        use_container_width=True,
    )

# ── Main header ───────────────────────────────────────────────────────────────

st.markdown("""
<h1 style="color:#ffffff;font-size:28px;font-weight:700;margin-bottom:4px;">
    AIP Data Pipeline
</h1>
<div class="accent-rule"></div>
""", unsafe_allow_html=True)

# ── Initialise session state ──────────────────────────────────────────────────
for key in ["metrics","train_metrics","log_text","plots","inf_result"]:
    if key not in st.session_state:
        st.session_state[key] = ({} if key in ("metrics","train_metrics","plots")
                                  else "" if key == "log_text" else None)

# ═══════════════════════════════════════════════════════════════════════════════
#  INFERENCE MODE
# ═══════════════════════════════════════════════════════════════════════════════

if mode == "🔍 Inference (use saved model)":

    if model_bundle is None:
        st.error("No trained model found at `models/aip_model.pkl`. "
                 "Run `python train.py` locally and commit the file to GitHub.")
        st.stop()

    tab_results, tab_lwc, tab_mvd, tab_log = st.tabs([
        "📊 Results", "💧 LWC", "🔵 MVD", "📋 Log"
    ])

    result_df = st.session_state.inf_result

    with tab_results:
        if result_df is None:
            st.info(
                "Upload a flight file and its log file(s) in the sidebar, "
                "then click **▶ Run Inference**."
            )
        else:
            import numpy as np
            import pandas as pd

            icing = result_df[result_df.get("REF_LWC", pd.Series(dtype=float)) >= 0.01] \
                    if "REF_LWC" in result_df.columns else result_df

            st.markdown("### LWC Estimates")
            lwc_cols = [f"LWC_{n}" for n in range(1,6)] + ["LWC_MEAN"]
            available = [c for c in lwc_cols if c in result_df.columns]
            st.dataframe(
                result_df[["TIME"] + available].describe().round(4),
                use_container_width=True,
            )

            if "REF_LWC" in result_df.columns:
                from sklearn.metrics import r2_score, mean_absolute_error
                st.markdown("### Accuracy vs Reference")
                c1, c2, c3, c4 = st.columns(4)
                ref_lwc = result_df.loc[result_df["REF_LWC"]>=0.01, "REF_LWC"]
                pred_lwc = result_df.loc[result_df["REF_LWC"]>=0.01, "LWC_MEAN"]
                v = ref_lwc.notna() & pred_lwc.notna()
                if v.sum() > 2:
                    r2  = r2_score(ref_lwc[v], pred_lwc[v])
                    mae = mean_absolute_error(ref_lwc[v], pred_lwc[v])
                    with c1: st.markdown(metric_html("LWC R²",  r2,  "lwc_r2"),  unsafe_allow_html=True)
                    with c2: st.markdown(metric_html("LWC MAE", mae, "lwc_mae", " g/m³"), unsafe_allow_html=True)

                if "REF_MVD" in result_df.columns and "MVD_EST" in result_df.columns:
                    ref_mvd  = result_df.loc[result_df["REF_MVD"]>=1.0, "REF_MVD"]
                    pred_mvd = result_df.loc[result_df["REF_MVD"]>=1.0, "MVD_EST"]
                    v2 = ref_mvd.notna() & pred_mvd.notna()
                    if v2.sum() > 2:
                        r2m  = r2_score(ref_mvd[v2], pred_mvd[v2])
                        maem = mean_absolute_error(ref_mvd[v2], pred_mvd[v2])
                        with c3: st.markdown(metric_html("MVD R²",  r2m,  "mvd_r2"),  unsafe_allow_html=True)
                        with c4: st.markdown(metric_html("MVD MAE", maem, "mvd_mae", " µm"), unsafe_allow_html=True)

    with tab_lwc:
        if result_df is not None and st.session_state.plots.get("inf_lwc"):
            st.image(st.session_state.plots["inf_lwc"], use_container_width=True)
        elif result_df is None:
            st.info("Run inference to see LWC timeseries.")

    with tab_mvd:
        if result_df is not None and st.session_state.plots.get("inf_mvd"):
            st.image(st.session_state.plots["inf_mvd"], use_container_width=True)
        elif result_df is None:
            st.info("Run inference to see MVD timeseries.")

    with tab_log:
        if st.session_state.log_text:
            st.code(st.session_state.log_text, language=None)
        else:
            st.info("Log will appear here after running inference.")

    # ── Run inference ─────────────────────────────────────────────────────────
    if run_inference_btn:
        if inf_flight is None or not inf_logs:
            st.sidebar.error("Upload a flight file and at least one log file.")
            st.stop()

        log_capture = io.StringIO()
        status_box  = st.empty()

        try:
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                plot_dir = tmp_path / "plots"
                plot_dir.mkdir()

                old_stdout = sys.stdout
                sys.stdout = log_capture

                sys.path.insert(0, str(Path(__file__).parent))
                from src.pipeline.inference import run_inference
                import matplotlib.pyplot as plt

                status_box.info("⏳ Ingesting files…")
                flight_path = save_uploaded(inf_flight, tmp_path)
                log_paths   = [save_uploaded(lf, tmp_path) for lf in inf_logs]

                status_box.info("⏳ Running inference…")
                result_df = run_inference(
                    str(flight_path),
                    [str(p) for p in log_paths],
                    model_bundle,
                )

                # Generate timeseries plots
                status_box.info("⏳ Generating plots…")
                plots = {}

                # LWC timeseries
                fig, ax = plt.subplots(figsize=(12, 4),
                                       facecolor="#0f1117")
                ax.set_facecolor("#1a1d27")
                lwc_cols_plot = [f"LWC_{n}" for n in range(1,6)
                                 if f"LWC_{n}" in result_df.columns]
                for col in lwc_cols_plot:
                    ax.plot(result_df["TIME"], result_df[col],
                            alpha=0.5, linewidth=0.8)
                ax.plot(result_df["TIME"], result_df["LWC_MEAN"],
                        color="#4f8ef7", linewidth=1.5, label="LWC Mean")
                if "REF_LWC" in result_df.columns:
                    ax.plot(result_df["TIME"], result_df["REF_LWC"],
                            color="#3ecf8e", linewidth=1.5,
                            label="REF_LWC", linestyle="--")
                ax.set_xlabel("Time (s)", color="#6b7280")
                ax.set_ylabel("LWC (g/m³)", color="#6b7280")
                ax.set_title("LWC Estimation", color="#ffffff")
                ax.tick_params(colors="#6b7280")
                ax.legend(facecolor="#1a1d27", labelcolor="#e8eaf0")
                for spine in ax.spines.values():
                    spine.set_edgecolor("#2a2d3e")
                buf = io.BytesIO()
                fig.tight_layout()
                fig.savefig(buf, format="png", dpi=100,
                            facecolor="#0f1117")
                plt.close(fig)
                plots["inf_lwc"] = buf.getvalue()

                # MVD timeseries
                if "MVD_EST" in result_df.columns:
                    fig, ax = plt.subplots(figsize=(12, 4),
                                           facecolor="#0f1117")
                    ax.set_facecolor("#1a1d27")
                    ax.plot(result_df["TIME"], result_df["MVD_EST"],
                            color="#7c5cbf", linewidth=1.2, label="MVD Est")
                    if "REF_MVD" in result_df.columns:
                        ax.plot(result_df["TIME"], result_df["REF_MVD"],
                                color="#3ecf8e", linewidth=1.2,
                                label="REF_MVD", linestyle="--")
                    ax.set_xlabel("Time (s)", color="#6b7280")
                    ax.set_ylabel("MVD (µm)", color="#6b7280")
                    ax.set_title("MVD Estimation", color="#ffffff")
                    ax.tick_params(colors="#6b7280")
                    ax.legend(facecolor="#1a1d27", labelcolor="#e8eaf0")
                    for spine in ax.spines.values():
                        spine.set_edgecolor("#2a2d3e")
                    buf2 = io.BytesIO()
                    fig.tight_layout()
                    fig.savefig(buf2, format="png", dpi=100,
                                facecolor="#0f1117")
                    plt.close(fig)
                    plots["inf_mvd"] = buf2.getvalue()

                sys.stdout = old_stdout
                st.session_state.log_text   = log_capture.getvalue()
                st.session_state.inf_result = result_df
                st.session_state.plots      = plots

            status_box.empty()
            st.success("✓ Inference complete!")
            st.rerun()

        except Exception:
            sys.stdout = old_stdout
            st.session_state.log_text = (
                log_capture.getvalue() + "\n" + traceback.format_exc()
            )
            status_box.empty()
            st.error("Inference failed — see the **Log** tab.")
            st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
#  TRAINING MODE
# ═══════════════════════════════════════════════════════════════════════════════

else:
    # File matching preview
    if uploaded_files:
        with st.spinner("Classifying files…"):
            flight_info, log_info = {}, {}
            errors = []
            for f in uploaded_files:
                try:
                    kind, t_min, t_max = classify_and_get_range(f)
                    if kind == "flight":
                        flight_info[f.name] = (f, t_min, t_max)
                    elif kind.startswith("log"):
                        log_info[f.name] = (kind, t_min, t_max)
                except Exception as e:
                    errors.append(f"{f.name}: {e}")

        for e in errors:
            st.warning(f"⚠️ {e}")

        if not flight_info:
            st.error("No flight files detected.")
            st.stop()
        if not log_info:
            st.error("No log files detected.")
            st.stop()

        groups, rejected, match_warnings = auto_match(flight_info, log_info)

        st.markdown("### Detected File Matching")
        for w in match_warnings:
            st.error(f"❌ {w}")
        if rejected:
            with st.expander(f"⚠️ {len(rejected)} file(s) unmatched"):
                for lname, score, best in rejected:
                    st.write(f"• **{lname}** — best {score:.0%} to `{best}`")

        cols = st.columns(max(len(groups), 1))
        all_ok = True
        for col, (fname, lmatches) in zip(cols, groups.items()):
            lnames = [l for l,_ in lmatches]
            scores = {l:s for l,s in lmatches}
            with col:
                f_obj, ft_min, ft_max = flight_info[fname]
                def fmt(s):
                    return (f"{int(s//3600):02d}:{int((s%3600)//60):02d} UTC"
                            if 0 < s < 86400 else f"{s:.0f}s")
                st.markdown(f"""
                <div style="background:#1a1d27;border:1px solid #2a2d3e;
                            border-radius:8px;padding:14px 18px;margin-bottom:10px;">
                    <div style="color:#fff;font-weight:600;font-size:14px;
                                margin-bottom:6px;">✈ {fname}</div>
                    <div style="color:#6b7280;font-size:12px;font-family:monospace;">
                        {fmt(ft_min)} – {fmt(ft_max)}</div>
                    <hr style="border-color:#2a2d3e;margin:8px 0;">
                """ + "".join(
                    f'<div style="color:#a8b4c8;font-size:12px;font-family:monospace;'
                    f'padding:2px 0;">📋 {l} '
                    f'<span style="color:#3ecf8e;">({scores[l]:.0%})</span></div>'
                    for l in lnames
                ) + (
                    "" if lnames else
                    '<div style="color:#e05252;font-size:12px;">⚠ No logs matched</div>'
                ) + "</div>", unsafe_allow_html=True)
                if not lnames:
                    all_ok = False

        # LOO selector
        if loo_enabled and flight_info:
            loo_flight = st.selectbox(
                "Flight to hold out (test only)",
                options=list(groups.keys()),
                key="loo_select_real",
            )
        else:
            loo_flight = None

        st.divider()

    else:
        st.info(
            "Upload all your flight and log `.xlsx` files using the sidebar. "
            "The app will detect which logs belong to which flight."
        )
        groups  = {}
        all_ok  = False

    # Results tabs
    if st.session_state.metrics or st.session_state.log_text:
        st.markdown("### Training Results")
        tab_r, tab_lwc, tab_mvd, tab_sum, tab_log = st.tabs([
            "📊 Metrics", "💧 LWC", "🔵 MVD", "📈 Summary", "📋 Log"
        ])
        plots = st.session_state.plots
        with tab_r:
            m  = st.session_state.metrics
            tm = st.session_state.train_metrics
            st.markdown("**Test Set**")
            c1,c2,c3,c4 = st.columns(4)
            with c1: st.markdown(metric_html("LWC R²",  m.get("lwc_r2"),  "lwc_r2"),          unsafe_allow_html=True)
            with c2: st.markdown(metric_html("LWC MAE", m.get("lwc_mae"), "lwc_mae", " g/m³"), unsafe_allow_html=True)
            with c3: st.markdown(metric_html("MVD R²",  m.get("mvd_r2"),  "mvd_r2"),           unsafe_allow_html=True)
            with c4: st.markdown(metric_html("MVD MAE", m.get("mvd_mae"), "mvd_mae", " µm"),   unsafe_allow_html=True)
        with tab_lwc:
            if plots.get("lwc_scatter"):
                st.image(plots["lwc_scatter"], use_container_width=True)
                st.download_button("⬇ Download", plots["lwc_scatter"], "lwc_scatter.png", "image/png")
            else: st.info("Run training to see plots.")
        with tab_mvd:
            if plots.get("mvd_scatter"):
                st.image(plots["mvd_scatter"], use_container_width=True)
                st.download_button("⬇ Download", plots["mvd_scatter"], "mvd_scatter.png", "image/png")
            else: st.info("Run training to see plots.")
        with tab_sum:
            if plots.get("summary"):
                st.image(plots["summary"], use_container_width=True)
                st.download_button("⬇ Download", plots["summary"], "summary.png", "image/png")
            else: st.info("Run training to see plots.")
        with tab_log:
            if st.session_state.log_text:
                st.code(st.session_state.log_text, language=None)
            else: st.info("Log will appear here after training.")

    # ── Run training ──────────────────────────────────────────────────────────
    if "run_training_btn" in dir() and run_training_btn and uploaded_files and all_ok:
        log_capture = io.StringIO()
        status_box  = st.empty()
        n_f         = len(groups)

        try:
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                plot_dir = tmp_path / "plots"
                plot_dir.mkdir()

                old_stdout = sys.stdout
                sys.stdout = log_capture

                sys.path.insert(0, str(Path(__file__).parent))
                import gc, pickle
                from src.pipeline import run_pipeline
                from src.pipeline.session import CombinedSession
                from src.pipeline.visualiser import CombinedVisualiser

                results = []
                for i, (fname, lmatches) in enumerate(groups.items()):
                    status_box.info(
                        f"⏳ Step {i+1}/{n_f+2} — Ingesting **{fname}**…")
                    f_obj  = flight_info[fname][0]
                    l_objs = [log_info[ln][0] for ln in [l for l,_ in lmatches]]
                    fp = save_uploaded(f_obj, tmp_path)
                    lps = [save_uploaded(l, tmp_path) for l in l_objs]
                    r = run_pipeline(str(fp), [str(p) for p in lps])
                    results.append((f"Flight {i+1}", r))
                    gc.collect()

                status_box.info(
                    f"⏳ Step {n_f+1}/{n_f+2} — Training models…")
                session = CombinedSession(
                    test_fraction=float(test_frac),
                    random_seed=int(seed),
                )
                for name, r in results:
                    session.add_flight(name, r)

                loo_name = (st.session_state.get("loo_select_real")
                            if loo_enabled else None)
                combined = session.run(held_out_flight=loo_name)
                gc.collect()

                status_box.info(
                    f"⏳ Step {n_f+2}/{n_f+2} — Generating plots…")
                vis = CombinedVisualiser(combined, output_dir=str(plot_dir))
                vis.plot_all()

                # Save model bundle for download
                model_bundle_new = {
                    "lwc_estimator": combined.lwc_estimator,
                    "mvd_estimator": combined.mvd_estimator,
                    "train_metrics": combined.train_metrics,
                    "test_metrics":  combined.test_metrics,
                    "flights":       [n for n,_ in results],
                }
                model_bytes = pickle.dumps(model_bundle_new)

                sys.stdout = old_stdout

                st.session_state.log_text      = log_capture.getvalue()
                st.session_state.metrics       = combined.test_metrics
                st.session_state.train_metrics = combined.train_metrics
                st.session_state["model_bytes"] = model_bytes

                def read_plot(name):
                    p = plot_dir / name
                    return p.read_bytes() if p.exists() else None

                st.session_state.plots = {
                    "lwc_scatter": read_plot("combined_lwc_scatter.png"),
                    "mvd_scatter": read_plot("combined_mvd_scatter.png"),
                    "summary":     read_plot("combined_summary.png"),
                }

            status_box.empty()
            st.success("✓ Training complete!")
            st.info(
                "⬇ Download the model below and place it at "
                "`models/aip_model.pkl` in your repo to use it in Inference mode."
            )
            st.download_button(
                "⬇ Download trained model (aip_model.pkl)",
                data=st.session_state["model_bytes"],
                file_name="aip_model.pkl",
                mime="application/octet-stream",
            )
            st.rerun()

        except Exception:
            sys.stdout = old_stdout
            st.session_state.log_text = (
                log_capture.getvalue() + "\n" + traceback.format_exc()
            )
            status_box.empty()
            st.error("Training failed — see the **Log** tab.")
            st.rerun()
