import io
import logging
import os
import tempfile
import pandas as pd
import streamlit as st
from core.graph import build_graph

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Data Analysis Agent",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(
    """
    <style>
    /* ---------- Global ---------- */
    .block-container { padding-top: 2rem; }
    /* ---------- Metric cards ---------- */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 18px 20px;
        color: #ffffff;
        box-shadow: 0 4px 14px rgba(102,126,234,0.25);
    }
    div[data-testid="stMetric"] label {
        color: rgba(255,255,255,0.85) !important;
        font-size: 0.85rem !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-weight: 700 !important;
    }
    /* ---------- Insight cards ---------- */
    .insight-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 20px 24px;
        border-left: 4px solid;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        margin-bottom: 12px;
    }
    .insight-card.findings  { border-color: #667eea; }
    .insight-card.anomalies { border-color: #f5576c; }
    .insight-card.recommendations { border-color: #43e97b; }
    .insight-card h4 { margin: 0 0 10px 0; }
    /* ---------- Pipeline step badges ---------- */
    .step-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 6px;
        color: #fff;
    }
    .step-done { background: #43e97b; }
    .step-active { background: #667eea; }
    .step-pending { background: #e0e0e0; color: #888; }
    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    section[data-testid="stSidebar"] * {
        color: #e0e0e0 !important;
    }
    section[data-testid="stSidebar"] .stButton > button {
        width: 100%;
        border-radius: 8px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border: none;
        font-weight: 600;
        padding: 0.6rem 1rem;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        opacity: 0.9;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

_DEFAULTS = {
    "analysis_result": None,
    "uploaded_file_name": None,
}
for key, val in _DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val

with st.sidebar:
    st.markdown("## Data Analysis Agent")
    st.divider()
    uploaded_file = st.file_uploader(
        "Upload your CSV file",
        type=["csv"],
        help="Max 200 MB. The file is processed in-memory and never stored.",
    )
    if uploaded_file:
        st.success(f"**{uploaded_file.name}** ({uploaded_file.size / 1024:.1f} KB)")
    run_clicked = st.button("Run Analysis", type="primary", disabled=not uploaded_file)
    st.divider()

if run_clicked and uploaded_file:
    progress = st.progress(0, text="Analyzing the data")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    try:
        steps = [
            ("Cleaning data...", 15),
            ("Computing statistics...", 40),
            ("Generating charts...", 65),
            ("Generating AI insights...", 85),
        ]
        for text, pct in steps[:1]:
            progress.progress(pct, text=text)
        result = build_graph().invoke({"file_path": tmp_path})
        progress.progress(100, text="Analysis complete!")
        st.session_state["analysis_result"] = result
        st.session_state["uploaded_file_name"] = uploaded_file.name
    except Exception as exc:
        st.error(f"Pipeline failed: {exc}")
        logger.exception("Pipeline error")
    finally:
        os.unlink(tmp_path)

result = st.session_state["analysis_result"]
if result is None:
    st.markdown(
        """
        # Welcome to Data Analysis Agent
        Upload a CSV file in the sidebar and click **Run Analysis** to get
        started.
        
        """
    )
    st.stop()


file_name = st.session_state.get("uploaded_file_name", "Dataset")
st.markdown(f"# Analysis Results: `{file_name}`")
# --- Errors banner ---
if result.get("errors"):
    with st.expander("Pipeline warnings", expanded=False):
        for err in result["errors"]:
            st.warning(err)

summary = result["stats_summary"]
shape = summary.get("shape", [0, 0])
total_nulls = sum(summary.get("nulls", {}).values())
total_outliers = len(summary.get("outliers", {}))
n_correlations = len(summary.get("top_correlations", []))
cols = st.columns(5)
cols[0].metric("Rows", f"{shape[0]:,}")
cols[1].metric("Columns", shape[1])
cols[2].metric("Missing Values", total_nulls)
cols[3].metric("Outlier Columns", total_outliers)
cols[4].metric("Strong Correlations", n_correlations)
st.markdown("") 
tab_charts, tab_insights, tab_stats, tab_data = st.tabs(
    ["Charts", "AI Insights", "Statistics", "Data Preview"]
)

with tab_charts:
    charts = result.get("charts", [])
    if not charts:
        st.info("No charts were generated for this dataset.")
    else:
        st.markdown(f"**{len(charts)} chart(s)** selected based on your data characteristics")
        for i in range(0, len(charts), 2):
            row = st.columns(2)
            row[0].plotly_chart(charts[i], use_container_width=True)
            if i + 1 < len(charts):
                row[1].plotly_chart(charts[i + 1], use_container_width=True)
# ---- Insights tab ----
with tab_insights:
    insights = result.get("insights", {})
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            '<div class="insight-card findings">'
            "<h4>Key Findings</h4>",
            unsafe_allow_html=True,
        )
        for item in insights.get("key_findings", []):
            st.markdown(f"- {item}")
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown(
            '<div class="insight-card anomalies">'
            "<h4>Anomalies</h4>",
            unsafe_allow_html=True,
        )
        for item in insights.get("anomalies", []):
            st.markdown(f"- {item}")
        st.markdown("</div>", unsafe_allow_html=True)
    with col3:
        st.markdown(
            '<div class="insight-card recommendations">'
            "<h4>Recommendations</h4>",
            unsafe_allow_html=True,
        )
        for item in insights.get("recommendations", []):
            st.markdown(f"- {item}")
        st.markdown("</div>", unsafe_allow_html=True)
# ---- Statistics tab ----
with tab_stats:
    describe = summary.get("describe", {})
    if describe:
        st.markdown("#### Descriptive Statistics")
        st.dataframe(
            pd.DataFrame(describe).T,
            use_container_width=True,
        )
    # Outliers
    outliers = summary.get("outliers", {})
    if outliers:
        st.markdown("#### Outlier Summary")
        outlier_data = {
            col: info["count"] for col, info in outliers.items()
        }
        st.dataframe(
            pd.DataFrame.from_dict(
                outlier_data, orient="index", columns=["Outlier Count"]
            ),
            use_container_width=True,
        )
    # Correlations
    top_corr = summary.get("top_correlations", [])
    if top_corr:
        st.markdown("#### Top Correlations (|r| > 0.5)")
        corr_df = pd.DataFrame(top_corr, columns=["Column A", "Column B", "r"])
        st.dataframe(corr_df, use_container_width=True)
    # Column types
    dtypes = summary.get("dtypes", {})
    if dtypes:
        st.markdown("#### Column Types")
        st.dataframe(
            pd.DataFrame.from_dict(dtypes, orient="index", columns=["Type"]),
            use_container_width=True,
        )
# ---- Data preview tab ----
with tab_data:
    raw = result.get("raw_df")
    clean = result.get("clean_df")
    if raw is not None:
        st.markdown("#### Raw Data (first 100 rows)")
        st.dataframe(raw.head(100), use_container_width=True)
        buf = io.BytesIO()
        raw.to_csv(buf, index=False)
        st.download_button(
            "Download raw data as CSV",
            data=buf.getvalue(),
            file_name="raw_data.csv",
            mime="text/csv",
        )
    if clean is not None:
        st.markdown("#### Cleaned Data (first 100 rows)")
        st.dataframe(clean.head(100), use_container_width=True)
        buf = io.BytesIO()
        clean.to_csv(buf, index=False)
        st.download_button(
            "Download cleaned data as CSV",
            data=buf.getvalue(),
            file_name="cleaned_data.csv",
            mime="text/csv",
        )
