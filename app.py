"""
AI Data Analyst · Streamlit frontend
Pipeline: Architect → Statistician → Visualizer → Summary → Insights
"""
import io
import logging
import os
import tempfile

import pandas as pd
import streamlit as st

from core.graph import build_graph

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)-8s %(message)s")
logger = logging.getLogger(__name__)

st.set_page_config(page_title="AI Data Analyst", page_icon="📊", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300..700;1,9..40,300..700&family=DM+Mono&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.block-container { padding-top: 1.5rem; max-width: 1300px; }

[data-testid="stSidebar"] { background: linear-gradient(160deg,#0f172a,#1e293b); }
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stFileUploader * { color: #cbd5e1 !important; }

div[data-testid="stMetric"] {
    background: var(--secondary-background-color);
    border: 1px solid rgba(148,163,184,0.18);
    border-radius: 14px; padding: 18px 20px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.07);
}
div[data-testid="stMetric"] label { font-size:0.72rem !important; font-weight:700 !important; letter-spacing:0.08em; text-transform:uppercase; opacity:0.5; }
div[data-testid="stMetric"] [data-testid="stMetricValue"] { font-size:1.8rem !important; font-weight:800 !important; }

.health-pill { display:inline-flex; align-items:center; gap:8px; padding:9px 22px; border-radius:999px; font-weight:700; font-size:1rem; margin-bottom:20px; }
.hp-great { background:#dcfce7; color:#166534; }
.hp-ok    { background:#fef9c3; color:#854d0e; }
.hp-poor  { background:#fee2e2; color:#991b1b; }

.kcard {
    background: var(--secondary-background-color);
    border-left: 4px solid #6366f1;
    border-radius: 10px;
    border: 1px solid rgba(148,163,184,0.15);
    border-left: 4px solid #6366f1;
    padding: 14px 18px;
    margin-bottom: 10px;
}
.kcard-label { font-size:0.7rem; font-weight:700; letter-spacing:0.1em; text-transform:uppercase; opacity:0.45; margin:0 0 4px 0; }
.kcard-val   { font-size:1.35rem; font-weight:800; line-height:1.2; }
.kcard-sub   { font-size:0.8rem; opacity:0.55; margin-top:3px; }

.icard {
    border-radius:12px; padding:20px 22px;
    border-top: 4px solid #6366f1;
    border-left:1px solid rgba(148,163,184,0.12);
    border-right:1px solid rgba(148,163,184,0.12);
    border-bottom:1px solid rgba(148,163,184,0.12);
    background: var(--secondary-background-color);
    min-height:200px; height:100%;
}
.icard-find { border-top-color:#6366f1; }
.icard-anom { border-top-color:#f43f5e; }
.icard-rec  { border-top-color:#10b981; }
.icard h4   { margin:0 0 12px 0; font-size:0.85rem; font-weight:800; letter-spacing:0.06em; text-transform:uppercase; }
.icard ul   { padding-left:16px; margin:0; font-size:0.92rem; line-height:1.8; }
.icard li   { margin-bottom:5px; }

.sec-title  { font-size:1rem; font-weight:800; letter-spacing:0.01em; margin:0 0 12px 0; padding-bottom:5px; border-bottom:2px solid rgba(99,102,241,0.2); }

.stTabs [data-baseweb="tab"] { height:42px; padding:0 16px; font-size:0.88rem; font-weight:500; }
.stTabs [aria-selected="true"] { font-weight:800 !important; }
.stButton > button {
    width:100%; border-radius:10px; border:none; font-weight:700; font-size:0.95rem;
    background:linear-gradient(135deg,#6366f1,#8b5cf6); color:white !important;
    padding:0.55rem 1rem; transition:opacity 0.2s,transform 0.15s;
}
.stButton > button:hover { opacity:0.85; transform:translateY(-1px); }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
for k, v in {"result": None, "fname": None, "fbytes": None}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:14px 2px 10px">
        <div style="font-size:1.5rem;font-weight:900;color:#f1f5f9;letter-spacing:-0.01em">📊 AI Data Analyst</div>
        <div style="font-size:0.78rem;color:#64748b;margin-top:3px">LangGraph · Groq · Streamlit</div>
    </div>""", unsafe_allow_html=True)
    st.divider()

    f = st.file_uploader("Upload CSV Dataset", type=["csv"], help="Max 200 MB")

    if f is not None:
        fbytes = f.read()
        if st.session_state["fname"] != f.name:
            st.session_state["fbytes"] = fbytes
            st.session_state["fname"]  = f.name
            st.session_state["result"] = None
        elif st.session_state["fbytes"] is None:
            st.session_state["fbytes"] = fbytes

    if st.session_state["fname"]:
        st.success(f"**{st.session_state['fname']}**")

    st.markdown("<br>", unsafe_allow_html=True)
    go = st.button("⚡ Generate Analysis", disabled=st.session_state["fbytes"] is None)

    st.divider()
    st.markdown("""<div style="font-size:0.75rem;color:#475569;line-height:1.8">
        <b style="color:#94a3b8">Pipeline</b><br>
        🏗 Architect → 📐 Statistician<br>
        🎨 Visualizer → 📝 Summary → 💡 Insights
    </div>""", unsafe_allow_html=True)

# ── Run ───────────────────────────────────────────────────────────────────────
if go and st.session_state["fbytes"]:
    tmp = None
    with st.spinner("🤖 Analysing…"):
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as t:
                t.write(st.session_state["fbytes"])
                tmp = t.name
            st.session_state["result"] = build_graph().invoke({"file_path": tmp})
        except Exception as e:
            st.error(f"Pipeline error: {e}")
            logger.exception("Pipeline error")
        finally:
            if tmp and os.path.exists(tmp):
                os.unlink(tmp)

R = st.session_state["result"]

# ── Landing ───────────────────────────────────────────────────────────────────
if R is None:
    st.markdown("<br><br>", unsafe_allow_html=True)
    _, c, _ = st.columns([1, 2, 1])
    with c:
        st.markdown("""
        <div style="text-align:center;padding:52px 36px;background:var(--secondary-background-color);
                    border-radius:18px;border:2px dashed rgba(148,163,184,0.2)">
            <div style="font-size:3.5rem">📊</div>
            <h2 style="margin:14px 0 8px;font-weight:900">AI Data Analyst</h2>
            <p style="opacity:0.6;font-size:0.97rem;max-width:340px;margin:0 auto;line-height:1.65">
                Upload a CSV in the sidebar and click <b>Generate Analysis</b> for instant
                visualisations, statistics, and AI insights.
            </p>
        </div>""", unsafe_allow_html=True)
    st.stop()

# ── Errors ────────────────────────────────────────────────────────────────────
errs = [e for e in (R.get("errors") or []) if e]
if errs:
    with st.expander("⚠️ Processing warnings", expanded=True):
        for e in errs:
            st.warning(e)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(f"<h1 style='font-size:1.6rem;font-weight:900;margin-bottom:2px'>"
            f"Analysis: <code style='font-size:1.3rem;color:#6366f1'>{st.session_state['fname']}</code></h1>",
            unsafe_allow_html=True)

# ── Metrics ───────────────────────────────────────────────────────────────────
STS  = R.get("stats_summary") or {}
SUMM = R.get("summary") or {}
shape = STS.get("shape", [0, 0])

st.markdown("<br>", unsafe_allow_html=True)
m1,m2,m3,m4,m5,m6 = st.columns(6)
m1.metric("Rows",            f"{shape[0]:,}")
m2.metric("Columns",         shape[1])
m3.metric("Missing Values",  sum((STS.get("nulls") or {}).values()))
m4.metric("Outlier Cols",    len(STS.get("outliers") or {}))
m5.metric("Correlations",    len(STS.get("top_correlations") or []))
m6.metric("Health Score",    f"{SUMM.get('health_score','—')}/100")
st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
t1, t2, t3, t4, t5 = st.tabs(["📝 Summary","📊 Charts","💡 AI Insights","📈 Statistics","🔍 Data Preview"])

# ── TAB 1 · SUMMARY ──────────────────────────────────────────────────────────
with t1:
    if not SUMM:
        st.info("No summary data — ensure `core/state.py` has `summary: dict = {}` and redeploy.", icon="ℹ️")
    else:
        hs = SUMM.get("health_score", 0)
        hcls = "hp-great" if hs>=75 else ("hp-ok" if hs>=50 else "hp-poor")
        hico = "✅" if hs>=75 else ("⚠️" if hs>=50 else "🚨")
        hlbl = "Great" if hs>=75 else ("Fair" if hs>=50 else "Needs Attention")
        st.markdown(f'<div class="health-pill {hcls}">{hico} Data Health: {hlbl} — {hs}/100</div>',
                    unsafe_allow_html=True)

        L, R2 = st.columns(2)
        with L:
            st.markdown('<p class="sec-title">📐 Dataset Overview</p>', unsafe_allow_html=True)
            a,b,c2 = st.columns(3)
            a.metric("Rows",         f"{SUMM.get('rows',0):,}")
            b.metric("Columns",      SUMM.get("cols",0))
            c2.metric("Missing %",   f"{SUMM.get('missing_rate_pct',0)}%")
            d,e,f2 = st.columns(3)
            d.metric("Numeric",      len(SUMM.get("numeric_cols",[])))
            e.metric("Categorical",  len(SUMM.get("cat_cols",[])))
            f2.metric("Date cols",   len(SUMM.get("date_cols",[])))

            dr = SUMM.get("date_range")
            if dr:
                st.markdown(f"""<div class="kcard" style="border-left-color:#0ea5e9;margin-top:14px">
                    <p class="kcard-label">📅 Date Range — {dr['column']}</p>
                    <div class="kcard-val">{dr['from']} → {dr['to']}</div>
                    <div class="kcard-sub">Span: {dr['span_days']:,} days</div>
                </div>""", unsafe_allow_html=True)

        with R2:
            st.markdown('<p class="sec-title">🔢 Numeric Highlights</p>', unsafe_allow_html=True)
            for h in SUMM.get("highlights", []):
                st.markdown(f"""<div class="kcard">
                    <p class="kcard-label">{h['column']}</p>
                    <div class="kcard-val">μ = {h['mean']:,}</div>
                    <div class="kcard-sub">Min {h['min']:,} · Max {h['max']:,} · σ {h['std']:,}</div>
                </div>""", unsafe_allow_html=True)
            if not SUMM.get("highlights"):
                st.info("No numeric columns.")

        st.markdown("<br>", unsafe_allow_html=True)
        L2, R3 = st.columns(2)

        with L2:
            cats = SUMM.get("top_categories", {})
            if cats:
                st.markdown('<p class="sec-title">🏷️ Top Categories</p>', unsafe_allow_html=True)
                for col, info in cats.items():
                    st.markdown(f"""<div class="kcard" style="border-left-color:#f59e0b">
                        <p class="kcard-label">{col}</p>
                        <div class="kcard-val">{info['top_value']}</div>
                        <div class="kcard-sub">{info['top_pct']}% of rows · {info['unique']} unique values</div>
                    </div>""", unsafe_allow_html=True)

        with R3:
            corrs = SUMM.get("top_correlations", [])
            st.markdown('<p class="sec-title">🔗 Correlations</p>', unsafe_allow_html=True)
            if corrs:
                for a2, b2, rv in corrs:
                    strength = "Strong" if abs(rv)>=0.8 else "Moderate"
                    clr = "#10b981" if rv>0 else "#f43f5e"
                    st.markdown(f"""<div class="kcard" style="border-left-color:{clr}">
                        <p class="kcard-label">{strength} {"positive" if rv>0 else "negative"}</p>
                        <div class="kcard-val">{a2} ↔ {b2}</div>
                        <div class="kcard-sub">r = {rv:.2f}</div>
                    </div>""", unsafe_allow_html=True)
            else:
                st.info("No strong correlations (|r| > 0.5) found.")

# ── TAB 2 · CHARTS ───────────────────────────────────────────────────────────
with t2:
    charts = R.get("charts") or []
    if not charts:
        st.info("No charts generated for this dataset.", icon="ℹ️")
    else:
        for i in range(0, len(charts), 2):
            cols = st.columns(2)
            cols[0].plotly_chart(charts[i], use_container_width=True)
            if i+1 < len(charts):
                cols[1].plotly_chart(charts[i+1], use_container_width=True)

# ── TAB 3 · AI INSIGHTS ──────────────────────────────────────────────────────
with t3:
    ins = R.get("insights") or {}
    c1, c2, c3 = st.columns(3)
    for col, css, icon, title, key in [
        (c1, "icard-find", "🔎", "Key Findings",    "key_findings"),
        (c2, "icard-anom", "🚨", "Anomalies",       "anomalies"),
        (c3, "icard-rec",  "🎯", "Recommendations", "recommendations"),
    ]:
        items = ins.get(key, [])
        bullets = "".join(f"<li>{x}</li>" for x in items) if items else "<li>None detected.</li>"
        col.markdown(f"""<div class="icard {css}">
            <h4>{icon} {title}</h4><ul>{bullets}</ul>
        </div>""", unsafe_allow_html=True)

# ── TAB 4 · STATISTICS ───────────────────────────────────────────────────────
with t4:
    desc = STS.get("describe", {})
    if desc:
        st.markdown('<p class="sec-title">Descriptive Statistics</p>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(desc).T, use_container_width=True)
        st.divider()

    cl, cr = st.columns(2)
    with cl:
        outs = STS.get("outliers", {})
        if outs:
            st.markdown('<p class="sec-title">Outlier Summary</p>', unsafe_allow_html=True)
            st.dataframe(pd.DataFrame.from_dict({c: v["count"] for c,v in outs.items()},
                         orient="index", columns=["Count"]), use_container_width=True)
        dts = STS.get("dtypes", {})
        if dts:
            st.markdown('<p class="sec-title">Column Types</p>', unsafe_allow_html=True)
            st.dataframe(pd.DataFrame.from_dict(dts, orient="index", columns=["Type"]),
                         use_container_width=True)
    with cr:
        tc = STS.get("top_correlations", [])
        if tc:
            st.markdown('<p class="sec-title">Top Correlations (|r| > 0.5)</p>', unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(tc, columns=["Feature A","Feature B","r"]),
                         use_container_width=True, hide_index=True)
        cc = STS.get("category_counts", {})
        if cc:
            st.markdown('<p class="sec-title">Category Counts (top 10)</p>', unsafe_allow_html=True)
            for col, cnts in list(cc.items())[:3]:
                with st.expander(f"📌 {col}"):
                    st.dataframe(pd.DataFrame.from_dict(cnts, orient="index", columns=["Count"]),
                                 use_container_width=True)

# ── TAB 5 · DATA PREVIEW ─────────────────────────────────────────────────────
with t5:
    raw   = R.get("raw_df")
    clean = R.get("clean_df")
    cl, cr = st.columns(2)
    with cl:
        if raw is not None:
            st.markdown('<p class="sec-title">Raw Dataset</p>', unsafe_allow_html=True)
            st.caption("First 100 rows before processing")
            st.dataframe(raw.head(100), use_container_width=True)
            buf = io.BytesIO()
            raw.to_csv(buf, index=False)
            st.download_button("📥 Download Raw CSV", buf.getvalue(), "raw_data.csv", "text/csv")
    with cr:
        if clean is not None:
            st.markdown('<p class="sec-title">Cleaned Dataset</p>', unsafe_allow_html=True)
            st.caption("First 100 rows after processing")
            st.dataframe(clean.head(100), use_container_width=True)
            buf = io.BytesIO()
            clean.to_csv(buf, index=False)
            st.download_button("📥 Download Cleaned CSV", buf.getvalue(), "cleaned_data.csv", "text/csv")