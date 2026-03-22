import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import tempfile
import streamlit as st
from core.graph import build_graph


st.set_page_config(page_title="Data Analysis Agent", layout="wide")
st.title("Data Analysis Agent")
st.caption("Upload a CSV and let the 4-agent pipeline analyse it")

file = st.file_uploader("Upload your CSV file", type=["csv"])

if file and st.button("Run Analysis", type="primary"):
    with st.spinner("Pipeline running..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name

        result = build_graph().invoke({"file_path": tmp_path})
        os.unlink(tmp_path)

    if result["errors"]:
        st.warning("Errors: " + str(result["errors"]))

    # metric cards
    summary = result["stats_summary"]
    shape = summary.get("shape", [0, 0])
    total_nulls = sum(summary.get("nulls", {}).values())
    total_outliers = len(summary.get("outliers", {}))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Rows", f"{shape[0]:,}")
    c2.metric("Total Columns", shape[1])
    c3.metric("Missing Values", total_nulls)
    c4.metric("Outlier Columns", total_outliers)

    st.divider()

    # charts
    st.subheader("Charts")
    charts = result["charts"]
    for i in range(0, len(charts), 2):
        cols = st.columns(2)
        cols[0].plotly_chart(charts[i], use_container_width=True)
        if i + 1 < len(charts):
            cols[1].plotly_chart(charts[i + 1], use_container_width=True)

    st.divider()

    # insights
    st.subheader("Insights")
    insights = result["insights"]
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Key Findings**")
        for item in insights.get("key_findings", []):
            st.markdown(f"- {item}")
    with c2:
        st.markdown("**Anomalies**")
        for item in insights.get("anomalies", []):
            st.markdown(f"- {item}")
    with c3:
        st.markdown("**Recommendations**")
        for item in insights.get("recommendations", []):
            st.markdown(f"- {item}")