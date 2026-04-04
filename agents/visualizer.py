"""Visualizer agent — selects the most relevant charts for the dataset."""
import logging
from typing import List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from core.state import AnalysisState

logger = logging.getLogger(__name__)

COLOR_PALETTE = px.colors.qualitative.Set2
TEMPLATE = "plotly_white"
MAX_CHARTS = 6
MAX_BAR_CATEGORIES = 15
MAX_PIE_CATEGORIES = 6


def _str_cols(df: pd.DataFrame) -> list:
    """Return string/object column names — compatible with pandas 2 and 3."""
    return [
        col for col in df.columns
        if pd.api.types.is_string_dtype(df[col])
        and not pd.api.types.is_bool_dtype(df[col])
        and not pd.api.types.is_numeric_dtype(df[col])
    ]


def _style(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        template=TEMPLATE,
        font=dict(family="'DM Sans', sans-serif", size=12),
        title_font_size=14,
        title_font_color="#1e293b",
        margin=dict(l=40, r=40, t=55, b=40),
        colorway=COLOR_PALETTE,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(font=dict(size=11)),
    )
    return fig


def _pick_charts(df: pd.DataFrame, stats: dict) -> List[go.Figure]:
    charts: List[go.Figure] = []

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = _str_cols(df)
    date_cols = df.select_dtypes(include="datetime").columns.tolist()

    logger.info("Cols — numeric: %s | cat: %s | date: %s", numeric_cols, cat_cols, date_cols)

    # 1. Time-series line chart
    if date_cols and numeric_cols:
        fig = px.line(
            df.sort_values(date_cols[0]),
            x=date_cols[0], y=numeric_cols[0],
            title=f"{numeric_cols[0]} Over Time",
        )
        charts.append(_style(fig))

    # 2. Bar chart: category vs numeric mean
    if cat_cols and numeric_cols:
        cat_col, num_col = cat_cols[0], numeric_cols[0]
        if df[cat_col].nunique() <= MAX_BAR_CATEGORIES:
            grouped = (
                df.groupby(cat_col)[num_col].mean()
                .reset_index().sort_values(num_col, ascending=False)
            )
            fig = px.bar(
                grouped, x=cat_col, y=num_col,
                title=f"Avg {num_col} by {cat_col}",
                color=cat_col, color_discrete_sequence=COLOR_PALETTE,
            )
            charts.append(_style(fig))

    # 3. Donut chart for first low-cardinality cat col
    for col in cat_cols:
        if df[col].nunique() <= MAX_PIE_CATEGORIES:
            counts = df[col].value_counts().reset_index()
            counts.columns = [col, "count"]
            fig = px.pie(
                counts, names=col, values="count",
                title=f"Distribution of {col}",
                hole=0.45, color_discrete_sequence=COLOR_PALETTE,
            )
            fig.update_traces(textposition="outside", textinfo="percent+label")
            charts.append(_style(fig))
            break

    # 4. Scatter for highest-correlation pair
    top_corr = stats.get("top_correlations", [])
    if top_corr:
        pair = max(top_corr, key=lambda x: abs(x[2]))
        x_col, y_col, corr_val = pair
        if x_col in df.columns and y_col in df.columns:
            color_arg = cat_cols[0] if cat_cols else None
            fig = px.scatter(
                df, x=x_col, y=y_col, color=color_arg,
                title=f"{x_col} vs {y_col}  (r = {corr_val:.2f})",
                trendline="ols" if len(df) > 5 else None,
                opacity=0.75, color_discrete_sequence=COLOR_PALETTE,
            )
            charts.append(_style(fig))

    # 5. Histogram for outlier col or first numeric
    outlier_cols = list(stats.get("outliers", {}).keys())
    hist_col = outlier_cols[0] if outlier_cols else (numeric_cols[0] if numeric_cols else None)
    if hist_col and hist_col in df.columns:
        label = f"Distribution of {hist_col}" + (" (outliers detected)" if outlier_cols else "")
        fig = px.histogram(
            df, x=hist_col, nbins=30, title=label, marginal="box",
            color_discrete_sequence=[COLOR_PALETTE[0]],
        )
        charts.append(_style(fig))

    # 6. Correlation heatmap
    if len(numeric_cols) >= 3 and top_corr:
        corr_matrix = df[numeric_cols].corr().round(2)
        fig = px.imshow(
            corr_matrix, text_auto=True, title="Correlation Heatmap",
            color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto",
        )
        charts.append(_style(fig))

    # Fallback
    if not charts and len(df.columns) >= 2:
        fig = px.bar(
            df.head(20), x=df.columns[0], y=df.columns[1],
            title=f"{df.columns[1]} by {df.columns[0]}",
        )
        charts.append(_style(fig))

    return charts[:MAX_CHARTS]


def run(state: AnalysisState) -> AnalysisState:
    """Generate relevant charts based on data characteristics."""
    logger.info("Visualizer starting")

    if state.clean_df is None:
        state.errors.append("visualizer skipped: no clean_df available")
        return state

    try:
        df = state.clean_df.copy()

        # Coerce string cols to datetime where possible
        for col in _str_cols(df):
            try:
                df[col] = pd.to_datetime(df[col], format="mixed", dayfirst=False)
            except Exception:
                pass

        # Coerce remaining string cols to numeric where possible
        for col in _str_cols(df):
            try:
                df[col] = pd.to_numeric(df[col], errors="raise")
            except (ValueError, TypeError):
                pass

        state.charts = _pick_charts(df, state.stats_summary)
        logger.info("Visualizer done — %d charts", len(state.charts))

    except Exception as exc:
        state.errors.append(f"visualizer error: {exc}")
        logger.exception("Visualizer error")

    return state