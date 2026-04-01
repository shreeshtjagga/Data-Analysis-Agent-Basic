"""Visualizer agent -- selects only the most relevant charts for the dataset."""

import logging
from typing import List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from core.state import AnalysisState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Consistent styling
# ---------------------------------------------------------------------------
COLOR_PALETTE = px.colors.qualitative.Set2
TEMPLATE = "plotly_white"

MAX_CHARTS = 5
MAX_BAR_CATEGORIES = 15
MAX_PIE_CATEGORIES = 6


def _style(fig: go.Figure) -> go.Figure:
    """Apply consistent styling to every chart."""
    fig.update_layout(
        template=TEMPLATE,
        font=dict(family="Inter, sans-serif", size=12),
        title_font_size=15,
        margin=dict(l=40, r=40, t=50, b=40),
        colorway=COLOR_PALETTE,
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _pick_charts(df: pd.DataFrame, stats: dict) -> List[go.Figure]:
    """Select the most informative charts based on data characteristics."""

    charts: List[go.Figure] = []
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    date_cols = df.select_dtypes(include="datetime").columns.tolist()

    logger.info(
        "Columns -- numeric: %s, categorical: %s, datetime: %s",
        numeric_cols, cat_cols, date_cols,
    )

    # 1. Time-series line chart (highest priority when dates exist)
    if date_cols and numeric_cols:
        date_col = date_cols[0]
        y_col = numeric_cols[0]
        sorted_df = df.sort_values(date_col)
        fig = px.line(
            sorted_df, x=date_col, y=y_col,
            title=f"{y_col} Over Time",
        )
        charts.append(_style(fig))

    # 2. Bar chart -- only when categorical has reasonable cardinality
    if cat_cols and numeric_cols:
        cat_col = cat_cols[0]
        if df[cat_col].nunique() <= MAX_BAR_CATEGORIES:
            num_col = numeric_cols[0]
            grouped = (
                df.groupby(cat_col)[num_col]
                .mean()
                .reset_index()
                .sort_values(num_col, ascending=False)
            )
            fig = px.bar(
                grouped, x=cat_col, y=num_col,
                title=f"Average {num_col} by {cat_col}",
                color=cat_col,
            )
            charts.append(_style(fig))

    # 3. Pie / donut -- only for low-cardinality categorical columns
    if cat_cols:
        for col in cat_cols:
            if df[col].nunique() <= MAX_PIE_CATEGORIES:
                counts = df[col].value_counts().reset_index()
                counts.columns = [col, "count"]
                fig = px.pie(
                    counts, names=col, values="count",
                    title=f"Distribution of {col}",
                    hole=0.4,
                )
                charts.append(_style(fig))
                break  # one pie chart is enough

    # 4. Scatter plot for the strongest correlated numeric pair
    top_corr = stats.get("top_correlations", [])
    if top_corr:
        pair = max(top_corr, key=lambda x: abs(x[2]))
        x_col, y_col, corr_val = pair[0], pair[1], pair[2]
        if x_col in df.columns and y_col in df.columns:
            fig = px.scatter(
                df, x=x_col, y=y_col,
                title=f"{x_col} vs {y_col} (r = {corr_val})",
                trendline="ols" if len(df) > 5 else None,
                opacity=0.7,
            )
            charts.append(_style(fig))

    # 5. Histogram -- only for columns that have outliers (or fallback)
    outlier_cols = list(stats.get("outliers", {}).keys())
    if outlier_cols:
        col = outlier_cols[0]
        if col in df.columns:
            fig = px.histogram(
                df, x=col, nbins=30,
                title=f"Distribution of {col} (has outliers)",
                marginal="box",
            )
            charts.append(_style(fig))
    elif numeric_cols and len(charts) < 3:
        col = numeric_cols[0]
        fig = px.histogram(
            df, x=col, nbins=30,
            title=f"Distribution of {col}",
            marginal="box",
        )
        charts.append(_style(fig))

    # 6. Correlation heatmap -- only when meaningful correlations exist
    if len(numeric_cols) >= 2 and top_corr:
        corr_matrix = df[numeric_cols].corr().round(2)
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            title="Correlation Heatmap",
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
        )
        charts.append(_style(fig))

    # Fallback -- guarantee at least one chart
    if not charts and len(df.columns) >= 2:
        fig = px.bar(
            df.head(20), x=df.columns[0], y=df.columns[1],
            title=f"{df.columns[1]} by {df.columns[0]} (first 20 rows)",
        )
        charts.append(_style(fig))

    return charts[:MAX_CHARTS]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(state: AnalysisState) -> AnalysisState:
    """Generate relevant charts based on data characteristics."""
    logger.info("Visualizer starting")
    try:
        df = state.clean_df.copy()

        # Auto-detect date columns stored as strings
        for col in df.select_dtypes(include="object").columns:
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

        # Auto-detect numeric columns stored as strings
        for col in df.select_dtypes(include="object").columns:
            try:
                df[col] = pd.to_numeric(df[col])
            except Exception:
                pass

        charts = _pick_charts(df, state.stats_summary)
        state.charts = charts
        logger.info("Visualizer done -- %d charts generated", len(charts))

    except Exception as e:
        state.errors.append(f"visualizer error: {str(e)}")
        logger.exception("Visualizer error")

    return state
