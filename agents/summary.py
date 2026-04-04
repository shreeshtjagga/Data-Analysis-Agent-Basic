"""Summary agent — produces a concise human-readable dataset summary."""
import logging

import numpy as np
import pandas as pd

from core.state import AnalysisState

logger = logging.getLogger(__name__)


def _str_cols(df: pd.DataFrame) -> list:
    """Return string/object column names — compatible with pandas 2 and 3."""
    return [
        col for col in df.columns
        if pd.api.types.is_string_dtype(df[col])
        and not pd.api.types.is_bool_dtype(df[col])
        and not pd.api.types.is_numeric_dtype(df[col])
    ]


def _safe_float(v) -> float:
    """Convert any numeric type to plain Python float."""
    try:
        return float(v)
    except Exception:
        return 0.0


def _safe_int(v) -> int:
    """Convert any numeric type to plain Python int."""
    try:
        return int(v)
    except Exception:
        return 0


def run(state: AnalysisState) -> AnalysisState:
    """Build a structured summary dict from the stats for display in the UI."""
    logger.info("Summary starting")

    if state.clean_df is None:
        state.errors.append("summary skipped: no clean_df available")
        return state

    try:
        df = state.clean_df
        stats = state.stats_summary

        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = _str_cols(df)
        date_cols = df.select_dtypes(include="datetime").columns.tolist()

        rows, cols = df.shape
        rows = _safe_int(rows)
        cols = _safe_int(cols)

        # --- data health score (0–100) ---
        raw_df = state.raw_df
        missing_rate = _safe_float(raw_df.isnull().mean().mean()) if raw_df is not None else 0.0
        dup_rate = _safe_float(1 - rows / len(raw_df)) if (raw_df is not None and len(raw_df) > 0) else 0.0
        outlier_penalty = min(len(stats.get("outliers", {})) * 5, 30)
        health_score = max(0, _safe_int(100 - missing_rate * 40 - dup_rate * 30 - outlier_penalty))

        # --- numeric highlights (all native Python types) ---
        highlights = []
        describe = stats.get("describe", {})
        for col in numeric_cols[:4]:
            if col in describe:
                d = describe[col]
                highlights.append({
                    "column": col,
                    "mean": round(_safe_float(d.get("mean", 0)), 2),
                    "min":  round(_safe_float(d.get("min", 0)), 2),
                    "max":  round(_safe_float(d.get("max", 0)), 2),
                    "std":  round(_safe_float(d.get("std", 0)), 2),
                })

        # --- top categories ---
        top_categories = {}
        for col in cat_cols[:3]:
            counts = df[col].value_counts()
            top_categories[col] = {
                "unique":    _safe_int(counts.shape[0]),
                "top_value": str(counts.index[0]) if len(counts) else "N/A",
                "top_pct":   round(_safe_float(counts.iloc[0]) / rows * 100, 1) if len(counts) else 0.0,
            }

        # --- date range ---
        date_range = None
        if date_cols:
            dc = df[date_cols[0]]
            date_range = {
                "column":    date_cols[0],
                "from":      str(dc.min().date()),
                "to":        str(dc.max().date()),
                "span_days": _safe_int((dc.max() - dc.min()).days),
            }

        state.summary = {
            "rows":               rows,
            "cols":               cols,
            "numeric_cols":       numeric_cols,
            "cat_cols":           cat_cols,
            "date_cols":          date_cols,
            "missing_rate_pct":   round(missing_rate * 100, 1),
            "duplicate_rate_pct": round(dup_rate * 100, 1),
            "health_score":       health_score,
            "highlights":         highlights,
            "top_categories":     top_categories,
            "date_range":         date_range,
            "top_correlations":   stats.get("top_correlations", []),
        }

        logger.info("Summary done — health score: %d", health_score)

    except Exception as exc:
        state.errors.append(f"summary error: {exc}")
        logger.exception("Summary error")

    return state