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


def run(state: AnalysisState) -> AnalysisState:
    """Compute descriptive stats, outliers, correlations, and category counts."""
    logger.info("Statistician starting")

    if state.clean_df is None:
        state.errors.append("statistician skipped: no clean_df available")
        return state

    try:
        df = state.clean_df
        summary: dict = {}

        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = _str_cols(df)

        summary["shape"] = list(df.shape)
        summary["columns"] = df.columns.tolist()
        summary["dtypes"] = {col: str(df[col].dtype) for col in df.columns}
        summary["nulls"] = df.isnull().sum().to_dict()

        if numeric_cols:
            summary["describe"] = df[numeric_cols].describe().round(2).to_dict()

        # --- IQR outlier detection ---
        outliers: dict = {}
        for col in numeric_cols:
            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr = q3 - q1
            mask = (df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)
            flagged = df[mask].index.tolist()
            if flagged:
                outliers[col] = {"count": len(flagged), "rows": flagged[:5]}
        summary["outliers"] = outliers

        # --- top correlations (|r| > 0.5) ---
        top_corr: list = []
        if len(numeric_cols) >= 2:
            corr = df[numeric_cols].corr().round(2)
            for i in range(len(corr.columns)):
                for j in range(i + 1, len(corr.columns)):
                    val = float(corr.iloc[i, j])
                    if abs(val) > 0.5:
                        top_corr.append([corr.columns[i], corr.columns[j], val])
        summary["top_correlations"] = top_corr

        # --- category value counts ---
        summary["category_counts"] = {
            col: df[col].value_counts().head(10).to_dict() for col in cat_cols
        }

        state.stats_summary = summary
        logger.info(
            "Statistician done — %d numeric, %d categorical",
            len(numeric_cols), len(cat_cols),
        )

    except Exception as exc:
        state.errors.append(f"statistician error: {exc}")
        logger.exception("Statistician error")

    return state