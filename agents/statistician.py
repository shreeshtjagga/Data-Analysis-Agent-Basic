import logging
import numpy as np
import pandas as pd
from core.state import AnalysisState
logger = logging.getLogger(__name__)
def run(state: AnalysisState) -> AnalysisState:
    logger.info("Statistician starting")
    try:
        df = state.clean_df
        summary: dict = {}
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df.select_dtypes(include="object").columns.tolist()
        summary["shape"] = list(df.shape)
        summary["columns"] = df.columns.tolist()
        summary["dtypes"] = {col: str(df[col].dtype) for col in df.columns}
        summary["nulls"] = df.isnull().sum().to_dict()
        if numeric_cols:
            summary["describe"] = df[numeric_cols].describe().round(2).to_dict()
        # outlier detection via IQR
        outliers: dict = {}
        for col in numeric_cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            mask = (df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)
            flagged = df[mask].index.tolist()
            if flagged:
                outliers[col] = {"count": len(flagged), "rows": flagged[:5]}
        summary["outliers"] = outliers
        # top correlations above 0.5
        if len(numeric_cols) >= 2:
            corr = df[numeric_cols].corr().round(2)
            top_corr = []
            for i in range(len(corr.columns)):
                for j in range(i + 1, len(corr.columns)):
                    val = corr.iloc[i, j]
                    if abs(val) > 0.5:
                        top_corr.append([corr.columns[i], corr.columns[j], val])
            summary["top_correlations"] = top_corr
        # category value counts
        summary["category_counts"] = {
            col: df[col].value_counts().head(10).to_dict()
            for col in cat_cols
        }
        state.stats_summary = summary
        logger.info(
            "Statistician done -- %d numeric, %d categorical",
            len(numeric_cols), len(cat_cols),
        )
    except Exception as e:
        state.errors.append(f"statistician error: {str(e)}")
        logger.exception("Statistician error")
    return state