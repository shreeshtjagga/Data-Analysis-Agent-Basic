import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from core.state import AnalysisState


def run(state: AnalysisState) -> AnalysisState:
    print("[statistician] starting")
    try:
        df = state.clean_df
        summary = {}

        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df.select_dtypes(include="object").columns.tolist()

        summary["shape"] = list(df.shape)
        summary["columns"] = df.columns.tolist()
        summary["nulls"] = df.isnull().sum().to_dict()

        if numeric_cols:
            summary["describe"] = df[numeric_cols].describe().round(2).to_dict()

        # outlier detection via IQR
        outliers = {}
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
        print(f"[statistician] done — {len(numeric_cols)} numeric, {len(cat_cols)} categorical")

    except Exception as e:
        state.errors.append(f"statistician error: {str(e)}")
        print(f"[statistician] error: {e}")

    return state


if __name__ == "__main__":
    state = AnalysisState(file_path="test_data.csv")
    state.clean_df = pd.read_csv("test_data.csv")
    result = run(state)
    if result.errors:
        print("Errors:", result.errors)
    else:
        print("Summary keys:", list(result.stats_summary.keys()))
        print("Outliers:", result.stats_summary.get("outliers"))
        print("Correlations:", result.stats_summary.get("top_correlations"))