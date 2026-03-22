import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import plotly.express as px
from core.state import AnalysisState


def run(state: AnalysisState) -> AnalysisState:
    print("[visualizer] starting")
    try:
        df = state.clean_df.copy()
        charts = []

        # try to convert any string columns that look like dates
        for col in df.select_dtypes(include="object").columns:
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

        # try to convert any string columns that look like numbers
        for col in df.select_dtypes(include="object").columns:
            try:
                df[col] = pd.to_numeric(df[col])
            except Exception:
                pass

        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df.select_dtypes(include="object").columns.tolist()
        date_cols = df.select_dtypes(include="datetime").columns.tolist()

        print(f"[visualizer] numeric: {numeric_cols}")
        print(f"[visualizer] categorical: {cat_cols}")
        print(f"[visualizer] datetime: {date_cols}")

        # bar chart
        if cat_cols and numeric_cols:
            grouped = df.groupby(cat_cols[0])[numeric_cols[0]].sum().reset_index()
            charts.append(px.bar(
                grouped,
                x=cat_cols[0],
                y=numeric_cols[0],
                title=f"{numeric_cols[0]} by {cat_cols[0]}"
            ))

        # line chart
        if date_cols and numeric_cols:
            charts.append(px.line(
                df.sort_values(date_cols[0]),
                x=date_cols[0],
                y=numeric_cols[0],
                title=f"{numeric_cols[0]} over time"
            ))

        # histogram for every numeric column
        for col in numeric_cols[:3]:
            charts.append(px.histogram(
                df,
                x=col,
                title=f"Distribution of {col}"
            ))

        # correlation heatmap
        if len(numeric_cols) >= 2:
            charts.append(px.imshow(
                df[numeric_cols].corr().round(2),
                text_auto=True,
                title="Correlation heatmap"
            ))

        # if still no charts, force a bar on first two columns
        if not charts and len(df.columns) >= 2:
            charts.append(px.bar(
                df,
                x=df.columns[0],
                y=df.columns[1],
                title=f"{df.columns[1]} by {df.columns[0]}"
            ))

        state.charts = charts
        print(f"[visualizer] done — {len(charts)} charts generated")

    except Exception as e:
        state.errors.append(f"visualizer error: {str(e)}")
        print(f"[visualizer] error: {e}")

    return state


if __name__ == "__main__":
    state = AnalysisState(file_path="test_data.csv")
    state.clean_df = pd.read_csv("test_data.csv", parse_dates=["date"])
    state.stats_summary = {"shape": [100, 5]}
    result = run(state)
    if result.errors:
        print("Errors:", result.errors)
    else:
        print("Charts generated:", len(result.charts))
        result.charts[0].show()