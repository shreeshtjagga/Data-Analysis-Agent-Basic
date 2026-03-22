from core.state import AnalysisState
import pandas as pd
import numpy as np

def run(state: AnalysisState) -> AnalysisState:
    print(f"[architect] starting — file: {state.file_path}")
    try:
        df = pd.read_csv(state.file_path)
        state.raw_df = df.copy()

        for col in df.columns:
            converted = pd.to_numeric(df[col], errors='ignore')
            if converted.dtype != df[col].dtype:
                df[col] = converted
            if df[col].dtype == object:
                try:
                    df[col] = pd.to_datetime(df[col])
                except:
                    pass

        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if df[col].dtype in [np.float64, np.int64]:
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(df[col].mode()[0])

        df = df.drop_duplicates()
        state.clean_df = df
        print(f"[architect] done — shape: {df.shape}")

    except Exception as e:
        state.errors.append(f"architect error: {str(e)}")
        print(f"[architect] error: {e}")

    return state


if __name__ == "__main__":
    state = AnalysisState(file_path="test_data.csv")
    result = run(state)
    print("Shape:", result.clean_df.shape)
    print("Columns:", result.clean_df.columns.tolist())
    print("Nulls:", result.clean_df.isnull().sum().to_dict())
    print("Errors:", result.errors)