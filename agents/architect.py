import logging
import numpy as np
import pandas as pd
from core.state import AnalysisState
logger = logging.getLogger(__name__)
def run(state: AnalysisState) -> AnalysisState:
    logger.info("Architect starting")
    try:
        df = pd.read_csv(state.file_path)
        state.raw_df = df.copy()
        # fix column types
        for col in df.columns:
            converted = pd.to_numeric(df[col], errors="ignore")
            if converted.dtype != df[col].dtype:
                df[col] = converted
            if df[col].dtype == object:
                try:
                    df[col] = pd.to_datetime(df[col])
                except Exception:
                    pass
        # fill missing values
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if df[col].dtype in [np.float64, np.int64]:
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(df[col].mode()[0])
        df = df.drop_duplicates()
        state.clean_df = df
        logger.info("Architect done -- shape: %s", df.shape)
    except Exception as e:
        state.errors.append(f"architect error: {str(e)}")
        logger.exception("Architect error")
    return state
