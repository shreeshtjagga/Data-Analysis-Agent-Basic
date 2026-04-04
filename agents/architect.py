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
    """Load CSV, fix column types, impute missing values, drop duplicates."""
    logger.info("Architect starting")
    try:
        df = pd.read_csv(state.file_path)
        state.raw_df = df.copy()

        # --- numeric coercion ---
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors="raise")
            except (ValueError, TypeError):
                pass

        # --- datetime detection on remaining string cols ---
        for col in _str_cols(df):
            try:
                df[col] = pd.to_datetime(df[col], format="mixed", dayfirst=False)
            except Exception:
                pass

        # --- impute missing values ---
        for col in df.columns:
            if df[col].isnull().sum() == 0:
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = df[col].ffill().bfill()
            else:
                mode_vals = df[col].mode()
                df[col] = df[col].fillna(mode_vals[0] if len(mode_vals) else "Unknown")

        df = df.drop_duplicates()
        state.clean_df = df
        logger.info("Architect done — shape: %s", df.shape)

    except Exception as exc:
        state.errors.append(f"architect error: {exc}")
        logger.exception("Architect error")

    return state