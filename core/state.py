from pydantic import BaseModel, ConfigDict
from typing import Any


class AnalysisState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    file_path: str
    raw_df: Any = None
    clean_df: Any = None
    stats_summary: dict = {}
    charts: list = []
    insights: dict = {}
    summary: dict = {}
    errors: list = []
    current_step: str = "start"