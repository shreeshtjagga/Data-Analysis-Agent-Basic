import json
import logging
import os
from dotenv import load_dotenv
from groq import Groq
from core.state import AnalysisState
load_dotenv()
logger = logging.getLogger(__name__)
def _get_api_key() -> str:
    try:
        import streamlit as st
        key = st.secrets.get("GROQ_API_KEY")
        if key:
            return key
    except Exception:
        pass
    return os.getenv("GROQ_API_KEY", "")
def run(state: AnalysisState) -> AnalysisState:
    """Generate AI-powered insights from the statistical summary."""
    logger.info("Insights starting")
    try:
        api_key = _get_api_key()
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not found. Set it in .env or Streamlit secrets."
            )
        client = Groq(api_key=api_key)
        prompt = f"""
You are a senior data analyst. Analyze the statistics below and return ONLY
a raw JSON object -- no explanation, no markdown, no code fences.
Statistics:
{json.dumps(state.stats_summary, indent=2)}
Required format:
{{
  "key_findings": ["finding 1", "finding 2", "finding 3"],
  "anomalies": ["anomaly 1", "anomaly 2"],
  "recommendations": ["recommendation 1", "recommendation 2"]
}}
"""
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.choices[0].message.content.strip()
        # Strip markdown fences if the model wraps them
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
        if raw.endswith("```"):
            raw = raw.rsplit("```", 1)[0]
        raw = raw.strip()
        state.insights = json.loads(raw)
        logger.info("Insights done")
    except Exception as e:
        state.errors.append(f"insights error: {str(e)}")
        state.insights = {
            "key_findings": ["Could not generate insights"],
            "anomalies": [],
            "recommendations": [],
        }
        logger.exception("Insights error")
    return state