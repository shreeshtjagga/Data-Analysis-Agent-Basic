"""Insights agent — generates AI-powered findings via Groq."""
import json
import logging
import os

from core.state import AnalysisState

logger = logging.getLogger(__name__)


def _get_api_key() -> str:
    """Resolve GROQ_API_KEY from Streamlit secrets or environment."""
    try:
        import streamlit as st
        key = st.secrets.get("GROQ_API_KEY", "")
        if key:
            return key
    except Exception:
        pass
    return os.getenv("GROQ_API_KEY", "")


def run(state: AnalysisState) -> AnalysisState:
    """Generate AI-powered insights from the statistical summary."""
    logger.info("Insights starting")
    try:
        from groq import Groq  # lazy import to avoid cold-start cost

        api_key = _get_api_key()
        if not api_key:
            raise ValueError("GROQ_API_KEY not found. Add it to .streamlit/secrets.toml or .env.")

        client = Groq(api_key=api_key)

        # Trim stats payload to avoid token bloat
        slim_stats = {
            k: v for k, v in state.stats_summary.items()
            if k in ("shape", "describe", "outliers", "top_correlations", "category_counts")
        }

        prompt = f"""You are a senior data analyst. Given the dataset statistics below, return ONLY
a valid JSON object with no markdown, no explanation, no code fences.

Statistics:
{json.dumps(slim_stats, indent=2)}

Required JSON format:
{{
  "key_findings": ["<concise finding 1>", "<concise finding 2>", "<concise finding 3>"],
  "anomalies": ["<anomaly 1>", "<anomaly 2>"],
  "recommendations": ["<actionable recommendation 1>", "<actionable recommendation 2>"]
}}
"""
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            max_tokens=800,
        )

        raw = response.choices[0].message.content.strip()

        # Defensive cleanup in case model wraps output anyway
        for fence in ("```json", "```", "json"):
            if raw.startswith(fence):
                raw = raw[len(fence):]
            if raw.endswith(fence):
                raw = raw[: -len(fence)]
        raw = raw.strip()

        state.insights = json.loads(raw)
        logger.info("Insights done")

    except Exception as exc:
        state.errors.append(f"insights error: {exc}")
        state.insights = {
            "key_findings": ["Could not generate AI insights — check your GROQ_API_KEY."],
            "anomalies": [],
            "recommendations": ["Verify the API key and retry."],
        }
        logger.exception("Insights error")

    return state