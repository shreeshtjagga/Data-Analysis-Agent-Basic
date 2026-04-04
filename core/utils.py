"""Utility functions for the analysis pipeline."""
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def get_api_key() -> Optional[str]:
    """
    Retrieve GROQ API key from environment or Streamlit secrets.
    
    Returns:
        API key string or None if not found
    """
    # Try Streamlit secrets first (for deployment)
    try:
        import streamlit as st
        key = st.secrets.get("GROQ_API_KEY")
        if key:
            logger.info("API key loaded from Streamlit secrets")
            return key
    except Exception as e:
        logger.debug(f"Could not load from Streamlit secrets: {e}")
    
    # Fallback to environment variable (for local development)
    key = os.getenv("GROQ_API_KEY", "")
    if key:
        logger.info("API key loaded from environment variable")
        return key
    
    logger.warning("No API key found in secrets or environment")
    return None


def safe_json_parse(raw_text: str) -> str:
    """
    Clean and prepare LLM output for JSON parsing.
    
    Args:
        raw_text: Raw text output from LLM
        
    Returns:
        Cleaned text ready for json.loads()
    """
    text = raw_text.strip()
    
    # Remove markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json or ```)
        lines = lines[1:]
        text = "\n".join(lines)
    
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]
    
    # Remove 'json' keyword if present at start
    if text.strip().startswith("json"):
        text = text.strip()[4:]
    
    return text.strip()