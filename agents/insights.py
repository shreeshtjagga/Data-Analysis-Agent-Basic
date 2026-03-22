import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import anthropic
from dotenv import load_dotenv
from core.state import AnalysisState

load_dotenv()


def run(state: AnalysisState) -> AnalysisState:
    print("[insights] starting")
    try:
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

        prompt = f"""
You are a senior data analyst. Analyze the statistics below and return ONLY
a raw JSON object — no explanation, no markdown, no code fences.

Statistics:
{json.dumps(state.stats_summary, indent=2)}

Required format:
{{
  "key_findings": ["finding 1", "finding 2", "finding 3"],
  "anomalies": ["anomaly 1", "anomaly 2"],
  "recommendations": ["recommendation 1", "recommendation 2"]
}}
"""
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )

        state.insights = json.loads(response.content[0].text.strip())
        print("[insights] done")

    except Exception as e:
        state.errors.append(f"insights error: {str(e)}")
        state.insights = {
            "key_findings": ["Could not generate insights"],
            "anomalies": [],
            "recommendations": []
        }
        print(f"[insights] error: {e}")

    return state


if __name__ == "__main__":
    state = AnalysisState(file_path="test_data.csv")
    state.stats_summary = {
        "shape": [100, 5],
        "columns": ["date", "region", "revenue", "age", "units_sold"],
        "nulls": {"revenue": 0, "age": 1},
        "top_correlations": [["age", "revenue", 0.83]],
        "outliers": {"revenue": {"count": 1, "rows": [50]}}
    }
    result = run(state)
    if result.errors:
        print("Errors:", result.errors)
    else:
        print(json.dumps(result.insights, indent=2))