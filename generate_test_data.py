"""Generates test_data.csv for local development and testing."""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

df = pd.DataFrame(
    {
        "date": pd.date_range("2024-01-01", periods=100, freq="D"),
        "region": rng.choice(["North", "South", "East", "West"], 100),
        "revenue": rng.integers(1_000, 50_000, 100).astype(float),
        "age": rng.integers(18, 65, 100).astype(float),
        "units_sold": rng.integers(1, 200, 100),
    }
)

# Inject missing values
df.loc[5, "revenue"] = None
df.loc[20, "age"] = None

# Inject an extreme outlier
df.loc[50, "revenue"] = 999_999

df.to_csv("test_data.csv", index=False)
print("✅ test_data.csv created")