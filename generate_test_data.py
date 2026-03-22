import pandas as pd
import numpy as np

df = pd.DataFrame({
    "date": pd.date_range("2024-01-01", periods=100),
    "region": np.random.choice(["North", "South", "East", "West"], 100),
    "revenue": np.random.randint(1000, 50000, 100),
    "age": np.random.randint(18, 65, 100),
    "units_sold": np.random.randint(1, 200, 100)
})

df.loc[5, "revenue"] = None
df.loc[20, "age"] = None
df.loc[50, "revenue"] = 999999

df.to_csv("test_data.csv", index=False)
print("test_data.csv created")