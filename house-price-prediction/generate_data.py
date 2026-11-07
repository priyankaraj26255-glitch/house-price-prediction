"""Generate a small synthetic housing dataset -> data/housing.csv"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 1000

df = pd.DataFrame({
    "area_sqft": rng.integers(500, 4000, n),
    "bedrooms": rng.integers(1, 6, n),
    "bathrooms": rng.integers(1, 4, n),
    "age_years": rng.integers(0, 50, n),
    "distance_to_city_km": rng.uniform(1, 40, n).round(1),
    "city": rng.choice(["Nagpur", "Pune", "Mumbai"], n),
})

city_bonus = df["city"].map({"Nagpur": 0, "Pune": 1_500_000, "Mumbai": 4_000_000})
df["price"] = (
    df["area_sqft"] * 4500
    + df["bedrooms"] * 250_000
    + df["bathrooms"] * 200_000
    - df["age_years"] * 30_000
    - df["distance_to_city_km"] * 20_000
    + city_bonus
    + rng.normal(0, 400_000, n)
).round(0)

# add a few missing values so preprocessing is meaningful
for col in ["area_sqft", "age_years"]:
    df.loc[rng.choice(n, 20, replace=False), col] = np.nan

df.to_csv("data/housing.csv", index=False)
print("Saved data/housing.csv", df.shape)
