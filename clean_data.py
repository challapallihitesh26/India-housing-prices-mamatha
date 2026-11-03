"""
clean_data.py  -  Data cleaning + feature engineering
Run:  python clean_data.py
In :  india_housing_prices.csv
Out:  cleaned_housing_data.csv
"""
import numpy as np
import pandas as pd

RAW, OUT = "india_housing_prices.csv", "cleaned_housing_data.csv"
CURRENT_YEAR = 2025          # dataset: Age_of_Property = 2025 - Year_Built

df = pd.read_csv(RAW)
print(f"Raw shape: {df.shape}")

# 1. Tidy text columns -----------------------------------------------------
for c in df.select_dtypes(include=["object", "string"]).columns:
    df[c] = df[c].astype(str).str.strip()
df = df.replace({"": np.nan, "nan": np.nan, "None": np.nan, "NA": np.nan})

# 2. Duplicates ------------------------------------------------------------
n0 = len(df)
df = df.drop_duplicates()
df = df.drop_duplicates(subset=[c for c in df.columns if c != "ID"])
df = df.drop_duplicates(subset="ID")
print(f"Duplicates removed: {n0 - len(df)}")

# 3. Missing values (median for numbers, mode for categories) ---------------
print(f"Missing values before: {int(df.isna().sum().sum())}")
for c in df.columns:
    if df[c].isna().any():
        df[c] = df[c].fillna(df[c].median() if df[c].dtype.kind in "if" else df[c].mode()[0])
print(f"Missing values after : {int(df.isna().sum().sum())}")

# 4. Fix inconsistent values -------------------------------------------------
df["Age_of_Property"] = CURRENT_YEAR - df["Year_Built"]
bad_floor = df["Floor_No"] > df["Total_Floors"]
print(f"Rows where Floor_No > Total_Floors (fixed): {int(bad_floor.sum())}")
df.loc[bad_floor, "Total_Floors"] = df.loc[bad_floor, "Floor_No"]      # building must be at least as tall as the floor
df["Price_in_Lakhs"] = df["Price_in_Lakhs"].abs()
df["Size_in_SqFt"] = df["Size_in_SqFt"].abs()
# raw Price_per_SqFt is in Lakhs/sqft rounded to 2 decimals -> recompute in INR
df["Price_per_SqFt_INR"] = (df["Price_in_Lakhs"] * 1e5 / df["Size_in_SqFt"]).round(0)
df = df.drop(columns=["Price_per_SqFt"])

# 5. Outlier flags (kept, not deleted - IQR rule) ----------------------------
def iqr_flag(s):
    q1, q3 = s.quantile([.25, .75]); i = q3 - q1
    return ((s < q1 - 1.5 * i) | (s > q3 + 1.5 * i)).astype(int)
for c in ["Price_in_Lakhs", "Size_in_SqFt", "Price_per_SqFt_INR"]:
    df[f"Outlier_{c}"] = iqr_flag(df[c])

# 6. Feature engineering -------------------------------------------------------
df["Amenity_Count"] = df["Amenities"].str.split(",").apply(lambda x: len([a for a in x if a.strip()]))
df["Transport_Score"] = df["Public_Transport_Accessibility"].map({"Low": 1, "Medium": 2, "High": 3})
df["School_Density_Score"] = (df["Nearby_Schools"] / df["Size_in_SqFt"] * 1000).round(3)
df["Infrastructure_Score"] = (
    0.30 * df["Nearby_Schools"] / 10 + 0.30 * df["Nearby_Hospitals"] / 10
    + 0.25 * (df["Transport_Score"] - 1) / 2 + 0.15 * df["Amenity_Count"] / df["Amenity_Count"].max()
).mul(10).round(2)                                                  # 0-10 scale
df["Age_Category"] = pd.cut(df["Age_of_Property"], [-1, 5, 15, 25, 100],
                            labels=["New (0-5y)", "Recent (6-15y)", "Mature (16-25y)", "Old (25y+)"]).astype(str)
df["Size_Category"] = pd.cut(df["Size_in_SqFt"], [0, 1000, 2000, 3500, 10000],
                             labels=["Compact", "Medium", "Large", "Luxury"]).astype(str)
df["Floor_Position"] = np.where(df["Floor_No"] <= 3, "Low", np.where(df["Floor_No"] <= 15, "Mid", "High"))

# 7. Target 1 : Good_Investment (domain rule, 4 criteria -> need >= 3) ----------
city_med_pps = df.groupby("City")["Price_per_SqFt_INR"].transform("median")
score = ((df["Price_per_SqFt_INR"] <= city_med_pps).astype(int)             # priced below city median
         + (df["Infrastructure_Score"] >= df["Infrastructure_Score"].median()).astype(int)
         + (df["Transport_Score"] >= 2).astype(int)                          # medium/high transport
         + (df["Availability_Status"] == "Ready_to_Move").astype(int)
         + (df["Age_of_Property"] <= 15).astype(int))
df["Investment_Score"] = score
df["Good_Investment"] = (score >= 4).astype(int)

# 8. Target 2 : Future_Price_5Y (assumption-based appreciation model) ----------
# Dataset has a single snapshot, so annual growth is modelled from fundamentals:
# 5% base + up to 3% infrastructure + 1.5% transport + 1% new build + 1% under-construction premium
growth = (0.05 + 0.03 * df["Infrastructure_Score"] / 10 + 0.015 * (df["Transport_Score"] - 1) / 2
          + 0.01 * (df["Age_of_Property"] <= 10) + 0.01 * (df["Availability_Status"] == "Under_Construction"))
df["Annual_Growth_Rate_%"] = (growth * 100).round(2)
df["Future_Price_5Y_Lakhs"] = (df["Price_in_Lakhs"] * (1 + growth) ** 5).round(2)
df["Expected_Gain_Lakhs"] = (df["Future_Price_5Y_Lakhs"] - df["Price_in_Lakhs"]).round(2)

df.to_csv(OUT, index=False)
print(f"Clean shape: {df.shape}  ->  saved {OUT}")
print(f"Good investments: {df['Good_Investment'].mean():.1%}")
