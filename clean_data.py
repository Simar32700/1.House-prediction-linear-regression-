from datetime import date
from pathlib import Path

import pandas as pd

data_path = Path(__file__).resolve().parent / "data" / "house_prices_practice.csv"
df = pd.read_csv(data_path)

print("First five rows:")
print(df.head())

print("\nMissing values in each column:")
print(df.isna().sum())

print("\nDuplicate rows:", df.duplicated().sum())

print("Duplicates excluding Id:", df.drop(columns="Id").duplicated().sum())

print("\nColumn data types:")
print(df.dtypes)

invalid_checks = {
    "GrLivArea must be positive": df["GrLivArea"] <= 0,
    "SalePrice must be positive": df["SalePrice"] <= 0,
    "LotArea must be positive": df["LotArea"] <= 0,
    "BedroomAbvGr cannot be negative": df["BedroomAbvGr"] < 0,
    "FullBath cannot be negative": df["FullBath"] < 0,
    "GarageCars cannot be negative": df["GarageCars"] < 0,
    "TotalBsmtSF cannot be negative": df["TotalBsmtSF"] < 0,
    "OverallQual must be between 1 and 10": ~df["OverallQual"].between(1, 10),
    "YearBuilt must be positive": df["YearBuilt"] <= 0,
    "YearBuilt is in the future (review)": df["YearBuilt"] > date.today().year,
}

print("\nInvalid-value checks (number of flagged rows):")
for check, flagged_rows in invalid_checks.items():
    print(f"{check}: {flagged_rows.sum()}")

print("\nText-label checks:")
for column in df.select_dtypes(include=["object", "string", "category"]).columns:
    labels = df[column].astype("string")
    trimmed_labels = labels.str.strip()
    print(f"\n{column} labels: {labels.dropna().unique().tolist()}")
    print(f"Rows with extra spaces: {(labels != trimmed_labels).sum()}")
    print(f"Blank labels: {(trimmed_labels == '').sum()}")
    print(
        "Labels differing only by spaces or capitalization:",
        labels.nunique() - trimmed_labels.str.casefold().nunique(),
    )

numeric_columns = df.select_dtypes(include="number").columns.drop("Id").tolist()
print("\nNumeric summary:")
print(df[numeric_columns].describe().round(2).to_string())

# IQR flags are for inspection only; unusual houses may still be valid.
print("\nPotential outliers in area and price (IQR method):")
for column in ["GrLivArea", "SalePrice"]:
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    flagged_rows = (df[column] < lower_bound) | (df[column] > upper_bound)
    print(f"{column}: {flagged_rows.sum()} flagged rows")
    print(f"  Review bounds: {lower_bound:.2f} to {upper_bound:.2f}")
    if flagged_rows.any():
        print(df.loc[flagged_rows, ["Id", column]].head(10).to_string(index=False))
