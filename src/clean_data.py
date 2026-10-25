
from pathlib import Path
import pandas as pd
import json

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "outputs"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = RAW_DIR / "air_quality_daily.csv"
OUTPUT_FILE = PROCESSED_DIR / "air_quality_cleaned.csv"
REPORT_FILE = OUTPUT_DIR / "data_quality_report.json"


print("Loading raw dataset...")
df = pd.read_csv(INPUT_FILE, low_memory=False)

initial_rows = len(df)
initial_columns = len(df.columns)

print(f"Initial rows: {initial_rows}")
print(f"Initial columns: {initial_columns}")


df.columns = (
    df.columns
    .str.strip()
    .str.lower()
)

required_columns = [
    "date",
    "city",
    "pm2_5_mean",
    "pm10_mean",
    "india_aqi",
    "india_aqi_category",
    "dominant_pollutant",
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Required columns are missing: {missing_columns}"
    )

df["date"] = pd.to_datetime(
    df["date"], errors="coerce"
)

df["city"] = df["city"].astype("string").str.strip()

numeric_columns = [
    col for col in df.columns
    if col not in [
        "date",
        "city",
        "india_aqi_category",
        "dominant_pollutant",
        "india_aqi_pm_category",
    ]
]

for col in numeric_columns:
    df[col] = pd.to_numeric(
        df[col], errors="coerce"
    )

missing_before = int(df.isna().sum().sum())

invalid_dates = int(df["date"].isna().sum())
invalid_cities = int(
    (df["city"].isna() | df["city"].eq("")).sum()
)

# Drop rows only when the date or city key is invalid.
df = df.dropna(subset=["date", "city"])
df = df[df["city"].str.len() > 0]

duplicate_city_dates = int(
    df.duplicated(subset=["date", "city"]).sum()
)

# Retain the first duplicate temporarily and report
# the issue explicitly rather than silently discarding data.
df = df.drop_duplicates(
    subset=["date", "city"],
    keep="first"
)

df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day
df["quarter"] = df["date"].dt.quarter
df["day_of_week"] = df["date"].dt.day_name()

extreme_value_flags = {}

for col in numeric_columns:
    if col not in df.columns:
        continue

    values = df[col].dropna()

    if len(values) == 0:
        continue

    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    extreme_value_flags[col] = int(
        ((df[col] < lower_bound) |
         (df[col] > upper_bound)).sum()
    )


df = df.sort_values(
    ["city", "date"]
).reset_index(drop=True)

df.to_csv(OUTPUT_FILE, index=False)

report = {
    "input_file": INPUT_FILE.name,
    "initial_rows": initial_rows,
    "initial_columns": initial_columns,
    "final_rows": len(df),
    "final_columns": len(df.columns),
    "cities": int(df["city"].nunique()),
    "start_date": str(df["date"].min().date()),
    "end_date": str(df["date"].max().date()),
    "invalid_dates_removed": invalid_dates,
    "invalid_cities_removed": invalid_cities,
    "duplicate_city_date_records": duplicate_city_dates,
    "missing_cells_after_cleaning": int(
        df.isna().sum().sum()
    ),
    "extreme_value_flags": extreme_value_flags,
    "output_file": str(OUTPUT_FILE.relative_to(ROOT)),
}

with open(REPORT_FILE, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=4)

print("\n========== CLEANING SUMMARY ==========")
print(json.dumps(report, indent=4))
print("\nCleaned dataset saved to:", OUTPUT_FILE)
print("Quality report saved to:", REPORT_FILE)
