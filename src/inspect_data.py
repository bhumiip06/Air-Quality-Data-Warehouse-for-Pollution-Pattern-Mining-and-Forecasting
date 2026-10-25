from pathlib import Path
import pandas as pd

PROJECT_ROOT=Path(__file__).resolve().parent.parent
RAW_DATA_DIR=PROJECT_ROOT /"data"/"raw"
csv_files=list(RAW_DATA_DIR.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError(
        f"No CSV files found in: {RAW_DATA_DIR}"
    )
print("\nCSV files found:")
for index, file in enumerate(csv_files, start=1):
    print(f"{index}. {file.name}")

# If there are multiple files, inspect the first one.
# We will select the correct file explicitly if necessary.
csv_path = csv_files[0]

print(f"\nInspecting: {csv_path.name}")

df = pd.read_csv(csv_path, low_memory=False)

print("\n========== DATASET OVERVIEW ==========")
print("Rows:", len(df))
print("Columns:", len(df.columns))

print("\n========== COLUMN NAMES ==========")
print(df.columns.tolist())

print("\n========== FIRST 5 ROWS ==========")
print(df.head().to_string())

print("\n========== DATA TYPES ==========")
print(df.dtypes.to_string())

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum().to_string())

print("\n========== DUPLICATE ROWS ==========")
print(df.duplicated().sum())

print("\n========== NUMERICAL SUMMARY ==========")
print(df.describe(include="all").to_string())

print("\n========== INSPECTION COMPLETE ==========")
