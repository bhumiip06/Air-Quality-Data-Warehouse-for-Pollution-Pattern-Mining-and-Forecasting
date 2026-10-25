
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    ROOT / "data" / "processed" / "air_quality_cleaned.csv"
)

OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        "Cleaned dataset not found. Run clean_data.py first."
    )

df = pd.read_csv(INPUT_FILE, parse_dates=["date"])

# We need a daily PM2.5 value for each city.
df = df[["date", "city", "pm2_5_mean"]].copy()

df = df.sort_values(["city", "date"])

# Generate past-value features separately for each city.
for lag in [1, 2, 3, 7, 14, 30]:
    df[f"lag_{lag}"] = (
        df.groupby("city")["pm2_5_mean"].shift(lag)
    )

# Rolling statistics use only previous days, avoiding leakage.
df["rolling_mean_7"] = (
    df.groupby("city")["pm2_5_mean"]
    .transform(lambda s: s.shift(1).rolling(7).mean())
)

df["rolling_mean_30"] = (
    df.groupby("city")["pm2_5_mean"]
    .transform(lambda s: s.shift(1).rolling(30).mean())
)

# Calendar features
df["month"] = df["date"].dt.month
df["day_of_year"] = df["date"].dt.day_of_year
df["day_of_week"] = df["date"].dt.day_of_week

feature_columns = [
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_7",
    "lag_14",
    "lag_30",
    "rolling_mean_7",
    "rolling_mean_30",
    "month",
    "day_of_year",
    "day_of_week",
]

df = df.dropna(
    subset=feature_columns + ["pm2_5_mean"]
).copy()

if df.empty:
    raise ValueError(
        "No rows remain after feature generation."
    )

# Chronological split: the latest 20% of dates form the test set.
unique_dates = np.sort(df["date"].unique())

if len(unique_dates) < 10:
    raise ValueError("Not enough dates for a reliable time split.")

split_index = int(len(unique_dates) * 0.8)
split_date = pd.Timestamp(unique_dates[split_index])

train = df[df["date"] < split_date].copy()
test = df[df["date"] >= split_date].copy()

if train.empty or test.empty:
    raise ValueError("Training or test dataset is empty.")

X_train = train[feature_columns]
y_train = train["pm2_5_mean"]

X_test = test[feature_columns]
y_test = test["pm2_5_mean"]

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=12,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
)

print("Training Random Forest model...")
model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

# Baseline: predict tomorrow's PM2.5 as today's value.
baseline_predictions = test["lag_1"]

baseline_mae = mean_absolute_error(
    y_test, baseline_predictions
)

print("\n========== FORECAST MODEL RESULTS ==========")
print("Training rows:", len(train))
print("Testing rows:", len(test))
print("Training ends before:", split_date.date())
print("Test starts on:", test["date"].min().date())
print("Test ends on:", test["date"].max().date())
print(f"Random Forest MAE: {mae:.3f}")
print(f"Random Forest RMSE: {rmse:.3f}")
print(f"Persistence baseline MAE: {baseline_mae:.3f}")

# Save actual and predicted values.
results = test[["date", "city", "pm2_5_mean"]].copy()
results = results.rename(
    columns={"pm2_5_mean": "actual_pm2_5"}
)
results["predicted_pm2_5"] = predictions
results["baseline_pm2_5"] = baseline_predictions.to_numpy()

results.to_csv(
    OUTPUT_DIR / "pm25_forecast_test_results.csv",
    index=False,
)

# Save evaluation metrics.
metrics = pd.DataFrame([{
    "model": "Random Forest",
    "train_rows": len(train),
    "test_rows": len(test),
    "test_start": str(test["date"].min().date()),
    "test_end": str(test["date"].max().date()),
    "mae": mae,
    "rmse": rmse,
    "baseline_mae": baseline_mae,
}])

metrics.to_csv(
    OUTPUT_DIR / "forecast_metrics.csv",
    index=False,
)

print("\nSaved:")
print(OUTPUT_DIR / "pm25_forecast_test_results.csv")
print(OUTPUT_DIR / "forecast_metrics.csv")
print("\nForecast model evaluation completed.")
