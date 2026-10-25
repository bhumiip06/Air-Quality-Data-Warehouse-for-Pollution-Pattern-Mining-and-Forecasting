
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load analysis results
city_file = OUTPUT_DIR / "city_aqi_ranking.csv"
monthly_file = OUTPUT_DIR / "monthly_pollution_trends.csv"
category_file = OUTPUT_DIR / "aqi_category_distribution.csv"

for file in [city_file, monthly_file, category_file]:
    if not file.exists():
        raise FileNotFoundError(
            f"{file.name} not found. Run analyze_pollution.py first."
        )

cities = pd.read_csv(city_file)
monthly = pd.read_csv(monthly_file)
categories = pd.read_csv(category_file)

monthly["month_start"] = pd.to_datetime(monthly["month_start"])

#Top 10 cities by average AQI
top_cities = cities.head(10).sort_values("average_aqi")

plt.figure(figsize=(10, 6))
plt.barh(top_cities["city_name"], top_cities["average_aqi"])
plt.xlabel("Average India AQI")
plt.ylabel("City")
plt.title("Top 10 Cities by Average AQI")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "top_cities_aqi.png", dpi=150)
plt.close()

# Monthly AQI trend
plt.figure(figsize=(12, 5))
plt.plot(
    monthly["month_start"],
    monthly["average_aqi"],
    marker="o",
    markersize=3
)
plt.xlabel("Month")
plt.ylabel("Average India AQI")
plt.title("Monthly Air Quality Trend")
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "monthly_aqi_trend.png", dpi=150)
plt.close()

#  Monthly PM2.5 trend
plt.figure(figsize=(12, 5))
plt.plot(
    monthly["month_start"],
    monthly["average_pm25"],
    marker="o",
    markersize=3
)
plt.xlabel("Month")
plt.ylabel("Average PM2.5")
plt.title("Monthly PM2.5 Trend")
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "monthly_pm25_trend.png", dpi=150)
plt.close()

#AQI category distribution
plt.figure(figsize=(9, 5))
plt.bar(
    categories["india_aqi_category"],
    categories["record_count"]
)
plt.xlabel("AQI Category")
plt.ylabel("Number of Daily Records")
plt.title("Distribution of Air Quality Categories")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "aqi_category_distribution.png", dpi=150)
plt.close()

print("Visualizations generated successfully!")

for filename in [
    "top_cities_aqi.png",
    "monthly_aqi_trend.png",
    "monthly_pm25_trend.png",
    "aqi_category_distribution.png"
]:
    print("Created:", OUTPUT_DIR / filename)
