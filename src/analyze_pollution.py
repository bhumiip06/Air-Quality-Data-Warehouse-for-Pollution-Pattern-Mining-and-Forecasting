
from pathlib import Path
import sqlite3
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DB_FILE = ROOT / "database" / "air_quality_warehouse.db"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not DB_FILE.exists():
    raise FileNotFoundError(
        f"Warehouse database not found: {DB_FILE}"
    )

conn = sqlite3.connect(DB_FILE)

try:
    # Overall warehouse summary
    summary = pd.read_sql_query(
        """
        SELECT
            COUNT(*) AS total_records,
            COUNT(DISTINCT city_key) AS total_cities,
            MIN(d.full_date) AS first_date,
            MAX(d.full_date) AS last_date,
            ROUND(AVG(f.india_aqi), 2) AS average_india_aqi,
            ROUND(AVG(f.pm2_5_mean), 2) AS average_pm25,
            ROUND(AVG(f.pm10_mean), 2) AS average_pm10
        FROM fact_air_quality f
        JOIN dim_date d
            ON f.date_key = d.date_key
        """,
        conn,
    )

    print("\n========== WAREHOUSE SUMMARY ==========")
    print(summary.to_string(index=False))

    # Cities ranked by average AQI
    city_aqi = pd.read_sql_query(
        """
        SELECT
            c.city_name,
            COUNT(*) AS days_recorded,
            ROUND(AVG(f.india_aqi), 2) AS average_aqi,
            MAX(f.india_aqi) AS maximum_aqi
        FROM fact_air_quality f
        JOIN dim_city c
            ON f.city_key = c.city_key
        GROUP BY c.city_name
        ORDER BY average_aqi DESC
        """,
        conn,
    )

    print("\n========== TOP 10 CITIES BY AVERAGE AQI ==========")
    print(city_aqi.head(10).to_string(index=False))

    city_aqi.to_csv(
        OUTPUT_DIR / "city_aqi_ranking.csv",
        index=False,
    )

    # Monthly pollution trend
    monthly = pd.read_sql_query(
        """
        SELECT
            d.year,
            d.month,
            ROUND(AVG(f.india_aqi), 2) AS average_aqi,
            ROUND(AVG(f.pm2_5_mean), 2) AS average_pm25,
            ROUND(AVG(f.pm10_mean), 2) AS average_pm10
        FROM fact_air_quality f
        JOIN dim_date d
            ON f.date_key = d.date_key
        GROUP BY d.year, d.month
        ORDER BY d.year, d.month
        """,
        conn,
    )

    monthly["month_start"] = pd.to_datetime(
        monthly["year"].astype(str)
        + "-"
        + monthly["month"].astype(str).str.zfill(2)
        + "-01"
    )

    print("\n========== FIRST 12 MONTHLY RECORDS ==========")
    print(monthly.head(12).to_string(index=False))

    monthly.to_csv(
        OUTPUT_DIR / "monthly_pollution_trends.csv",
        index=False,
    )

    #  AQI category distribution
    categories = pd.read_sql_query(
        """
        SELECT
            india_aqi_category,
            COUNT(*) AS record_count,
            ROUND(
                100.0 * COUNT(*) /
                (SELECT COUNT(*) FROM fact_air_quality),
                2
            ) AS percentage
        FROM fact_air_quality
        GROUP BY india_aqi_category
        ORDER BY record_count DESC
        """,
        conn,
    )

    print("\n========== AQI CATEGORY DISTRIBUTION ==========")
    print(categories.to_string(index=False))

    categories.to_csv(
        OUTPUT_DIR / "aqi_category_distribution.csv",
        index=False,
    )

    print("\nAnalysis completed successfully.")
    print("Results saved in:", OUTPUT_DIR)

finally:
    conn.close()
