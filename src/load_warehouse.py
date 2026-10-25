
from pathlib import Path
import sqlite3
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    ROOT / "data" / "processed" / "air_quality_cleaned.csv"
)

DB_DIR = ROOT / "database"
DB_DIR.mkdir(parents=True, exist_ok=True)

DB_FILE = DB_DIR / "air_quality_warehouse.db"

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Cleaned dataset not found: {INPUT_FILE}. "
        "Run src/clean_data.py first."
    )

df = pd.read_csv(INPUT_FILE, low_memory=False)
df["date"] = pd.to_datetime(df["date"], errors="raise")

df["city"] = df["city"].astype(str).str.strip()

if df["city"].eq("").any() or df["city"].isna().any():
    raise ValueError("City contains missing or empty values.")

if df.duplicated(["date", "city"]).any():
    raise ValueError("Duplicate city-date records detected.")

# Date key format: YYYYMMDD
df["date_key"] = df["date"].dt.strftime("%Y%m%d").astype(int)

schema_sql = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS dim_city (
    city_key INTEGER PRIMARY KEY,
    city_name TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS dim_date (
    date_key INTEGER PRIMARY KEY,
    full_date TEXT NOT NULL UNIQUE,
    year INTEGER NOT NULL,
    month INTEGER NOT NULL,
    day INTEGER NOT NULL,
    quarter INTEGER NOT NULL,
    day_of_week TEXT NOT NULL,
    is_weekend INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS fact_air_quality (
    fact_key INTEGER PRIMARY KEY,
    date_key INTEGER NOT NULL,
    city_key INTEGER NOT NULL,

    pm2_5_mean REAL,
    pm2_5_max REAL,
    pm10_mean REAL,
    pm10_max REAL,
    no2_mean REAL,
    so2_mean REAL,
    o3_8h_max REAL,
    co_8h_max REAL,
    dust_mean REAL,
    aod_mean REAL,
    us_aqi_max REAL,
    hours_available INTEGER,

    india_aqi INTEGER,
    india_aqi_category TEXT,
    dominant_pollutant TEXT,
    india_aqi_pm INTEGER,
    india_aqi_pm_category TEXT,

    FOREIGN KEY (date_key) REFERENCES dim_date(date_key),
    FOREIGN KEY (city_key) REFERENCES dim_city(city_key),

    UNIQUE (date_key, city_key)
);

CREATE INDEX IF NOT EXISTS idx_fact_date
ON fact_air_quality(date_key);

CREATE INDEX IF NOT EXISTS idx_fact_city
ON fact_air_quality(city_key);

CREATE INDEX IF NOT EXISTS idx_fact_aqi
ON fact_air_quality(india_aqi);
"""

conn = sqlite3.connect(DB_FILE)

try:
    conn.execute("PRAGMA foreign_keys = ON;")

    with conn:
        conn.executescript(schema_sql)

        # Rebuild the warehouse contents on each run.
        # This makes the script repeatable without
        # accumulating duplicate fact records.
        conn.execute("DELETE FROM fact_air_quality")
        conn.execute("DELETE FROM dim_date")
        conn.execute("DELETE FROM dim_city")

        # City dimension
        cities = sorted(df["city"].unique())

        conn.executemany(
            """
            INSERT INTO dim_city (city_name)
            VALUES (?)
            """,
            [(city,) for city in cities]
        )

        city_lookup = {
            name: key
            for key, name in conn.execute(
                "SELECT city_key, city_name FROM dim_city"
            ).fetchall()
        }

        # Date dimension
        unique_dates = df["date"].drop_duplicates().sort_values()

        date_rows = []

        for date in unique_dates:
            date_rows.append((
                int(date.strftime("%Y%m%d")),
                date.strftime("%Y-%m-%d"),
                date.year,
                date.month,
                date.day,
                date.quarter,
                date.day_name(),
                int(date.day_of_week >= 5)
            ))

        conn.executemany(
            """
            INSERT INTO dim_date (
                date_key, full_date, year, month,
                day, quarter, day_of_week, is_weekend
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            date_rows
        )

        # Fact table
        fact_columns = [
            "pm2_5_mean", "pm2_5_max",
            "pm10_mean", "pm10_max",
            "no2_mean", "so2_mean",
            "o3_8h_max", "co_8h_max",
            "dust_mean", "aod_mean",
            "us_aqi_max", "hours_available",
            "india_aqi", "india_aqi_category",
            "dominant_pollutant",
            "india_aqi_pm", "india_aqi_pm_category"
        ]

        fact_rows = []

        for row in df.itertuples(index=False):
            date_key = int(row.date.strftime("%Y%m%d"))
            city_key = city_lookup[row.city]

            values = [
                getattr(row, col) for col in fact_columns
            ]

            # Convert pandas missing values to SQL NULL.
            values = [
                None if pd.isna(value) else value
                for value in values
            ]

            fact_rows.append(
                (date_key, city_key, *values)
            )

        placeholders = ", ".join(["?"] * 19)

        conn.executemany(
            f"""
            INSERT INTO fact_air_quality (
                date_key, city_key,
                {", ".join(fact_columns)}
            )
            VALUES ({placeholders})
            """,
            fact_rows
        )

    print("Warehouse created and loaded successfully!")

    print(f"\nDatabase location: {DB_FILE}")
    print(f"Source records: {len(df):,}")

    for table in [
        "dim_city",
        "dim_date",
        "fact_air_quality"
    ]:
        count = conn.execute(
            f"SELECT COUNT(*) FROM {table}"
        ).fetchone()[0]

        print(f"{table}: {count:,} records")

    violations = conn.execute(
        "PRAGMA foreign_key_check"
    ).fetchall()

    print("Foreign key violations:", len(violations))

    if violations:
        print(violations[:10])
        raise RuntimeError("Warehouse foreign key check failed.")

    print("\nWarehouse validation completed.")

finally:
    conn.close()
