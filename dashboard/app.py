
from pathlib import Path
import sqlite3

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
DB_FILE = ROOT / "database" / "air_quality_warehouse.db"
OUTPUT_DIR = ROOT / "outputs"
FORECAST_FILE = OUTPUT_DIR / "pm25_forecast_test_results.csv"

st.set_page_config(
    page_title="Air Quality Intelligence",
    page_icon="🌍",
    layout="wide",
)

st.title("🌍 Air Quality Data Warehouse")
st.subheader("Pollution Pattern Mining & Forecasting")
st.caption(
    "Explore daily air quality, compare cities, "
    "and evaluate PM2.5 forecasting results."
)

if not DB_FILE.exists():
    st.error("Warehouse not found. Run load_warehouse.py first.")
    st.stop()

# ------------------------------------
# Load warehouse data
# ------------------------------------
@st.cache_data
def load_data(db_path):
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(
            """
            SELECT
                d.full_date AS date,
                d.year,
                d.month,
                d.quarter,
                d.day_of_week,
                c.city_name AS city,
                f.pm2_5_mean,
                f.pm10_mean,
                f.no2_mean,
                f.so2_mean,
                f.o3_8h_max,
                f.co_8h_max,
                f.india_aqi,
                f.india_aqi_category,
                f.dominant_pollutant
            FROM fact_air_quality f
            JOIN dim_date d
                ON f.date_key = d.date_key
            JOIN dim_city c
                ON f.city_key = c.city_key
            ORDER BY d.full_date
            """,
            conn,
        )


df = load_data(str(DB_FILE))
df["date"] = pd.to_datetime(df["date"])

# ------------------------------------
# Sidebar filters
# ------------------------------------
st.sidebar.header("Dashboard Filters")

cities = sorted(df["city"].dropna().unique().tolist())

selected_cities = st.sidebar.multiselect(
    "Select cities",
    cities,
    default=cities[:5],
)

min_date = df["date"].min().date()
max_date = df["date"].max().date()

date_range = st.sidebar.date_input(
    "Select date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

if not selected_cities:
    st.warning("Select at least one city.")
    st.stop()

if not isinstance(date_range, (tuple, list)) or len(date_range) != 2:
    st.info("Select both a start date and an end date.")
    st.stop()

start_date, end_date = date_range

if start_date > end_date:
    st.error("Start date must not be after end date.")
    st.stop()

filtered = df[
    df["city"].isin(selected_cities)
    & df["date"].dt.date.between(start_date, end_date)
].copy()

if filtered.empty:
    st.warning("No records found for these filters.")
    st.stop()

# ------------------------------------
# Key performance indicators
# ------------------------------------
st.markdown("### Air Quality Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Daily Records", f"{len(filtered):,}")
c2.metric("Cities Selected", filtered["city"].nunique())
c3.metric("Average India AQI", f"{filtered['india_aqi'].mean():.1f}")
c4.metric("Average PM2.5", f"{filtered['pm2_5_mean'].mean():.2f}")

st.divider()

# ------------------------------------
# City comparison
# ------------------------------------
st.markdown("### 1. City-wise Pollution Comparison")

city_summary = (
    filtered.groupby("city", as_index=False)
    .agg(
        average_aqi=("india_aqi", "mean"),
        average_pm25=("pm2_5_mean", "mean"),
        average_pm10=("pm10_mean", "mean"),
    )
)

fig_city = px.bar(
    city_summary.sort_values("average_aqi", ascending=False),
    x="city",
    y="average_aqi",
    color="average_aqi",
    title="Average India AQI by City",
    labels={
        "city": "City",
        "average_aqi": "Average India AQI",
    },
)
st.plotly_chart(fig_city, use_container_width=True)

# ------------------------------------
# Pollution trends
# ------------------------------------
st.markdown("### 2. Pollution Trends Over Time")

pollutant = st.selectbox(
    "Choose a measurement",
    ["india_aqi", "pm2_5_mean", "pm10_mean", "no2_mean", "so2_mean"],
    format_func=lambda x: {
        "india_aqi": "India AQI",
        "pm2_5_mean": "PM2.5",
        "pm10_mean": "PM10",
        "no2_mean": "NO₂",
        "so2_mean": "SO₂",
    }[x],
)

monthly = (
    filtered.assign(
        month_start=filtered["date"].dt.to_period("M").dt.to_timestamp()
    )
    .groupby(["month_start", "city"], as_index=False)[pollutant]
    .mean()
)

fig_trend = px.line(
    monthly,
    x="month_start",
    y=pollutant,
    color="city",
    markers=True,
    title="Monthly Average Pollution Trend",
    labels={
        "month_start": "Month",
        pollutant: "Monthly Average",
        "city": "City",
    },
)
st.plotly_chart(fig_trend, use_container_width=True)

# ------------------------------------
# AQI category distribution
# ------------------------------------
st.markdown("### 3. AQI Category Distribution")

category_data = (
    filtered.groupby("india_aqi_category")
    .size()
    .reset_index(name="records")
    .sort_values("records", ascending=False)
)

fig_category = px.pie(
    category_data,
    names="india_aqi_category",
    values="records",
    title="Distribution of Recorded AQI Categories",
    hole=0.35,
)
st.plotly_chart(fig_category, use_container_width=True)

# ------------------------------------
# Dominant pollutants
# ------------------------------------
st.markdown("### 4. Dominant Pollutants")

dominant = (
    filtered.groupby("dominant_pollutant")
    .size()
    .reset_index(name="records")
    .sort_values("records", ascending=False)
)

fig_dominant = px.bar(
    dominant,
    x="dominant_pollutant",
    y="records",
    title="Most Frequently Reported Dominant Pollutants",
    labels={
        "dominant_pollutant": "Pollutant",
        "records": "Daily Records",
    },
)
st.plotly_chart(fig_dominant, use_container_width=True)

# ------------------------------------
# Forecast evaluation
# ------------------------------------
st.markdown("### 5. PM2.5 Forecast Evaluation")

if FORECAST_FILE.exists():
    forecast = pd.read_csv(
        FORECAST_FILE,
        parse_dates=["date"],
    )

    forecast = forecast[
        forecast["city"].isin(selected_cities)
        & forecast["date"].dt.date.between(start_date, end_date)
    ].copy()

    if not forecast.empty:
        forecast_city = st.selectbox(
            "Choose a city for forecast comparison",
            sorted(forecast["city"].unique()),
        )

        city_forecast = forecast[
            forecast["city"] == forecast_city
        ].sort_values("date")

        fig_forecast = px.line(
            city_forecast,
            x="date",
            y=[
                "actual_pm2_5",
                "predicted_pm2_5",
                "baseline_pm2_5",
            ],
            title=f"Actual vs Predicted PM2.5 — {forecast_city}",
            labels={
                "date": "Date",
                "value": "PM2.5",
                "variable": "Series",
            },
        )
        st.plotly_chart(fig_forecast, use_container_width=True)

        st.caption(
            "Historical test-period predictions, not future forecasts."
        )
    else:
        st.info(
            "No test-period predictions are available for "
            "the selected cities and dates."
        )
else:
    st.info(
        "Forecast results not found. Run src/forecast_pollution.py first."
    )

# ------------------------------------
# Data explorer and download
# ------------------------------------
st.markdown("### 6. Explore and Download Data")

st.dataframe(
    filtered.sort_values(["date", "city"], ascending=[False, True]),
    use_container_width=True,
    hide_index=True,
)

st.download_button(
    "Download filtered data as CSV",
    data=filtered.to_csv(index=False).encode("utf-8"),
    file_name="filtered_air_quality.csv",
    mime="text/csv",
)

st.success("Dashboard loaded successfully.")
