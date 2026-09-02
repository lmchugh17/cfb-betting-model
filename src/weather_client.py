"""Open-Meteo client (free, no API key). Historical archive for past games,
forecast endpoint for upcoming games (used later by the weekly live pipeline).
"""
import time

import requests

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

HOURLY_FIELDS = "temperature_2m,precipitation,windspeed_10m,winddirection_10m,relativehumidity_2m"
UNIT_PARAMS = {
    "temperature_unit": "fahrenheit",
    "windspeed_unit": "mph",
    "precipitation_unit": "inch",
    "timezone": "UTC",
}


def _hourly_by_timestamp(payload: dict) -> dict:
    hourly = payload.get("hourly", {})
    times = hourly.get("time", [])
    result = {}
    for i, ts in enumerate(times):
        result[ts] = {
            "temperature_f": hourly.get("temperature_2m", [None] * len(times))[i],
            "wind_speed_mph": hourly.get("windspeed_10m", [None] * len(times))[i],
            "wind_direction_deg": hourly.get("winddirection_10m", [None] * len(times))[i],
            "precipitation_in": hourly.get("precipitation", [None] * len(times))[i],
            "humidity_pct": hourly.get("relativehumidity_2m", [None] * len(times))[i],
        }
    return result


# Short on purpose: backfill_weather.py and pull_weather_forecast.py both make one
# call per venue in a loop (up to ~130+ in a full week), sequentially. A slow/stuck
# request needs to fail fast, not eat 30s each -- confirmed 2026-09-02 that GitHub
# Actions' shared runner IPs can see Open-Meteo hang far longer than requests from a
# normal residential/office network (~0.6s there vs. a run that never finished a
# single venue in 13+ minutes on Actions), most likely rate-limiting or throttling
# aimed at datacenter/CI traffic on this free, unauthenticated API.
REQUEST_TIMEOUT_S = 10


def fetch_historical(lat: float, lon: float, start_date: str, end_date: str) -> dict:
    """Returns {ISO hour string ('YYYY-MM-DDTHH:00'): {weather fields}}."""
    params = {"latitude": lat, "longitude": lon, "start_date": start_date, "end_date": end_date,
              "hourly": HOURLY_FIELDS, **UNIT_PARAMS}
    resp = requests.get(ARCHIVE_URL, params=params, timeout=REQUEST_TIMEOUT_S)
    resp.raise_for_status()
    return _hourly_by_timestamp(resp.json())


def fetch_forecast(lat: float, lon: float) -> dict:
    """Forecast API covers ~16 days ahead; used for the upcoming weekend's games."""
    params = {"latitude": lat, "longitude": lon, "hourly": HOURLY_FIELDS,
              "forecast_days": 16, **UNIT_PARAMS}
    resp = requests.get(FORECAST_URL, params=params, timeout=REQUEST_TIMEOUT_S)
    resp.raise_for_status()
    return _hourly_by_timestamp(resp.json())


def nearest_hour(hourly: dict, kickoff_iso_utc: str):
    """kickoff_iso_utc like '2025-08-23T16:00:00.000Z' -> match against 'YYYY-MM-DDTHH:00' keys."""
    key = kickoff_iso_utc[:13] + ":00"
    return hourly.get(key)


def polite_sleep():
    time.sleep(0.2)  # stay well under Open-Meteo's fair-use rate limits across ~700 calls
