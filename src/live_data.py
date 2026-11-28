"""Fetch recent hourly air quality and weather from Open-Meteo.

Plain functions only. Parameter names were taken from the current docs:

- Air Quality API, https://open-meteo.com/en/docs/air-quality-api :
  hourly pm10, pm2_5, carbon_monoxide, nitrogen_dioxide, sulphur_dioxide,
  ozone, dust, us_aqi (units ug/m3 and US AQI). past_days is an integer
  0-92 (default 0). forecast_days is an integer 0-7 (default 5). timezone
  is any IANA name.
- Weather API, https://open-meteo.com/en/docs : hourly temperature_2m (C),
  relative_humidity_2m (%), surface_pressure (hPa), wind_speed_10m (km/h),
  wind_direction_10m (degrees), precipitation (mm).

forecast_days=0 and past_days are sent to both endpoints. No API key.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import streamlit as st

from src import alerts, utils

AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
TIMEZONE = "Asia/Karachi"
CITY_COORDINATES = {
    "Peshawar": (34.0151, 71.5249),
    "Lahore": (31.5204, 74.3587),
}
AIR_QUALITY_HOURLY = (
    "pm2_5,pm10,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,dust,us_aqi"
)
WEATHER_HOURLY = (
    "temperature_2m,relative_humidity_2m,surface_pressure,"
    "wind_speed_10m,wind_direction_10m,precipitation"
)
REQUEST_TIMEOUT_SECONDS = 20.0
MAX_ATTEMPTS = 2
CACHE_DIR = utils.PROCESSED_DIR / "live_cache"

_RANGES = {
    "pm2_5": (0.0, 1000.0),
    "pm10": (0.0, 2000.0),
    "carbon_monoxide": (0.0, 20000.0),
    "nitrogen_dioxide": (0.0, 1000.0),
    "sulphur_dioxide": (0.0, 1000.0),
    "ozone": (0.0, 1000.0),
    "dust": (0.0, 2000.0),
    "us_aqi": (0.0, 1000.0),
    "temperature_2m": (-40.0, 55.0),
    "relative_humidity_2m": (0.0, 100.0),
    "surface_pressure": (800.0, 1100.0),
    "wind_speed_10m": (0.0, 200.0),
    "wind_direction_10m": (0.0, 360.0),
    "precipitation": (0.0, 500.0),
}


def _cache_paths(city: str) -> tuple[Path, Path]:
    """Return the CSV and JSON paths of one city's last successful fetch."""
    slug = city.lower()
    return CACHE_DIR / f"{slug}.csv", CACHE_DIR / f"{slug}.json"


def _now_local() -> pd.Timestamp:
    """Current Pakistan time, naive, so it compares with Open-Meteo timestamps."""
    now = datetime.now(ZoneInfo(TIMEZONE)).replace(tzinfo=None)
    return pd.Timestamp(now)


def _get_json(url: str, params: dict) -> dict:
    """GET JSON with a timeout and one retry on timeout, connection loss, or HTTP 5xx.

    A 4xx is not retried. The URL is wrong, and a second call will not fix it.
    """
    last_error = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
            if response.status_code >= 500 and attempt + 1 < MAX_ATTEMPTS:
                last_error = requests.HTTPError(response=response)
                continue
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, dict) and payload.get("error"):
                raise RuntimeError(str(payload.get("reason", "Open-Meteo error")))
            return payload
        except (requests.Timeout, requests.ConnectionError) as exc:
            last_error = exc
            if attempt + 1 >= MAX_ATTEMPTS:
                raise
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"No response from {url}")


def _hourly_frame(payload: dict) -> pd.DataFrame:
    """Turn an Open-Meteo hourly object into a DataFrame."""
    hourly = payload.get("hourly")
    if not isinstance(hourly, dict) or "time" not in hourly:
        raise ValueError("Open-Meteo response has no hourly.time array")
    frame = pd.DataFrame(hourly)
    if frame.empty:
        raise ValueError("Open-Meteo response has no hourly rows")
    return frame


def _aqi_category(us_aqi: float) -> str:
    """Map one AQI to the project tier name, or Unknown when it is missing."""
    if pd.isna(us_aqi):
        return "Unknown"
    name, _color = alerts.get_aqi_tier(float(us_aqi))
    return name


def _assemble(city: str, air: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Merge the two hourly frames and add the CSV calendar columns."""
    latitude, longitude = CITY_COORDINATES[city]
    merged = air.merge(weather, on="time", how="inner")
    if merged.empty:
        raise ValueError(f"No overlapping hours for {city}")
    stamps = pd.to_datetime(merged["time"], format="%Y-%m-%dT%H:%M")
    out = pd.DataFrame({
        "city": city,
        "latitude": float(latitude),
        "longitude": float(longitude),
        "time": stamps.dt.strftime("%Y-%m-%dT%H:%M"),
        "pm10": pd.to_numeric(merged["pm10"], errors="coerce"),
        "pm2_5": pd.to_numeric(merged["pm2_5"], errors="coerce"),
        "carbon_monoxide": pd.to_numeric(merged["carbon_monoxide"], errors="coerce"),
        "nitrogen_dioxide": pd.to_numeric(merged["nitrogen_dioxide"], errors="coerce"),
        "sulphur_dioxide": pd.to_numeric(merged["sulphur_dioxide"], errors="coerce"),
        "ozone": pd.to_numeric(merged["ozone"], errors="coerce"),
        "dust": pd.to_numeric(merged["dust"], errors="coerce"),
        "us_aqi": pd.to_numeric(merged["us_aqi"], errors="coerce"),
        "temperature_2m": pd.to_numeric(merged["temperature_2m"], errors="coerce"),
        "relative_humidity_2m": pd.to_numeric(merged["relative_humidity_2m"], errors="coerce").round(),
        "surface_pressure": pd.to_numeric(merged["surface_pressure"], errors="coerce"),
        "wind_speed_10m": pd.to_numeric(merged["wind_speed_10m"], errors="coerce"),
        "wind_direction_10m": pd.to_numeric(merged["wind_direction_10m"], errors="coerce").round(),
        "precipitation": pd.to_numeric(merged["precipitation"], errors="coerce"),
    })
    out["datetime"] = pd.to_datetime(stamps)
    out["year"] = stamps.dt.year.astype(int)
    out["month"] = stamps.dt.month.astype(int)
    out["day"] = stamps.dt.day.astype(int)
    out["hour"] = stamps.dt.hour.astype(int)
    out["day_of_week"] = stamps.dt.day_name()
    out["is_weekend"] = (stamps.dt.dayofweek >= 5).astype(int)
    out["aqi_category"] = out["us_aqi"].map(_aqi_category)
    return out[utils.REQUIRED_COLUMNS].sort_values("datetime").reset_index(drop=True)


def drop_future_rows(frame: pd.DataFrame, now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Remove rows later than the Pakistan clock.

    forecast_days=0 omits tomorrow, but later hours of today are still in the
    payload. Those hours have not happened.
    """
    clock = _now_local() if now is None else pd.Timestamp(now)
    stamps = pd.to_datetime(frame["datetime"])
    kept = frame.loc[stamps <= clock].copy()
    return kept.reset_index(drop=True)


def validate_frame(frame: pd.DataFrame) -> None:
    """Check required columns, one-hour steps, and plausible ranges."""
    missing = [name for name in utils.REQUIRED_COLUMNS if name not in frame.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    if frame.empty:
        raise ValueError("No rows left after removing future timestamps")
    stamps = pd.to_datetime(frame["datetime"]).sort_values().reset_index(drop=True)
    if len(stamps) >= 2:
        steps = stamps.diff().iloc[1:]
        one_hour = pd.Timedelta(1, unit="h")
        bad_at = steps[steps != one_hour]
        if not bad_at.empty:
            raise ValueError(
                f"Hourly sequence breaks at row {int(bad_at.index[0])}: step is {bad_at.iloc[0]}"
            )
    for column, (low, high) in _RANGES.items():
        values = pd.to_numeric(frame[column], errors="coerce")
        outside = values.notna() & ((values < low) | (values > high))
        if bool(outside.any()):
            raise ValueError(f"{column} has {int(outside.sum())} values outside {low} to {high}")


def _request_city(city: str, past_days: int) -> pd.DataFrame:
    """Call both APIs for one city and return a validated frame."""
    if city not in CITY_COORDINATES:
        raise ValueError(f"Unknown city {city!r}. Known cities: {sorted(CITY_COORDINATES)}")
    if not isinstance(past_days, int) or isinstance(past_days, bool) or not 1 <= past_days <= 92:
        raise ValueError("past_days must be an integer from 1 to 92")
    latitude, longitude = CITY_COORDINATES[city]
    common = {
        "latitude": latitude,
        "longitude": longitude,
        "timezone": TIMEZONE,
        "past_days": past_days,
        "forecast_days": 0,
    }
    air = _hourly_frame(_get_json(AIR_QUALITY_URL, {**common, "hourly": AIR_QUALITY_HOURLY}))
    weather = _hourly_frame(_get_json(WEATHER_URL, {**common, "hourly": WEATHER_HOURLY}))
    frame = drop_future_rows(_assemble(city, air, weather))
    validate_frame(frame)
    return frame


def _write_cache(city: str, frame: pd.DataFrame) -> None:
    """Save a successful fetch so a later outage can reuse it."""
    csv_path, meta_path = _cache_paths(city)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    stored = frame.copy()
    stored["datetime"] = pd.to_datetime(stored["datetime"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    stored.to_csv(csv_path, index=False)
    meta_path.write_text(
        json.dumps({
            "city": city,
            "fetched_at": _now_local().strftime("%Y-%m-%d %H:%M"),
            "rows": int(len(frame)),
        }),
        encoding="utf-8",
    )


def _read_cache(city: str):
    """Load the last saved fetch, or None when there is no cache file."""
    csv_path, meta_path = _cache_paths(city)
    if not csv_path.is_file():
        return None
    frame = pd.read_csv(csv_path)
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    fetched_at = "an unknown time"
    if meta_path.is_file():
        fetched_at = json.loads(meta_path.read_text(encoding="utf-8")).get("fetched_at", fetched_at)
    return frame, fetched_at


def _read_csv_fallback(city: str, past_days: int) -> pd.DataFrame:
    """Take the latest non-future hours of one city from the master CSV."""
    master = pd.read_csv(utils.MASTER_PATH)
    city_rows = master.loc[master["city"] == city].copy()
    city_rows["datetime"] = pd.to_datetime(city_rows["datetime"])
    city_rows = drop_future_rows(city_rows)
    keep = past_days * 24 + 24
    return city_rows.tail(keep).reset_index(drop=True)


def _notice(frame: pd.DataFrame, source: str, text: str) -> dict:
    """Status dict. based_on is the last hour the forecast may use."""
    based_on = pd.to_datetime(frame["datetime"]).max().strftime("%Y-%m-%d %H:%M")
    return {"source": source, "notice": text, "based_on": based_on}


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_recent(city: str, past_days: int = 7) -> pd.DataFrame:
    """Fetch the recent hourly rows for one city from Open-Meteo.

    The result is cached for one hour. A failed request is not cached,
    because this function raises instead of returning a fallback.

    Args:
        city: Peshawar or Lahore.
        past_days: How many past days to request. Today is included.

    Returns:
        A frame with the master-file columns.
    """
    frame = _request_city(city, past_days)
    _write_cache(city, frame)
    return frame


def load_recent(city: str, past_days: int = 7) -> tuple[pd.DataFrame, dict]:
    """Fetch recent hours, then the last saved fetch, then the CSV.

    Args:
        city: Peshawar or Lahore.
        past_days: Days requested from the API.

    Returns:
        The frame and a status dict with source, notice, and based_on.
    """
    try:
        frame = fetch_recent(city, past_days)
    except (requests.RequestException, RuntimeError, ValueError, OSError):
        cached = _read_cache(city)
        if cached is not None:
            frame, fetched_at = cached
            frame = drop_future_rows(frame)
            status = _notice(frame, "cache", "")
            status["notice"] = (
                f"Open-Meteo could not be reached. Using the last saved fetch "
                f"from {fetched_at}. Forecast is based on {status['based_on']}."
            )
            return frame, status
        frame = _read_csv_fallback(city, past_days)
        status = _notice(frame, "csv", "")
        status["notice"] = (
            "Open-Meteo could not be reached and no saved fetch was found. "
            f"Forecast is based on the historical file through {status['based_on']}."
        )
        return frame, status
    status = _notice(frame, "api", "")
    status["notice"] = f"Forecast is based on {status['based_on']} (Open-Meteo, {TIMEZONE})."
    return frame, status
