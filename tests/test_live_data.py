"""Tests for the Open-Meteo fetch. HTTP is mocked; no real network call."""

from __future__ import annotations

import pandas as pd
import pytest
import requests
import streamlit as st

from src import live_data, utils

NOW = pd.Timestamp("2024-06-15 12:00:00")
HOURS = ["2024-06-15T10:00", "2024-06-15T11:00", "2024-06-15T12:00", "2024-06-15T13:00"]


def _payload(url: str) -> dict:
    if "air-quality" in url:
        values = {
            "pm2_5": [10.0, 11.0, 12.0, 13.0],
            "pm10": [20.0, 21.0, 22.0, 23.0],
            "carbon_monoxide": [100.0, 101.0, 102.0, 103.0],
            "nitrogen_dioxide": [5.0, 5.0, 5.0, 5.0],
            "sulphur_dioxide": [1.0, 1.0, 1.0, 1.0],
            "ozone": [40.0, 41.0, 42.0, 43.0],
            "dust": [2.0, 2.0, 2.0, 2.0],
            "us_aqi": [42, 45, 48, 51],
        }
    else:
        values = {
            "temperature_2m": [30.0, 31.0, 32.0, 33.0],
            "relative_humidity_2m": [40, 41, 42, 43],
            "surface_pressure": [1000.0, 1000.0, 999.0, 999.0],
            "wind_speed_10m": [3.0, 4.0, 5.0, 6.0],
            "wind_direction_10m": [10, 20, 30, 40],
            "precipitation": [0.0, 0.0, 0.1, 0.0],
        }
    return {"hourly": {"time": HOURS, **values}}


class _Response:
    status_code = 200

    def __init__(self, url: str) -> None:
        self._url = url

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return _payload(self._url)


@pytest.fixture(autouse=True)
def _clear_cache():
    st.cache_data.clear()
    yield
    st.cache_data.clear()


def test_schema_and_future_rows_removed(monkeypatch: pytest.MonkeyPatch) -> None:
    """A mocked payload becomes the master-file schema, without future hours."""
    monkeypatch.setattr(live_data, "_now_local", lambda: NOW)
    monkeypatch.setattr(live_data, "_write_cache", lambda city, frame: None)
    calls: list[str] = []

    def fake_get(url, params, timeout):
        calls.append(url)
        assert params["timezone"] == "Asia/Karachi"
        assert params["forecast_days"] == 0
        assert params["past_days"] == 7
        assert timeout == live_data.REQUEST_TIMEOUT_SECONDS
        return _Response(url)

    monkeypatch.setattr(live_data.requests, "get", fake_get)
    frame = live_data.fetch_recent("Lahore")
    assert list(frame.columns) == utils.REQUIRED_COLUMNS
    assert set(frame["time"]) == {"2024-06-15T10:00", "2024-06-15T11:00", "2024-06-15T12:00"}
    assert "2024-06-15T13:00" not in set(frame["time"])
    assert frame["city"].eq("Lahore").all()
    assert frame["aqi_category"].iloc[0] == "Good"
    assert frame["day_of_week"].iloc[0] == "Saturday"
    assert int(frame["is_weekend"].iloc[0]) == 1
    assert any("air-quality" in url for url in calls)
    assert any(url == live_data.WEATHER_URL for url in calls)


def test_hourly_gap_is_rejected() -> None:
    """A two-hour jump is not a valid hourly series."""
    frame = pd.DataFrame({name: [0, 0] for name in utils.REQUIRED_COLUMNS})
    frame["datetime"] = [pd.Timestamp("2024-01-01 00:00"), pd.Timestamp("2024-01-01 02:00")]
    frame["city"] = "Lahore"
    with pytest.raises(ValueError, match="Hourly sequence breaks"):
        live_data.validate_frame(frame)


def test_offline_uses_cache_then_csv(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """A network error uses the saved fetch, and then the CSV when no save exists."""
    monkeypatch.setattr(live_data, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(live_data, "_now_local", lambda: NOW)

    def explode(city, past_days=7):
        raise requests.ConnectionError("offline")

    monkeypatch.setattr(live_data, "fetch_recent", explode)
    cached = pd.DataFrame({name: [1, 1] for name in utils.REQUIRED_COLUMNS})
    cached["datetime"] = ["2024-06-15 10:00:00", "2024-06-15 11:00:00"]
    cached["city"] = "Peshawar"
    live_data._write_cache("Peshawar", cached)
    frame, status = live_data.load_recent("Peshawar")
    assert status["source"] == "cache"
    assert status["based_on"] == "2024-06-15 11:00"
    assert "2024-06-15 11:00" in status["notice"]
    assert "last saved fetch" in status["notice"]

    monkeypatch.setattr(live_data, "_read_cache", lambda city: None)
    fallback = cached.tail(1).copy()
    monkeypatch.setattr(live_data, "_read_csv_fallback", lambda city, past_days: fallback)
    frame, status = live_data.load_recent("Peshawar")
    assert status["source"] == "csv"
    assert "historical file" in status["notice"]
    assert status["based_on"] == "2024-06-15 11:00"


def test_fetch_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    """A second call within the hour does not call the network again."""
    monkeypatch.setattr(live_data, "_now_local", lambda: NOW)
    monkeypatch.setattr(live_data, "_write_cache", lambda city, frame: None)
    calls = {"n": 0}

    def fake_get(url, params, timeout):
        calls["n"] += 1
        return _Response(url)

    monkeypatch.setattr(live_data.requests, "get", fake_get)
    first = live_data.fetch_recent("Peshawar", 3)
    second = live_data.fetch_recent("Peshawar", 3)
    assert calls["n"] == 2
    assert first.equals(second)
