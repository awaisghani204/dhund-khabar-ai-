"""Shared pytest fixtures and synthetic-data helpers for Dhund-Khabar AI.

All data here is generated in memory from a seeded generator. Tests must never
read the real files under ``data/``, so results do not depend on local data.
"""

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import pytest

from src.pipeline import AirQualityDataPipeline
from src.utils import CITIES, REQUIRED_COLUMNS

# Distinct base levels per city so cross-city leakage in lag, rolling, and
# target shifts shows up as a clearly wrong value instead of a near match.
CITY_COORDS: dict[str, tuple[float, float]] = {
    "Peshawar": (34.0151, 71.5249),
    "Lahore": (31.5497, 74.3436),
}
DEFAULT_CITY_OFFSETS: dict[str, float] = {"Peshawar": 0.0, "Lahore": 100.0}

# (upper bound of us_aqi, label); mirrors the usual US AQI tier names.
_AQI_LABELS: list[tuple[float, str]] = [
    (50, "Good"), (100, "Moderate"), (150, "Unhealthy for Sensitive Groups"),
    (200, "Unhealthy"), (300, "Very Unhealthy"), (float("inf"), "Hazardous"),
]


def _aqi_label(value: float) -> str:
    """Return the AQI tier label for one us_aqi value."""
    for upper, label in _AQI_LABELS:
        if value <= upper:
            return label
    return "Hazardous"  # unreachable; keeps the return type total


def make_synthetic_master(
    n_hours_per_city: int = 240,
    start: str = "2024-12-01 00:00",
    seed: int = 0,
    cities: list[str] | None = None,
    city_offsets: dict[str, float] | None = None,
    ramp: bool = False,
) -> pd.DataFrame:
    """Build a synthetic master frame with all Required_Columns in file order.

    Args:
        n_hours_per_city: Consecutive hourly rows generated per city.
        start: First timestamp of every city's hourly grid.
        seed: Seed for ``np.random.default_rng``; same seed gives same frame.
        cities: City names; defaults to ``CITIES`` (Peshawar, Lahore).
        city_offsets: Added to every pollutant level per city so the cities
            sit at different levels (default: Lahore is 100 above Peshawar).
        ramp: If True, pm2_5, pm10 and us_aqi are exact linear ramps over the
            row index (plus the city offset) with no noise, which makes lag
            and rolling-mean expectations computable by hand. Otherwise they
            are a daily cycle plus noise. Either way each row's value is unique
            within a city, so shifted rows can be identified.

    Returns:
        DataFrame with ``REQUIRED_COLUMNS`` in order. ``time`` is a
        ``YYYY-MM-DDTHH:MM`` string, ``datetime`` is ``YYYY-MM-DD HH:MM:SS``,
        ``day_of_week`` is the weekday name, and pollutants are non-negative.
    """
    rng = np.random.default_rng(seed)
    cities = list(CITIES) if cities is None else list(cities)
    offsets = DEFAULT_CITY_OFFSETS if city_offsets is None else city_offsets
    stamps = pd.date_range(start=start, periods=n_hours_per_city, freq="h")
    idx = np.arange(n_hours_per_city, dtype=float)
    n = n_hours_per_city

    frames: list[pd.DataFrame] = []
    for k, city in enumerate(cities):
        off = float(offsets.get(city, 100.0 * k))
        hour = stamps.hour.to_numpy()
        daily = np.sin(2 * np.pi * hour / 24.0)
        if ramp:
            pm25 = 20.0 + 0.5 * idx + off
            pm10 = 40.0 + 1.0 * idx + off
            aqi = 50.0 + 0.7 * idx + off
        else:
            # Tiny index term keeps values unique even if noise were tiny.
            pm25 = 60.0 + off + 15.0 * daily + rng.normal(0, 3, n) + 0.001 * idx
            pm10 = 1.8 * pm25 + rng.normal(0, 4, n)
            aqi = 1.5 * pm25 + rng.normal(0, 5, n)
        pm25 = np.clip(pm25, 0.1, None)
        pm10 = np.clip(pm10, 0.1, None)
        aqi = np.clip(aqi, 0.0, None)

        lat, lon = CITY_COORDS.get(city, (30.0 + k, 70.0 + k))
        df = pd.DataFrame({
            "city": city,
            "latitude": lat,
            "longitude": lon,
            "time": stamps.strftime("%Y-%m-%dT%H:%M"),
            "pm10": pm10,
            "pm2_5": pm25,
            "carbon_monoxide": np.abs(rng.normal(600, 100, n)),
            "nitrogen_dioxide": np.abs(rng.normal(30, 8, n)),
            "sulphur_dioxide": np.abs(rng.normal(15, 4, n)),
            "ozone": np.abs(rng.normal(50, 12, n)),
            "dust": np.abs(rng.normal(20, 6, n)),
            "us_aqi": aqi,
            "temperature_2m": 12.0 + 6.0 * daily + rng.normal(0, 1, n),
            "relative_humidity_2m": np.clip(70 - 15 * daily + rng.normal(0, 5, n), 0, 100),
            "surface_pressure": 1000.0 + rng.normal(0, 3, n),
            "wind_speed_10m": np.abs(rng.normal(8, 4, n)),
            "wind_direction_10m": rng.integers(0, 360, n),
            "precipitation": np.abs(rng.normal(0, 0.3, n)),
            "datetime": stamps.strftime("%Y-%m-%d %H:%M:%S"),
            "year": stamps.year,
            "month": stamps.month,
            "day": stamps.day,
            "hour": stamps.hour,
            "day_of_week": stamps.day_name(),
            "is_weekend": (stamps.dayofweek >= 5).astype(int),
            "aqi_category": [_aqi_label(v) for v in aqi],
        })
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    return out[REQUIRED_COLUMNS]


def write_synthetic_csv(df: pd.DataFrame, tmp_path: Path,
                        name: str = "synthetic_master.csv") -> Path:
    """Write a synthetic master frame to ``tmp_path`` as CSV and return the path.

    Timestamps are already strings, so the file matches the real master format.
    """
    path = tmp_path / name
    df.to_csv(path, index=False)
    return path


def build_features_frame(df: pd.DataFrame, tmp_path: Path,
                         with_targets: bool = True) -> pd.DataFrame:
    """Run a synthetic master through the pipeline feature steps.

    Writes the CSV, then load_data -> generate_time_features ->
    create_lag_and_rolling_features (-> create_targets). ``flags_path`` points
    to a file that does not exist so ``is_forecast_like`` is 0 and the real
    ``audit_flags.json`` is never read.
    """
    path = write_synthetic_csv(df, tmp_path)
    pipe = AirQualityDataPipeline(flags_path=tmp_path / "no_such_flags.json")
    out = pipe.load_data(path)
    out = pipe.generate_time_features(out)
    out = pipe.create_lag_and_rolling_features(out)
    if with_targets:
        out = pipe.create_targets(out)
    return out


@pytest.fixture
def synthetic_master() -> Callable[..., pd.DataFrame]:
    """Factory fixture exposing ``make_synthetic_master`` to tests."""
    return make_synthetic_master


@pytest.fixture
def small_master() -> pd.DataFrame:
    """Two cities, 240 hourly rows each, seed 0."""
    return make_synthetic_master(240, seed=0)


@pytest.fixture
def boundary_frame() -> pd.DataFrame:
    """Two cities hourly from 2024-12-20 00:00 to 2025-01-10 23:00.

    Spans the 2025-01-01 split date and the year change, for split, embargo,
    and boundary tests. 22 days x 24 hours = 528 rows per city.
    """
    return make_synthetic_master(528, start="2024-12-20 00:00", seed=1)


@pytest.fixture
def tiny_model_data() -> pd.DataFrame:
    """300 rows with numeric features ``f0``..``f7`` and a ``target`` column.

    The target is a noisy linear function of the features, so even a small
    model can learn it quickly.
    """
    rng = np.random.default_rng(7)
    x = rng.normal(size=(300, 8))
    coefs = np.array([3.0, -2.0, 1.5, 0.0, 0.5, 0.0, -1.0, 2.0])
    frame = pd.DataFrame(x, columns=[f"f{i}" for i in range(8)])
    frame["target"] = x @ coefs + rng.normal(0, 0.5, 300) + 50.0
    return frame


@pytest.fixture
def csv_writer(tmp_path: Path) -> Callable[..., Path]:
    """Fixture returning ``f(df, name=...) -> Path`` that writes CSVs to tmp_path."""
    def _write(df: pd.DataFrame, name: str = "synthetic_master.csv") -> Path:
        return write_synthetic_csv(df, tmp_path, name)
    return _write


@pytest.fixture
def features_frame(tmp_path: Path) -> pd.DataFrame:
    """Feature frame (time, lag, rolling, targets) built from synthetic data."""
    return build_features_frame(make_synthetic_master(300, seed=2), tmp_path)
