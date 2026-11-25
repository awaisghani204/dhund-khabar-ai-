"""Smoke tests for the shared fixtures in conftest.py."""

import pandas as pd

from src.pipeline import AirQualityDataPipeline
from src.utils import CITIES, REQUIRED_COLUMNS
from tests.conftest import make_synthetic_master, write_synthetic_csv


def test_master_columns_formats_and_values(small_master):
    df = small_master
    assert list(df.columns) == REQUIRED_COLUMNS and len(REQUIRED_COLUMNS) == 26
    assert set(df["city"]) == set(CITIES)
    assert df["time"].str.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}").all()
    assert df["datetime"].str.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}").all()
    parsed = pd.to_datetime(df["datetime"], format="%Y-%m-%d %H:%M:%S")
    assert (parsed.dt.day_name() == df["day_of_week"]).all()
    assert (df[["pm10", "pm2_5", "us_aqi", "ozone", "dust"]] >= 0).all().all()
    assert df["relative_humidity_2m"].between(0, 100).all()


def test_master_seeded_unique_and_city_levels_differ():
    a = make_synthetic_master(48, seed=3)
    assert a.equals(make_synthetic_master(48, seed=3))
    for _, g in a.groupby("city"):
        assert g["pm2_5"].is_unique
    means = a.groupby("city")["pm2_5"].mean()
    assert means["Lahore"] - means["Peshawar"] > 50


def test_ramp_is_linear():
    g = make_synthetic_master(10, ramp=True).query("city == 'Peshawar'")
    assert (g["pm2_5"].diff().dropna().round(9) == 0.5).all()


def test_boundary_frame_range(boundary_frame):
    for _, g in boundary_frame.groupby("city"):
        assert g["datetime"].iloc[0] == "2024-12-20 00:00:00"
        assert g["datetime"].iloc[-1] == "2025-01-10 23:00:00"
        assert len(g) == 528


def test_tiny_model_data_shape(tiny_model_data):
    assert tiny_model_data.shape == (300, 9)
    assert list(tiny_model_data.columns[:8]) == [f"f{i}" for i in range(8)]
    assert "target" in tiny_model_data.columns


def test_csv_helper_round_trips_through_load_data(tmp_path):
    df = make_synthetic_master(72, seed=5)
    path = write_synthetic_csv(df, tmp_path)
    pipe = AirQualityDataPipeline(flags_path=tmp_path / "missing.json")
    loaded = pipe.load_data(path)
    assert len(loaded) == len(df)
    assert (loaded["is_forecast_like"] == 0).all()


def test_features_frame_fixture(features_frame):
    assert len(features_frame) > 0
    assert set(features_frame["city"]) == set(CITIES)
