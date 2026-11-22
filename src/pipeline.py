"""Data pipeline for Dhund-Khabar AI.

This module holds the class ``AirQualityDataPipeline``. It loads the master
hourly CSV, validates the schema, sorts the rows, applies the Gap_Strategy the
user approved at the Phase 1 checkpoint, and flags Forecast_Like_Rows from
``data/processed/audit_flags.json``. The same class adds the time, lag, and
rolling features, creates the future targets, builds the embargo-aware time
split, selects the feature columns per target and mode, and saves the
processed files to ``data/processed/``.

Run ``python -m src.pipeline`` from the project root to build
``features_all.csv`` and ``model_dataset.csv`` and print the Phase 2 summary.
"""

import json
import re
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import (
    CITIES,
    HORIZONS,
    MASTER_PATH,
    POLLUTANT_COLUMNS,
    PROCESSED_DIR,
    REQUIRED_COLUMNS,
    SPLIT_DATE,
    TARGET_TYPES,
    WEATHER_COLUMNS,
    WINTER_MONTHS,
    ensure_dir,
    inversion_proxy,
    stagnation_flag,
)

DATETIME_FORMAT: str = "%Y-%m-%d %H:%M:%S"
TIME_FORMAT: str = "%Y-%m-%dT%H:%M"
DEFAULT_FLAGS_PATH: Path = PROCESSED_DIR / "audit_flags.json"

# us_aqi gets the same short-gap forward fill as the concentrations (approved
# Gap_Strategy step 3), but it is not clipped (step 4), so it is kept separate
# from POLLUTANT_COLUMNS.
FILL_COLUMNS: list[str] = POLLUTANT_COLUMNS + ["us_aqi"]
# Columns that are constant within a city, so a row inserted by reindexing can
# take them from the city's existing rows without imputing a measurement.
STATIC_COLUMNS: list[str] = ["latitude", "longitude"]
# Columns that are pure functions of the timestamp. Filling them on inserted
# rows keeps the calendar consistent and adds no measured value.
CALENDAR_COLUMNS: list[str] = ["time", "year", "month", "day", "hour",
                               "day_of_week", "is_weekend"]
# Trailing-window features as (output column, source column, window in hours,
# statistic). One table keeps the column names and windows in a single place,
# so the design's feature list and the code cannot drift apart.
ROLLING_FEATURES: list[tuple[str, str, int, str]] = [
    ("pm2_5_rolling_mean_24h", "pm2_5", 24, "mean"),
    ("pm2_5_rolling_mean_72h", "pm2_5", 72, "mean"),
    ("pm2_5_rolling_std_24h", "pm2_5", 24, "std"),
    ("wind_speed_rolling_mean_6h", "wind_speed_10m", 6, "mean"),
    ("humidity_rolling_mean_6h", "relative_humidity_2m", 6, "mean"),
    ("temperature_rolling_mean_6h", "temperature_2m", 6, "mean"),
]
# Matched by suffix rather than listed, so the split also recognises pm2_5
# lags built from a custom lag_steps list.
LAG_COLUMN_PATTERN: re.Pattern[str] = re.compile(r"_lag_\d+$")
# Differences against a lagged value share the lag warm-up, so the split
# treats them like lag and rolling columns.
CHANGE_FEATURES: list[str] = ["pm2_5_change_24h"]
# Target columns are "{target}_t_plus_{h}"; the split reads h back from the
# name when it is called without an explicit horizon.
TARGET_COLUMN_PATTERN: re.Pattern[str] = re.compile(
    rf"^(?:{'|'.join(re.escape(t) for t in TARGET_TYPES)})_t_plus_(?P<hours>\d+)$"
)
ONE_HOUR: pd.Timedelta = pd.Timedelta(1, unit="h")

# Feature groups for get_feature_columns, one list per row of the design's
# feature table and in table order, so each list can be checked against the
# design at a glance. The pm2_5 lags and the us_aqi family are built from the
# class lag constants instead, so they always match the created columns.
FEATURE_MODES: list[str] = ["forecast", "simulation"]
# Concentrations at time t. pm2_5 leads, as in the design table; the file
# order (POLLUTANT_COLUMNS) has pm10 first.
FORECAST_POLLUTANT_FEATURES: list[str] = [
    "pm2_5", "pm10", "carbon_monoxide", "nitrogen_dioxide",
    "sulphur_dioxide", "ozone", "dust",
]
PM25_ROLLING_FEATURES: list[str] = [
    "pm2_5_rolling_mean_24h", "pm2_5_rolling_mean_72h",
    "pm2_5_rolling_std_24h", "pm2_5_change_24h",
]
WEATHER_ROLLING_FEATURES: list[str] = [
    "wind_speed_rolling_mean_6h", "humidity_rolling_mean_6h",
    "temperature_rolling_mean_6h",
]
# Both are instantaneous functions of the current hour's weather, so the
# What-If scenario builder can recompute them from the slider values.
DERIVED_WEATHER_FEATURES: list[str] = ["stagnation_flag", "inversion_proxy"]
TIME_FEATURES: list[str] = [
    "hour_sin", "hour_cos", "month_sin", "month_cos", "is_weekend", "is_winter",
]
# The single pollution input of Simulation_Mode: a trailing 24-hour mean that
# gives the model the city's current baseline load without autoregressive
# hour-by-hour lags.
SIMULATION_POLLUTION_FEATURES: list[str] = ["pm2_5_rolling_mean_24h"]

FEATURES_FILENAME: str = "features_all.csv"
MODEL_DATASET_FILENAME: str = "model_dataset.csv"


def _invalid_hour_steps(steps: list[object]) -> list[object]:
    """Return the entries that are not whole numbers of hours of at least 1.

    Args:
        steps: Candidate shift lengths in hours.

    Returns:
        The invalid entries in input order; an empty list when all are valid.
    """
    # bool is a subclass of int, so True would otherwise pass as a 1-hour step.
    return [k for k in steps
            if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 1]


def _format_breakdown(counts: dict[str, int]) -> str:
    """Format the non-zero per-column counts for a printed Gap_Strategy line.

    Args:
        counts: Count per column name.

    Returns:
        Text such as ``" (nitrogen_dioxide 2, ozone 59)"``, or an empty string
        when every count is zero.
    """
    parts = [f"{col} {n}" for col, n in counts.items() if n > 0]
    return f" ({', '.join(parts)})" if parts else ""


class AirQualityDataPipeline:
    """Load, validate, and feature-engineer the smog data, and build the time-based split.

    Attributes:
        DEFAULT_LAG_STEPS: pm2_5 lag lengths in hours used when no lag list is given.
        US_AQI_LAG_STEPS: us_aqi lag lengths in hours.
        flags_path: Location of ``audit_flags.json``, or None to skip it.
        ffill_limit: Longest interior gap, in consecutive hours, that is
            forward-filled.
        data: The frame from the last ``load_data`` call, or None before it.
        features: The feature frame (all rows, no targets) from the last
            ``build_features`` call, or None before it.
        gap_counts: The four Gap_Strategy counts from the last ``load_data`` call.
        target_drop_counts: Rows per city dropped by the last ``create_targets``
            call because a target was NaN.
        warmup_drop_counts: Rows per city dropped by the last
            ``prepare_train_test_split`` call because a lag, rolling, or change
            feature was NaN.
    """

    DEFAULT_LAG_STEPS: tuple[int, ...] = (1, 3, 6, 12, 24, 48)
    US_AQI_LAG_STEPS: tuple[int, ...] = (1, 24)

    def __init__(self, flags_path: str | Path | None = DEFAULT_FLAGS_PATH,
                 ffill_limit: int = 3) -> None:
        """Create a pipeline.

        Args:
            flags_path: Path to ``audit_flags.json`` written by
                ``scripts/audit_data.py``. None, or a path that does not exist,
                sets ``is_forecast_like`` to 0 for every row.
            ffill_limit: Longest interior gap of pollutant or us_aqi values, in
                consecutive hours, that is forward-filled within a city. Longer
                gaps stay entirely missing. The approved Gap_Strategy uses 3.

        Raises:
            ValueError: If ``ffill_limit`` is negative.
        """
        if ffill_limit < 0:
            raise ValueError(f"ffill_limit must be 0 or more, got {ffill_limit}")
        self.flags_path: Path | None = Path(flags_path) if flags_path is not None else None
        self.ffill_limit: int = int(ffill_limit)
        self.data: pd.DataFrame | None = None
        self.features: pd.DataFrame | None = None
        self.gap_counts: dict[str, int] = {}
        self.target_drop_counts: dict[str, int] = {}
        self.warmup_drop_counts: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Loading and validation
    # ------------------------------------------------------------------

    def load_data(self, filepath: str | Path) -> pd.DataFrame:
        """Read, validate, sort, and clean the master hourly CSV.

        Steps: read the CSV; check that every Required_Column is present;
        parse ``datetime`` with ``%Y-%m-%d %H:%M:%S``; stable-sort by city and
        then datetime; apply the approved Gap_Strategy
        (``_apply_gap_strategy``); add ``is_forecast_like`` (0 or 1) from
        ``audit_flags.json``. The result is stored in ``self.data``.

        Args:
            filepath: Path to a CSV with the Master_Dataset columns.

        Returns:
            A copy of the cleaned frame, sorted by city and then datetime, with
            the original columns in file order plus ``is_forecast_like``.

        Raises:
            ValueError: If Required_Columns are missing (the message names every
                missing column in file order), if a ``datetime`` value does not
                match the format, or if a ``datetime`` value is empty.
        """
        df = pd.read_csv(filepath)

        missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(missing)}")

        # An explicit format makes a malformed timestamp fail loudly instead of
        # being guessed, and it is much faster than format inference.
        df["datetime"] = pd.to_datetime(df["datetime"], format=DATETIME_FORMAT)
        n_missing_ts = int(df["datetime"].isna().sum())
        if n_missing_ts:
            raise ValueError(
                f"datetime has {n_missing_ts} empty values; every row needs a "
                "timestamp to be placed on the hourly grid"
            )

        # A stable sort keeps duplicate timestamps in file order, so "keep the
        # first duplicate" in the Gap_Strategy means the first one in the file.
        df = df.sort_values(["city", "datetime"], kind="mergesort").reset_index(drop=True)
        df = self._apply_gap_strategy(df)
        df["is_forecast_like"] = self._forecast_like_flags(df["datetime"])

        self.data = df
        return df.copy()

    def filter_city(self, city_name: str) -> pd.DataFrame:
        """Return the loaded rows of one city.

        Args:
            city_name: City name exactly as in the ``city`` column.

        Returns:
            A copy of the rows whose ``city`` equals ``city_name``, in
            datetime order, with a fresh 0..n-1 index.

        Raises:
            RuntimeError: If ``load_data`` has not been called yet.
            ValueError: If ``city_name`` is not in the loaded data. The message
                lists the available cities.
        """
        if self.data is None:
            raise RuntimeError("Call load_data first")
        available = sorted(self.data["city"].dropna().unique().tolist())
        if city_name not in available:
            raise ValueError(f"Unknown city '{city_name}'. Available cities: {available}")
        return self.data[self.data["city"] == city_name].reset_index(drop=True).copy()

    # ------------------------------------------------------------------
    # Gap_Strategy (approved at the Phase 1 checkpoint)
    # ------------------------------------------------------------------

    def _apply_gap_strategy(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the Gap_Strategy approved at the Phase 1 checkpoint.

        Input must be sorted by city and then datetime, with ``datetime``
        parsed. The steps run in this order:

        1. Drop duplicate (city, datetime) rows, keeping the first, then
           reindex each city onto a complete hourly grid from its first to its
           last timestamp. Inserted rows get city, latitude, longitude, and the
           calendar columns (time, year, month, day, hour, day_of_week,
           is_weekend) from the timestamp and the city; every measured column
           (pollutants, us_aqi, weather) and aqi_category stay NaN.
        2. Leave the Leading_Gap (missing values at the start of a city's
           series) unimputed.
        3. Clip negative values in ``POLLUTANT_COLUMNS`` to 0. This runs before
           the fill, so the count is the number of negative values read.
           us_aqi is not clipped.
        4. Forward-fill interior gaps of pollutant and us_aqi values within
           each city, only when the whole run of missing hours is at most
           ``ffill_limit`` hours long and has an observed value both before and
           after it in the same city. A longer run stays entirely NaN; a plain
           ``groupby("city").ffill(limit=3)`` would wrongly fill the first 3
           hours of it. Leading and trailing runs are not interior and stay
           NaN. Weather columns are never filled.
        5. Keep us_aqi values above 500, keep all rows, and keep aqi_category
           as-is (including "Unknown").

        The four counts are printed and stored in ``self.gap_counts`` under
        ``duplicates_dropped``, ``rows_inserted``, ``values_clipped``, and
        ``values_filled``.

        Args:
            df: Frame sorted by city and datetime with a parsed ``datetime``.

        Returns:
            The cleaned frame, sorted by city and datetime, with a 0..n-1
            index and the same columns in the same order as the input.
        """
        columns = list(df.columns)
        original_dtypes = df.dtypes

        # Step 1a: duplicates. The input is stably sorted, so keep="first"
        # keeps the row that came first in the file.
        dup_mask = df.duplicated(subset=["city", "datetime"], keep="first")
        n_duplicates = int(dup_mask.sum())
        df = df.loc[~dup_mask]

        # Step 1b: complete hourly grid per city, so that a shift of k rows in
        # the lag and target code always means k hours.
        city_frames: list[pd.DataFrame] = []
        n_inserted = 0
        for city, group in df.groupby("city", sort=True, dropna=False):
            grid = pd.date_range(group["datetime"].iloc[0], group["datetime"].iloc[-1],
                                 freq="h", name="datetime")
            # The union keeps any off-the-hour timestamp instead of silently
            # dropping it, because all rows must be kept; create_targets then
            # reports the bad step.
            grid = grid.union(pd.DatetimeIndex(group["datetime"]))
            grid.name = "datetime"
            full = group.set_index("datetime").reindex(grid)
            inserted = ~grid.isin(group["datetime"])
            n_city_inserted = int(inserted.sum())
            if n_city_inserted:
                stamps = grid[inserted]
                full.loc[inserted, "city"] = city
                for col in STATIC_COLUMNS:
                    full.loc[inserted, col] = group[col].iloc[0]
                full.loc[inserted, "time"] = stamps.strftime(TIME_FORMAT).to_numpy()
                full.loc[inserted, "year"] = stamps.year.to_numpy()
                full.loc[inserted, "month"] = stamps.month.to_numpy()
                full.loc[inserted, "day"] = stamps.day.to_numpy()
                full.loc[inserted, "hour"] = stamps.hour.to_numpy()
                full.loc[inserted, "day_of_week"] = stamps.day_name().to_numpy()
                full.loc[inserted, "is_weekend"] = (stamps.dayofweek >= 5).astype(int)
                n_inserted += n_city_inserted
            city_frames.append(full.reset_index())

        if city_frames:
            df = pd.concat(city_frames, ignore_index=True)[columns]
        else:
            df = df.reset_index(drop=True)

        # Reindexing turns integer columns into float when rows are inserted.
        # Calendar columns are complete again after the fill, so they get
        # their file dtype back; measured columns keep NaN and stay float.
        for col in CALENDAR_COLUMNS:
            if (pd.api.types.is_integer_dtype(original_dtypes[col])
                    and df[col].notna().all()):
                df[col] = df[col].astype(original_dtypes[col])

        # Step 3: negative concentrations are model artifacts, not physical
        # values. NaN compares as False, so missing values are not counted.
        negative = df[POLLUTANT_COLUMNS] < 0
        clipped_per_col = {col: int(n) for col, n in negative.sum().items()}
        n_clipped = sum(clipped_per_col.values())
        df[POLLUTANT_COLUMNS] = df[POLLUTANT_COLUMNS].clip(lower=0)

        # Step 4: short interior gaps only. Forward fill copies the last value
        # at or before the gap, so it uses no future value.
        city_codes = df.groupby("city", sort=False, dropna=False).ngroup().to_numpy()
        filled_per_col: dict[str, int] = {}
        for col in FILL_COLUMNS:
            fill_mask = self._short_interior_gap_mask(df[col], city_codes, self.ffill_limit)
            filled_per_col[col] = int(fill_mask.sum())
            if filled_per_col[col]:
                forward = df[col].groupby(city_codes).ffill()
                df[col] = df[col].mask(fill_mask, forward)
        n_filled = sum(filled_per_col.values())

        self.gap_counts = {
            "duplicates_dropped": n_duplicates,
            "rows_inserted": n_inserted,
            "values_clipped": n_clipped,
            "values_filled": n_filled,
        }
        print("Gap_Strategy (approved at the Phase 1 checkpoint):")
        print(f"  duplicate (city, datetime) rows dropped: {n_duplicates}")
        print(f"  rows inserted by hourly reindexing: {n_inserted}")
        print(f"  negative pollutant values clipped to 0: {n_clipped}"
              f"{_format_breakdown(clipped_per_col)}")
        print(f"  values forward-filled (interior gaps of at most {self.ffill_limit} h): "
              f"{n_filled}{_format_breakdown(filled_per_col)}")
        return df

    @staticmethod
    def _short_interior_gap_mask(values: pd.Series, city_codes: np.ndarray,
                                 max_len: int) -> np.ndarray:
        """Mark missing values that belong to a short interior gap.

        A gap is a run of consecutive missing values within one city. It is
        short when its whole length is at most ``max_len`` rows, and interior
        when the same city has an observed value right before and right after
        it. Rows must be sorted by city and then datetime on a complete hourly
        grid, so one row is one hour.

        Args:
            values: One column of the sorted frame.
            city_codes: Integer city code per row, aligned with ``values``.
            max_len: Longest gap length, in rows, that is marked.

        Returns:
            Boolean array, True where the value should be forward-filled.
        """
        is_na = values.isna().to_numpy()
        n_rows = len(is_na)
        if n_rows == 0 or not is_na.any():
            return np.zeros(n_rows, dtype=bool)

        # A new run starts at the first row, where missingness flips, and at a
        # city boundary, so a run never spans two cities.
        run_start = np.ones(n_rows, dtype=bool)
        run_start[1:] = (is_na[1:] != is_na[:-1]) | (city_codes[1:] != city_codes[:-1])
        run_id = np.cumsum(run_start) - 1
        run_length = np.bincount(run_id)[run_id]

        # Within a city, a missing value has an observed value somewhere before
        # it exactly when forward fill reaches it, and one after it exactly
        # when backward fill reaches it. This excludes leading and trailing runs.
        grouped = values.groupby(city_codes)
        has_before = grouped.ffill().notna().to_numpy()
        has_after = grouped.bfill().notna().to_numpy()

        return is_na & has_before & has_after & (run_length <= max_len)

    # ------------------------------------------------------------------
    # Forecast_Like_Row flag
    # ------------------------------------------------------------------

    def _forecast_like_flags(self, timestamps: pd.Series) -> np.ndarray:
        """Compute ``is_forecast_like`` from ``audit_flags.json``.

        Rows at or after ``forecast_like_start`` get 1, all others 0. Rows are
        only flagged, never removed (Requirement 2.4).

        Args:
            timestamps: Parsed ``datetime`` values of the cleaned frame.

        Returns:
            Integer array of 0 and 1 aligned with ``timestamps``. All 0 when
            the flags file is absent (a notice is printed) or when
            ``forecast_like_start`` is null.
        """
        if self.flags_path is None or not self.flags_path.exists():
            print(f"Notice: audit flags file not found ({self.flags_path}); "
                  "is_forecast_like is 0 for every row. Run "
                  "`python scripts/audit_data.py` to create it.")
            return np.zeros(len(timestamps), dtype=int)

        with self.flags_path.open(encoding="utf-8") as handle:
            flags = json.load(handle)
        start = flags.get("forecast_like_start")
        if start is None:
            print("is_forecast_like: 0 rows flagged (audit_flags.json has no "
                  "forecast_like_start)")
            return np.zeros(len(timestamps), dtype=int)

        start_ts = pd.to_datetime(start, format=DATETIME_FORMAT)
        flags_array = (timestamps >= start_ts).to_numpy().astype(int)
        print(f"is_forecast_like: {int(flags_array.sum())} rows flagged "
              f"(from {start_ts:%Y-%m-%d %H:%M:%S})")
        return flags_array

    # ------------------------------------------------------------------
    # Time features
    # ------------------------------------------------------------------

    def generate_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add cyclical hour and month features and the weekend and winter flags.

        Every value is derived from the parsed ``datetime`` column:

        - ``hour_sin = sin(2*pi*hour/24)`` and ``hour_cos = cos(2*pi*hour/24)``,
          with hour in 0..23.
        - ``month_sin = sin(2*pi*month/12)`` and ``month_cos = cos(2*pi*month/12)``,
          with month in 1..12 used as-is.
        - ``is_weekend``: 1 for Saturday and Sunday, 0 for other days. The
          file's ``is_weekend`` column is overwritten.
        - ``is_winter``: 1 for months in ``WINTER_MONTHS`` (11, 12, 1, 2), 0
          for other months.

        Args:
            df: Frame with a parsed (datetime64) ``datetime`` column.

        Returns:
            A copy of ``df`` with hour_sin, hour_cos, month_sin, month_cos, and
            is_winter added and is_weekend recomputed. The input frame is not
            changed.
        """
        df = df.copy()
        stamps = df["datetime"].dt
        hour = stamps.hour.to_numpy(dtype=float)
        month = stamps.month.to_numpy(dtype=float)

        # Sine and cosine together place each hour and month on a circle, so
        # 23:00 sits next to 00:00 and December next to January; a raw hour or
        # month number would put them at opposite ends of the scale.
        hour_angle = 2.0 * np.pi * hour / 24.0
        month_angle = 2.0 * np.pi * month / 12.0
        df["hour_sin"] = np.sin(hour_angle)
        df["hour_cos"] = np.cos(hour_angle)
        df["month_sin"] = np.sin(month_angle)
        df["month_cos"] = np.cos(month_angle)

        # Recomputed from the timestamp instead of trusting the file's column,
        # so the flag is also correct on rows inserted by hourly reindexing
        # and cannot disagree with the parsed datetime. dayofweek: Monday = 0,
        # so Saturday = 5 and Sunday = 6.
        df["is_weekend"] = (stamps.dayofweek.to_numpy() >= 5).astype(int)
        df["is_winter"] = np.isin(stamps.month.to_numpy(), WINTER_MONTHS).astype(int)
        return df

    # ------------------------------------------------------------------
    # Lag and rolling features
    # ------------------------------------------------------------------

    def create_lag_and_rolling_features(self, df: pd.DataFrame,
                                        lag_steps: list[int] | None = None) -> pd.DataFrame:
        """Add per-city lag, trailing rolling, change, and derived weather features.

        The frame must be sorted by city and then datetime, as ``load_data``
        returns it, and each city must lie on a complete hourly grid (the
        Gap_Strategy guarantees this), so a shift of k rows within a city
        means k hours. Every lag, rolling, and change feature is computed with
        ``groupby("city")``, so no value ever crosses from one city into
        another.

        Added columns:

        - ``pm2_5_lag_{k}`` for each k in ``lag_steps``: pm2_5 k hours earlier.
        - ``us_aqi_lag_1`` and ``us_aqi_lag_24``: us_aqi 1 and 24 hours earlier.
        - ``pm2_5_rolling_mean_24h``, ``pm2_5_rolling_mean_72h``, and
          ``pm2_5_rolling_std_24h`` (sample std, ddof=1).
        - ``wind_speed_rolling_mean_6h``, ``humidity_rolling_mean_6h``, and
          ``temperature_rolling_mean_6h`` from wind_speed_10m,
          relative_humidity_2m, and temperature_2m.
        - ``pm2_5_change_24h = pm2_5 - (pm2_5 24 hours earlier)``.
        - ``stagnation_flag``: 1 where wind_speed_10m < 5 km/h
          (``STAGNATION_WIND_KMH``), 0 elsewhere, from ``utils.stagnation_flag``.
        - ``inversion_proxy``, from ``utils.inversion_proxy``, with the exact
          formula::

              inversion_proxy = relative_humidity_2m / 100 - temperature_2m / 40

          temperature_2m in degrees C, relative_humidity_2m in %, no unit. It
          increases as temperature_2m decreases and as relative_humidity_2m
          increases. It is a heuristic for cold, moist air, not a measured
          inversion.

        Every rolling window is trailing: the value at hour t uses hours
        t-w+1 through t of the same city, and nothing after t. ``min_periods``
        equals the window, so a window with any missing hour (the warm-up at
        the start of a city, the Leading_Gap, or a long interior gap) gives
        NaN instead of a partial-window value.

        Args:
            df: Frame sorted by city and datetime, with at least city,
                datetime, pm2_5, us_aqi, wind_speed_10m, relative_humidity_2m,
                and temperature_2m.
            lag_steps: pm2_5 lag lengths in hours. None means
                ``DEFAULT_LAG_STEPS`` (1, 3, 6, 12, 24, 48).

        Returns:
            A copy of ``df`` with the feature columns added. Row order and
            index are unchanged, and the input frame is not changed.

        Raises:
            ValueError: If the frame is not sorted by city and datetime, or if
                a lag step is not a whole number of hours of at least 1.
        """
        # A MultiIndex is monotonic exactly when the rows are in
        # lexicographic (city, datetime) order, which is the order
        # load_data's sort_values(["city", "datetime"]) produces.
        order = pd.MultiIndex.from_arrays([df["city"], df["datetime"]])
        if not order.is_monotonic_increasing:
            raise ValueError("Frame must be sorted by city and datetime")

        steps = list(self.DEFAULT_LAG_STEPS) if lag_steps is None else list(lag_steps)
        # A zero or negative shift would copy the current or a future value
        # into a "lag" column, which is exactly the leakage this method exists
        # to prevent.
        bad_steps = [k for k in steps
                     if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 1]
        if bad_steps:
            raise ValueError(f"lag_steps must be whole hours of at least 1, got {bad_steps}")

        df = df.copy()
        # sort=False keeps the groups in the frame's own order. Every result
        # below is aligned back to the original rows by pandas, so it is
        # correct for any index.
        by_city = df.groupby("city", sort=False)

        for k in steps:
            df[f"pm2_5_lag_{k}"] = by_city["pm2_5"].shift(k)
        for k in self.US_AQI_LAG_STEPS:
            df[f"us_aqi_lag_{k}"] = by_city["us_aqi"].shift(k)

        # transform runs the trailing window on each city's rows alone, so a
        # window at the start of the second city cannot reach back into the
        # last hours of the first city.
        for out_col, src_col, window, stat in ROLLING_FEATURES:
            df[out_col] = by_city[src_col].transform(
                lambda s, w=window, f=stat: getattr(s.rolling(w, min_periods=w), f)()
            )

        # Taken from its own grouped shift rather than from pm2_5_lag_24, so
        # the change feature exists even when lag_steps leaves out 24.
        pm25_24h_ago = by_city["pm2_5"].shift(24).to_numpy(dtype=float)
        df["pm2_5_change_24h"] = df["pm2_5"].to_numpy(dtype=float) - pm25_24h_ago

        # Both helpers are row-wise functions of the current hour, so they
        # need no grouping; they return NumPy arrays, assigned by position.
        df["stagnation_flag"] = stagnation_flag(df["wind_speed_10m"].to_numpy(dtype=float))
        df["inversion_proxy"] = inversion_proxy(df["temperature_2m"].to_numpy(dtype=float),
                                                df["relative_humidity_2m"].to_numpy(dtype=float))
        return df

    # ------------------------------------------------------------------
    # Targets and the embargo-aware time split
    # ------------------------------------------------------------------

    def create_targets(self, df: pd.DataFrame,
                       horizons: list[int] | None = None) -> pd.DataFrame:
        """Add future pm2_5 and us_aqi targets per city and drop rows without them.

        For each horizon h, ``pm2_5_t_plus_{h}`` and ``us_aqi_t_plus_{h}`` at
        hour t hold the city's pm2_5 and us_aqi at hour t + h. They are made by
        ``groupby("city").shift(-h)``, so a target never comes from another
        city.

        Before shifting, the method checks that consecutive timestamps within
        each city are exactly 1 hour apart, in row order. Only then does a
        shift of h rows equal h hours. A gap, a duplicate, an off-the-hour
        timestamp, or rows out of time order all fail this check.

        Rows with NaN in any created target are dropped: the last
        ``max(horizons)`` hours of each city, rows whose future hour falls in
        the Leading_Gap, and rows whose future hour falls in an unfilled
        interior gap. Per-city drop counts go to ``self.target_drop_counts``.
        The input frame should already hold the lag and rolling features, so
        those were computed on the complete hourly grid before rows are
        removed here.

        Args:
            df: Frame with ``city``, a parsed (datetime64) ``datetime``,
                ``pm2_5``, and ``us_aqi``, each city on an hourly grid in time
                order, as ``load_data`` returns it.
            horizons: Horizons in hours. None means ``HORIZONS`` (24, 48, 72).

        Returns:
            A copy of ``df`` with the target columns added (for each horizon in
            order, pm2_5 then us_aqi), only the rows where every created target
            is present, and a fresh 0..n-1 index. The input frame is not
            changed.

        Raises:
            ValueError: If ``city`` or ``datetime`` is missing or ``datetime``
                is not parsed, if a horizon is not a whole number of hours of
                at least 1, or if two consecutive timestamps within a city are
                not exactly 1 hour apart. The last message names the city and
                the first bad timestamp.
        """
        self._check_city_and_datetime(df, "create_targets")
        hours = list(HORIZONS) if horizons is None else list(horizons)
        # A shift of 0 would copy the current value into the target and a
        # negative one would turn the "future" target into a lag.
        bad_hours = _invalid_hour_steps(hours)
        if not hours or bad_hours:
            raise ValueError("horizons must be a non-empty list of whole hours of at "
                             f"least 1, got {hours}")

        steps = df.groupby("city", sort=False)["datetime"].diff()
        # The first row of each city has no previous hour (NaT) and is fine.
        bad_step = (steps.notna() & (steps != ONE_HOUR)).to_numpy()
        if bad_step.any():
            pos = int(np.flatnonzero(bad_step)[0])
            city = df["city"].iloc[pos]
            stamp = df["datetime"].iloc[pos]
            step_hours = steps.iloc[pos] / ONE_HOUR
            raise ValueError(
                "create_targets needs consecutive timestamps exactly 1 hour apart "
                f"within each city; city '{city}' has its first bad timestamp at "
                f"{stamp:%Y-%m-%d %H:%M:%S} ({step_hours:g} h after the previous row)"
            )

        df = df.copy()
        by_city = df.groupby("city", sort=False)
        target_cols: list[str] = []
        for h in hours:
            for target in TARGET_TYPES:
                col = f"{target}_t_plus_{h}"
                df[col] = by_city[target].shift(-h)
                target_cols.append(col)

        keep = df[target_cols].notna().all(axis=1).to_numpy()
        dropped = pd.Series(~keep).groupby(df["city"].to_numpy(), sort=False).sum()
        self.target_drop_counts = {str(city): int(n) for city, n in dropped.items()}
        return df.loc[keep].reset_index(drop=True)

    def prepare_train_test_split(self, df: pd.DataFrame,
                                 split_date: str | pd.Timestamp = SPLIT_DATE,
                                 horizon: int | None = None
                                 ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split by timestamp into training and test rows, with the per-horizon Embargo.

        Steps:

        1. Drop warm-up rows: rows with NaN in any lag column (name ends in
           ``_lag_<k>``), any rolling column from ``ROLLING_FEATURES``, or
           ``pm2_5_change_24h``. Per-city counts go to
           ``self.warmup_drop_counts``.
        2. Training rows: ``datetime < split`` and ``datetime + h hours <
           split``. This is the Embargo. A row at t carries the label at
           t + h, so without the second condition a late-December row would
           be trained on a Test_Period value. On a complete hourly grid with
           split 2025-01-01, the last training feature row is
           2024-12-31 23:00 minus h hours and the last training label is at
           2024-12-31 23:00.
        3. Test rows: ``datetime >= split``.

        The h rows per city just before the split, from split - h through
        split - 1 hour, are in neither set: their features are training-period
        rows but their labels are Test_Period values.

        The us_aqi lags count as warm-up columns for every split, also when the
        frame is meant for a pm2_5 model. The method has no target argument,
        and one rule for both targets means the pm2_5 and us_aqi models of a
        city and horizon are trained and tested on the same rows, so their
        metrics compare like with like. Weather rolling means count too, with a
        warm-up of only 5 hours.

        Two guards then raise ``AssertionError`` if the embargo logic is ever
        broken: the latest training timestamp must be earlier than the
        earliest test timestamp, and the latest training timestamp plus h
        hours must be earlier than the split. Both are skipped when the set
        they inspect is empty, because an empty set breaks neither rule.

        Args:
            df: Frame from ``create_targets``, with ``city``, a parsed
                ``datetime``, the lag and rolling features, and the target
                columns.
            split_date: First timestamp of the Test_Period, as a string
                ``pd.Timestamp`` accepts (default ``SPLIT_DATE``, "2025-01-01",
                which means 2025-01-01 00:00).
            horizon: Horizon h in hours for the Embargo. None means the largest
                h found among the ``{target}_t_plus_{h}`` columns (72 for the
                default targets). That keeps the split safe for every horizon
                in the frame but drops the most training rows; pass the
                model's own horizon to keep every row it is allowed.

        Returns:
            ``(train, test)``: copies of the selected rows with all columns,
            each with a fresh 0..n-1 index. Either can be empty, for example
            when ``split_date`` lies outside the frame.

        Raises:
            ValueError: If ``city`` or ``datetime`` is missing or ``datetime``
                is not parsed, if ``horizon`` is not a whole number of hours of
                at least 1, or if ``horizon`` is None and the frame has no
                target columns.
            AssertionError: If either guard above fails.
        """
        self._check_city_and_datetime(df, "prepare_train_test_split")
        split_ts = pd.Timestamp(split_date)
        h = self._resolve_horizon(df, horizon)
        offset = pd.Timedelta(h, unit="h")

        warmup_cols = self._warmup_columns(df)
        if warmup_cols:
            is_warmup = df[warmup_cols].isna().any(axis=1).to_numpy()
        else:
            is_warmup = np.zeros(len(df), dtype=bool)
        dropped = pd.Series(is_warmup).groupby(df["city"].to_numpy(), sort=False).sum()
        self.warmup_drop_counts = {str(city): int(n) for city, n in dropped.items()}
        clean = df.loc[~is_warmup]

        # Masks are NumPy arrays so selection works by position and does not
        # depend on the frame's index being unique.
        stamps = clean["datetime"]
        # For h >= 1 the embargo condition implies the first one; both are kept
        # so the code reads like the rule it implements.
        train_mask = ((stamps < split_ts) & (stamps + offset < split_ts)).to_numpy()
        test_mask = (stamps >= split_ts).to_numpy()
        train = clean.loc[train_mask].reset_index(drop=True)
        test = clean.loc[test_mask].reset_index(drop=True)

        # Explicit raises instead of assert statements, so the guards still run
        # under python -O.
        if len(train) and len(test) and not train["datetime"].max() < test["datetime"].min():
            raise AssertionError(
                f"Train and test overlap: latest training timestamp "
                f"{train['datetime'].max()} is not earlier than earliest test "
                f"timestamp {test['datetime'].min()}"
            )
        if len(train) and not (train["datetime"] + offset).max() < split_ts:
            raise AssertionError(
                f"Embargo broken for h={h}: a training label at "
                f"{(train['datetime'] + offset).max()} is not earlier than {split_ts}"
            )
        return train, test

    @staticmethod
    def _check_city_and_datetime(df: pd.DataFrame, caller: str) -> None:
        """Check that ``city`` exists and ``datetime`` exists and is parsed.

        Args:
            df: Frame passed to a public method.
            caller: Name of that method, used in the error message.

        Raises:
            ValueError: If a column is missing or ``datetime`` is not datetime64.
        """
        missing = [col for col in ("city", "datetime") if col not in df.columns]
        if missing:
            raise ValueError(f"{caller} needs the columns {missing}")
        # String timestamps (for example from a reloaded CSV) would compare as
        # text, which silently breaks both the 1-hour check and the split.
        if not pd.api.types.is_datetime64_any_dtype(df["datetime"]):
            raise ValueError(f"{caller} needs a parsed datetime64 'datetime' column, "
                             f"got dtype {df['datetime'].dtype}")

    @staticmethod
    def _resolve_horizon(df: pd.DataFrame, horizon: int | None) -> int:
        """Return the embargo horizon for ``prepare_train_test_split``.

        Args:
            df: Frame whose target columns are read when ``horizon`` is None.
            horizon: Explicit horizon in hours, or None for the largest one
                found among the ``{target}_t_plus_{h}`` columns.

        Returns:
            The horizon in hours.

        Raises:
            ValueError: If ``horizon`` is not a whole number of hours of at
                least 1, or if it is None and no target column is found.
        """
        if horizon is None:
            found = [int(match.group("hours")) for col in df.columns
                     if isinstance(col, str) and (match := TARGET_COLUMN_PATTERN.match(col))]
            if not found:
                raise ValueError("horizon is None but the frame has no target columns "
                                 "named '{target}_t_plus_{h}'; call create_targets first "
                                 "or pass horizon")
            return max(found)
        if _invalid_hour_steps([horizon]):
            raise ValueError(f"horizon must be a whole number of hours of at least 1, "
                             f"got {horizon!r}")
        return int(horizon)

    @staticmethod
    def _warmup_columns(df: pd.DataFrame) -> list[str]:
        """List the lag, rolling, and change feature columns present in ``df``.

        Args:
            df: Frame from ``create_lag_and_rolling_features`` or
                ``create_targets``.

        Returns:
            The matching column names in frame order. Empty when the frame has
            none of them, in which case the split drops no warm-up rows.
        """
        rolling_cols = {out_col for out_col, _, _, _ in ROLLING_FEATURES}
        return [col for col in df.columns
                if isinstance(col, str)
                and (LAG_COLUMN_PATTERN.search(col) is not None
                     or col in rolling_cols or col in CHANGE_FEATURES)]

    # ------------------------------------------------------------------
    # Feature columns, processed files, and the full build
    # ------------------------------------------------------------------

    def get_feature_columns(self, target_type: str, mode: str) -> list[str]:
        """Return the model input columns for one target and feature mode.

        The lists follow the design's feature table, group by group in table
        order.

        Mode ``"forecast"`` (Forecast_Mode), 34 columns for pm2_5 and 37 for
        us_aqi:

        - pollutants at t: pm2_5, pm10, carbon_monoxide, nitrogen_dioxide,
          sulphur_dioxide, ozone, dust;
        - us_aqi target only: us_aqi at t, us_aqi_lag_1, us_aqi_lag_24;
        - pm2_5 lags: pm2_5_lag_1, _3, _6, _12, _24, _48;
        - pm2_5 rolling: pm2_5_rolling_mean_24h, pm2_5_rolling_mean_72h,
          pm2_5_rolling_std_24h, pm2_5_change_24h;
        - weather at t: temperature_2m, relative_humidity_2m,
          surface_pressure, wind_speed_10m, wind_direction_10m, precipitation;
        - weather rolling: wind_speed_rolling_mean_6h,
          humidity_rolling_mean_6h, temperature_rolling_mean_6h;
        - derived weather: stagnation_flag, inversion_proxy;
        - time: hour_sin, hour_cos, month_sin, month_cos, is_weekend,
          is_winter.

        Mode ``"simulation"`` (Simulation_Mode), the same 15 columns for both
        targets: pm2_5_rolling_mean_24h as the only pollution feature, the six
        weather values at t, stagnation_flag, inversion_proxy, and the six
        time features. It has no autoregressive lags, no pollutant values at
        t, and no weather rolling means.

        Never returned: any ``*_t_plus_*`` target, city, latitude, longitude,
        time, datetime, year, month, day, hour, day_of_week, aqi_category, and
        is_forecast_like. Every returned column is known at time t: lags and
        rolling windows use only hours at or before t.

        Args:
            target_type: ``"pm2_5"`` or ``"us_aqi"``.
            mode: ``"forecast"`` or ``"simulation"``.

        Returns:
            A new list of column names, so callers can change it freely.

        Raises:
            ValueError: If ``target_type`` or ``mode`` is not an accepted
                value. The message names the bad value or values and lists the
                accepted ones.
        """
        bad: list[str] = []
        # The isinstance checks keep a non-string such as None or an array
        # from reaching the membership test, which could raise a confusing
        # error or match by accident.
        if not isinstance(target_type, str) or target_type not in TARGET_TYPES:
            bad.append(f"target_type {target_type!r}")
        if not isinstance(mode, str) or mode not in FEATURE_MODES:
            bad.append(f"mode {mode!r}")
        if bad:
            raise ValueError(f"Invalid {' and '.join(bad)}: target_type must be one of "
                             f"{TARGET_TYPES}; mode must be one of {FEATURE_MODES}")

        if mode == "simulation":
            # The What-If sliders set only the current hour's weather. Lags,
            # pollutant values at t, and 6-hour weather means would hold the
            # observed situation fixed and contradict the slider, so the
            # baseline pollution load enters only through the 24-hour mean.
            return (SIMULATION_POLLUTION_FEATURES + WEATHER_COLUMNS
                    + DERIVED_WEATHER_FEATURES + TIME_FEATURES)

        columns = list(FORECAST_POLLUTANT_FEATURES)
        if target_type == "us_aqi":
            # us_aqi at t and its lags are published values known at
            # prediction time. The pm2_5 model never gets them, so it stays a
            # pure concentration forecaster and cannot lean on an index that
            # is partly computed from other pollutants.
            columns += ["us_aqi"] + [f"us_aqi_lag_{k}" for k in self.US_AQI_LAG_STEPS]
        columns += [f"pm2_5_lag_{k}" for k in self.DEFAULT_LAG_STEPS]
        columns += (PM25_ROLLING_FEATURES + WEATHER_COLUMNS + WEATHER_ROLLING_FEATURES
                    + DERIVED_WEATHER_FEATURES + TIME_FEATURES)
        return columns

    def save_processed(self, df: pd.DataFrame, filename: str) -> Path:
        """Write a frame as CSV to ``data/processed/<filename>``.

        A parsed ``datetime`` column is written as text in the format
        ``YYYY-MM-DD HH:MM:SS``, so readers can parse it back with the same
        format that ``load_data`` uses. The index is not written, and floats
        keep full precision so a reloaded file matches the frame in memory.
        The folder is created if needed; an existing file is overwritten.

        Args:
            df: Frame to save. It is not changed.
            filename: Plain file name such as ``"features_all.csv"``, without
                any folder part.

        Returns:
            The path of the written file.

        Raises:
            ValueError: If ``filename`` is empty or contains a folder part.
        """
        # A folder part could place the file outside data/processed/, which
        # the dashboard and training script would then not find.
        if (not isinstance(filename, str) or not filename
                or Path(filename).name != filename or filename in (".", "..")):
            raise ValueError(f"filename must be a plain file name without a folder, "
                             f"got {filename!r}")
        path = ensure_dir(PROCESSED_DIR) / filename

        out = df
        if "datetime" in df.columns and pd.api.types.is_datetime64_any_dtype(df["datetime"]):
            # Formatted explicitly: pandas would write a date-only value such
            # as "2022-08-01" when every timestamp in a column is at midnight.
            out = df.assign(datetime=df["datetime"].dt.strftime(DATETIME_FORMAT))
        out.to_csv(path, index=False)
        return path

    def build_features(self, filepath: str | Path) -> pd.DataFrame:
        """Build and save both processed files from the master CSV.

        Steps: ``load_data`` -> ``generate_time_features`` ->
        ``create_lag_and_rolling_features`` -> save ``features_all.csv`` ->
        ``create_targets`` -> save ``model_dataset.csv``.

        ``features_all.csv`` keeps every row, including the last hours of each
        city that have no future targets, because the dashboard predicts from
        the latest available row. Features are built before
        ``create_targets`` removes rows, so every lag and rolling window is
        computed on the complete hourly grid. The feature frame is kept in
        ``self.features``.

        Args:
            filepath: Path to the Master_Dataset CSV.

        Returns:
            The target frame written to ``model_dataset.csv``: every feature
            column plus ``pm2_5_t_plus_{h}`` and ``us_aqi_t_plus_{h}`` for h in
            24, 48, 72, only rows where all six targets are present, with
            ``datetime`` parsed.

        Raises:
            ValueError: Passed on from ``load_data``,
                ``create_lag_and_rolling_features``, or ``create_targets``.
        """
        df = self.load_data(filepath)
        df = self.generate_time_features(df)
        df = self.create_lag_and_rolling_features(df)
        self.features = df
        self.save_processed(df, FEATURES_FILENAME)

        model_df = self.create_targets(df)
        self.save_processed(model_df, MODEL_DATASET_FILENAME)
        return model_df


def _format_timestamp(ts: pd.Timestamp | None) -> str:
    """Format a timestamp as ``YYYY-MM-DD HH:MM:SS`` for the printed summary.

    Args:
        ts: Timestamp, NaT, or None.

    Returns:
        The formatted text, or ``"n/a"`` for NaT or None (an empty split side).
    """
    return "n/a" if ts is None or pd.isna(ts) else f"{ts:%Y-%m-%d %H:%M:%S}"


def main() -> None:
    """Build both processed files from the master CSV and print the Phase 2 summary.

    Run from the project root with ``python -m src.pipeline``. The summary
    lists the Gap_Strategy counts, the row and column counts of both files,
    per city the rows dropped for NaN targets and as NaN warm-up rows, per
    city and horizon the train and test sizes with the last training feature
    and label timestamps (the embargo table), and the feature columns for
    each target and mode. Every number comes from the class methods run on
    the real file.
    """
    pipeline = AirQualityDataPipeline()
    model_df = pipeline.build_features(MASTER_PATH)
    features = pipeline.features
    split_ts = pd.Timestamp(SPLIT_DATE)

    print("\n=== Phase 2 pipeline summary ===")
    print("Gap_Strategy counts:")
    for name, count in pipeline.gap_counts.items():
        print(f"  {name}: {count}")

    print("\nProcessed files:")
    for filename, frame in ((FEATURES_FILENAME, features), (MODEL_DATASET_FILENAME, model_df)):
        print(f"  {PROCESSED_DIR / filename}: {len(frame)} rows x {frame.shape[1]} columns")

    # Only cities present in the file, in the project's usual order.
    cities = [city for city in CITIES if city in set(model_df["city"])]
    city_rows: list[dict[str, object]] = []
    split_rows: list[dict[str, object]] = []
    for city in cities:
        city_df = model_df[model_df["city"] == city]
        warmup = None
        for h in HORIZONS:
            train, test = pipeline.prepare_train_test_split(city_df, SPLIT_DATE, horizon=h)
            # Warm-up rows depend only on the feature columns, so the count is
            # the same for every horizon; it is taken from the first split.
            if warmup is None:
                warmup = pipeline.warmup_drop_counts.get(city, 0)
            last_feature = train["datetime"].max() if len(train) else None
            last_label = (last_feature + pd.Timedelta(h, unit="h")
                          if last_feature is not None else None)
            split_rows.append({
                "city": city,
                "horizon_h": h,
                "n_train": len(train),
                "n_test": len(test),
                "last_train_feature": _format_timestamp(last_feature),
                "last_train_label": _format_timestamp(last_label),
                "first_test": _format_timestamp(test["datetime"].min() if len(test) else None),
                # Rows before the split that are neither warm-up nor training
                # rows: the embargoed hours whose labels lie after the split.
                "embargo_dropped": len(city_df) - warmup - len(train) - len(test),
            })
        city_rows.append({
            "city": city,
            "features_all_rows": int((features["city"] == city).sum()),
            "target_nan_dropped": pipeline.target_drop_counts.get(city, 0),
            "model_dataset_rows": len(city_df),
            "warmup_nan_dropped": warmup,
        })

    print("\nRows per city (target NaN rows are dropped by create_targets, "
          "warm-up NaN rows by prepare_train_test_split):")
    print(pd.DataFrame(city_rows).to_string(index=False))

    print(f"\nTime split at {split_ts:%Y-%m-%d %H:%M:%S} with the per-horizon embargo:")
    print(pd.DataFrame(split_rows).to_string(index=False))

    print("\nFeature columns per target and mode:")
    for target_type in TARGET_TYPES:
        for mode in FEATURE_MODES:
            columns = pipeline.get_feature_columns(target_type, mode)
            absent = [col for col in columns if col not in model_df.columns]
            print(f"  {target_type} / {mode} ({len(columns)} columns):")
            print(textwrap.fill(", ".join(columns), width=100,
                                initial_indent="    ", subsequent_indent="    "))
            if absent:
                print(f"    WARNING: not in {MODEL_DATASET_FILENAME}: {absent}")


if __name__ == "__main__":
    main()
