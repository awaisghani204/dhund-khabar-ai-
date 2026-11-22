"""Phase 1 data audit for Dhund-Khabar AI.

Run from the project root with ``python scripts/audit_data.py``. The script
reads ``data/pakistan_smog_master_hourly.csv`` with pandas only (no pipeline
or model code), prints every audit section, and writes the same text to
``docs/data_audit.md``. It also writes ``data/processed/audit_flags.json``
with the start of the Forecast_Like_Row range for the pipeline. Every number
in the report is computed here from the real file; the only hand-written text
is the fixed interpretation notes, the method descriptions, and the approved
Gap_Strategy (whose only numbers are rule parameters).

Each ``audit_*`` function takes the raw master frame and returns a string, a
DataFrame, or a tuple. ``build_sections`` turns those results into Markdown
sections in report order, so a new audit only needs one new function and one
line in ``build_sections``.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

# The script lives in scripts/, so the project root must be on sys.path before
# `src` can be imported when it is started as `python scripts/audit_data.py`.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.utils import (  # noqa: E402
    CITIES,
    MASTER_PATH,
    POLLUTANT_COLUMNS,
    PROCESSED_DIR,
    PROJECT_ROOT,
    RANDOM_STATE,
    WINTER_MONTHS,
    ensure_dir,
    pm25_sub_index,
)

REPORT_PATH: Path = PROJECT_ROOT / "docs" / "data_audit.md"
FLAGS_PATH: Path = PROCESSED_DIR / "audit_flags.json"
ANALYSIS_COLUMNS: list[str] = [
    "pm2_5", "us_aqi", "temperature_2m", "relative_humidity_2m", "wind_speed_10m",
]
DATETIME_FORMAT: str = "%Y-%m-%d %H:%M:%S"
TIME_FORMAT: str = "%Y-%m-%dT%H:%M"
# A long gap list would bury the rest of the report; the total is always shown.
MAX_GAP_ROWS_SHOWN: int = 100
# Three months is enough to show whether the high season is a block of winter
# months rather than one noisy month.
TOP_MONTHS: int = 3

# EPA's PM2.5 index is defined on a 24-hour average, so the comparison uses a
# trailing 24-hour mean per city rather than the hourly value.
SUBINDEX_WINDOW_HOURS: int = 24
SUBINDEX_SAMPLE_SIZE: int = 5000
SUBINDEX_TABLES: list[str] = ["2024", "2012"]

FORECAST_SIGNAL_COLUMNS: list[str] = ["pm2_5", "us_aqi"]
FORECAST_WINDOW_DAYS: int = 180
# 2023 is the first complete calendar year in the file, so it is the reference
# for what "normal" observation-like statistics look like.
FORECAST_BASELINE_YEAR: int = 2023
FORECAST_REPEAT_FACTOR: float = 3.0
# Values are stored with at most a few decimals; 6 is far beyond that and
# keeps the decimal-place search short.
MAX_DECIMAL_PLACES: int = 6
FORECAST_METHOD: str = (
    "Two checks per city. (1) Rows with a timestamp later than the audit run "
    "time are flagged because they cannot be observations. (2) For the last "
    f"{FORECAST_WINDOW_DAYS} days of each city, weekly statistics (Monday-Sunday "
    f"weeks) are computed for {' and '.join(FORECAST_SIGNAL_COLUMNS)}: the share of "
    "hours equal to the previous hour, the median number of decimal places, and "
    "the lag-1 autocorrelation. A week is flagged when its repeated-value share "
    f"exceeds {FORECAST_REPEAT_FACTOR:g} times the {FORECAST_BASELINE_YEAR} "
    "baseline of the same city and column, or when its median decimal places "
    f"fall below the {FORECAST_BASELINE_YEAR} baseline. The lag-1 autocorrelation "
    "is reported for review but does not flag a week. forecast_like_start is the "
    "earliest flagged timestamp over both checks and both cities. Rows are "
    "flagged only, never removed."
)

FIXED_STATEMENTS: list[str] = [
    "us_aqi may be the maximum of several pollutant sub-indices (for example "
    "PM2.5, PM10, ozone, nitrogen dioxide, sulphur dioxide, and carbon monoxide), "
    "not the PM2.5 sub-index alone. A mismatch between us_aqi and a PM2.5-only "
    "sub-index is therefore expected and is not by itself a data error.",
    "The provenance of the Master_Dataset must be confirmed by the student. The "
    "values may be modeled or reanalysis data (for example from an air-quality "
    "forecast service) rather than readings from ground sensors, and the project "
    "conclusions must be worded accordingly.",
]

# The Gap_Strategy the user approved at the Phase 1 checkpoint. It is fixed
# text: the only numbers are rule parameters, because the counts it acts on
# come from the audit sections above and from the pipeline's printed output.
GAP_STRATEGY_APPROVED_ON: str = "2026-11-21"
APPROVED_GAP_STRATEGY: list[str] = [
    "Reindex each city to a complete hourly grid and drop duplicate (city, "
    "datetime) timestamps, keeping the first row. This is a safety net: the "
    "audit's counts are in \"Duplicate timestamps per city\" and \"Hourly gaps\", "
    "and the pipeline prints how many rows it inserts and drops.",
    "Leave the Leading_Gap (the leading missing pollutant and us_aqi hours per "
    "city, shown as \"leading run\" in \"Missing values per column per city\") "
    "unimputed. Those rows are removed as warm-up rows by the split.",
    "Forward-fill interior pollutant and us_aqi gaps of up to 3 consecutive "
    "hours, within each city only. A gap longer than 3 hours is left entirely "
    "missing, not partly filled. Weather columns are unchanged.",
    "Clip negative pollutant concentrations to 0, treated as model artifacts. "
    "At approval they were found in ozone and nitrogen_dioxide (see \"Impossible "
    "values per city\"). The pipeline prints how many values it changed.",
    "Keep us_aqi values above 500. They map to Hazardous (Emergency) and are "
    "listed in the README limitations.",
    "Keep all rows. At approval no Forecast_Like_Rows were flagged (see "
    "\"Forecast-like rows\"). The rows after the master file's last-modified time "
    "(column \"rows after file modified time\" in that section) are kept and "
    "mentioned in the README limitations.",
    "aqi_category is kept as-is and is never used as a feature.",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_master(path: Path = MASTER_PATH) -> pd.DataFrame:
    """Read the master CSV exactly as stored, without cleaning or features.

    Args:
        path: Location of the master hourly CSV.

    Returns:
        The raw DataFrame with pandas' default dtype inference.
    """
    return pd.read_csv(path)


def parse_datetime(df: pd.DataFrame) -> pd.Series:
    """Parse the ``datetime`` column with the documented file format.

    Unparseable values become NaT instead of raising, because the audit must
    report bad values rather than stop on the first one.

    Args:
        df: Raw master frame.

    Returns:
        A datetime64 Series aligned with ``df``.
    """
    return pd.to_datetime(df["datetime"], format=DATETIME_FORMAT, errors="coerce")


def city_order(df: pd.DataFrame) -> list[str]:
    """Return the cities in the frame, known cities first in project order.

    Args:
        df: Frame with a ``city`` column.

    Returns:
        City names: those in ``CITIES`` that are present, then any others sorted.
    """
    present = set(df["city"].dropna().unique())
    known = [c for c in CITIES if c in present]
    return known + sorted(present - set(known))


def _format_cell(value: object) -> str:
    """Format one table cell for Markdown with fixed precision.

    Args:
        value: Cell value of any type.

    Returns:
        The cell as text; floats get two decimals, whole floats print as ints.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    if isinstance(value, np.datetime64):
        value = pd.Timestamp(value)
    if isinstance(value, pd.Timestamp):
        return "NaT" if pd.isna(value) else value.strftime(DATETIME_FORMAT)
    if isinstance(value, (float, np.floating)):
        if float(value).is_integer():
            return str(int(value))
        return f"{float(value):.2f}"
    return str(value)


def df_to_markdown(df: pd.DataFrame, index: bool = True) -> str:
    """Render a DataFrame as a GitHub Markdown table.

    pandas' own ``to_markdown`` needs the tabulate package, which is not in the
    project's allowed dependency list, so the table is built by hand.

    Args:
        df: Table to render. A MultiIndex index is joined with " / ".
        index: Whether to include the index as the first column.

    Returns:
        The table as Markdown text, or "(no rows)" for an empty frame.
    """
    if df.empty:
        return "(no rows)"
    headers = [str(c) for c in df.columns]
    rows = [[_format_cell(v) for v in row] for row in df.itertuples(index=False)]
    if index:
        index_name = " / ".join(str(n) for n in df.index.names if n is not None)
        headers = [index_name] + headers
        labels = [
            " / ".join(str(p) for p in idx) if isinstance(idx, tuple) else _format_cell(idx)
            for idx in df.index
        ]
        rows = [[label] + row for label, row in zip(labels, rows)]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def make_section(title: str, body: str) -> str:
    """Wrap a section body under a level-2 Markdown heading.

    Args:
        title: Section heading text.
        body: Markdown body.

    Returns:
        The section text ending with a newline.
    """
    return f"## {title}\n\n{body.strip()}\n"


# ---------------------------------------------------------------------------
# Audit functions
# ---------------------------------------------------------------------------

def audit_shape_and_dtypes(df: pd.DataFrame) -> str:
    """Report the frame shape and the dtype pandas inferred for every column.

    Args:
        df: Raw master frame.

    Returns:
        Markdown text with the shape line and a column/dtype table.
    """
    dtypes = pd.DataFrame({
        "position": np.arange(1, df.shape[1] + 1),
        "column": df.columns,
        "dtype": [str(t) for t in df.dtypes],
    })
    return (
        f"Rows: {df.shape[0]}, columns: {df.shape[1]}\n\n"
        + df_to_markdown(dtypes, index=False)
    )


def audit_date_range_per_city(df: pd.DataFrame) -> pd.DataFrame:
    """Report the first and last timestamp and the row count for each city.

    The expected hourly count (span in hours + 1) is included so a missing or
    extra row shows up as a difference without reading the gap table.

    Args:
        df: Raw master frame.

    Returns:
        One row per city with first_timestamp, last_timestamp, rows, and
        expected_hourly_rows.
    """
    ts = parse_datetime(df)
    records = []
    for city in city_order(df):
        city_ts = ts[df["city"] == city]
        first, last = city_ts.min(), city_ts.max()
        expected = int((last - first).total_seconds() // 3600) + 1
        records.append({
            "city": city,
            "first_timestamp": first,
            "last_timestamp": last,
            "rows": int(city_ts.size),
            "expected_hourly_rows": expected,
        })
    return pd.DataFrame(records).set_index("city")


def audit_missing_per_city(df: pd.DataFrame) -> pd.DataFrame:
    """Count missing values per column per city, plus the leading missing run.

    The leading run (consecutive missing rows from the first timestamp) is
    reported separately because the Gap_Strategy treats a Leading_Gap
    differently from interior gaps.

    Args:
        df: Raw master frame.

    Returns:
        One row per column, with "<city> missing" and "<city> leading run"
        columns for each city.
    """
    ts = parse_datetime(df)
    result = pd.DataFrame(index=pd.Index(df.columns, name="column"))
    for city in city_order(df):
        mask = (df["city"] == city).to_numpy()
        city_df = df.loc[mask].iloc[np.argsort(ts[mask].to_numpy(), kind="stable")]
        isna = city_df.isna().to_numpy()
        # argmin finds the first non-missing row; an all-missing column has no
        # such row, so its whole length is the leading run.
        leading = np.where(isna.all(axis=0), isna.shape[0], isna.argmin(axis=0))
        result[f"{city} missing"] = isna.sum(axis=0)
        result[f"{city} leading run"] = leading
    return result


def audit_duplicate_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Count repeated (city, datetime) pairs per city.

    Args:
        df: Raw master frame.

    Returns:
        One row per city with the number of extra rows that repeat an earlier
        timestamp of the same city.
    """
    ts = parse_datetime(df)
    dup = pd.DataFrame({"city": df["city"], "ts": ts}).duplicated(keep="first")
    counts = [int(dup[df["city"] == city].sum()) for city in city_order(df)]
    return pd.DataFrame(
        {"duplicate_timestamps": counts},
        index=pd.Index(city_order(df), name="city"),
    )


def audit_hourly_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """List every step between consecutive timestamps that is not 1 hour.

    Duplicates show up as 0-hour steps and missing hours as steps above 1, so
    one table covers both problems.

    Args:
        df: Raw master frame.

    Returns:
        One row per non-1-hour step with city, previous_timestamp,
        next_timestamp, and step_hours.
    """
    ts = parse_datetime(df)
    # Plain lists instead of pd.concat: concatenating empty datetime frames
    # triggers a NumPy generic-timedelta deprecation inside pandas.
    cities: list[str] = []
    previous: list[pd.Timestamp] = []
    following: list[pd.Timestamp] = []
    steps: list[float] = []
    for city in city_order(df):
        city_ts = ts[df["city"] == city].sort_values(kind="stable").reset_index(drop=True)
        # total_seconds avoids NumPy's deprecated generic-unit timedelta division.
        step = city_ts.diff().dt.total_seconds() / 3600.0
        bad = step.notna() & (step != 1.0)
        for i in np.flatnonzero(bad.to_numpy()):
            cities.append(city)
            previous.append(city_ts.iloc[i - 1])
            following.append(city_ts.iloc[i])
            steps.append(float(step.iloc[i]))
    return pd.DataFrame({
        "city": pd.Series(cities, dtype="object"),
        "previous_timestamp": pd.Series(previous, dtype="datetime64[ns]"),
        "next_timestamp": pd.Series(following, dtype="datetime64[ns]"),
        "step_hours": pd.Series(steps, dtype="float64"),
    })


def audit_analysis_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Summarise each Analysis_Column per city with ``describe()``.

    Args:
        df: Raw master frame.

    Returns:
        One row per (city, column) with count, mean, std, min, quartiles, max.
    """
    pieces = {}
    for city in city_order(df):
        desc = df.loc[df["city"] == city, ANALYSIS_COLUMNS].describe().T
        pieces[city] = desc
    out = pd.concat(pieces, names=["city", "column"])
    return out


def audit_impossible_values(df: pd.DataFrame) -> pd.DataFrame:
    """Count physically impossible values per city.

    Checks: each pollutant below 0, relative_humidity_2m above 100, and us_aqi
    below 0. Missing values are not counted as impossible.

    Args:
        df: Raw master frame.

    Returns:
        One row per check with one count column per city.
    """
    checks: dict[str, pd.Series] = {
        f"{col} < 0": df[col] < 0 for col in POLLUTANT_COLUMNS
    }
    checks["relative_humidity_2m > 100"] = df["relative_humidity_2m"] > 100
    checks["us_aqi < 0"] = df["us_aqi"] < 0
    result = pd.DataFrame(index=pd.Index(list(checks), name="check"))
    for city in city_order(df):
        in_city = df["city"] == city
        result[city] = [int((mask & in_city).sum()) for mask in checks.values()]
    return result


def audit_datetime_vs_time(df: pd.DataFrame) -> int:
    """Count rows where ``datetime`` disagrees with the parsed ``time`` column.

    A value that cannot be parsed in either column counts as a mismatch,
    because NaT never equals a timestamp.

    Args:
        df: Raw master frame.

    Returns:
        Number of mismatching rows.
    """
    parsed_time = pd.to_datetime(df["time"], format=TIME_FORMAT, errors="coerce")
    return int((parsed_time != parse_datetime(df)).sum())


def audit_monthly_means(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Compute calendar-month means of pm2_5 and us_aqi per city.

    Months are pooled across years, so partial years (the data starts in
    August 2022 and ends in November 2026) weight some months more than
    others; the sentence names the peak months so the reader can judge.

    Args:
        df: Raw master frame.

    Returns:
        A table indexed by month 1-12 with one column per city and variable,
        and a sentence per city and variable stating the top months and
        whether they fall in the Winter_Months.
    """
    month = parse_datetime(df).dt.month
    table = pd.DataFrame(index=pd.Index(range(1, 13), name="month"))
    sentences = []
    for city in city_order(df):
        in_city = df["city"] == city
        for col in ("pm2_5", "us_aqi"):
            means = df.loc[in_city, col].groupby(month[in_city]).mean()
            table[f"{city} {col}"] = means
            top = means.dropna().sort_values(ascending=False).head(TOP_MONTHS)
            if top.empty:
                sentences.append(f"- {city} {col}: no non-missing values.")
                continue
            top_months = [int(m) for m in top.index]
            in_winter = [m for m in top_months if m in WINTER_MONTHS]
            if len(in_winter) == len(top_months):
                verdict = "all in the Winter_Months"
            elif in_winter:
                verdict = f"{len(in_winter)} of {len(top_months)} in the Winter_Months"
            else:
                verdict = "none in the Winter_Months"
            sentences.append(
                f"- {city} {col}: peak month {top_months[0]} "
                f"(mean {top.iloc[0]:.2f}); top {len(top_months)} months "
                f"{', '.join(str(m) for m in top_months)}, {verdict} "
                f"({', '.join(str(m) for m in WINTER_MONTHS)})."
            )
    return table, "\n".join(sentences)


def audit_aqi_above_500(df: pd.DataFrame) -> str:
    """Count us_aqi values above the EPA scale top of 500 and find the maximum.

    Args:
        df: Raw master frame.

    Returns:
        Markdown text with the overall count and maximum, and a per-city table
        including the timestamp of each city's maximum.
    """
    above = df["us_aqi"] > 500
    records = []
    for city in city_order(df):
        city_df = df[df["city"] == city]
        aqi = city_df["us_aqi"]
        max_ts = city_df.loc[aqi.idxmax(), "datetime"] if aqi.notna().any() else ""
        records.append({
            "city": city,
            "us_aqi > 500": int((aqi > 500).sum()),
            "max us_aqi": aqi.max(),
            "timestamp of max": max_ts,
        })
    table = pd.DataFrame(records).set_index("city")
    return (
        f"Rows with us_aqi above 500: {int(above.sum())}. "
        f"Maximum us_aqi: {_format_cell(df['us_aqi'].max())}.\n\n"
        + df_to_markdown(table)
    )


def _sorted_city_frames(df: pd.DataFrame, columns: list[str]) -> dict[str, pd.DataFrame]:
    """Split the master frame into one time-sorted frame per city.

    Rows whose ``datetime`` cannot be parsed are dropped here, because time
    windows and lag-1 comparisons need a real timestamp; the
    "datetime vs time" section already counts such rows.

    Args:
        df: Raw master frame.
        columns: Value columns to keep next to the timestamp.

    Returns:
        City name mapped to a frame with a ``ts`` column followed by
        ``columns``, sorted by ``ts`` with a fresh 0..n-1 index.
    """
    ts = parse_datetime(df)
    frames: dict[str, pd.DataFrame] = {}
    for city in city_order(df):
        mask = (df["city"] == city).to_numpy()
        part = df.loc[mask, columns].copy()
        part.insert(0, "ts", ts[mask].to_numpy())
        part = part.dropna(subset=["ts"])
        frames[city] = part.sort_values("ts", kind="stable").reset_index(drop=True)
    return frames


def _round_half_up(values: np.ndarray) -> np.ndarray:
    """Round to the nearest integer with halves going up, as EPA reports AQI.

    ``np.round`` rounds halves to the nearest even integer (100.5 -> 100),
    which would count some exact EPA matches as mismatches.

    Args:
        values: Float array; NaN stays NaN.

    Returns:
        Float array of rounded values.
    """
    return np.floor(np.asarray(values, dtype=float) + 0.5)


def _decimal_places(values: np.ndarray) -> np.ndarray:
    """Count the decimal places each value is stored with.

    The CSV is parsed into floats, so the count is the smallest number of
    places d for which ``value * 10**d`` is a whole number within float noise.

    Args:
        values: Float array.

    Returns:
        Float array of decimal places (0..MAX_DECIMAL_PLACES), NaN where the
        value is NaN.
    """
    vals = np.asarray(values, dtype=float)
    places = np.full(vals.shape, np.nan)
    for d in range(MAX_DECIMAL_PLACES + 1):
        scaled = vals * 10.0 ** d
        whole = np.isclose(scaled, np.rint(scaled), rtol=0.0, atol=1e-6)
        places = np.where(np.isnan(places) & whole, float(d), places)
    # Values that never became whole get the maximum, so they are not NaN.
    return np.where(np.isnan(places) & ~np.isnan(vals), float(MAX_DECIMAL_PLACES), places)


def _lag1_autocorr(values: np.ndarray) -> float:
    """Pearson correlation between consecutive values of a series.

    Args:
        values: Time-ordered float array.

    Returns:
        The lag-1 autocorrelation, or NaN when fewer than 3 consecutive pairs
        exist or either side is constant (the correlation is undefined then).
    """
    vals = np.asarray(values, dtype=float)
    x, y = vals[:-1], vals[1:]
    ok = ~np.isnan(x) & ~np.isnan(y)
    if ok.sum() < 3:
        return float("nan")
    x, y = x[ok], y[ok]
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _ts_text(value: pd.Timestamp | None) -> str:
    """Format a timestamp for the report, or "none" when it is missing.

    Args:
        value: Timestamp, NaT, or None.

    Returns:
        ``YYYY-MM-DD HH:MM:SS`` text or "none".
    """
    if value is None or pd.isna(value):
        return "none"
    return pd.Timestamp(value).strftime(DATETIME_FORMAT)


def audit_aqi_subindex(df: pd.DataFrame) -> str:
    """Compare us_aqi with the EPA PM2.5 sub-index for a seeded random sample.

    The sub-index is computed from the trailing 24-hour mean of pm2_5 per city
    (all 24 hourly values required), because the EPA PM2.5 index is defined on
    a 24-hour average. Both values are rounded half up before the equality
    check. The comparison is run for the 2024 and the 2012 breakpoint tables
    so the reader can see which one the data source follows.

    Args:
        df: Raw master frame.

    Returns:
        Markdown text stating the averaging choice, the sample size, the
        per-table and per-city comparison table, and the offset between the
        first pm2_5 and the first us_aqi value per city.
    """
    frames = _sorted_city_frames(df, ["pm2_5", "us_aqi"])
    pieces = []
    offset_lines = []
    for city, frame in frames.items():
        frame = frame.copy()
        # A time-based window, not a 24-row window, so an hourly gap can never
        # stretch the average over more than 24 hours.
        frame["pm2_5_24h"] = (
            frame.set_index("ts")["pm2_5"]
            .rolling(f"{SUBINDEX_WINDOW_HOURS}h", min_periods=SUBINDEX_WINDOW_HOURS)
            .mean()
            .to_numpy()
        )
        frame["city"] = city
        pieces.append(frame)

        first_pm = frame.loc[frame["pm2_5"].notna(), "ts"].min()
        first_aqi = frame.loc[frame["us_aqi"].notna(), "ts"].min()
        if pd.isna(first_pm) or pd.isna(first_aqi):
            offset_lines.append(f"- {city}: pm2_5 or us_aqi has no values.")
        else:
            offset = (first_aqi - first_pm).total_seconds() / 3600.0
            offset_lines.append(
                f"- {city}: first pm2_5 at {_ts_text(first_pm)}, first us_aqi at "
                f"{_ts_text(first_aqi)}, {offset:g} hours later."
            )

    rows = pd.concat(pieces, ignore_index=True)
    eligible = rows.dropna(subset=["pm2_5_24h", "us_aqi"])
    n_sample = min(SUBINDEX_SAMPLE_SIZE, len(eligible))
    sample = eligible.sample(n=n_sample, random_state=RANDOM_STATE)

    aqi = sample["us_aqi"].to_numpy(dtype=float)
    aqi_r = _round_half_up(aqi)
    groups = [("All", np.ones(n_sample, dtype=bool))] + [
        (city, (sample["city"] == city).to_numpy()) for city in frames
    ]
    records = []
    for table in SUBINDEX_TABLES:
        sub = pm25_sub_index(sample["pm2_5_24h"].to_numpy(dtype=float), table)
        sub_r = _round_half_up(sub)
        for label, mask in groups:
            if not mask.any():
                continue
            records.append({
                "table": table,
                "group": label,
                "rows": int(mask.sum()),
                "mismatch %": 100.0 * np.mean(sub_r[mask] != aqi_r[mask]),
                "mean abs diff": float(np.mean(np.abs(sub[mask] - aqi[mask]))),
                "us_aqi > sub-index %": 100.0 * np.mean(aqi_r[mask] > sub_r[mask]),
                "us_aqi < sub-index %": 100.0 * np.mean(aqi_r[mask] < sub_r[mask]),
            })
    table_df = pd.DataFrame(records).set_index(["table", "group"])

    return (
        f"- PM2.5 averaging: trailing {SUBINDEX_WINDOW_HOURS}-hour mean of pm2_5 per "
        f"city (time-based window, all {SUBINDEX_WINDOW_HOURS} hourly values "
        "required). The hourly pm2_5 value is not used, because the EPA PM2.5 "
        f"index is defined on a {SUBINDEX_WINDOW_HOURS}-hour average.\n"
        f"- Rows with both the {SUBINDEX_WINDOW_HOURS}-hour mean and us_aqi present: "
        f"{len(eligible)}. Sample: {n_sample} rows drawn with "
        f"random_state={RANDOM_STATE}.\n"
        "- Mismatch means round(sub-index) != round(us_aqi), both rounded half up. "
        "Mean abs diff uses the unrounded sub-index. Sub-index values above the "
        "top breakpoint row are capped at 500.\n\n"
        + df_to_markdown(table_df)
        + "\n\nFirst non-missing pm2_5 and us_aqi per city (the sub-index window is "
        f"{SUBINDEX_WINDOW_HOURS} hours):\n\n"
        + "\n".join(offset_lines)
        + "\n\nus_aqi may be the maximum of several pollutant sub-indices (see "
        "Interpretation notes), so us_aqi above the PM2.5 sub-index is expected. "
        "us_aqi below the PM2.5 sub-index is not explained by that and points to a "
        "different averaging or breakpoint table."
    )


def audit_forecast_like_rows(
    df: pd.DataFrame,
    run_time: datetime,
    source_modified: datetime | None = None,
) -> tuple[str, str | None]:
    """Look for recent rows that look like model forecasts, not observations.

    Applies the two checks described in ``FORECAST_METHOD`` to each city: rows
    later than ``run_time``, and weekly repeated-value share and decimal
    precision of pm2_5 and us_aqi over the last ``FORECAST_WINDOW_DAYS`` days
    compared with the ``FORECAST_BASELINE_YEAR`` baseline. The lag-1
    autocorrelation is computed for review only. No rows are removed.

    Args:
        df: Raw master frame.
        run_time: Time the audit started (naive local time, like the data).
        source_modified: Last-modified time of the master file. Rows after it
            are counted for review only, because copying a file can change
            that time.

    Returns:
        Markdown text with the method, a per-city summary, the baseline
        table, and one weekly table per city; and the earliest flagged
        timestamp over both cities as ``YYYY-MM-DD HH:MM:SS``, or None when
        nothing is flagged.
    """
    run_ts = pd.Timestamp(run_time)
    frames = _sorted_city_frames(df, FORECAST_SIGNAL_COLUMNS)
    summary_records = []
    baseline_records = []
    weekly_blocks = []
    city_starts: list[pd.Timestamp] = []

    for city, frame in frames.items():
        ts = frame["ts"]
        last_ts = ts.max()
        future = (ts > run_ts).to_numpy()
        first_future = ts[future].min() if future.any() else pd.NaT
        # Value plus explicit unit: the days= keyword form goes through NumPy's
        # deprecated generic timedelta unit under pandas 2.3 with NumPy 2.5.
        window = pd.Timedelta(FORECAST_WINDOW_DAYS, unit="D")
        in_window = (ts > last_ts - window).to_numpy()
        in_base = (ts.dt.year == FORECAST_BASELINE_YEAR).to_numpy()
        week = ts.dt.to_period("W-SUN").dt.start_time
        weeks = sorted(week[in_window].unique())

        weekly = pd.DataFrame({"week_start": weeks})
        weekly["hours"] = [int((in_window & (week == wk).to_numpy()).sum()) for wk in weeks]
        week_first_ts = [ts[in_window & (week == wk).to_numpy()].min() for wk in weeks]
        week_flagged = np.zeros(len(weeks), dtype=bool)

        for col in FORECAST_SIGNAL_COLUMNS:
            vals = frame[col].to_numpy(dtype=float)
            prev = np.r_[np.nan, vals[:-1]]
            pair_ok = ~np.isnan(vals) & ~np.isnan(prev)
            repeated = pair_ok & (vals == prev)
            decimals = _decimal_places(vals)

            base_pairs = pair_ok[in_base].sum()
            base_rep = repeated[in_base].sum() / base_pairs if base_pairs else np.nan
            base_dec = (
                float(np.nanmedian(decimals[in_base]))
                if np.any(~np.isnan(decimals[in_base])) else np.nan
            )
            baseline_records.append({
                "city": city,
                "column": col,
                "hours": int(in_base.sum()),
                "repeated %": 100.0 * base_rep,
                "median decimals": base_dec,
                "lag-1 autocorr": _lag1_autocorr(vals[in_base]),
                "repeat flag threshold %": 100.0 * FORECAST_REPEAT_FACTOR * base_rep,
            })

            rep_col, dec_col, ac_col, flag_col = [], [], [], []
            for wk in weeks:
                m = in_window & (week == wk).to_numpy()
                pairs = pair_ok[m].sum()
                rep = repeated[m].sum() / pairs if pairs else np.nan
                dec = (
                    float(np.nanmedian(decimals[m]))
                    if np.any(~np.isnan(decimals[m])) else np.nan
                )
                # NaN compares False, so a week with no data is never flagged.
                flag = bool(rep > FORECAST_REPEAT_FACTOR * base_rep or dec < base_dec)
                rep_col.append(100.0 * rep)
                dec_col.append(dec)
                ac_col.append(_lag1_autocorr(vals[m]))
                flag_col.append("yes" if flag else "no")
            weekly[f"{col} repeated %"] = rep_col
            weekly[f"{col} median decimals"] = dec_col
            weekly[f"{col} lag-1 autocorr"] = ac_col
            weekly[f"{col} flagged"] = flag_col
            week_flagged |= np.array(flag_col) == "yes"

        flagged_starts = [t for t, f in zip(week_first_ts, week_flagged) if f]
        earliest_week = min(flagged_starts) if flagged_starts else pd.NaT
        candidates = [t for t in (first_future, earliest_week) if not pd.isna(t)]
        city_start = min(candidates) if candidates else pd.NaT
        if not pd.isna(city_start):
            city_starts.append(city_start)

        record = {
            "city": city,
            "last timestamp": _ts_text(last_ts),
            "rows after run time": int(future.sum()),
            "first row after run time": _ts_text(first_future),
            "flagged weeks": f"{int(week_flagged.sum())} of {len(weeks)}",
            "earliest flagged week row": _ts_text(earliest_week),
            "forecast-like start": _ts_text(city_start),
            "affected range": (
                f"{_ts_text(city_start)} to {_ts_text(last_ts)}"
                if not pd.isna(city_start) else "none"
            ),
            "affected rows": int((ts >= city_start).sum()) if not pd.isna(city_start) else 0,
        }
        if source_modified is not None:
            record["rows after file modified time"] = int(
                (ts > pd.Timestamp(source_modified)).sum()
            )
        summary_records.append(record)
        weekly_blocks.append(
            f"### Weekly statistics, last {FORECAST_WINDOW_DAYS} days, {city}\n\n"
            + df_to_markdown(weekly, index=False)
        )

    overall = min(city_starts) if city_starts else None
    overall_text = _ts_text(overall) if overall is not None else None

    header_lines = [f"- Audit run time: {run_ts.strftime(DATETIME_FORMAT)}"]
    if source_modified is not None:
        header_lines.append(
            "- Master file last modified: "
            f"{pd.Timestamp(source_modified).strftime(DATETIME_FORMAT)} "
            "(rows after this time are counted for review only and are not part "
            "of the flag rule)"
        )
    header_lines.append(
        "- forecast_like_start (earliest flagged timestamp, both cities): "
        + (overall_text if overall_text else "none, no row was flagged")
    )
    body = (
        "\n".join(header_lines)
        + "\n\nMethod: " + FORECAST_METHOD
        + "\n\n### Summary per city\n\n"
        + df_to_markdown(pd.DataFrame(summary_records).set_index("city"))
        + f"\n\n### {FORECAST_BASELINE_YEAR} baseline per city and column\n\n"
        + df_to_markdown(pd.DataFrame(baseline_records).set_index(["city", "column"]))
        + "\n\n" + "\n\n".join(weekly_blocks)
    )
    return body, overall_text


def write_flags(forecast_like_start: str | None, run_time: datetime, path: Path) -> dict:
    """Save the Forecast_Like_Row result for the pipeline to read.

    Only the start timestamp is stored; the pipeline flags rows at or after
    it with ``is_forecast_like`` and never removes them (Requirement 2.4).

    Args:
        forecast_like_start: Earliest flagged timestamp, or None.
        run_time: Time the audit started.
        path: Output file, normally ``data/processed/audit_flags.json``.

    Returns:
        The dict that was written.
    """
    payload = {
        "forecast_like_start": forecast_like_start,
        "method": FORECAST_METHOD,
        "audit_run": run_time.strftime(DATETIME_FORMAT),
    }
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def audit_fixed_statements() -> str:
    """Return the interpretation notes that every report must contain.

    These are not computed, so they contain no numbers (Requirement 2.1).

    Returns:
        Markdown bullet list of the fixed statements.
    """
    return "\n".join(f"- {s}" for s in FIXED_STATEMENTS)


def approved_gap_strategy() -> str:
    """Return the Gap_Strategy approved at the Phase 1 checkpoint.

    The text is fixed and carries no data counts (Requirement 2.1); it points
    to the audit sections that hold the counts.

    Returns:
        Markdown with the approval line and the numbered strategy rules.
    """
    rules = "\n".join(f"{i}. {rule}" for i, rule in enumerate(APPROVED_GAP_STRATEGY, 1))
    return (
        f"Approved by the user at the Phase 1 checkpoint on {GAP_STRATEGY_APPROVED_ON}. "
        "Phase 2 applies it in `AirQualityDataPipeline._apply_gap_strategy` "
        "(`src/pipeline.py`).\n\n"
        + rules
    )


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------

def build_sections(df: pd.DataFrame, run_time: datetime) -> tuple[list[str], str | None]:
    """Run every audit and render the results as Markdown sections in order.

    To add an audit, write an ``audit_*`` function and append one
    ``make_section`` call here; printing and the report pick it up.

    Args:
        df: Raw master frame.
        run_time: Time the audit started, shown in the first section and used
            as the cut-off for rows that cannot be observations yet.

    Returns:
        Markdown sections in report order, and the Forecast_Like_Row start
        timestamp (or None) that ``main`` writes to ``audit_flags.json``.
    """
    gaps = audit_hourly_gaps(df)
    gap_body = f"Steps other than 1 hour: {len(gaps)}.\n\n" + df_to_markdown(
        gaps.head(MAX_GAP_ROWS_SHOWN), index=False
    )
    if len(gaps) > MAX_GAP_ROWS_SHOWN:
        gap_body += f"\n\nShowing the first {MAX_GAP_ROWS_SHOWN} of {len(gaps)} gaps."

    monthly_table, monthly_sentence = audit_monthly_means(df)
    mismatches = audit_datetime_vs_time(df)

    source_modified = datetime.fromtimestamp(MASTER_PATH.stat().st_mtime)
    forecast_body, forecast_start = audit_forecast_like_rows(df, run_time, source_modified)
    forecast_body += (
        f"\n\nResult saved to `{FLAGS_PATH.relative_to(PROJECT_ROOT).as_posix()}`. "
        "Rows are flagged only and never removed."
    )

    sections = [
        make_section("Run information", (
            f"- Source file: `{MASTER_PATH.relative_to(PROJECT_ROOT).as_posix()}`\n"
            f"- Audit run: {run_time.strftime(DATETIME_FORMAT)}"
        )),
        make_section("Shape and dtypes", audit_shape_and_dtypes(df)),
        make_section("Date range and rows per city",
                     df_to_markdown(audit_date_range_per_city(df))),
        make_section("Missing values per column per city",
                     df_to_markdown(audit_missing_per_city(df))),
        make_section("Duplicate timestamps per city",
                     df_to_markdown(audit_duplicate_timestamps(df))),
        make_section("Hourly gaps", gap_body),
        make_section("Analysis column statistics per city",
                     df_to_markdown(audit_analysis_stats(df))),
        make_section("Impossible values per city",
                     df_to_markdown(audit_impossible_values(df))),
        make_section("datetime vs time",
                     f"Rows where `datetime` differs from the parsed `time`: {mismatches}."),
        make_section("Monthly means of pm2_5 and us_aqi",
                     df_to_markdown(monthly_table) + "\n\n" + monthly_sentence),
        make_section("us_aqi vs EPA PM2.5 sub-index", audit_aqi_subindex(df)),
        make_section("us_aqi above 500", audit_aqi_above_500(df)),
        make_section("Forecast-like rows", forecast_body),
        make_section("Approved Gap_Strategy", approved_gap_strategy()),
        make_section("Interpretation notes", audit_fixed_statements()),
    ]
    return sections, forecast_start


def write_report(sections: list[str], path: Path) -> None:
    """Write the audit sections to a Markdown file under a title.

    Args:
        sections: Markdown sections in report order.
        path: Output file, normally ``docs/data_audit.md``.
    """
    ensure_dir(path.parent)
    text = "# Dhund-Khabar AI: Phase 1 data audit\n\n" + "\n".join(sections)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    """Load the master file, print every section, and save the report and flags."""
    run_time = datetime.now()
    df = load_master(MASTER_PATH)
    sections, forecast_start = build_sections(df, run_time)
    for section in sections:
        print(section)
    write_report(sections, REPORT_PATH)
    write_flags(forecast_start, run_time, FLAGS_PATH)
    print(f"Report written to {REPORT_PATH.relative_to(PROJECT_ROOT).as_posix()}")
    print(f"Flags written to {FLAGS_PATH.relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
