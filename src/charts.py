"""Dark-themed Matplotlib charts for the Dhund-Khabar AI report and dashboard.

Every chart function is a plain function (Requirement 3.2) that returns a
Matplotlib Figure. The same Figure can be saved for the report with
:func:`save_figure` or shown in the dashboard with ``st.pyplot(fig)``. Nothing
here calls ``plt.show``.

The colours match the dashboard design: figure #0B0F19, axes #161F30, grid
#2A3B53, light grey text, and cyan #00F2FE accents, with amber #FFB300 and
crimson #FF3366 from the same palette where a second colour is needed. Tier
bands take their colours from ``alerts.TIERS``, so a band on a chart always
has the colour of the alert the dashboard would show for that value. The
correlation heatmap is drawn with seaborn; every other chart uses Matplotlib
directly.
"""

from collections.abc import Iterable
from datetime import date
from pathlib import Path

import matplotlib

# Agg renders to memory only. Selecting it before pyplot is imported lets the
# training script, the tests, and Streamlit build figures with no display.
matplotlib.use("Agg")

import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from src.alerts import TIERS  # noqa: E402
from src.utils import (  # noqa: E402
    CITIES,
    WINTER_MONTHS,
    ensure_dir,
    mae,
    pm25_breakpoints,
    r2,
)

BG_COLOR: str = "#0B0F19"
SURFACE_COLOR: str = "#161F30"
GRID_COLOR: str = "#2A3B53"
ACCENT_COLOR: str = "#00F2FE"
TEXT_COLOR: str = "#C9D1D9"
# Near-white keeps the observed series clearly apart from the cyan prediction
# and from every tier band colour.
ACTUAL_COLOR: str = "#E6EDF3"
TIER_BAND_ALPHA: float = 0.12
FIGURE_DPI: int = 150

# First full winter inside the Test_Period (design, Requirement 14.2).
DEFAULT_WINTER_START: str = "2025-11-01"
DEFAULT_WINTER_END: str = "2026-01-31"
# The top tier is open-ended, so its band stops a little above the data instead
# of at infinity; 5 % headroom keeps the highest peak off the frame edge.
_Y_HEADROOM: float = 1.05
# Bands thinner than this share of the y-range get no text label, because the
# label would overlap the neighbouring band's label.
_MIN_LABEL_SHARE: float = 0.04

UNIT_LABELS: dict[str, str] = {"pm2_5": "PM2.5 (µg/m³)", "us_aqi": "US AQI"}
TARGET_NAMES: dict[str, str] = {"pm2_5": "PM2.5", "us_aqi": "US AQI"}

# Second and third colours of the dashboard palette (Requirement 16.2).
AMBER_COLOR: str = "#FFB300"
CRIMSON_COLOR: str = "#FF3366"
VIOLET_COLOR: str = "#9933FF"

# Heatmap rows and columns, in the order Requirement 14.3 lists them.
CORRELATION_WEATHER_COLUMNS: list[str] = [
    "wind_speed_10m", "temperature_2m", "relative_humidity_2m",
    "surface_pressure", "precipitation",
]
CORRELATION_POLLUTANT_COLUMNS: list[str] = ["pm2_5", "pm10", "us_aqi"]
# Negative correlations run to cyan, positive ones to crimson, and zero sits on
# the grid colour, so weak links fade into the dark theme and strong ones
# stand out. Crimson marks the "more pollution" direction, as on the dashboard.
CORRELATION_CMAP: LinearSegmentedColormap = LinearSegmentedColormap.from_list(
    "dk_diverging", [ACCENT_COLOR, GRID_COLOR, CRIMSON_COLOR]
)

# Fixed colours per city so Peshawar and Lahore look the same in every chart.
# Any other city (synthetic test data) takes the next unused palette colour.
CITY_COLORS: dict[str, str] = {"Peshawar": ACCENT_COLOR, "Lahore": AMBER_COLOR}
_EXTRA_CITY_COLORS: list[str] = [CRIMSON_COLOR, VIOLET_COLOR, ACTUAL_COLOR]
MONTH_LABELS: list[str] = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
# A neutral near-white wash marks the winter months without clashing with the
# cyan and amber city lines.
WINTER_SHADE_COLOR: str = ACTUAL_COLOR
WINTER_SHADE_ALPHA: float = 0.10


def apply_dark_theme(fig: Figure, axes: Iterable[Axes] | None = None,
                     grid: bool = True) -> Figure:
    """Apply the dashboard's dark colours to a figure and its axes.

    Sets the figure background to #0B0F19, each axes background to #161F30,
    spines and grid lines to #2A3B53, tick labels, axis labels, and legend
    text to light grey, and axes titles to cyan #00F2FE. Call it after the
    legend is created so the legend is styled too.

    Args:
        fig: Figure to style.
        axes: Axes to style. None means every axes in ``fig``.
        grid: Draw #2A3B53 grid lines. Charts such as heatmaps pass False.

    Returns:
        The same Figure, so the call can be chained.
    """
    fig.patch.set_facecolor(BG_COLOR)
    for ax in (fig.axes if axes is None else axes):
        ax.set_facecolor(SURFACE_COLOR)
        for spine in ax.spines.values():
            spine.set_color(GRID_COLOR)
        ax.tick_params(which="both", colors=TEXT_COLOR)
        ax.xaxis.label.set_color(TEXT_COLOR)
        ax.yaxis.label.set_color(TEXT_COLOR)
        # Date formatters put the year in the offset text, which keeps the
        # default black colour unless it is set here.
        ax.xaxis.get_offset_text().set_color(TEXT_COLOR)
        ax.yaxis.get_offset_text().set_color(TEXT_COLOR)
        ax.title.set_color(ACCENT_COLOR)
        if grid:
            ax.grid(True, color=GRID_COLOR, linewidth=0.6)
            ax.set_axisbelow(True)
        else:
            ax.grid(False)
        legend = ax.get_legend()
        if legend is not None:
            legend.get_frame().set_facecolor(SURFACE_COLOR)
            legend.get_frame().set_edgecolor(GRID_COLOR)
            for text in legend.get_texts():
                text.set_color(TEXT_COLOR)
    return fig


def save_figure(fig: Figure, path: str | Path) -> Path:
    """Save a figure as a 150 dpi PNG on the dark background, then close it.

    The parent folder is created if needed. Once writing starts, the figure
    is closed even when saving fails, so a loop that saves many charts cannot
    pile up open figures in pyplot's memory. A path with the wrong suffix is
    rejected before the figure is touched, so the caller can retry with a
    corrected path.

    Args:
        fig: Figure to save.
        path: Target file; it must end in ``.png``.

    Returns:
        The path written, as a Path.

    Raises:
        ValueError: If ``path`` does not end in ``.png``. The figure stays open.
    """
    path = Path(path)
    if path.suffix.lower() != ".png":
        raise ValueError(f"Figures are saved as PNG; path must end in .png, got '{path}'")
    try:
        ensure_dir(path.parent)
        # facecolor is passed explicitly because savefig would otherwise use
        # the rcParams default (white) for the area around the axes.
        fig.savefig(path, format="png", dpi=FIGURE_DPI, facecolor=BG_COLOR)
    finally:
        plt.close(fig)
    return path


def _tier_bands(target_type: str) -> list[tuple[float, float, str, str]]:
    """Return the EPA tier bands for a target as (lower, upper, name, colour).

    us_aqi bands are the AQI ranges of ``alerts.TIERS``. pm2_5 bands are the
    concentration ranges of the 2024 EPA breakpoint table from
    ``utils.pm25_breakpoints``, one row per tier. Each pm2_5 band starts at the
    previous row's C_hi rather than its own C_lo, because the table leaves
    0.1 ug/m3 gaps between rows (9.0 then 9.1) that would show as dark lines.
    The top band of both targets is open-ended (upper = inf); concentrations
    above the last row still map to Hazardous because the sub-index is capped
    at AQI 500.

    Args:
        target_type: ``"pm2_5"`` or ``"us_aqi"``.

    Returns:
        Six bands ordered from Good to Hazardous (Emergency).

    Raises:
        RuntimeError: If the breakpoint table and the tier list ever differ in
            length, since the bands would then get the wrong colours.
    """
    if target_type == "us_aqi":
        return [(t["lower"], t["upper"], t["name"], t["color"]) for t in TIERS]

    table = pm25_breakpoints("2024")
    if len(table) != len(TIERS):
        raise RuntimeError(
            f"PM2.5 breakpoint table has {len(table)} rows but there are {len(TIERS)} tiers"
        )
    uppers = list(table[:-1, 1]) + [float("inf")]
    lowers = [0.0] + list(table[:-1, 1])
    return [
        (float(lo), float(hi), tier["name"], tier["color"])
        for lo, hi, tier in zip(lowers, uppers, TIERS)
    ]


def _as_1d_float(values: pd.Series | np.ndarray, name: str) -> np.ndarray:
    """Convert chart input values to a 1-D float array.

    Args:
        values: Array-like of numbers.
        name: Argument name used in the error message.

    Returns:
        A 1-D float array.

    Raises:
        ValueError: If the values are not one-dimensional.
    """
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {arr.shape}")
    return arr


def _check_frame(df: pd.DataFrame, required: list[str], name: str) -> None:
    """Check that a chart input is a non-empty DataFrame with the needed columns.

    Args:
        df: Frame passed to a chart function.
        required: Column names the chart reads.
        name: Argument name used in the error messages.

    Raises:
        TypeError: If ``df`` is not a pandas DataFrame.
        ValueError: If any required column is missing (all of them are named,
            in the order of ``required``) or the frame has no rows.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame, got {type(df).__name__}")
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")
    if df.empty:
        raise ValueError(f"{name} is empty; there is nothing to plot")


def _window_mask(stamps: pd.DatetimeIndex, start: str | date | None,
                 end: str | date | None) -> np.ndarray:
    """Select the timestamps inside a date window.

    A date-only ``end`` such as "2026-01-31" (or a ``datetime.date`` from a
    Streamlit date picker) includes the whole day up to 23:00; an ``end``
    with a time of day is used as given. Both bounds are inclusive.

    Args:
        stamps: Parsed timestamps.
        start: First timestamp to keep, or None for no lower bound.
        end: Last day or timestamp to keep, or None for no upper bound.

    Returns:
        A boolean array, True for rows inside the window.

    Raises:
        ValueError: If ``start`` is later than ``end``.
    """
    start_ts = None if start is None else pd.Timestamp(start)
    end_ts = None if end is None else pd.Timestamp(end)
    if start_ts is not None and end_ts is not None and start_ts > end_ts:
        raise ValueError(f"Window start {start_ts} is after window end {end_ts}")

    mask = np.ones(len(stamps), dtype=bool)
    if start_ts is not None:
        mask &= np.asarray(stamps >= start_ts)
    if end_ts is not None:
        if end_ts == end_ts.normalize():
            # Positional form: with numpy 2.5, pd.Timedelta(days=1) emits a
            # DeprecationWarning about the generic timedelta unit.
            mask &= np.asarray(stamps < end_ts + pd.Timedelta(1, unit="D"))
        else:
            mask &= np.asarray(stamps <= end_ts)
    return mask


def plot_actual_vs_predicted(
    dates: pd.Series | pd.DatetimeIndex | np.ndarray,
    actual: pd.Series | np.ndarray,
    predicted: pd.Series | np.ndarray,
    target_type: str,
    city: str,
    horizon: int,
    start: str | date | None = DEFAULT_WINTER_START,
    end: str | date | None = DEFAULT_WINTER_END,
) -> Figure:
    """Plot actual and predicted PM2.5 or AQI over a date window with tier bands.

    Behind the two lines, ``axhspan`` bands at alpha 0.12 mark the EPA tiers:
    AQI ranges for us_aqi and 2024 EPA concentration breakpoints for pm2_5,
    in the colours of ``alerts.TIERS``. The top band stops 5 % above the
    highest value in the window. NaN values leave gaps in the lines rather
    than being dropped, so missing hours stay visible.

    Args:
        dates: Timestamp each value refers to. For a forecast this is the
            target time t + h, so the actual line sits on the calendar hour it
            was measured. Strings or datetimes; any order.
        actual: Observed values, same length as ``dates``.
        predicted: Model predictions, same length as ``dates``.
        target_type: ``"pm2_5"`` or ``"us_aqi"``.
        city: City name for the title.
        horizon: Forecast horizon in hours for the title and legend.
        start: First day of the window (inclusive). Default 2025-11-01; None
            means from the first timestamp.
        end: Last day of the window (inclusive, whole day). Default
            2026-01-31; None means up to the last timestamp.

    Returns:
        A 12 x 5 inch Figure with the dark theme applied.

    Raises:
        ValueError: If ``target_type`` is unknown, ``horizon`` is not a
            positive whole number, the inputs differ in length or are not 1-D,
            ``start`` is after ``end``, the window holds no rows, or the window
            holds no finite value.
    """
    if target_type not in UNIT_LABELS:
        raise ValueError(
            f"target_type must be one of {list(UNIT_LABELS)}, got '{target_type}'"
        )
    if isinstance(horizon, bool) or not float(horizon).is_integer() or horizon < 1:
        raise ValueError(f"horizon must be a positive whole number of hours, got {horizon}")
    horizon = int(horizon)

    stamps = pd.DatetimeIndex(pd.to_datetime(np.asarray(dates)))
    actual_arr = _as_1d_float(actual, "actual")
    predicted_arr = _as_1d_float(predicted, "predicted")
    if not len(stamps) == len(actual_arr) == len(predicted_arr):
        raise ValueError(
            "dates, actual, and predicted must have the same length, got "
            f"{len(stamps)}, {len(actual_arr)}, and {len(predicted_arr)}"
        )

    mask = _window_mask(stamps, start, end)
    if not mask.any():
        data_range = (
            f"{stamps.min()} to {stamps.max()}" if len(stamps) else "no timestamps"
        )
        raise ValueError(
            f"No rows in the window {start or 'first timestamp'} to "
            f"{end or 'last timestamp'}; the data covers {data_range}"
        )
    # Sorting by time keeps the lines from zig-zagging when the caller passes
    # rows in another order, e.g. both cities' test rows stacked.
    order = np.argsort(stamps[mask].to_numpy(), kind="stable")
    win_stamps = stamps[mask][order]
    win_actual = actual_arr[mask][order]
    win_predicted = predicted_arr[mask][order]

    values = np.concatenate([win_actual, win_predicted])
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("The window has no finite actual or predicted values to plot")
    y_top = max(float(finite.max()) * _Y_HEADROOM, 1.0)
    # Linear models can predict slightly below 0; the axis follows them down
    # instead of hiding those points.
    y_bottom = min(0.0, float(finite.min()) * _Y_HEADROOM)

    fig, ax = plt.subplots(figsize=(12, 5), layout="constrained")
    y_span = y_top - y_bottom
    for lower, upper, name, color in _tier_bands(target_type):
        if lower >= y_top:
            break
        band_top = min(upper, y_top)
        ax.axhspan(lower, band_top, color=color, alpha=TIER_BAND_ALPHA,
                   linewidth=0, zorder=0)
        if (band_top - lower) / y_span >= _MIN_LABEL_SHARE:
            # x in axes units and y in data units puts the label just right of
            # the plot area, level with its band, so it never covers the lines.
            # Constrained layout shrinks the axes to make room for it.
            ax.text(1.005, (lower + band_top) / 2, name,
                    transform=ax.get_yaxis_transform(), ha="left", va="center",
                    fontsize=7, color=TEXT_COLOR, alpha=0.75, clip_on=False)

    ax.plot(win_stamps, win_actual, color=ACTUAL_COLOR, linewidth=1.0,
            label="Actual", zorder=2)
    ax.plot(win_stamps, win_predicted, color=ACCENT_COLOR, linewidth=1.0,
            alpha=0.9, label=f"Predicted ({horizon} h ahead)", zorder=3)

    ax.set_ylim(y_bottom, y_top)
    ax.set_xlim(win_stamps.min(), win_stamps.max())
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    # The default offset shows only the last tick's year and month ("2026-Jan"),
    # which mislabels the Nov and Dec 2025 ticks of the default window. Month
    # starts carry their own year instead, and the title holds the full range.
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(
        locator, zero_formats=["", "%Y", "%b %Y", "%b-%d", "%H:%M", "%H:%M"],
        show_offset=False))
    ax.set_xlabel("Date")
    ax.set_ylabel(UNIT_LABELS[target_type])
    ax.set_title(
        f"{city}: actual vs predicted {TARGET_NAMES[target_type]}, "
        f"{horizon} h horizon\n"
        f"{win_stamps.min():%d %b %Y} to {win_stamps.max():%d %b %Y}"
    )
    ax.legend(loc="upper left")
    return apply_dark_theme(fig, [ax])


def _correlation_matrix(df_city: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Compute the weather-by-pollutant Pearson correlation matrix of one city.

    Only rows where all eight columns have a value are used, so every cell is
    computed on the same hours and the row count shown on the heatmap holds
    for every cell. On the real data this drops the Leading_Gap, where the
    pollutant columns and us_aqi are still missing.

    Args:
        df_city: One city's rows with the five weather columns of
            ``CORRELATION_WEATHER_COLUMNS`` and the three pollutant columns of
            ``CORRELATION_POLLUTANT_COLUMNS``.

    Returns:
        ``(matrix, n_rows)``: a 5 x 3 DataFrame with the weather columns as the
        index and the pollutant columns as columns, and the number of complete
        rows behind it. A cell is NaN when one of its columns is constant over
        those rows.

    Raises:
        ValueError: If a column cannot be read as numbers, or fewer than 2
            rows have all eight values.
    """
    columns = CORRELATION_WEATHER_COLUMNS + CORRELATION_POLLUTANT_COLUMNS
    complete = df_city[columns].astype(float).dropna()
    if len(complete) < 2:
        raise ValueError(
            f"Need at least 2 rows with all of {columns} present to compute "
            f"correlations, got {len(complete)}"
        )
    matrix = complete.corr(method="pearson").loc[
        CORRELATION_WEATHER_COLUMNS, CORRELATION_POLLUTANT_COLUMNS
    ]
    return matrix, len(complete)


def plot_correlation_heatmap(df_city: pd.DataFrame, city: str) -> Figure:
    """Plot Pearson correlations between weather and pollution for one city.

    Rows are wind_speed_10m, temperature_2m, relative_humidity_2m,
    surface_pressure, and precipitation; columns are pm2_5, pm10, and us_aqi.
    Each cell is annotated with r to two decimals. The diverging colour scale
    is fixed to -1 .. 1 and centred at 0 (cyan for negative, crimson for
    positive), so two cities' heatmaps can be compared cell by cell. The
    correlations use only rows where all eight columns are present (see
    :func:`_correlation_matrix`).

    Args:
        df_city: Rows of a single city, e.g. ``pipeline.filter_city(city)`` or
            the city's rows of ``features_all.csv``.
        city: City name for the title. If ``df_city`` has a ``city`` column,
            every non-missing value in it must equal this name.

    Returns:
        A 7.5 x 5.2 inch Figure with the dark theme applied.

    Raises:
        TypeError: If ``df_city`` is not a DataFrame.
        ValueError: If a required column is missing, the frame is empty, its
            ``city`` column holds other cities, a column is not numeric, or
            fewer than 2 rows have all eight values.
    """
    _check_frame(df_city, CORRELATION_WEATHER_COLUMNS + CORRELATION_POLLUTANT_COLUMNS,
                 "df_city")
    if "city" in df_city.columns:
        # Pooling two cities would mix their different pollution levels into
        # one correlation and label the result with a single city's name.
        others = sorted(set(df_city["city"].dropna().astype(str)) - {city})
        if others:
            raise ValueError(
                f"df_city must hold only rows for '{city}', but it also contains "
                f"{others}; filter it to one city first"
            )
    matrix, n_rows = _correlation_matrix(df_city)

    fig, ax = plt.subplots(figsize=(7.5, 5.2), layout="constrained")
    sns.heatmap(
        matrix, ax=ax, annot=True, fmt=".2f", annot_kws={"fontsize": 10},
        cmap=CORRELATION_CMAP, center=0.0, vmin=-1.0, vmax=1.0,
        linewidths=1.0, linecolor=BG_COLOR, cbar_kws={"label": "Pearson r"},
    )
    ax.set_xlabel("Pollution variable")
    ax.set_ylabel("Weather variable")
    ax.tick_params(axis="y", rotation=0)
    ax.tick_params(length=0)
    ax.set_title(
        f"{city}: weather vs pollution, Pearson correlation\n"
        f"{n_rows:,} hourly rows with all eight values"
    )
    # grid=False because grid lines would cut through the coloured cells; the
    # colour bar's axes are styled too since it is one of fig.axes.
    return apply_dark_theme(fig, grid=False)


def plot_feature_importance(importance_df: pd.DataFrame, title: str,
                            top_n: int = 10) -> Figure:
    """Plot the top features by importance as horizontal cyan bars.

    The frame is re-sorted here, highest first, so the chart is correct even
    if the caller changed the order; the highest bar is drawn at the top.
    Each bar carries its value to three significant digits, which suits both
    Random Forest importances (shares that sum to 1) and the larger
    ``|coef_|`` values of a linear model.

    Args:
        importance_df: Columns ``feature`` and ``importance``, as returned by
            ``SmogPredictorModel.get_feature_importance()``.
        title: Chart title, e.g. "Peshawar: US AQI 24 h Random Forest".
        top_n: Number of features to show. If the frame has fewer rows, all
            of them are shown.

    Returns:
        A Figure 9 inches wide whose height grows with the number of bars,
        with the dark theme applied.

    Raises:
        TypeError: If ``importance_df`` is not a DataFrame.
        ValueError: If ``feature`` or ``importance`` is missing, the frame is
            empty, ``top_n`` is not a positive whole number, or an importance
            value is not a finite, non-negative number.
    """
    _check_frame(importance_df, ["feature", "importance"], "importance_df")
    if isinstance(top_n, bool) or not isinstance(top_n, (int, np.integer)) or top_n < 1:
        raise ValueError(f"top_n must be a positive whole number, got {top_n!r}")
    values = importance_df["importance"].to_numpy(dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("importance values must be finite numbers")
    if np.any(values < 0):
        # Bars start at 0 and the label reads "importance", which only makes
        # sense for magnitudes such as feature_importances_ or |coef_|.
        raise ValueError("importance values must be non-negative")

    ranked = pd.DataFrame({
        "feature": importance_df["feature"].astype(str).to_numpy(),
        "importance": values,
    })
    # A stable sort keeps tied features in the caller's order on every run.
    ranked = ranked.sort_values("importance", ascending=False,
                                kind="mergesort").head(int(top_n))
    n_shown = len(ranked)

    fig, ax = plt.subplots(figsize=(9, max(3.0, 1.2 + 0.42 * n_shown)),
                           layout="constrained")
    positions = np.arange(n_shown)
    bars = ax.barh(positions, ranked["importance"].to_numpy(), height=0.65,
                   color=ACCENT_COLOR, zorder=2)
    ax.set_yticks(positions, labels=ranked["feature"].tolist())
    # barh draws the first row at the bottom; inverting puts the highest
    # importance on top, where the eye starts reading.
    ax.invert_yaxis()
    ax.bar_label(bars, labels=[f"{v:.3g}" for v in ranked["importance"]],
                 padding=3, color=TEXT_COLOR, fontsize=8)
    top_value = float(ranked["importance"].max())
    # 15 % headroom leaves space for the value label of the longest bar.
    ax.set_xlim(0.0, top_value * 1.15 if top_value > 0 else 1.0)
    ax.set_xlabel(f"Importance (top {n_shown} of {len(importance_df)} features)")
    ax.set_title(title)
    apply_dark_theme(fig, [ax])
    # Horizontal grid lines would run through the bars; only the value axis
    # keeps its grid.
    ax.yaxis.grid(False)
    return fig


def plot_predicted_vs_actual_scatter(actual: pd.Series | np.ndarray,
                                     predicted: pd.Series | np.ndarray,
                                     title: str) -> Figure:
    """Plot predicted against actual values with a dashed y = x line.

    Both axes share the same limits and an equal aspect ratio, so a point's
    vertical distance from the dashed line is its error at true scale. The
    box in the top-left corner shows the MAE and R2 computed here with
    ``utils.mae`` and ``utils.r2`` on the plotted pairs, and the number of
    pairs. Pairs where either value is NaN or infinite are dropped before
    plotting and before the metrics, and the dropped count is shown.

    Args:
        actual: Observed values (x axis).
        predicted: Model predictions (y axis), same length as ``actual``.
        title: Chart title, e.g. "Lahore: US AQI, 24 h Random Forest".

    Returns:
        A 6.5 x 6.5 inch Figure with the dark theme applied.

    Raises:
        ValueError: If the inputs are not 1-D, differ in length, or have
            fewer than 2 pairs with both values finite (R2 needs at least 2).
    """
    actual_arr = _as_1d_float(actual, "actual")
    predicted_arr = _as_1d_float(predicted, "predicted")
    if len(actual_arr) != len(predicted_arr):
        raise ValueError(
            "actual and predicted must have the same length, got "
            f"{len(actual_arr)} and {len(predicted_arr)}"
        )
    keep = np.isfinite(actual_arr) & np.isfinite(predicted_arr)
    n_pairs = int(keep.sum())
    if n_pairs < 2:
        raise ValueError(
            f"Need at least 2 pairs where actual and predicted are both finite, got {n_pairs}"
        )
    x, y = actual_arr[keep], predicted_arr[keep]
    n_dropped = len(actual_arr) - n_pairs
    mae_value = mae(x, y)
    r2_value = r2(x, y)

    # Starting at 0 keeps the scale honest for AQI and concentrations; the
    # lower limit only goes below 0 when a linear model predicts negatives.
    data_low = min(float(x.min()), float(y.min()))
    data_high = max(float(x.max()), float(y.max()))
    low = min(0.0, data_low)
    span = data_high - low if data_high > low else 1.0
    pad = 0.03 * span
    lim_low = low - pad if low < 0 else low
    lim_high = data_high + pad if data_high > low else low + span

    fig, ax = plt.subplots(figsize=(6.5, 6.5), layout="constrained")
    # Small, faint points let dense clusters show as brighter areas instead of
    # one solid blob when tens of thousands of hours overlap.
    ax.scatter(x, y, s=6, alpha=0.25, color=ACCENT_COLOR, edgecolors="none",
               label="Test hours", zorder=2)
    ax.plot([lim_low, lim_high], [lim_low, lim_high], linestyle="--",
            linewidth=1.2, color=ACTUAL_COLOR, label="y = x (perfect prediction)",
            zorder=3)
    ax.set_xlim(lim_low, lim_high)
    ax.set_ylim(lim_low, lim_high)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(title)

    stats = f"MAE = {mae_value:.2f}\nR² = {r2_value:.3f}\nn = {n_pairs:,}"
    if n_dropped:
        stats += f" ({n_dropped:,} non-finite dropped)"
    ax.text(0.03, 0.97, stats, transform=ax.transAxes, ha="left", va="top",
            fontsize=9, color=TEXT_COLOR, zorder=4,
            bbox={"boxstyle": "round,pad=0.4", "facecolor": BG_COLOR,
                  "edgecolor": GRID_COLOR})
    legend = ax.legend(loc="lower right", markerscale=3)
    # The legend marker copies the faint point alpha; full opacity keeps it
    # readable.
    for handle in legend.legend_handles:
        handle.set_alpha(1.0)
    return apply_dark_theme(fig, [ax])


def plot_city_comparison(df: pd.DataFrame) -> Figure:
    """Plot the monthly mean us_aqi of each city with the winter months shaded.

    Months are calendar months 1 to 12. Each point is the mean of all hourly
    us_aqi values of that city in that calendar month over every year in
    ``df``, so a month with more years of data contributes more hours.
    Missing us_aqi values (the Leading_Gap) are left out. Peshawar is drawn in
    cyan and Lahore in amber; the Winter_Months 11, 12, 1, and 2 are shaded.

    Args:
        df: Hourly rows of one or more cities with columns ``city``,
            ``datetime`` (datetime64 or parseable strings), and ``us_aqi``,
            e.g. the frame from ``load_data`` or ``features_all.csv``.

    Returns:
        A 10 x 5 inch Figure with the dark theme applied.

    Raises:
        TypeError: If ``df`` is not a DataFrame.
        ValueError: If ``city``, ``datetime``, or ``us_aqi`` is missing, the
            frame is empty, ``us_aqi`` is not numeric, ``datetime`` cannot be
            parsed, or no row has city, datetime, and us_aqi all present.
    """
    _check_frame(df, ["city", "datetime", "us_aqi"], "df")
    frame = pd.DataFrame({
        "city": df["city"].to_numpy(),
        "datetime": pd.to_datetime(df["datetime"]).to_numpy(),
        "us_aqi": df["us_aqi"].astype(float).to_numpy(),
    }).dropna()
    if frame.empty:
        raise ValueError("df has no row with city, datetime, and us_aqi all present")
    frame["city"] = frame["city"].astype(str)
    frame["month"] = frame["datetime"].dt.month

    # Reindexing to all 12 months leaves a gap in a city's line for a month
    # without data instead of joining its neighbours.
    monthly = (frame.groupby(["city", "month"])["us_aqi"].mean()
               .unstack("month").reindex(columns=range(1, 13)))
    present = list(monthly.index)
    ordered = ([c for c in CITIES if c in present]
               + sorted(c for c in present if c not in CITIES))

    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    for month in WINTER_MONTHS:
        # One span per month, so November-December and January-February each
        # form one even block without overlapping alpha.
        ax.axvspan(month - 0.5, month + 0.5, color=WINTER_SHADE_COLOR,
                   alpha=WINTER_SHADE_ALPHA, linewidth=0, zorder=0)

    months = np.arange(1, 13)
    n_extra = 0
    for city in ordered:
        if city in CITY_COLORS:
            color = CITY_COLORS[city]
        else:
            color = _EXTRA_CITY_COLORS[n_extra % len(_EXTRA_CITY_COLORS)]
            n_extra += 1
        ax.plot(months, monthly.loc[city].to_numpy(), color=color, linewidth=2.0,
                marker="o", markersize=5, label=city, zorder=3)

    ax.set_xticks(months, labels=MONTH_LABELS)
    ax.set_xlim(0.5, 12.5)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Calendar month")
    ax.set_ylabel("Mean US AQI")
    first, last = frame["datetime"].min(), frame["datetime"].max()
    ax.set_title(
        "Monthly mean US AQI by city\n"
        f"hourly values averaged over all years, {first:%b %Y} to {last:%b %Y}"
    )
    handles, _ = ax.get_legend_handles_labels()
    # The legend swatch is tiny and sits on the legend frame, so the wash
    # alpha alone is barely visible there; double it to keep the key readable.
    handles.append(Patch(facecolor=WINTER_SHADE_COLOR, alpha=WINTER_SHADE_ALPHA * 2,
                         label="Winter months (Nov to Feb)"))
    # Mid-year months have the lowest AQI, so the top centre is free space.
    ax.legend(handles=handles, loc="upper center")
    return apply_dark_theme(fig, [ax])
