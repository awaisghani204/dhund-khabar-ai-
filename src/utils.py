"""Shared constants and pure helper functions for Dhund-Khabar AI.

This module is imported by the audit script, the pipeline, the model, the
training script, and the dashboard. It must stay free of module-level imports
from other ``src`` modules so that it can never take part in a circular import.
The tier metrics need ``src.alerts`` and import it inside the function body,
which only runs after both modules have finished loading. Its only third-party
dependencies are numpy, pandas, and scikit-learn metrics.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

# Paths are anchored to the project root rather than the working directory, so
# pytest, Streamlit, and the scripts find the same files from any start folder.
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
MASTER_PATH: Path = PROJECT_ROOT / "data" / "pakistan_smog_master_hourly.csv"
PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"
MODELS_DIR: Path = PROJECT_ROOT / "models"
RESULTS_DIR: Path = PROJECT_ROOT / "results"
FIGURES_DIR: Path = RESULTS_DIR / "figures"

# Spelled and ordered exactly as in the master file header (pm10 comes before
# pm2_5), so a missing-column error can list names in file order.
REQUIRED_COLUMNS: list[str] = [
    "city", "latitude", "longitude", "time", "pm10", "pm2_5", "carbon_monoxide",
    "nitrogen_dioxide", "sulphur_dioxide", "ozone", "dust", "us_aqi",
    "temperature_2m", "relative_humidity_2m", "surface_pressure", "wind_speed_10m",
    "wind_direction_10m", "precipitation", "datetime", "year", "month", "day",
    "hour", "day_of_week", "is_weekend", "aqi_category",
]
POLLUTANT_COLUMNS: list[str] = [
    "pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide",
    "sulphur_dioxide", "ozone", "dust",
]
WEATHER_COLUMNS: list[str] = [
    "temperature_2m", "relative_humidity_2m", "surface_pressure",
    "wind_speed_10m", "wind_direction_10m", "precipitation",
]
CITIES: list[str] = ["Peshawar", "Lahore"]
# Forecast horizons in hours. With the complete hourly grid from the
# Gap_Strategy, a shift of h rows within a city equals h hours.
HORIZONS: list[int] = [24, 48, 72]
TARGET_TYPES: list[str] = ["pm2_5", "us_aqi"]
# Everything before this date is training data and everything from it onward is
# test data; a fixed date keeps the split reproducible and strictly time-based.
SPLIT_DATE: str = "2025-01-01"
WINTER_MONTHS: list[int] = [11, 12, 1, 2]
RANDOM_STATE: int = 42
# Below this wind speed (km/h) the air is treated as stagnant, because weak
# wind lets locally emitted particles accumulate instead of dispersing.
STAGNATION_WIND_KMH: float = 5.0
# What-If slider bounds (low, high). They cover the usual range of the two
# cities' weather; observed values outside them are clamped by the dashboard.
SLIDER_RANGES: dict[str, tuple[float, float]] = {
    "wind_speed_10m": (0.0, 30.0),
    "temperature_2m": (5.0, 35.0),
    "relative_humidity_2m": (20.0, 100.0),
}
# (lower bound, label) pairs checked from the highest bound down, so the first
# match wins. Anything below the last bound is "Weak". The bounds are fixed
# heuristics, not physical constants.
INVERSION_LEVELS: list[tuple[float, str]] = [(0.50, "Strong"), (0.25, "Moderate")]
_INVERSION_DEFAULT_LEVEL: str = "Weak"

# EPA PM2.5 breakpoint rows as (C_lo, C_hi, I_lo, I_hi) in ug/m3 and AQI units.
# Both tables are kept because the audit has to find out which one the data
# source follows; the 2024 revision lowered the Good and Unhealthy+ limits.
_PM25_BREAKPOINTS: dict[str, np.ndarray] = {
    "2024": np.array([
        [0.0, 9.0, 0.0, 50.0],
        [9.1, 35.4, 51.0, 100.0],
        [35.5, 55.4, 101.0, 150.0],
        [55.5, 125.4, 151.0, 200.0],
        [125.5, 225.4, 201.0, 300.0],
        [225.5, 325.4, 301.0, 500.0],
    ]),
    "2012": np.array([
        [0.0, 12.0, 0.0, 50.0],
        [12.1, 35.4, 51.0, 100.0],
        [35.5, 55.4, 101.0, 150.0],
        [55.5, 150.4, 151.0, 200.0],
        [150.5, 250.4, 201.0, 300.0],
        [250.5, 500.4, 301.0, 500.0],
    ]),
}
_AQI_CAP: float = 500.0
# Unhealthy_Or_Worse_Event: an hour whose AQI is above 150, that is in the
# Unhealthy, Very Unhealthy, or Hazardous (Emergency) tier.
UNHEALTHY_AQI_THRESHOLD: float = 150.0


def ensure_dir(path: Path) -> Path:
    """Create a directory (and its parents) if it does not exist yet.

    Args:
        path: Directory to create.

    Returns:
        The same path, so the call can be used inline, e.g.
        ``ensure_dir(RESULTS_DIR) / "metrics.csv"``.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def pm25_breakpoints(table: str = "2024") -> np.ndarray:
    """Return a copy of one EPA PM2.5 breakpoint table.

    Each row is ``(C_lo, C_hi, I_lo, I_hi)``: the concentration range in ug/m3
    and the AQI range it maps to. Rows run from Good to Hazardous, one row per
    EPA tier. The charts use this for the PM2.5 tier bands, so there is only
    one copy of the breakpoint numbers in the project.

    Args:
        table: Breakpoint table to return, ``"2024"`` or ``"2012"``.

    Returns:
        A new float array of shape (6, 4). It is a copy, so a caller that edits
        it cannot change the values :func:`pm25_sub_index` uses.

    Raises:
        ValueError: If ``table`` is not ``"2024"`` or ``"2012"``.
    """
    if table not in _PM25_BREAKPOINTS:
        raise ValueError(
            f"Unknown breakpoint table '{table}'. Use one of {sorted(_PM25_BREAKPOINTS)}"
        )
    return _PM25_BREAKPOINTS[table].copy()


def pm25_sub_index(pm25_24h: np.ndarray, table: str) -> np.ndarray:
    """Compute the US EPA PM2.5 AQI sub-index from 24-hour mean concentrations.

    Within a breakpoint row the index is linearly interpolated:
    ``I = (I_hi - I_lo) / (C_hi - C_lo) * (C - C_lo) + I_lo``.
    Concentrations above the top row of the table are capped at AQI 500.
    Negative or NaN concentrations give NaN.

    Args:
        pm25_24h: 24-hour trailing mean PM2.5 concentrations in ug/m3
            (array-like or scalar).
        table: Breakpoint table to use, ``"2024"`` or ``"2012"``.

    Returns:
        Float array of sub-index values with the same shape as ``pm25_24h``.
        The values are not rounded, so callers decide how to compare them.

    Raises:
        ValueError: If ``table`` is not ``"2024"`` or ``"2012"``.
    """
    if table not in _PM25_BREAKPOINTS:
        raise ValueError(
            f"Unknown breakpoint table '{table}'. Use one of {sorted(_PM25_BREAKPOINTS)}"
        )
    bp = _PM25_BREAKPOINTS[table]
    c_lo, c_hi, i_lo, i_hi = bp[:, 0], bp[:, 1], bp[:, 2], bp[:, 3]

    conc = np.asarray(pm25_24h, dtype=float)
    # EPA truncates PM2.5 to one decimal before the lookup. Without it a value
    # such as 9.05 would fall in the 0.1-wide gap between two rows. The small
    # epsilon stops float noise (e.g. 35.4 stored as 35.39999...) from
    # truncating a value into the row below.
    trunc = np.floor(conc * 10.0 + 1e-9) / 10.0

    # side="left" picks the first row whose C_hi is >= the concentration, so a
    # value equal to C_hi stays in that row (9.0 -> 50, not 51).
    idx = np.searchsorted(c_hi, trunc, side="left")
    above_top = idx >= len(c_hi)
    safe_idx = np.minimum(idx, len(c_hi) - 1)

    result = (i_hi[safe_idx] - i_lo[safe_idx]) / (c_hi[safe_idx] - c_lo[safe_idx]) \
        * (trunc - c_lo[safe_idx]) + i_lo[safe_idx]
    result = np.where(above_top, _AQI_CAP, result)
    result = np.where(np.isnan(conc) | (conc < 0), np.nan, result)
    return result


def inversion_proxy(
    temperature_2m: np.ndarray | float,
    relative_humidity_2m: np.ndarray | float,
) -> np.ndarray | float:
    """Compute the thermal inversion proxy from temperature and humidity.

    Exact formula::

        inversion_proxy = relative_humidity_2m / 100 - temperature_2m / 40

    temperature_2m is in degrees C and relative_humidity_2m in %; the result
    has no unit. The proxy rises by 1/40 for every degree C drop in
    temperature and by 1/100 for every % rise in humidity, so it increases as
    the air gets colder and more humid. It is a heuristic for cold, moist
    near-surface air that favours trapped smog, not a measured inversion.

    Examples: (8 C, 90 %) -> 0.70, (20 C, 60 %) -> 0.10, (38 C, 30 %) -> -0.65.

    Args:
        temperature_2m: Air temperature at 2 m in degrees C (scalar or
            array-like).
        relative_humidity_2m: Relative humidity at 2 m in % (scalar or
            array-like, broadcastable against ``temperature_2m``).

    Returns:
        A float for scalar inputs, otherwise a float array. NaN inputs give
        NaN.
    """
    temp = np.asarray(temperature_2m, dtype=float)
    rh = np.asarray(relative_humidity_2m, dtype=float)
    proxy = rh / 100.0 - temp / 40.0
    # Scalars come back as floats so the dashboard card can format them
    # directly without unwrapping a 0-d array.
    return float(proxy) if proxy.ndim == 0 else proxy


def inversion_level(proxy: np.ndarray | float) -> np.ndarray | str:
    """Map inversion proxy values to "Strong", "Moderate", or "Weak".

    "Strong" if proxy >= 0.50, "Moderate" if proxy >= 0.25, otherwise "Weak"
    (thresholds from ``INVERSION_LEVELS``). A value equal to a bound belongs to
    the higher level, so 0.25 is "Moderate" and 0.50 is "Strong".

    Args:
        proxy: Inversion proxy value(s) from :func:`inversion_proxy`.

    Returns:
        A label string for scalar input, otherwise an array of label strings
        with the same shape. NaN compares False with every bound and therefore
        maps to "Weak", so callers should check for NaN first.
    """
    values = np.asarray(proxy, dtype=float)
    # np.select takes the first true condition, and INVERSION_LEVELS is listed
    # from the highest bound down, so each value gets the highest level it
    # reaches.
    conditions = [values >= bound for bound, _ in INVERSION_LEVELS]
    labels = [label for _, label in INVERSION_LEVELS]
    levels = np.select(conditions, labels, default=_INVERSION_DEFAULT_LEVEL)
    return str(levels) if levels.ndim == 0 else levels


def stagnation_flag(wind_speed_10m: np.ndarray | float) -> np.ndarray | int:
    """Flag stagnant air: 1 where wind speed is below 5 km/h, else 0.

    The bound comes from ``STAGNATION_WIND_KMH``. The comparison is strict, so
    exactly 5.0 km/h gives 0.

    Args:
        wind_speed_10m: Wind speed at 10 m in km/h (scalar or array-like).

    Returns:
        An int for scalar input, otherwise an int array with the same shape.
        NaN wind speed gives 0, because NaN is not below the bound; the
        master file has no missing wind values.
    """
    wind = np.asarray(wind_speed_10m, dtype=float)
    flag = np.where(wind < STAGNATION_WIND_KMH, 1, 0)
    return int(flag) if flag.ndim == 0 else flag


def _paired_vectors(
    y_true: np.ndarray | pd.Series | list[float],
    y_pred: np.ndarray | pd.Series | list[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Convert two metric inputs to equal-length, non-empty 1-D float arrays.

    Args:
        y_true: Observed values, 1-D or a single column of shape (n, 1).
        y_pred: Predicted values, 1-D or a single column of shape (n, 1).

    Returns:
        ``(y_true, y_pred)`` as 1-D float arrays of the same length.

    Raises:
        ValueError: If either input has more than one column or is a scalar,
            if the lengths differ, or if the inputs are empty.
    """
    arrays = []
    for name, values in (("y_true", y_true), ("y_pred", y_pred)):
        arr = np.asarray(values, dtype=float)
        # An (n, 1) column next to an (n,) vector would broadcast to an (n, n)
        # matrix and give a wrong metric without any error, so single columns
        # are flattened and anything wider is rejected.
        if arr.ndim == 2 and arr.shape[1] == 1:
            arr = arr[:, 0]
        if arr.ndim != 1:
            raise ValueError(
                f"{name} must be 1-D or a single column, got shape {arr.shape}"
            )
        arrays.append(arr)
    true, pred = arrays
    if true.shape[0] != pred.shape[0]:
        raise ValueError(
            "y_true and y_pred must have the same length, "
            f"got {true.shape[0]} and {pred.shape[0]}"
        )
    if true.shape[0] == 0:
        raise ValueError("y_true and y_pred must not be empty")
    return true, pred


def mae(
    y_true: np.ndarray | pd.Series | list[float],
    y_pred: np.ndarray | pd.Series | list[float],
) -> float:
    """Mean absolute error, ``mean(|y_true - y_pred|)``, computed with NumPy.

    Written in the project rather than taken from scikit-learn because
    Requirement 11.5 asks for MAE implemented in project code.

    Args:
        y_true: Observed values (1-D or a single column).
        y_pred: Predicted values, same length as ``y_true``.

    Returns:
        The MAE as a Python float, in the units of the target. NaN in either
        input gives NaN, so callers drop NaN rows first.

    Raises:
        ValueError: If the inputs differ in length, are empty, or have more
            than one column.
    """
    true, pred = _paired_vectors(y_true, y_pred)
    return float(np.mean(np.abs(true - pred)))


def rmse(
    y_true: np.ndarray | pd.Series | list[float],
    y_pred: np.ndarray | pd.Series | list[float],
) -> float:
    """Root mean squared error, ``sqrt(mean((y_true - y_pred) ** 2))``, in NumPy.

    Written in the project rather than taken from scikit-learn because
    Requirement 11.5 asks for RMSE implemented in project code.

    Args:
        y_true: Observed values (1-D or a single column).
        y_pred: Predicted values, same length as ``y_true``.

    Returns:
        The RMSE as a Python float, in the units of the target. NaN in either
        input gives NaN, so callers drop NaN rows first.

    Raises:
        ValueError: If the inputs differ in length, are empty, or have more
            than one column.
    """
    true, pred = _paired_vectors(y_true, y_pred)
    return float(np.sqrt(np.mean((true - pred) ** 2)))


def r2(
    y_true: np.ndarray | pd.Series | list[float],
    y_pred: np.ndarray | pd.Series | list[float],
) -> float:
    """Coefficient of determination, a thin wrapper around ``sklearn.metrics.r2_score``.

    The inputs go through the same checks as :func:`mae` and :func:`rmse`, so
    all three metrics reject the same bad inputs with the same messages.

    Args:
        y_true: Observed values (1-D or a single column).
        y_pred: Predicted values, same length as ``y_true``.

    Returns:
        R2 as a Python float. 1.0 is a perfect fit and values below 0 mean the
        predictions are worse than always predicting the mean of ``y_true``.
        scikit-learn returns NaN with a warning for a single sample, because
        R2 is undefined there.

    Raises:
        ValueError: If the inputs differ in length, are empty, or have more
            than one column, or (from scikit-learn) if they contain NaN.
    """
    true, pred = _paired_vectors(y_true, y_pred)
    return float(r2_score(true, pred))


def _tier_index_pair(
    y_true_aqi: np.ndarray | pd.Series | list[float],
    y_pred_aqi: np.ndarray | pd.Series | list[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Map actual and predicted AQI to EPA tier indices for scoring.

    Predictions are clipped at 0 first. LinearRegression can predict small
    negative AQI values, and for scoring a negative prediction means "Good";
    without the clip ``aqi_to_tier_index`` would reject the whole test set.
    Actual values are not clipped, because a negative observed AQI is a data
    error that should surface.

    Args:
        y_true_aqi: Observed AQI values (1-D or a single column).
        y_pred_aqi: Predicted AQI values, same length as ``y_true_aqi``.

    Returns:
        ``(true_idx, pred_idx)`` as integer arrays of tier indices, 0 (Good)
        to 5 (Hazardous (Emergency)).

    Raises:
        ValueError: If the inputs differ in length, are empty, or have more
            than one column; if any actual value is negative; or if any
            value is NaN or +inf (messages from ``alerts.aqi_to_tier_index``).
    """
    # Imported here rather than at module level so utils never imports another
    # src module while it is itself being loaded (no circular-import risk).
    from src.alerts import aqi_to_tier_index

    true, pred = _paired_vectors(y_true_aqi, y_pred_aqi)
    # np.clip keeps NaN as NaN, so a NaN prediction still raises instead of
    # being scored as "Good".
    pred = np.clip(pred, 0.0, None)
    return aqi_to_tier_index(true), aqi_to_tier_index(pred)


def tier_accuracy(
    y_true_aqi: np.ndarray | pd.Series | list[float],
    y_pred_aqi: np.ndarray | pd.Series | list[float],
) -> float:
    """Share of rows whose predicted EPA tier equals the actual EPA tier.

    Tiers follow ``alerts.aqi_to_tier_index`` (lower < AQI <= upper, Good from
    0 inclusive). Predictions are clipped at 0 before the mapping.

    Args:
        y_true_aqi: Observed AQI values (1-D or a single column).
        y_pred_aqi: Predicted AQI values, same length as ``y_true_aqi``.

    Returns:
        Tier accuracy as a Python float between 0 and 1.

    Raises:
        ValueError: If the inputs differ in length, are empty, or have more
            than one column, if any actual AQI is negative, or if any value
            is NaN or +inf. Callers drop NaN rows first.
    """
    true_idx, pred_idx = _tier_index_pair(y_true_aqi, y_pred_aqi)
    return float(np.mean(true_idx == pred_idx))


def unhealthy_precision_recall(
    y_true_aqi: np.ndarray | pd.Series | list[float],
    y_pred_aqi: np.ndarray | pd.Series | list[float],
) -> tuple[float, float]:
    """Precision and recall of Unhealthy_Or_Worse_Event detection (AQI > 150).

    An hour is an event when its AQI is above 150, i.e. its tier is Unhealthy,
    Very Unhealthy, or Hazardous (Emergency). Predictions are clipped at 0
    before the mapping, which cannot change whether they exceed 150.

    precision = true positives / predicted events
    recall    = true positives / actual events

    Args:
        y_true_aqi: Observed AQI values (1-D or a single column).
        y_pred_aqi: Predicted AQI values, same length as ``y_true_aqi``.

    Returns:
        ``(precision, recall)`` as Python floats. Precision is NaN when no
        row is predicted as an event and recall is NaN when no row is an
        actual event, because the ratio is undefined there; reporting 0
        would wrongly read as "every alarm was false" or "every event missed".

    Raises:
        ValueError: If the inputs differ in length, are empty, or have more
            than one column, if any actual AQI is negative, or if any value
            is NaN or +inf. Callers drop NaN rows first.
    """
    from src.alerts import aqi_to_tier_index

    true_idx, pred_idx = _tier_index_pair(y_true_aqi, y_pred_aqi)
    # Going through tier indices keeps one boundary rule for every tier metric.
    # 150 is the upper bound of Unhealthy for Sensitive Groups, and a value
    # equal to an upper bound stays in that tier, so a tier index above the
    # index of 150 holds exactly when AQI > 150.
    event_floor = int(aqi_to_tier_index(UNHEALTHY_AQI_THRESHOLD))
    true_event = true_idx > event_floor
    pred_event = pred_idx > event_floor

    true_positives = int(np.sum(true_event & pred_event))
    n_pred_events = int(np.sum(pred_event))
    n_true_events = int(np.sum(true_event))
    precision = true_positives / n_pred_events if n_pred_events else float("nan")
    recall = true_positives / n_true_events if n_true_events else float("nan")
    return float(precision), float(recall)


def persistence_forecast(df: pd.DataFrame, target_type: str) -> np.ndarray:
    """Persistence_Baseline: predict the value at t + h as the value at time t.

    Each row of ``df`` is a feature row at time t for one city. The prediction
    for the label ``{target_type}_t_plus_{h}`` is the column ``target_type`` in
    the same row, so the same prediction serves every horizon h. The baseline
    uses no future data and needs no fitting, which makes it the reference any
    trained model has to beat.

    Args:
        df: Feature rows (one city or several) containing the ``target_type``
            column at time t.
        target_type: ``"pm2_5"`` or ``"us_aqi"``.

    Returns:
        A new float array with one prediction per row of ``df``, in row order.

    Raises:
        ValueError: If ``target_type`` is not in ``TARGET_TYPES`` or the column
            is absent from ``df``.
    """
    if target_type not in TARGET_TYPES:
        raise ValueError(
            f"target_type must be one of {TARGET_TYPES}, got '{target_type}'"
        )
    if target_type not in df.columns:
        raise ValueError(f"Column '{target_type}' not found in the frame")
    # np.array copies, so callers that clip or edit the predictions cannot
    # change the feature frame by accident.
    return np.array(df[target_type], dtype=float)


def model_path(city: str, target: str, horizon: int, mode: str, model_type: str) -> Path:
    """Build the joblib path for one saved model under ``MODELS_DIR``.

    The file name is ``{city.lower()}_{target}_h{horizon}_{mode}_{model_type}.joblib``,
    for example ``peshawar_us_aqi_h24_forecast_random_forest.joblib``. The
    Training_Script and the Dashboard both call this function, so they always
    agree on where a model lives. The directory is not created here; callers
    that write use :func:`ensure_dir` first.

    Args:
        city: City name, e.g. ``"Peshawar"``. Lower-cased in the file name.
        target: Target_Type, ``"pm2_5"`` or ``"us_aqi"``.
        horizon: Forecast horizon in hours (24, 48, or 72).
        mode: ``"forecast"`` or ``"simulation"``.
        model_type: Estimator name, e.g. ``"random_forest"``.

    Returns:
        Absolute path of the model file.
    """
    return MODELS_DIR / f"{city.lower()}_{target}_h{int(horizon)}_{mode}_{model_type}.joblib"


def xgboost_available() -> bool:
    """Report whether the optional xgboost package can be found.

    ``importlib.util.find_spec`` looks the package up without importing it, so
    the check is cheap and never fails when xgboost is absent (Requirement
    1.5). It only shows that the package is installed; a broken install would
    still fail later, at the import inside the model class.

    Returns:
        True if xgboost is installed, otherwise False.
    """
    try:
        return importlib.util.find_spec("xgboost") is not None
    except (ImportError, ValueError):
        # find_spec raises ValueError when sys.modules holds an "xgboost" entry
        # without a spec (for example a test stub), and ImportError for some
        # broken installs. Either way xgboost is not usable.
        return False
