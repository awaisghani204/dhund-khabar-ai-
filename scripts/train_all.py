"""Training script for Dhund-Khabar AI (Phase 3).

Run from the project root with ``python scripts/train_all.py``. The script
rebuilds the processed data from the master CSV with
``AirQualityDataPipeline.build_features``. Then, for every city, Target_Type,
and Horizon, it splits the rows by timestamp with the per-horizon Embargo,
trains the Forecast_Mode models (Linear Regression, Gradient Boosting, Random
Forest, and XGBoost when installed), and scores them and the
Persistence_Baseline on the whole Test_Period ("all") and on its Winter_Months
rows ("winter"). Every fitted model is saved under ``models/`` and every score
goes to ``results/metrics.csv``. Each Forecast_Mode Random Forest is compared
with persistence on MAE, and the verdict is printed and written to the CSV.

After the forecast loop it trains one Simulation_Mode Random Forest per city
(us_aqi, Horizon 24) for the dashboard's What-If lab, scores it the same way,
compares it with persistence on the same test rows, and adds its rows to
``results/metrics.csv``. It then measures how the
Simulation_Mode model and the Forecast_Mode us_aqi h = 24 Random Forest react
to stronger wind and to more humid air on cold Test_Period hours, prints the
numbers without adjustment, and writes them to ``results/sensitivity.csv``.

Finally it draws the report figures (Phase 4, Requirement 14.7) from the
saved models into ``results/figures/``: actual vs predicted and a scatter per
city and horizon for the us_aqi Random Forest, one PM2.5 h = 24 actual vs
predicted chart per city, a correlation heatmap and the feature importance of
the us_aqi h = 24 forecast and simulation models per city, and the city
comparison.

Command line::

    python scripts/train_all.py                 # full training run, then figures
    python scripts/train_all.py --figures-only  # figures only, nothing retrained

``--figures-only`` reads ``data/processed/model_dataset.csv`` and
``features_all.csv`` as they are and loads the saved models. It writes only
PNG files, so models, ``metrics.csv``, ``sensitivity.csv``, and the processed
data stay unchanged.

The script is plain functions. ``run_forecast_loop`` returns the metric rows
instead of writing them, so later steps can append their own rows (the
Simulation_Mode models) before ``write_metrics`` saves one CSV.
"""

import argparse
import math
import sys
import time
from pathlib import Path

# The script lives in scripts/, so the project root must be on sys.path before
# `src` can be imported when it is started as `python scripts/train_all.py`.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from matplotlib.figure import Figure  # noqa: E402

from src import charts, utils  # noqa: E402
from src.model import SmogPredictorModel  # noqa: E402
from src.pipeline import (  # noqa: E402
    DATETIME_FORMAT,
    FEATURES_FILENAME,
    MODEL_DATASET_FILENAME,
    AirQualityDataPipeline,
)
from src.utils import (  # noqa: E402
    CITIES,
    FIGURES_DIR,
    HORIZONS,
    MASTER_PATH,
    MODELS_DIR,
    PROCESSED_DIR,
    PROJECT_ROOT,
    RESULTS_DIR,
    SPLIT_DATE,
    TARGET_TYPES,
    WINTER_MONTHS,
)

# One row of results/metrics.csv, keyed by the names in METRICS_COLUMNS.
MetricRow = dict[str, object]

METRICS_FILENAME: str = "metrics.csv"
# Column order of results/metrics.csv, exactly as in the design schema.
METRICS_COLUMNS: list[str] = [
    "city", "target", "horizon", "mode", "model", "subset", "n_train", "n_test",
    "MAE", "RMSE", "R2", "tier_accuracy", "unhealthy_recall", "unhealthy_precision",
    "beats_persistence_mae", "note",
]
FORECAST_MODE: str = "forecast"
PERSISTENCE: str = "persistence"
# The project's main model; every other model is a comparison.
REFERENCE_MODEL: str = "random_forest"
# Fitted in this order, cheapest first, so a broken setup fails within seconds
# instead of after the slow Gradient Boosting fit.
BASE_MODEL_TYPES: list[str] = ["linear_regression", "gradient_boosting", "random_forest"]
SUBSETS: tuple[str, ...] = ("all", "winter")
# EPA tiers and Unhealthy_Or_Worse_Events are defined on the AQI scale only, so
# tier metrics are computed for this target and left empty for pm2_5.
TIER_TARGET: str = "us_aqi"
NAN_TIER_SCORES: dict[str, float] = {
    "tier_accuracy": float("nan"),
    "unhealthy_recall": float("nan"),
    "unhealthy_precision": float("nan"),
}

SIMULATION_MODE: str = "simulation"
# Requirement 13.2 asks for one Simulation_Mode model per city at this target
# and horizon only; the What-If lab shows next-day AQI.
SIMULATION_TARGET: str = "us_aqi"
SIMULATION_HORIZON: int = 24

# One row of results/sensitivity.csv, keyed by the names in SENSITIVITY_COLUMNS.
SensitivityRow = dict[str, object]

SENSITIVITY_FILENAME: str = "sensitivity.csv"
# Column order of results/sensitivity.csv, exactly as in the design.
SENSITIVITY_COLUMNS: list[str] = [
    "city", "mode", "perturbation", "delta", "n_rows", "mean_delta_aqi",
]
# Requirement 13.9 asks about humidity "under cold temperature_2m values"; the
# design fixes "cold" as at or below this temperature in degrees C.
COLD_TEMPERATURE_C: float = 15.0
# (column, amount added) for each perturbation, in report order.
PERTURBATIONS: list[tuple[str, float]] = [
    ("wind_speed_10m", 5.0),
    ("relative_humidity_2m", 20.0),
]
# Humidity above 100 % cannot occur, so a perturbed value is capped there. On
# very humid hours the applied change is therefore smaller than the nominal
# delta, and the script prints the mean change it really applied.
PERTURBATION_UPPER_LIMITS: dict[str, float] = {"relative_humidity_2m": 100.0}
# Sign of the change the physics suggests: stronger wind disperses smog, and
# more humid cold air favours trapped smog. Used only to phrase the printed
# verdict; nothing is tuned or filtered towards it.
EXPECTED_SIGN: dict[str, int] = {"wind_speed_10m": -1, "relative_humidity_2m": 1}

# Figures (Phase 4). The us_aqi Random Forests get the full chart set because
# the alerts and the dashboard work on the AQI scale.
FIGURE_TARGET: str = "us_aqi"
# One PM2.5 chart per city shows the concentration tier bands; the longer
# horizons would repeat the same picture, so they are left to the dashboard.
PM25_FIGURE_TARGET: str = "pm2_5"
PM25_FIGURE_HORIZON: int = 24
# The forecast model's importance chart uses the simulation model's horizon,
# so the two charts of a city compare models that answer the same question.
IMPORTANCE_HORIZON: int = SIMULATION_HORIZON
FEATURE_IMPORTANCE_TOP_N: int = 10
CITY_COMPARISON_FIGURE: str = "city_comparison_monthly_aqi.png"
# The heatmap and the city comparison need only these columns of
# features_all.csv; reading just them keeps --figures-only quick.
FIGURE_FEATURE_COLUMNS: list[str] = (["city", "datetime"]
                                     + charts.CORRELATION_WEATHER_COLUMNS
                                     + charts.CORRELATION_POLLUTANT_COLUMNS)
MODE_LABELS: dict[str, str] = {FORECAST_MODE: "forecast",
                               SIMULATION_MODE: "simulation (What-If)"}
PIPELINE_COMMAND: str = "python -m src.pipeline"
TRAINING_COMMAND: str = "python scripts/train_all.py"


def forecast_model_types() -> list[str]:
    """List the trained Forecast_Mode model types, adding xgboost only when installed.

    Returns:
        ``["linear_regression", "gradient_boosting", "random_forest"]``, plus
        ``"xgboost"`` at the end when ``utils.xgboost_available()`` is True
        (Requirements 1.4, 1.5). Persistence is not listed, because it is a
        function and is always scored separately.
    """
    model_types = list(BASE_MODEL_TYPES)
    if utils.xgboost_available():
        model_types.append("xgboost")
    return model_types


def target_column(target: str, horizon: int) -> str:
    """Return the label column name for a Target_Type and Horizon.

    Args:
        target: ``"pm2_5"`` or ``"us_aqi"``.
        horizon: Horizon in hours.

    Returns:
        The name created by ``AirQualityDataPipeline.create_targets``, e.g.
        ``"us_aqi_t_plus_24"``.
    """
    return f"{target}_t_plus_{int(horizon)}"


def model_file(models_dir: Path, city: str, target: str, horizon: int, mode: str,
               model_type: str) -> Path:
    """Return the joblib path for one model inside ``models_dir``.

    Args:
        models_dir: Folder for the saved models; ``MODELS_DIR`` in real runs.
        city: City name, e.g. ``"Lahore"``.
        target: ``"pm2_5"`` or ``"us_aqi"``.
        horizon: Horizon in hours.
        mode: ``"forecast"`` or ``"simulation"``.
        model_type: Estimator name, e.g. ``"random_forest"``.

    Returns:
        ``models_dir / <file name from utils.model_path>``. With the default
        folder this is exactly ``utils.model_path(...)``.
    """
    # utils.model_path owns the file-name rule that the dashboard also uses.
    # Only the folder may change, so a dry run can write to a temporary folder
    # without leaving files that look like final outputs in models/.
    return Path(models_dir) / utils.model_path(city, target, horizon, mode, model_type).name


def drop_incomplete_rows(frame: pd.DataFrame,
                         columns: list[str]) -> tuple[pd.DataFrame, int]:
    """Drop rows with NaN in any of ``columns``.

    The split already removed warm-up rows, but an interior pollutant or
    us_aqi gap longer than 3 hours stays NaN under the approved Gap_Strategy,
    and scikit-learn estimators cannot fit or predict on NaN.

    Args:
        frame: Training or test rows.
        columns: Feature columns plus the label column of one model.

    Returns:
        ``(complete_rows, n_dropped)``. The kept rows get a fresh 0..n-1 index.
    """
    complete = frame[columns].notna().all(axis=1).to_numpy()
    return frame.loc[complete].reset_index(drop=True), int((~complete).sum())


def subset_masks(test: pd.DataFrame) -> dict[str, np.ndarray]:
    """Build the row masks for the "all" and "winter" evaluation subsets.

    "winter" uses the month of the feature row's ``datetime`` (the time t at
    which the forecast is issued), which is how Test_Period rows are defined.

    Args:
        test: Test rows with a parsed ``datetime`` column.

    Returns:
        ``{"all": all-True mask, "winter": mask of months 11, 12, 1, 2}`` as
        boolean NumPy arrays, in the order of ``SUBSETS``.
    """
    months = test["datetime"].dt.month.to_numpy()
    return {
        "all": np.ones(len(test), dtype=bool),
        "winter": np.isin(months, WINTER_MONTHS),
    }


def score_predictions(y_true: np.ndarray, y_pred: np.ndarray,
                      target: str) -> dict[str, float]:
    """Score one set of predictions with the metrics of the metrics.csv schema.

    MAE, RMSE, and R2 come from ``utils.mae``, ``utils.rmse``, and
    ``utils.r2``, the same functions ``SmogPredictorModel.evaluate`` uses. For
    us_aqi, tier accuracy and Unhealthy-or-worse precision and recall come from
    ``utils.tier_accuracy`` and ``utils.unhealthy_precision_recall``, which
    clip predictions at 0 first. For pm2_5 the three tier scores are NaN.

    Args:
        y_true: Observed label values.
        y_pred: Predictions, same length as ``y_true``.
        target: ``"pm2_5"`` or ``"us_aqi"``.

    Returns:
        Dict with MAE, RMSE, R2, tier_accuracy, unhealthy_recall, and
        unhealthy_precision. Every value is NaN when the subset is empty,
        because no metric is defined on zero rows.
    """
    if len(y_true) == 0:
        return {"MAE": float("nan"), "RMSE": float("nan"), "R2": float("nan"),
                **NAN_TIER_SCORES}
    scores = {
        "MAE": utils.mae(y_true, y_pred),
        "RMSE": utils.rmse(y_true, y_pred),
        "R2": utils.r2(y_true, y_pred),
    }
    if target != TIER_TARGET:
        return {**scores, **NAN_TIER_SCORES}
    precision, recall = utils.unhealthy_precision_recall(y_true, y_pred)
    return {
        **scores,
        "tier_accuracy": utils.tier_accuracy(y_true, y_pred),
        "unhealthy_recall": recall,
        "unhealthy_precision": precision,
    }


def metric_rows(city: str, target: str, horizon: int, mode: str, model_name: str,
                n_train: int, y_true: np.ndarray, y_pred: np.ndarray,
                masks: dict[str, np.ndarray]) -> list[MetricRow]:
    """Build one metrics.csv row per evaluation subset for one model.

    The model is trained once on all training rows; only the test rows differ
    between subsets.

    Args:
        city: City name.
        target: ``"pm2_5"`` or ``"us_aqi"``.
        horizon: Horizon in hours.
        mode: ``"forecast"`` or ``"simulation"``.
        model_name: ``"persistence"`` or a model type.
        n_train: Training rows the model was fitted on (0 for persistence).
        y_true: Observed labels for all test rows.
        y_pred: Predictions for all test rows, same order as ``y_true``.
        masks: Subset masks from :func:`subset_masks`.

    Returns:
        One row per subset in ``SUBSETS`` order, with empty
        ``beats_persistence_mae`` and ``note``.
    """
    rows: list[MetricRow] = []
    for subset in SUBSETS:
        mask = masks[subset]
        rows.append({
            "city": city,
            "target": target,
            "horizon": int(horizon),
            "mode": mode,
            "model": model_name,
            "subset": subset,
            "n_train": int(n_train),
            "n_test": int(mask.sum()),
            **score_predictions(y_true[mask], y_pred[mask], target),
            # None is written as an empty cell; only Random Forest rows get a
            # verdict.
            "beats_persistence_mae": None,
            "note": "",
        })
    return rows


def persistence_verdict(rf_mae: float, persistence_mae: float,
                        model_label: str = "Random Forest",
                        short_label: str = "RF") -> tuple[bool | None, str]:
    """Compare a Random Forest's MAE with persistence MAE and phrase the result plainly.

    Args:
        rf_mae: Random Forest MAE on one subset.
        persistence_mae: Persistence MAE on the same test rows.
        model_label: Model name at the start of the note, e.g.
            ``"Simulation Random Forest"``.
        short_label: Model name inside the numbers, e.g. ``"sim"``.

    Returns:
        ``(beats, note)``. ``beats`` is True only when the Random Forest MAE is
        strictly lower; a tie counts as not beating persistence, because
        persistence needs no model at all. ``beats`` is None and the note says
        so when either MAE is NaN (an empty subset).
    """
    if math.isnan(rf_mae) or math.isnan(persistence_mae):
        return None, f"{model_label} not compared with persistence: no test rows in this subset"
    numbers = f"({short_label} {rf_mae:.2f} vs persistence {persistence_mae:.2f})"
    if rf_mae < persistence_mae:
        return True, f"{model_label} beat persistence on MAE {numbers}"
    return False, f"{model_label} did not beat persistence on MAE {numbers}"


def _find_row(rows: list[MetricRow], model_name: str, subset: str) -> MetricRow:
    """Return the row of one model and subset from one combination's rows.

    Args:
        rows: Metric rows of a single city, target, and horizon.
        model_name: Value of the ``model`` column.
        subset: ``"all"`` or ``"winter"``.

    Returns:
        The matching row (the same dict object, so callers can fill it in).

    Raises:
        KeyError: If no row matches.
    """
    for row in rows:
        if row["model"] == model_name and row["subset"] == subset:
            return row
    raise KeyError(f"No metric row for model '{model_name}' and subset '{subset}'")


def _display_path(path: Path) -> str:
    """Show a path relative to the project root when it lies inside it.

    Args:
        path: Any path.

    Returns:
        The relative path as text, or the full path for files elsewhere (for
        example a temporary folder in a dry run).
    """
    try:
        return str(Path(path).resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _format_scores(rows: list[MetricRow]) -> str:
    """Format the MAE per subset of one model for the progress line.

    Args:
        rows: The subset rows of one model.

    Returns:
        Text such as ``"MAE all 21.40 | winter 35.02"``.
    """
    return "MAE " + " | ".join(f"{row['subset']} {row['MAE']:.2f}" for row in rows)


def run_combination(pipeline: AirQualityDataPipeline, city_df: pd.DataFrame, city: str,
                    target: str, horizon: int, model_types: list[str],
                    models_dir: Path) -> list[MetricRow]:
    """Train, score, and save every Forecast_Mode model for one city, target, and horizon.

    Steps: embargo-aware time split for this horizon; forecast feature columns
    for this target; drop rows with NaN features or label (counts printed);
    score persistence; fit, score, and save each model type; compare Random
    Forest MAE with persistence MAE on each subset and print the verdict.

    Args:
        pipeline: Pipeline that provides the split and the feature columns.
        city_df: Rows of one city from ``build_features`` (features + targets).
        city: City name.
        target: ``"pm2_5"`` or ``"us_aqi"``.
        horizon: Horizon in hours.
        model_types: Model types to fit, e.g. from :func:`forecast_model_types`.
        models_dir: Folder for the joblib files.

    Returns:
        Metric rows: persistence first, then each model type in order, each
        with one row per subset. Random Forest rows carry
        ``beats_persistence_mae`` and ``note``.

    Raises:
        ValueError: If no training or test rows remain after the NaN drop.
    """
    print(f"\n=== {city} | {target} | h={horizon} ===", flush=True)
    label_col = target_column(target, horizon)
    feature_cols = pipeline.get_feature_columns(target, FORECAST_MODE)

    train, test = pipeline.prepare_train_test_split(city_df, SPLIT_DATE, horizon=horizon)
    needed = feature_cols + [label_col]
    train, train_dropped = drop_incomplete_rows(train, needed)
    test, test_dropped = drop_incomplete_rows(test, needed)
    print(f"  Rows dropped for NaN features or label: train {train_dropped}, test {test_dropped}")
    if train.empty or test.empty:
        raise ValueError(f"{city} {target} h={horizon}: no rows left after the NaN drop "
                         f"(train {len(train)}, test {len(test)})")

    masks = subset_masks(test)
    last_feature = train["datetime"].max()
    print(f"  Rows used: train {len(train)}, test {len(test)} "
          f"(winter {int(masks['winter'].sum())}), {len(feature_cols)} features")
    print(f"  Last training feature row {last_feature:%Y-%m-%d %H:%M}, last training label "
          f"{last_feature + pd.Timedelta(horizon, unit='h'):%Y-%m-%d %H:%M}, first test row "
          f"{test['datetime'].min():%Y-%m-%d %H:%M}")

    X_train = train[feature_cols]
    y_train = train[label_col].to_numpy(dtype=float)
    X_test = test[feature_cols]
    y_test = test[label_col].to_numpy(dtype=float)

    # Persistence is scored on exactly the test rows the models see, so every
    # MAE in this combination is measured on the same hours.
    persistence_rows = metric_rows(city, target, horizon, FORECAST_MODE, PERSISTENCE,
                                   0, y_test, utils.persistence_forecast(test, target), masks)
    rows: list[MetricRow] = list(persistence_rows)
    print(f"  {PERSISTENCE:<18} no fit       | {_format_scores(persistence_rows)}", flush=True)

    for model_type in model_types:
        print(f"  {model_type:<18} fitting ...", end="", flush=True)
        model = SmogPredictorModel(model_type=model_type, city=city, horizon=horizon,
                                   target=target, mode=FORECAST_MODE)
        start = time.perf_counter()
        model.train(X_train, y_train)
        fit_seconds = time.perf_counter() - start
        model_rows = metric_rows(city, target, horizon, FORECAST_MODE, model_type,
                                 len(train), y_test, model.predict(X_test), masks)
        path = model_file(models_dir, city, target, horizon, FORECAST_MODE, model_type)
        model.save_model(path)
        rows.extend(model_rows)
        print(f" {fit_seconds:6.1f} s | {_format_scores(model_rows)} | "
              f"saved {_display_path(path)}", flush=True)

    if REFERENCE_MODEL in model_types:
        for subset in SUBSETS:
            rf_row = _find_row(rows, REFERENCE_MODEL, subset)
            persistence_row = _find_row(rows, PERSISTENCE, subset)
            beats, note = persistence_verdict(float(rf_row["MAE"]),
                                              float(persistence_row["MAE"]))
            rf_row["beats_persistence_mae"] = beats
            rf_row["note"] = note
            print(f"  VERDICT {city} {target} h={horizon} [{subset}]: {note}")
    return rows


def run_forecast_loop(pipeline: AirQualityDataPipeline, model_df: pd.DataFrame,
                      models_dir: Path = MODELS_DIR, cities: list[str] | None = None,
                      targets: list[str] | None = None, horizons: list[int] | None = None,
                      model_types: list[str] | None = None) -> list[MetricRow]:
    """Run :func:`run_combination` for every city x target x horizon.

    The defaults cover the full Requirement 13 run: Peshawar and Lahore,
    pm2_5 and us_aqi, 24, 48, and 72 hours, which is 12 Random Forest models
    plus their baselines. The optional arguments narrow the run, for example
    for a quick check on one combination.

    Args:
        pipeline: Pipeline that provides the split and the feature columns.
        model_df: Frame from ``build_features`` with features and targets.
        models_dir: Folder for the joblib files (default ``MODELS_DIR``).
        cities: Cities to run, default ``CITIES``.
        targets: Target_Types to run, default ``TARGET_TYPES``.
        horizons: Horizons to run, default ``HORIZONS``.
        model_types: Model types to fit, default :func:`forecast_model_types`.

    Returns:
        All metric rows in loop order (city, then target, then horizon). The
        rows are not written here, so a caller can add more rows before
        :func:`write_metrics`.

    Raises:
        ValueError: If a requested city has no rows in ``model_df``.
    """
    cities = list(CITIES) if cities is None else list(cities)
    targets = list(TARGET_TYPES) if targets is None else list(targets)
    horizons = list(HORIZONS) if horizons is None else list(horizons)
    model_types = forecast_model_types() if model_types is None else list(model_types)

    present = set(model_df["city"].unique())
    missing = [city for city in cities if city not in present]
    if missing:
        raise ValueError(f"No rows for cities {missing}; cities in the data: {sorted(present)}")

    if not utils.xgboost_available():
        print("xgboost is not installed, so the XGBoost comparison model is skipped.")
    print(f"Forecast models per combination: {PERSISTENCE} (baseline), {', '.join(model_types)}")
    print(f"Combinations: {len(cities)} cities x {len(targets)} targets x "
          f"{len(horizons)} horizons; test subsets {', '.join(SUBSETS)} "
          f"(winter months {WINTER_MONTHS})")

    rows: list[MetricRow] = []
    for city in cities:
        city_df = model_df[model_df["city"] == city]
        for target in targets:
            for horizon in horizons:
                rows.extend(run_combination(pipeline, city_df, city, target, horizon,
                                            model_types, models_dir))
    return rows


def print_verdict_summary(rows: list[MetricRow]) -> None:
    """Print every Random Forest vs persistence verdict in one block.

    Forecast_Mode and Simulation_Mode verdicts are counted separately, because
    the two models answer different questions and should not share one total.

    Args:
        rows: Metric rows; only rows with a filled ``note`` are printed.

    Returns:
        None.
    """
    verdict_rows = [row for row in rows if row["model"] == REFERENCE_MODEL and row["note"]]
    print("\n=== Random Forest vs persistence (MAE) ===")
    for row in verdict_rows:
        print(f"  {row['city']:<9} {row['target']:<7} h={row['horizon']:<3} "
              f"{row['mode']:<10} [{row['subset']}]: {row['note']}")
    for mode, label in ((FORECAST_MODE, "Forecast"), (SIMULATION_MODE, "Simulation")):
        mode_rows = [row for row in verdict_rows if row["mode"] == mode]
        if not mode_rows:
            continue
        not_beaten = sum(1 for row in mode_rows if row["beats_persistence_mae"] is False)
        print(f"  {label} Random Forest did not beat persistence in {not_beaten} of "
              f"{len(mode_rows)} comparisons.")


def write_metrics(rows: list[MetricRow], path: Path) -> Path:
    """Write metric rows to a CSV with the design's metrics.csv columns.

    Args:
        rows: Rows from :func:`run_forecast_loop` and any later steps.
        path: Destination file, ``results/metrics.csv`` in real runs.

    Returns:
        The written path.

    Raises:
        ValueError: If a row is missing a schema column or has an extra key;
            ``pd.DataFrame`` would otherwise fill or drop it silently.
    """
    expected = set(METRICS_COLUMNS)
    for i, row in enumerate(rows):
        if set(row) != expected:
            raise ValueError(f"Metric row {i} does not match the schema. "
                             f"Missing: {sorted(expected - set(row))}. "
                             f"Unexpected: {sorted(set(row) - expected)}.")
    path = Path(path)
    utils.ensure_dir(path.parent)
    # na_rep="" writes undefined values (pm2_5 tier scores, NaN precision or
    # recall, no verdict) as empty cells, as the design schema asks.
    pd.DataFrame(rows, columns=METRICS_COLUMNS).to_csv(path, index=False, na_rep="")
    return path


def _complete_split(pipeline: AirQualityDataPipeline, city_df: pd.DataFrame, horizon: int,
                    columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, int, int]:
    """Split one city's rows with the Embargo and keep rows complete in ``columns``.

    Args:
        pipeline: Pipeline that provides the split.
        city_df: Rows of one city from ``build_features``.
        horizon: Horizon in hours, which sets the Embargo.
        columns: Columns that must be non-NaN (features, and the label when
            the rows are used for fitting or scoring).

    Returns:
        ``(train, test, train_dropped, test_dropped)``.
    """
    train, test = pipeline.prepare_train_test_split(city_df, SPLIT_DATE, horizon=horizon)
    train, train_dropped = drop_incomplete_rows(train, columns)
    test, test_dropped = drop_incomplete_rows(test, columns)
    return train, test, train_dropped, test_dropped


def run_simulation_model(pipeline: AirQualityDataPipeline, city_df: pd.DataFrame, city: str,
                         models_dir: Path) -> list[MetricRow]:
    """Train, score, and save the Simulation_Mode Random Forest of one city.

    The model predicts us_aqi 24 hours ahead from the Simulation_Mode
    features (weather and time at t plus pm2_5_rolling_mean_24h). It uses the
    same Embargo split, NaN drop, and "all" / "winter" scoring as the
    Forecast_Mode models (Requirements 13.2, 13.4).

    Persistence (us_aqi at t) is scored on exactly the test rows the model
    is evaluated on, and the MAE verdict fills ``beats_persistence_mae`` and
    ``note`` on both subset rows. No persistence rows are added for
    Simulation_Mode: they would only repeat the forecast us_aqi h = 24
    persistence rows' role, and the metrics.csv layout stays unchanged.

    Args:
        pipeline: Pipeline that provides the split and the feature columns.
        city_df: Rows of one city from ``build_features``.
        city: City name.
        models_dir: Folder for the joblib file.

    Returns:
        One metric row per subset with ``mode`` "simulation" and ``model``
        "random_forest", each with its persistence verdict.

    Raises:
        ValueError: If no training or test rows remain after the NaN drop, or
            if us_aqi at t is NaN on a test row, because persistence could
            then not be scored on the same rows as the model.
    """
    print(f"\n=== {city} | {SIMULATION_TARGET} | h={SIMULATION_HORIZON} | "
          f"{SIMULATION_MODE} ===", flush=True)
    label_col = target_column(SIMULATION_TARGET, SIMULATION_HORIZON)
    feature_cols = pipeline.get_feature_columns(SIMULATION_TARGET, SIMULATION_MODE)
    train, test, train_dropped, test_dropped = _complete_split(
        pipeline, city_df, SIMULATION_HORIZON, feature_cols + [label_col])
    print(f"  Rows dropped for NaN features or label: train {train_dropped}, test {test_dropped}")
    if train.empty or test.empty:
        raise ValueError(f"{city} {SIMULATION_MODE}: no rows left after the NaN drop "
                         f"(train {len(train)}, test {len(test)})")

    masks = subset_masks(test)
    print(f"  Rows used: train {len(train)}, test {len(test)} "
          f"(winter {int(masks['winter'].sum())}), {len(feature_cols)} features: "
          f"{', '.join(feature_cols)}")

    model = SmogPredictorModel(model_type=REFERENCE_MODEL, city=city,
                               horizon=SIMULATION_HORIZON, target=SIMULATION_TARGET,
                               mode=SIMULATION_MODE)
    print(f"  {REFERENCE_MODEL:<18} fitting ...", end="", flush=True)
    start = time.perf_counter()
    model.train(train[feature_cols], train[label_col].to_numpy(dtype=float))
    fit_seconds = time.perf_counter() - start
    y_test = test[label_col].to_numpy(dtype=float)
    rows = metric_rows(city, SIMULATION_TARGET, SIMULATION_HORIZON, SIMULATION_MODE,
                       REFERENCE_MODEL, len(train), y_test,
                       model.predict(test[feature_cols]), masks)
    path = model_file(models_dir, city, SIMULATION_TARGET, SIMULATION_HORIZON,
                      SIMULATION_MODE, REFERENCE_MODEL)
    model.save_model(path)
    print(f" {fit_seconds:6.1f} s | {_format_scores(rows)} | saved {_display_path(path)}",
          flush=True)

    # us_aqi is not a Simulation_Mode feature, so the NaN drop above did not
    # check it. Dropping rows here would score persistence on other hours than
    # the model, so a NaN stops the run instead.
    persistence_pred = utils.persistence_forecast(test, SIMULATION_TARGET)
    n_missing = int(np.isnan(persistence_pred).sum())
    if n_missing:
        raise ValueError(f"{city} {SIMULATION_MODE}: {SIMULATION_TARGET} is NaN on {n_missing} "
                         "test rows, so persistence cannot be scored on the model's rows")
    for row in rows:
        mask = masks[str(row["subset"])]
        persistence_mae = score_predictions(y_test[mask], persistence_pred[mask],
                                            SIMULATION_TARGET)["MAE"]
        beats, note = persistence_verdict(float(row["MAE"]), persistence_mae,
                                          model_label="Simulation Random Forest",
                                          short_label="sim")
        row["beats_persistence_mae"] = beats
        row["note"] = note
        print(f"  VERDICT {city} {SIMULATION_TARGET} h={SIMULATION_HORIZON} "
              f"{SIMULATION_MODE} [{row['subset']}]: {note}")
    return rows


def perturb_weather(X: pd.DataFrame, column: str, delta: float) -> pd.DataFrame:
    """Return a copy of ``X`` with ``delta`` added to one weather column.

    stagnation_flag and inversion_proxy are recomputed with
    ``utils.stagnation_flag`` and ``utils.inversion_proxy`` when they are
    present, because they are functions of the current hour's wind,
    temperature, and humidity and must follow the perturbed value. Rolling
    weather means are left unchanged: they summarise the past 6 hours, which
    the perturbation does not alter. Values are capped by
    ``PERTURBATION_UPPER_LIMITS`` (relative_humidity_2m at 100).

    Args:
        X: Feature rows of one model.
        column: Weather column to change, e.g. ``"wind_speed_10m"``.
        delta: Amount added to every row.

    Returns:
        A new DataFrame with the same columns and index as ``X``.

    Raises:
        ValueError: If ``column`` is not in ``X``.
    """
    if column not in X.columns:
        raise ValueError(f"Column '{column}' is not among the model features")
    out = X.copy()
    values = out[column].to_numpy(dtype=float) + float(delta)
    upper = PERTURBATION_UPPER_LIMITS.get(column)
    if upper is not None:
        values = np.minimum(values, upper)
    out[column] = values
    if "stagnation_flag" in out.columns:
        out["stagnation_flag"] = utils.stagnation_flag(out["wind_speed_10m"].to_numpy(dtype=float))
    if "inversion_proxy" in out.columns:
        out["inversion_proxy"] = utils.inversion_proxy(
            out["temperature_2m"].to_numpy(dtype=float),
            out["relative_humidity_2m"].to_numpy(dtype=float))
    return out


def sensitivity_verdict(perturbation: str, delta: float, mean_delta_aqi: float) -> str:
    """Say plainly whether a perturbation moved the mean predicted AQI the expected way.

    Args:
        perturbation: Perturbed column, a key of ``EXPECTED_SIGN``.
        delta: Nominal amount added.
        mean_delta_aqi: Mean change in predicted AQI, NaN when no rows exist.

    Returns:
        One sentence, e.g. "wind_speed_10m +5 lowered predicted AQI on
        average (expected: lower) - matches the expected direction".
    """
    label = f"{perturbation} +{delta:g}"
    if math.isnan(mean_delta_aqi):
        return f"{label}: not measured, no cold test rows"
    if mean_delta_aqi < 0:
        moved, sign = "lowered", -1
    elif mean_delta_aqi > 0:
        moved, sign = "raised", 1
    else:
        moved, sign = "did not change", 0
    expected_sign = EXPECTED_SIGN[perturbation]
    expected = "lower" if expected_sign < 0 else "raise"
    match = ("matches" if sign == expected_sign else "does NOT match")
    return (f"{label} {moved} predicted AQI on average (expected: {expected}) - "
            f"{match} the expected direction")


def _load_saved_model(models_dir: Path, city: str, mode: str) -> SmogPredictorModel:
    """Load the saved us_aqi h = 24 Random Forest of one city and mode.

    Args:
        models_dir: Folder with the joblib files.
        city: City name.
        mode: ``"forecast"`` or ``"simulation"``.

    Returns:
        The loaded model.

    Raises:
        FileNotFoundError: If the model has not been trained into ``models_dir``.
    """
    path = model_file(models_dir, city, SIMULATION_TARGET, SIMULATION_HORIZON, mode,
                      REFERENCE_MODEL)
    if not path.is_file():
        raise FileNotFoundError(f"Sensitivity analysis needs {path}; train it first")
    return SmogPredictorModel().load_model(path)


def run_sensitivity(pipeline: AirQualityDataPipeline, city_df: pd.DataFrame, city: str,
                    models_dir: Path) -> list[SensitivityRow]:
    """Measure how both us_aqi h = 24 Random Forests react to wind and humidity on cold hours.

    Rows: Test_Period rows of the h = 24 split with temperature_2m at or
    below ``COLD_TEMPERATURE_C``, complete in the features of both models.
    Using the same rows for both models keeps their numbers comparable. For
    each model and each entry of ``PERTURBATIONS`` the result is
    ``mean(predict(perturbed X) - predict(X))`` (Requirements 13.9, 13.10).
    Both models are loaded from their saved files, so the measured models are
    exactly the ones the dashboard loads.

    Args:
        pipeline: Pipeline that provides the split.
        city_df: Rows of one city from ``build_features``.
        city: City name.
        models_dir: Folder with the saved simulation and forecast models.

    Returns:
        One row per mode (simulation, then forecast) and perturbation, with
        the measured mean change as computed, without adjustment.
    """
    print(f"\n=== Sensitivity | {city} | {SIMULATION_TARGET} h={SIMULATION_HORIZON} | "
          f"cold test rows (temperature_2m <= {COLD_TEMPERATURE_C:g} C) ===", flush=True)
    models = {mode: _load_saved_model(models_dir, city, mode)
              for mode in (SIMULATION_MODE, FORECAST_MODE)}
    union_cols = list(dict.fromkeys(col for model in models.values()
                                    for col in model.feature_columns))
    _, test, _, test_dropped = _complete_split(pipeline, city_df, SIMULATION_HORIZON,
                                               union_cols)
    cold = test.loc[test["temperature_2m"].to_numpy(dtype=float) <= COLD_TEMPERATURE_C]
    n_rows = len(cold)
    if n_rows:
        print(f"  {n_rows} cold rows of {len(test)} complete test rows "
              f"({test_dropped} test rows dropped for NaN features), "
              f"{cold['datetime'].min():%Y-%m-%d %H:%M} to {cold['datetime'].max():%Y-%m-%d %H:%M}")
    else:
        print(f"  No cold rows among {len(test)} complete test rows")

    rows: list[SensitivityRow] = []
    for mode, model in models.items():
        X = cold[model.feature_columns]
        # sklearn cannot predict on zero rows, so an empty set is reported as NaN.
        base = model.predict(X) if n_rows else np.empty(0)
        for column, delta in PERTURBATIONS:
            if n_rows:
                X_new = perturb_weather(X, column, delta)
                mean_delta = float(np.mean(model.predict(X_new) - base))
                applied = float(np.mean(X_new[column].to_numpy() - X[column].to_numpy()))
            else:
                mean_delta = applied = float("nan")
            rows.append({
                "city": city,
                "mode": mode,
                "perturbation": column,
                "delta": float(delta),
                "n_rows": int(n_rows),
                "mean_delta_aqi": mean_delta,
            })
            print(f"  {mode:<10} {column} +{delta:g} (mean applied {applied:+.2f}): "
                  f"mean change in predicted AQI {mean_delta:+.3f} | "
                  f"{sensitivity_verdict(column, delta, mean_delta)}", flush=True)
    return rows


def print_sensitivity_summary(rows: list[SensitivityRow]) -> None:
    """Print every sensitivity result with its plain verdict, as measured.

    Args:
        rows: Rows from :func:`run_sensitivity`.

    Returns:
        None.
    """
    print(f"\n=== Sensitivity on cold test rows (temperature_2m <= {COLD_TEMPERATURE_C:g} C), "
          "reported without adjustment ===")
    for row in rows:
        verdict = sensitivity_verdict(str(row["perturbation"]), float(row["delta"]),
                                      float(row["mean_delta_aqi"]))
        print(f"  {row['city']:<9} {row['mode']:<10} n={row['n_rows']:<5} "
              f"mean change {float(row['mean_delta_aqi']):+8.3f} AQI | {verdict}")
    for column, delta in PERTURBATIONS:
        subset = [row for row in rows if row["perturbation"] == column]
        as_expected = sum(1 for row in subset
                          if np.sign(float(row["mean_delta_aqi"])) == EXPECTED_SIGN[column])
        print(f"  {column} +{delta:g} moved predicted AQI in the expected direction in "
              f"{as_expected} of {len(subset)} city/mode cases.")


def write_sensitivity(rows: list[SensitivityRow], path: Path) -> Path:
    """Write sensitivity rows to a CSV with the design's sensitivity.csv columns.

    Args:
        rows: Rows from :func:`run_sensitivity`.
        path: Destination file, ``results/sensitivity.csv`` in real runs.

    Returns:
        The written path.

    Raises:
        ValueError: If a row is missing a schema column or has an extra key.
    """
    expected = set(SENSITIVITY_COLUMNS)
    for i, row in enumerate(rows):
        if set(row) != expected:
            raise ValueError(f"Sensitivity row {i} does not match the schema. "
                             f"Missing: {sorted(expected - set(row))}. "
                             f"Unexpected: {sorted(set(row) - expected)}.")
    path = Path(path)
    utils.ensure_dir(path.parent)
    pd.DataFrame(rows, columns=SENSITIVITY_COLUMNS).to_csv(path, index=False, na_rep="")
    return path


def run_training(pipeline: AirQualityDataPipeline, model_df: pd.DataFrame,
                 models_dir: Path = MODELS_DIR, results_dir: Path = RESULTS_DIR,
                 cities: list[str] | None = None, targets: list[str] | None = None,
                 horizons: list[int] | None = None,
                 model_types: list[str] | None = None
                 ) -> tuple[list[MetricRow], list[SensitivityRow]]:
    """Run the forecast loop, the Simulation_Mode models, and the sensitivity analysis.

    Writes ``metrics.csv`` (forecast rows, then simulation rows) and
    ``sensitivity.csv`` into ``results_dir``, and every model into
    ``models_dir``. The defaults are the full run; the optional arguments
    narrow it, for example for a quick check in a temporary folder.

    Args:
        pipeline: Pipeline that provides the split and the feature columns.
        model_df: Frame from ``build_features`` with features and targets.
        models_dir: Folder for the joblib files (default ``MODELS_DIR``).
        results_dir: Folder for the CSV files (default ``RESULTS_DIR``).
        cities: Cities to run, default ``CITIES``.
        targets: Target_Types for the forecast loop, default ``TARGET_TYPES``.
        horizons: Horizons for the forecast loop, default ``HORIZONS``.
        model_types: Forecast model types, default :func:`forecast_model_types`.

    Returns:
        ``(metric_rows, sensitivity_rows)`` as written.

    Raises:
        ValueError: If the narrowed run would not train the forecast us_aqi
            h = 24 Random Forest. The sensitivity analysis needs it, and the
            check runs before any training so a stale file in ``models_dir``
            can never be measured instead.
    """
    cities = list(CITIES) if cities is None else list(cities)
    run_targets = list(TARGET_TYPES) if targets is None else list(targets)
    run_horizons = list(HORIZONS) if horizons is None else list(horizons)
    run_models = forecast_model_types() if model_types is None else list(model_types)
    if (SIMULATION_TARGET not in run_targets or SIMULATION_HORIZON not in run_horizons
            or REFERENCE_MODEL not in run_models):
        raise ValueError(f"The sensitivity analysis needs the forecast {SIMULATION_TARGET} "
                         f"h={SIMULATION_HORIZON} {REFERENCE_MODEL}; include it in the run")

    rows = run_forecast_loop(pipeline, model_df, models_dir=models_dir, cities=cities,
                             targets=run_targets, horizons=run_horizons,
                             model_types=run_models)
    city_frames = {city: model_df[model_df["city"] == city] for city in cities}
    for city in cities:
        rows.extend(run_simulation_model(pipeline, city_frames[city], city, models_dir))

    sensitivity_rows: list[SensitivityRow] = []
    for city in cities:
        sensitivity_rows.extend(run_sensitivity(pipeline, city_frames[city], city, models_dir))

    print_verdict_summary(rows)
    print_sensitivity_summary(sensitivity_rows)
    sensitivity_path = write_sensitivity(sensitivity_rows, Path(results_dir) / SENSITIVITY_FILENAME)
    metrics_path = write_metrics(rows, Path(results_dir) / METRICS_FILENAME)
    print(f"\nWrote {len(rows)} metric rows to {_display_path(metrics_path)}")
    print(f"Wrote {len(sensitivity_rows)} sensitivity rows to {_display_path(sensitivity_path)}")
    return rows, sensitivity_rows


def prediction_figure_name(city: str, target: str, horizon: int, kind: str) -> str:
    """Return the PNG file name of a chart drawn from one model's test predictions.

    Args:
        city: City name, lower-cased in the name like the model files.
        target: ``"pm2_5"`` or ``"us_aqi"``.
        horizon: Horizon in hours.
        kind: ``"actual_vs_predicted"`` or ``"scatter"``.

    Returns:
        For example ``"lahore_us_aqi_h24_actual_vs_predicted.png"``.
    """
    return f"{city.lower()}_{target}_h{int(horizon)}_{kind}.png"


def heatmap_figure_name(city: str) -> str:
    """Return the PNG file name of a city's correlation heatmap.

    Args:
        city: City name.

    Returns:
        For example ``"peshawar_correlation_heatmap.png"``.
    """
    return f"{city.lower()}_correlation_heatmap.png"


def importance_figure_name(city: str, target: str, horizon: int, mode: str) -> str:
    """Return the PNG file name of one model's feature importance chart.

    Args:
        city: City name.
        target: ``"pm2_5"`` or ``"us_aqi"``.
        horizon: Horizon in hours.
        mode: ``"forecast"`` or ``"simulation"``.

    Returns:
        For example ``"lahore_us_aqi_h24_simulation_feature_importance.png"``.
    """
    return f"{city.lower()}_{target}_h{int(horizon)}_{mode}_feature_importance.png"


def figure_model_files(models_dir: Path = MODELS_DIR) -> list[Path]:
    """List every saved model the figures read, in the order they are used.

    Per city: the forecast us_aqi Random Forest of each horizon, the forecast
    pm2_5 h = 24 Random Forest, and the Simulation_Mode Random Forest.

    Args:
        models_dir: Folder with the joblib files.

    Returns:
        The model paths, whether or not they exist.
    """
    paths: list[Path] = []
    for city in CITIES:
        paths += [model_file(models_dir, city, FIGURE_TARGET, horizon, FORECAST_MODE,
                             REFERENCE_MODEL) for horizon in HORIZONS]
        paths.append(model_file(models_dir, city, PM25_FIGURE_TARGET, PM25_FIGURE_HORIZON,
                                FORECAST_MODE, REFERENCE_MODEL))
        paths.append(model_file(models_dir, city, SIMULATION_TARGET, SIMULATION_HORIZON,
                                SIMULATION_MODE, REFERENCE_MODEL))
    return paths


def missing_figure_inputs(processed_dir: Path = PROCESSED_DIR,
                          models_dir: Path = MODELS_DIR) -> list[str]:
    """List the files ``--figures-only`` needs that do not exist yet.

    Args:
        processed_dir: Folder with ``model_dataset.csv`` and ``features_all.csv``.
        models_dir: Folder with the joblib files.

    Returns:
        One line per missing file, naming the file and the command that
        creates it. Empty when everything is present.
    """
    lines: list[str] = []
    for name in (MODEL_DATASET_FILENAME, FEATURES_FILENAME):
        path = Path(processed_dir) / name
        if not path.is_file():
            lines.append(f"{_display_path(path)} (create it with {PIPELINE_COMMAND} "
                         f"or {TRAINING_COMMAND})")
    for path in figure_model_files(models_dir):
        if not path.is_file():
            lines.append(f"{_display_path(path)} (create it with {TRAINING_COMMAND})")
    return lines


def load_processed_frames(processed_dir: Path = PROCESSED_DIR
                          ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the processed files as saved, without rebuilding or rewriting them.

    Args:
        processed_dir: Folder with ``model_dataset.csv`` and ``features_all.csv``.

    Returns:
        ``(model_df, features_df)``: every column of ``model_dataset.csv``, and
        the ``FIGURE_FEATURE_COLUMNS`` of ``features_all.csv``, both with
        ``datetime`` parsed.

    Raises:
        FileNotFoundError: If either file is missing.
    """
    processed_dir = Path(processed_dir)
    model_df = pd.read_csv(processed_dir / MODEL_DATASET_FILENAME)
    features_df = pd.read_csv(processed_dir / FEATURES_FILENAME, usecols=FIGURE_FEATURE_COLUMNS)
    for frame in (model_df, features_df):
        # The split compares timestamps and rejects text, so the column is
        # parsed with the exact format save_processed wrote.
        frame["datetime"] = pd.to_datetime(frame["datetime"], format=DATETIME_FORMAT)
    return model_df, features_df


def load_reference_model(models_dir: Path, city: str, target: str, horizon: int,
                         mode: str) -> SmogPredictorModel:
    """Load one saved Random Forest and check that it is the model its file name says.

    Args:
        models_dir: Folder with the joblib files.
        city: City name.
        target: ``"pm2_5"`` or ``"us_aqi"``.
        horizon: Horizon in hours.
        mode: ``"forecast"`` or ``"simulation"``.

    Returns:
        The loaded model.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the stored city, target, horizon, mode, or model type
            differs from the requested one.
    """
    path = model_file(models_dir, city, target, horizon, mode, REFERENCE_MODEL)
    model = SmogPredictorModel().load_model(path)
    # A file copied or renamed by hand would otherwise put one model's
    # predictions under another model's title.
    stored = (model.city, model.target, model.horizon, model.mode, model.model_type)
    expected = (city, target, int(horizon), mode, REFERENCE_MODEL)
    if stored != expected:
        raise ValueError(f"{path.name} holds the model {stored}, expected {expected}")
    return model


def rebuild_test_predictions(pipeline: AirQualityDataPipeline, city_df: pd.DataFrame,
                             model: SmogPredictorModel) -> tuple[pd.DataFrame, np.ndarray]:
    """Recreate a saved model's Test_Period rows and predict them again.

    Uses the same embargo split for the model's horizon and the same NaN drop
    on its stored ``feature_columns`` plus label as training did, so the rows
    match the ``n_test`` rows behind the model's ``metrics.csv`` scores.

    Args:
        pipeline: Pipeline that provides the split.
        city_df: Rows of the model's city from ``model_dataset.csv``.
        model: Loaded model.

    Returns:
        ``(test, predicted)``: the complete test rows and one prediction per row.

    Raises:
        ValueError: If no complete test rows remain.
    """
    label_col = target_column(model.target, model.horizon)
    _, test, _, _ = _complete_split(pipeline, city_df, model.horizon,
                                    model.feature_columns + [label_col])
    if test.empty:
        raise ValueError(f"{model.city} {model.target} h={model.horizon} {model.mode}: "
                         "no complete test rows to predict")
    return test, model.predict(test[model.feature_columns])


def _save_and_report(fig: Figure, path: Path) -> Path:
    """Save and close a figure with ``charts.save_figure`` and print where it went.

    Args:
        fig: Figure to save.
        path: Target PNG path.

    Returns:
        The written path.
    """
    saved = charts.save_figure(fig, path)
    print(f"  saved {_display_path(saved)}", flush=True)
    return saved


def save_prediction_figures(pipeline: AirQualityDataPipeline, city_df: pd.DataFrame,
                            model: SmogPredictorModel, figures_dir: Path,
                            with_scatter: bool) -> list[Path]:
    """Draw the actual-vs-predicted chart, and optionally the scatter, for one model.

    Args:
        pipeline: Pipeline that provides the split.
        city_df: Rows of the model's city from ``model_dataset.csv``.
        model: Loaded Forecast_Mode model.
        figures_dir: Folder for the PNG files.
        with_scatter: Also draw the predicted-vs-actual scatter of all test rows.

    Returns:
        The written paths, actual vs predicted first.
    """
    city, target, horizon = model.city, model.target, model.horizon
    test, predicted = rebuild_test_predictions(pipeline, city_df, model)
    actual = test[target_column(target, horizon)].to_numpy(dtype=float)
    # Row t holds the value measured at t + h, so both lines are drawn at the
    # hour the forecast is for, and the winter window selects those hours.
    target_times = test["datetime"] + pd.Timedelta(horizon, unit="h")
    first, last = target_times.min(), target_times.max()
    print(f"  {city} {target} h={horizon}: {len(test):,} test rows, target hours "
          f"{first:%Y-%m-%d %H:%M} to {last:%Y-%m-%d %H:%M}, "
          f"MAE {utils.mae(actual, predicted):.2f}", flush=True)

    figures_dir = Path(figures_dir)
    saved = [_save_and_report(
        charts.plot_actual_vs_predicted(target_times, actual, predicted, target, city, horizon),
        figures_dir / prediction_figure_name(city, target, horizon, "actual_vs_predicted"))]
    if with_scatter:
        title = (f"{city}: predicted vs actual {charts.TARGET_NAMES[target]}, "
                 f"{horizon} h Random Forest\n"
                 f"all test hours, {first:%d %b %Y} to {last:%d %b %Y}")
        saved.append(_save_and_report(
            charts.plot_predicted_vs_actual_scatter(actual, predicted, title),
            figures_dir / prediction_figure_name(city, target, horizon, "scatter")))
    return saved


def save_importance_figure(model: SmogPredictorModel, figures_dir: Path) -> Path:
    """Draw the top-10 feature importance chart of one model.

    Args:
        model: Loaded model.
        figures_dir: Folder for the PNG file.

    Returns:
        The written path.
    """
    title = (f"{model.city}: feature importance, {charts.TARGET_NAMES[model.target]} "
             f"{model.horizon} h {MODE_LABELS[model.mode]} Random Forest")
    fig = charts.plot_feature_importance(model.get_feature_importance(), title,
                                         top_n=FEATURE_IMPORTANCE_TOP_N)
    return _save_and_report(fig, Path(figures_dir) / importance_figure_name(
        model.city, model.target, model.horizon, model.mode))


def generate_figures(pipeline: AirQualityDataPipeline, model_df: pd.DataFrame,
                     features_df: pd.DataFrame, models_dir: Path = MODELS_DIR,
                     figures_dir: Path = FIGURES_DIR) -> list[Path]:
    """Draw every report figure from the saved models (Requirement 14.7).

    Per city: actual vs predicted and scatter for the forecast us_aqi Random
    Forest of each horizon, actual vs predicted for the forecast pm2_5
    h = 24 Random Forest, feature importance for the forecast us_aqi h = 24
    and the Simulation_Mode Random Forests, and the correlation heatmap. Then
    one city comparison chart. Existing files with the same names are
    overwritten; other files in ``figures_dir`` are left alone.

    Args:
        pipeline: Pipeline that provides the split.
        model_df: Frame of ``model_dataset.csv`` (features and targets) with
            ``datetime`` parsed.
        features_df: Frame of ``features_all.csv`` with at least
            ``FIGURE_FEATURE_COLUMNS`` and ``datetime`` parsed.
        models_dir: Folder with the joblib files (default ``MODELS_DIR``).
        figures_dir: Folder for the PNG files (default ``FIGURES_DIR``).

    Returns:
        The written paths in drawing order.

    Raises:
        ValueError: If a city of ``CITIES`` has no rows in either frame.
        FileNotFoundError: If a needed model file is missing.
    """
    for name, frame in (("model_df", model_df), ("features_df", features_df)):
        present = set(frame["city"].unique())
        absent = [city for city in CITIES if city not in present]
        if absent:
            raise ValueError(f"{name} has no rows for {absent}; cities in it: {sorted(present)}")

    figures_dir = Path(figures_dir)
    print(f"\n=== Figures -> {_display_path(figures_dir)} ===", flush=True)
    saved: list[Path] = []
    for city in CITIES:
        city_df = model_df[model_df["city"] == city]
        for horizon in HORIZONS:
            model = load_reference_model(models_dir, city, FIGURE_TARGET, horizon, FORECAST_MODE)
            saved += save_prediction_figures(pipeline, city_df, model, figures_dir,
                                             with_scatter=True)
            if horizon == IMPORTANCE_HORIZON:
                # The model is already in memory, so its file is not read twice.
                saved.append(save_importance_figure(model, figures_dir))
        pm25_model = load_reference_model(models_dir, city, PM25_FIGURE_TARGET,
                                          PM25_FIGURE_HORIZON, FORECAST_MODE)
        saved += save_prediction_figures(pipeline, city_df, pm25_model, figures_dir,
                                         with_scatter=False)
        simulation_model = load_reference_model(models_dir, city, SIMULATION_TARGET,
                                                SIMULATION_HORIZON, SIMULATION_MODE)
        saved.append(save_importance_figure(simulation_model, figures_dir))
        saved.append(_save_and_report(
            charts.plot_correlation_heatmap(features_df[features_df["city"] == city], city),
            figures_dir / heatmap_figure_name(city)))

    saved.append(_save_and_report(
        charts.plot_city_comparison(features_df[features_df["city"].isin(CITIES)]),
        figures_dir / CITY_COMPARISON_FIGURE))
    print(f"Saved {len(saved)} figures to {_display_path(figures_dir)}")
    return saved


def run_figures_only(processed_dir: Path = PROCESSED_DIR, models_dir: Path = MODELS_DIR,
                     figures_dir: Path = FIGURES_DIR) -> int:
    """Run ``--figures-only``: draw every figure from saved files and train nothing.

    The processed CSVs are read as they are, never rebuilt, so only PNG files
    are written; models, ``metrics.csv``, ``sensitivity.csv``, and
    ``data/processed`` stay unchanged.

    Args:
        processed_dir: Folder with the processed CSVs (default ``PROCESSED_DIR``).
        models_dir: Folder with the joblib files (default ``MODELS_DIR``).
        figures_dir: Folder for the PNG files (default ``FIGURES_DIR``).

    Returns:
        0 when the figures were written; 1 when an input file is missing, in
        which case nothing is drawn and every missing file is printed with the
        command that creates it.
    """
    missing = missing_figure_inputs(processed_dir, models_dir)
    if missing:
        print("--figures-only needs these files, which do not exist yet:")
        for line in missing:
            print(f"  {line}")
        print(f"Nothing was drawn. Run {TRAINING_COMMAND} once to build the processed "
              "data and train the models, then rerun with --figures-only.")
        return 1
    model_df, features_df = load_processed_frames(processed_dir)
    print(f"Read {MODEL_DATASET_FILENAME} ({len(model_df):,} rows) and {FEATURES_FILENAME} "
          f"({len(features_df):,} rows) from {_display_path(Path(processed_dir))}; "
          "nothing is retrained.")
    generate_figures(AirQualityDataPipeline(), model_df, features_df, models_dir, figures_dir)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command-line options of the Training_Script.

    Args:
        argv: Arguments without the program name; None reads ``sys.argv``.

    Returns:
        Namespace with ``figures_only`` (bool).
    """
    parser = argparse.ArgumentParser(
        description="Train the Dhund-Khabar AI models, write results/metrics.csv and "
                    "results/sensitivity.csv, and save the figures to results/figures/.")
    parser.add_argument(
        "--figures-only", action="store_true",
        help="Skip training and only draw the figures from the saved models and "
             "the existing data/processed files.")
    return parser.parse_args(argv)


def main(figures_only: bool = False) -> int:
    """Build the processed data, train every model, write the results, and draw the figures.

    Args:
        figures_only: When True, train nothing and only draw the figures with
            :func:`run_figures_only`.

    Returns:
        Exit status: 0 on success, 1 when ``--figures-only`` is missing input
        files. Models go to ``models/``, scores to ``results/metrics.csv`` and
        ``results/sensitivity.csv``, figures to ``results/figures/``.
    """
    if figures_only:
        return run_figures_only()
    start = time.perf_counter()
    pipeline = AirQualityDataPipeline()
    model_df = pipeline.build_features(MASTER_PATH)
    run_training(pipeline, model_df)
    # The figures load the files just saved, which are the models the
    # dashboard will load, rather than the objects still in memory.
    generate_figures(pipeline, model_df, pipeline.features)
    print(f"Total time {(time.perf_counter() - start) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main(figures_only=parse_args().figures_only))
