"""Model wrapper for Dhund-Khabar AI.

This module holds the class ``SmogPredictorModel``, the second of the
project's two classes. One instance wraps one scikit-learn (or optional
xgboost) regressor together with the facts that describe it: city, horizon,
target, mode, and the exact feature column names it was trained on. Keeping
those facts next to the fitted estimator, in memory and in the saved joblib
file, means a loaded model can always check that it receives the columns it
was trained on.

xgboost is optional (Requirement 1.5). It is imported inside
``_build_estimator`` only, so importing this module never needs it.
"""

import copy
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression

from src import alerts, utils

SUPPORTED_MODEL_TYPES: tuple[str, ...] = (
    "random_forest", "linear_regression", "gradient_boosting", "xgboost",
)
# Same values as pipeline.FEATURE_MODES. They are repeated here because the
# design lets model.py import only utils and alerts, which keeps the model
# usable without the pipeline (for example when the dashboard loads it).
SUPPORTED_MODES: tuple[str, ...] = ("forecast", "simulation")
# Every key save_model writes. load_model checks all of them, so a file from
# another tool fails with a clear message instead of a KeyError later.
PAYLOAD_KEYS: tuple[str, ...] = (
    "estimator", "model_type", "city", "horizon", "target", "mode",
    "feature_columns", "sklearn_version",
)
NOT_FITTED_MESSAGE: str = "Model is not fitted. Call train() or load_model() first."


def _check_model_type(model_type: str) -> None:
    """Raise a ValueError if a model type is unsupported or needs a missing package.

    Args:
        model_type: Name of the estimator to check.

    Returns:
        None. The function only raises.

    Raises:
        ValueError: If ``model_type`` is not in ``SUPPORTED_MODEL_TYPES``
            (the message names every supported type), or if it is
            ``"xgboost"`` and xgboost is not installed.
    """
    if model_type not in SUPPORTED_MODEL_TYPES:
        raise ValueError(
            f"Unsupported model_type '{model_type}'. "
            f"Supported: {', '.join(SUPPORTED_MODEL_TYPES)}"
        )
    if model_type == "xgboost" and not utils.xgboost_available():
        raise ValueError("xgboost is not installed")


class SmogPredictorModel:
    """Train, evaluate, predict, explain, save, and load one smog forecasting model.

    Attributes:
        model_type: One of ``SUPPORTED_MODEL_TYPES``.
        city: City the model belongs to, or None when not tied to one city.
        horizon: Forecast horizon in hours.
        target: Target_Type, ``"pm2_5"`` or ``"us_aqi"``.
        mode: ``"forecast"`` or ``"simulation"``.
        feature_columns: Column names of ``X_train`` in training order. Empty
            until the model is trained or loaded.
        estimator: The fitted regressor, or None before training or loading.
        sklearn_version: scikit-learn version the estimator was fitted with,
            or None before training or loading.
        is_fitted: True once ``train`` or ``load_model`` has succeeded.
    """

    def __init__(self, model_type: str = "random_forest", city: str | None = None,
                 horizon: int = 24, target: str = "pm2_5",
                 mode: str = "forecast") -> None:
        """Create an unfitted model and validate its settings.

        Args:
            model_type: ``"random_forest"``, ``"linear_regression"``,
                ``"gradient_boosting"``, or ``"xgboost"``.
            city: City name such as ``"Lahore"``, or None.
            horizon: Forecast horizon in hours, a positive integer.
            target: ``"pm2_5"`` or ``"us_aqi"``.
            mode: ``"forecast"`` or ``"simulation"``.

        Returns:
            None.

        Raises:
            ValueError: If model_type is unsupported (the message names the
                supported types), if xgboost is requested but not installed,
                or if horizon, target, or mode is invalid.
        """
        _check_model_type(model_type)
        # target and mode end up in the saved file name (utils.model_path), so
        # a typo here would save a model that the dashboard can never find.
        if target not in utils.TARGET_TYPES:
            raise ValueError(f"target must be one of {utils.TARGET_TYPES}, got '{target}'")
        if mode not in SUPPORTED_MODES:
            raise ValueError(f"mode must be one of {list(SUPPORTED_MODES)}, got '{mode}'")
        # bool is a subclass of int, so True would otherwise pass as horizon 1.
        if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)) \
                or horizon <= 0:
            raise ValueError(f"horizon must be a positive integer, got {horizon!r}")

        self.model_type: str = model_type
        self.city: str | None = city
        self.horizon: int = int(horizon)
        self.target: str = target
        self.mode: str = mode
        self.feature_columns: list[str] = []
        self.estimator: object | None = None
        self.sklearn_version: str | None = None
        self.is_fitted: bool = False

    def _build_estimator(self) -> object:
        """Create a new, unfitted estimator for ``self.model_type``.

        Every estimator that accepts ``random_state`` gets
        ``utils.RANDOM_STATE`` (42), so repeated runs give the same model
        (Requirement 1.6).

        Returns:
            RandomForestRegressor(n_estimators=100, max_depth=12, n_jobs=-1,
            random_state=42), LinearRegression(),
            GradientBoostingRegressor(random_state=42), or
            XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
            n_jobs=-1, random_state=42).

        Raises:
            ValueError: If the model type is unsupported or xgboost is missing.
        """
        # Checked again here because model_type is a public attribute and may
        # have been changed after construction; a clear ValueError beats an
        # ImportError from deep inside a fit.
        _check_model_type(self.model_type)
        if self.model_type == "random_forest":
            # max_depth=12 bounds the size of each tree, which keeps training
            # time and the saved file size manageable for 12+ city/horizon
            # models while still capturing non-linear weather effects.
            return RandomForestRegressor(n_estimators=100, max_depth=12, n_jobs=-1,
                                         random_state=utils.RANDOM_STATE)
        if self.model_type == "linear_regression":
            # LinearRegression has no random_state parameter: its fit is
            # deterministic.
            return LinearRegression()
        if self.model_type == "gradient_boosting":
            return GradientBoostingRegressor(random_state=utils.RANDOM_STATE)
        # Imported here, not at module level, so the whole project runs when
        # xgboost is not installed (Requirement 1.5).
        from xgboost import XGBRegressor
        return XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                            n_jobs=-1, random_state=utils.RANDOM_STATE)

    def _check_fitted(self) -> None:
        """Raise a RuntimeError if the model has not been trained or loaded.

        Returns:
            None. The method only raises.

        Raises:
            RuntimeError: "Model is not fitted. Call train() or load_model() first."
        """
        if not self.is_fitted or self.estimator is None:
            raise RuntimeError(NOT_FITTED_MESSAGE)

    def _align_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return ``X`` with exactly the training columns, in training order.

        The same columns in a different order are reordered silently, because
        order carries no meaning in a DataFrame. Any missing or extra column is
        an error, because a model must never predict from features it was not
        trained on or without features it needs.

        Args:
            X: Input features with named columns.

        Returns:
            A DataFrame with the columns of ``self.feature_columns`` in order.

        Raises:
            TypeError: If ``X`` is not a pandas DataFrame.
            ValueError: If ``X`` has duplicate column names, or if its columns
                differ from the training columns. The message names the
                missing and the unexpected columns.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                f"X must be a pandas DataFrame with named columns, got {type(X).__name__}"
            )
        if X.columns.duplicated().any():
            duplicates = sorted({str(c) for c in X.columns[X.columns.duplicated()]})
            raise ValueError(f"Duplicate columns in input: {duplicates}")
        given = set(X.columns)
        expected = set(self.feature_columns)
        if given != expected:
            missing = [c for c in self.feature_columns if c not in given]
            unexpected = [c for c in X.columns if c not in expected]
            raise ValueError(
                f"Feature mismatch. Missing columns: {missing}. "
                f"Unexpected columns: {unexpected}."
            )
        return X.loc[:, self.feature_columns]

    def train(self, X_train: pd.DataFrame,
              y_train: pd.Series | np.ndarray) -> "SmogPredictorModel":
        """Fit a fresh estimator and remember the training column names.

        A new estimator is built on every call, so training twice never mixes
        state from an earlier fit. The instance is only updated after the fit
        succeeds, so a failed fit leaves the previous state intact.

        Args:
            X_train: Training features. Must be a DataFrame, because its
                column names become ``feature_columns`` and are checked on
                every later ``predict`` call.
            y_train: Training labels, one per row of ``X_train`` (1-D or a
                single column).

        Returns:
            self, so calls can be chained, e.g.
            ``SmogPredictorModel().train(X, y).predict(X_new)``.

        Raises:
            TypeError: If ``X_train`` is not a pandas DataFrame.
            ValueError: If ``X_train`` has duplicate or no columns, or if
                ``y_train`` is not 1-D or its length differs from ``X_train``.
        """
        if not isinstance(X_train, pd.DataFrame):
            raise TypeError(
                "X_train must be a pandas DataFrame with named columns, "
                f"got {type(X_train).__name__}"
            )
        if X_train.shape[1] == 0:
            raise ValueError("X_train has no columns")
        if X_train.columns.duplicated().any():
            duplicates = sorted({str(c) for c in X_train.columns[X_train.columns.duplicated()]})
            raise ValueError(f"Duplicate columns in X_train: {duplicates}")

        y = np.asarray(y_train, dtype=float)
        # A (n, 1) label column would make LinearRegression return (n, 1)
        # predictions, so it is flattened to keep predict's shape at (n,).
        if y.ndim == 2 and y.shape[1] == 1:
            y = y[:, 0]
        if y.ndim != 1:
            raise ValueError(f"y_train must be 1-D or a single column, got shape {y.shape}")
        if y.shape[0] != X_train.shape[0]:
            raise ValueError(
                f"X_train has {X_train.shape[0]} rows but y_train has {y.shape[0]} values"
            )

        estimator = self._build_estimator()
        estimator.fit(X_train, y)

        self.estimator = estimator
        self.feature_columns = list(X_train.columns)
        self.sklearn_version = sklearn.__version__
        self.is_fitted = True
        return self

    def predict(self, X_new: pd.DataFrame) -> np.ndarray:
        """Predict one value per input row.

        Args:
            X_new: Features with exactly the training columns, in any order.

        Returns:
            A float NumPy array of shape ``(len(X_new),)``.

        Raises:
            RuntimeError: If the model is not trained or loaded.
            TypeError: If ``X_new`` is not a pandas DataFrame.
            ValueError: If the columns differ from the training columns; the
                message names the missing and unexpected columns.
        """
        self._check_fitted()
        X = self._align_columns(X_new)
        estimator = self.estimator
        if isinstance(estimator, RandomForestRegressor) and estimator.n_jobs != 1:
            # With n_jobs=-1 the forest adds tree outputs from several threads
            # in a varying order, so the same input can differ by ~1e-14
            # between calls. A single thread fixes the order and makes every
            # reported number reproducible. A shallow copy shares the fitted
            # trees and leaves the stored estimator's n_jobs=-1 untouched, so
            # concurrent dashboard sessions never see a half-changed setting.
            estimator = copy.copy(estimator)
            estimator.n_jobs = 1
        # reshape(-1) guarantees (n,) for every estimator type, including any
        # that return a single-column 2-D array.
        return np.asarray(estimator.predict(X), dtype=float).reshape(-1)

    def evaluate(self, X_test: pd.DataFrame,
                 y_test: pd.Series | np.ndarray) -> dict[str, float]:
        """Score the model on held-out rows with MAE, RMSE, and R2.

        MAE and RMSE come from the project's own NumPy code in ``utils``
        (Requirement 11.5); R2 wraps scikit-learn's ``r2_score``.

        Args:
            X_test: Test features with exactly the training columns.
            y_test: Observed values, one per row of ``X_test``.

        Returns:
            ``{"MAE": float, "RMSE": float, "R2": float}``.

        Raises:
            RuntimeError: If the model is not trained or loaded.
            TypeError: If ``X_test`` is not a pandas DataFrame.
            ValueError: If the columns differ from the training columns, or if
                ``y_test`` has a different length from ``X_test``.
        """
        predictions = self.predict(X_test)
        return {
            "MAE": utils.mae(y_test, predictions),
            "RMSE": utils.rmse(y_test, predictions),
            "R2": utils.r2(y_test, predictions),
        }

    def get_risk_advisory(self, predicted_aqi: float) -> dict[str, str]:
        """Return the EPA advisory for a predicted AQI value.

        The rules live in ``alerts`` only, so the model, the training script,
        and the dashboard can never disagree on a tier or advisory text.

        Args:
            predicted_aqi: One AQI value, for example an element of
                ``predict`` output from a us_aqi model.

        Returns:
            The dict from ``alerts.get_aqi_advisory``: tier, color, warnings,
            school_status, respiratory_guidance, and industrial_advisory.

        Raises:
            TypeError: If predicted_aqi is not a real number.
            ValueError: If predicted_aqi is NaN, negative, or +inf. Callers
                that use a model able to predict below 0 (linear regression)
                should clip at 0 first.
        """
        return alerts.get_aqi_advisory(predicted_aqi)

    def get_feature_importance(self) -> pd.DataFrame:
        """Return feature importances sorted from highest to lowest.

        Tree models (random forest, gradient boosting, xgboost) use the
        estimator's ``feature_importances_``. Linear regression uses
        ``abs(coef_)``; these magnitudes depend on each feature's units, so
        they rank features only roughly unless the inputs share a scale.

        Returns:
            A DataFrame with columns ``feature`` and ``importance`` and one row
            per training column, sorted by importance descending. Ties keep the
            training column order.

        Raises:
            RuntimeError: If the model is not trained or loaded, or if the
                estimator exposes neither ``feature_importances_`` nor
                ``coef_``.
        """
        self._check_fitted()
        if hasattr(self.estimator, "feature_importances_"):
            values = np.asarray(self.estimator.feature_importances_, dtype=float)
        elif hasattr(self.estimator, "coef_"):
            values = np.abs(np.asarray(self.estimator.coef_, dtype=float)).reshape(-1)
        else:
            raise RuntimeError(
                f"Estimator {type(self.estimator).__name__} has no feature importances"
            )
        importance = pd.DataFrame({"feature": self.feature_columns, "importance": values})
        # A stable sort keeps tied features in training order, so the result
        # is the same on every run.
        return importance.sort_values("importance", ascending=False,
                                      kind="mergesort").reset_index(drop=True)

    def save_model(self, filepath: str | Path) -> None:
        """Save the fitted estimator and its metadata to one joblib file.

        The payload holds estimator, model_type, city, horizon, target, mode,
        feature_columns, and sklearn_version. ``compress=3`` keeps 100-tree,
        depth-12 forests to a manageable size on disk. Parent directories are
        created if needed.

        Args:
            filepath: Destination path, usually from ``utils.model_path``.

        Returns:
            None.

        Raises:
            RuntimeError: If the model is not trained or loaded.
        """
        self._check_fitted()
        path = Path(filepath)
        utils.ensure_dir(path.parent)
        payload = {
            "estimator": self.estimator,
            "model_type": self.model_type,
            "city": self.city,
            "horizon": self.horizon,
            "target": self.target,
            "mode": self.mode,
            "feature_columns": list(self.feature_columns),
            "sklearn_version": self.sklearn_version or sklearn.__version__,
        }
        joblib.dump(payload, path, compress=3)

    def load_model(self, filepath: str | Path) -> "SmogPredictorModel":
        """Restore a model saved by :meth:`save_model` into this instance.

        Every saved field replaces the current one, so
        ``SmogPredictorModel().load_model(path)`` gives a ready-to-use model
        whatever the constructor defaults were. joblib files are pickles and
        can run code when loaded, so only load files this project created.

        Args:
            filepath: Path of a joblib file written by ``save_model``.

        Returns:
            self, with ``is_fitted`` set to True.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file does not hold a SmogPredictorModel payload
                (the message names the missing keys) or names an unsupported
                model type.

        Warns:
            UserWarning: If the file was saved with a different scikit-learn
                version than the one installed. Loading still succeeds,
                because small version steps usually predict the same.
        """
        path = Path(filepath)
        if not path.is_file():
            raise FileNotFoundError(f"Model file not found: {path}")
        payload = joblib.load(path)
        if not isinstance(payload, dict):
            raise ValueError(f"{path} does not contain a SmogPredictorModel payload")
        missing = [key for key in PAYLOAD_KEYS if key not in payload]
        if missing:
            raise ValueError(f"{path} is missing model payload keys: {missing}")
        if payload["model_type"] not in SUPPORTED_MODEL_TYPES:
            raise ValueError(
                f"Unsupported model_type '{payload['model_type']}' in {path}. "
                f"Supported: {', '.join(SUPPORTED_MODEL_TYPES)}"
            )

        saved_version = str(payload["sklearn_version"])
        if saved_version != sklearn.__version__:
            # A warning instead of an error, so the dashboard still works
            # after a minor library upgrade; the message says how to fix it.
            warnings.warn(
                f"{path.name} was saved with scikit-learn {saved_version}, but "
                f"{sklearn.__version__} is installed. Predictions may differ; "
                "rerun python scripts/train_all.py to retrain.",
                UserWarning,
                stacklevel=2,
            )

        self.estimator = payload["estimator"]
        self.model_type = payload["model_type"]
        self.city = payload["city"]
        self.horizon = int(payload["horizon"])
        self.target = payload["target"]
        self.mode = payload["mode"]
        self.feature_columns = list(payload["feature_columns"])
        self.sklearn_version = saved_version
        self.is_fitted = True
        return self
