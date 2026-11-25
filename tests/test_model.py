"""Tests for src.model.SmogPredictorModel (synthetic data only)."""

import numpy as np
import pandas as pd
import pytest

from src import utils
from src.model import SmogPredictorModel

N_FEATURES = 8
FEATURES = [f"f{i}" for i in range(N_FEATURES)]


def _installed_model_types() -> list[str]:
    types = ["random_forest", "linear_regression", "gradient_boosting"]
    if utils.xgboost_available():
        types.append("xgboost")
    return types


@pytest.fixture(scope="module")
def trained_models() -> dict[str, SmogPredictorModel]:
    """One default-config model per installed type, fitted once per module."""
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(300, N_FEATURES)), columns=FEATURES)
    y = 3.0 * X["f0"] - 2.0 * X["f1"] + X["f2"] ** 2 + rng.normal(scale=0.1, size=300)
    models = {}
    for model_type in _installed_model_types():
        model = SmogPredictorModel(model_type=model_type, city="Lahore", horizon=24,
                                   target="us_aqi", mode="forecast")
        models[model_type] = model.train(X, y)
    return models


def test_prediction_shape_and_feature_importance_property(trained_models):
    """Feature: dhund-khabar-ai, Property 13: One prediction per input row

    **Validates: Requirements 11.6, 11.8, 18.10**

    For 120 seeded cases (model type, row count 1..500, column order all
    varied), predict returns a float ndarray of shape (n,), and feature
    importance has one row per feature, sorted descending and non-negative.
    """
    rng = np.random.default_rng(42)
    model_types = list(trained_models)
    n_cases = 120
    for case in range(n_cases):
        model_type = model_types[case % len(model_types)]
        model = trained_models[model_type]
        n = int(rng.integers(1, 501))
        columns = list(rng.permutation(FEATURES))
        frame = pd.DataFrame(rng.normal(size=(n, N_FEATURES)), columns=FEATURES)
        frame = frame.loc[:, columns]

        pred = model.predict(frame)
        ctx = f"case={case} type={model_type} n={n}"
        assert isinstance(pred, np.ndarray), ctx
        assert pred.dtype.kind == "f", ctx
        assert pred.shape == (n,), ctx
        assert np.isfinite(pred).all(), ctx

        imp = model.get_feature_importance()
        assert len(imp) == N_FEATURES, ctx
        assert sorted(imp["feature"]) == sorted(FEATURES), ctx
        values = imp["importance"].to_numpy()
        assert (values >= 0).all(), ctx
        assert (np.diff(values) <= 0).all(), ctx


def test_save_load_round_trip_property(trained_models, tmp_path):
    """Feature: dhund-khabar-ai, Property 14: Save and load round trip

    **Validates: Requirements 11.9, 11.10, 18.11**

    24 seeded save/load round trips (model type, city, horizon, target, mode
    varied), each checked on 5 random batches (120 cases). Predictions must be
    identical, metadata and feature order restored, and the payload complete.
    """
    import copy

    import joblib
    import sklearn

    rng = np.random.default_rng(42)
    model_types = list(trained_models)
    cities = ["Lahore", "Peshawar"]
    horizons = [24, 48, 72]
    targets = ["us_aqi", "pm2_5"]
    modes = ["forecast", "simulation"]
    n_round_trips, batches_per_trip = 24, 5
    cases = 0

    for trip in range(n_round_trips):
        model_type = model_types[trip % len(model_types)]
        # Shallow copy shares the fitted estimator but carries its own metadata.
        original = copy.copy(trained_models[model_type])
        original.city = str(rng.choice(cities))
        original.horizon = int(rng.choice(horizons))
        original.target = str(rng.choice(targets))
        original.mode = str(rng.choice(modes))

        path = tmp_path / f"trip_{trip}" / "model.joblib"
        original.save_model(path)
        assert path.is_file()

        payload = joblib.load(path)
        ctx = f"trip={trip} type={model_type}"
        for key in ("estimator", "model_type", "city", "horizon", "target",
                    "mode", "feature_columns", "sklearn_version"):
            assert key in payload, f"{ctx} missing {key}"
        assert payload["sklearn_version"] == sklearn.__version__, ctx

        loaded = SmogPredictorModel().load_model(path)
        assert loaded.is_fitted is True, ctx
        assert loaded.model_type == original.model_type, ctx
        assert loaded.city == original.city, ctx
        assert loaded.horizon == original.horizon, ctx
        assert loaded.target == original.target, ctx
        assert loaded.mode == original.mode, ctx
        assert loaded.feature_columns == original.feature_columns, ctx

        for batch in range(batches_per_trip):
            n = int(rng.integers(1, 201))
            frame = pd.DataFrame(rng.normal(size=(n, N_FEATURES)), columns=FEATURES)
            assert np.array_equal(original.predict(frame), loaded.predict(frame)), \
                f"{ctx} batch={batch} n={n}"
            cases += 1

    assert cases >= 100


def test_column_mismatch_rejected_after_load_property(trained_models, tmp_path):
    """Feature: dhund-khabar-ai, Property 15: Column mismatch is rejected after load

    **Validates: Requirements 11.11, 18.12**

    For 150 seeded cases on models loaded from disk: dropping columns, adding
    unexpected columns, or both makes predict raise ValueError naming every
    missing and every unexpected column; a pure shuffle of the right columns
    does not raise and predicts the same values.
    """
    rng = np.random.default_rng(42)
    loaded = {}
    for model_type, model in trained_models.items():
        path = tmp_path / f"{model_type}.joblib"
        model.save_model(path)
        loaded[model_type] = SmogPredictorModel().load_model(path)
    model_types = list(loaded)

    n_cases = 150
    for case in range(n_cases):
        model_type = model_types[case % len(model_types)]
        model = loaded[model_type]
        n = int(rng.integers(1, 40))
        base = pd.DataFrame(rng.normal(size=(n, N_FEATURES)), columns=FEATURES)
        ctx = f"case={case} type={model_type}"

        kind = case % 4  # 0 drop, 1 add, 2 both, 3 shuffle only
        if kind == 3:
            shuffled = base.loc[:, list(rng.permutation(FEATURES))]
            assert np.array_equal(model.predict(shuffled), model.predict(base)), ctx
            continue

        n_drop = int(rng.integers(1, N_FEATURES + 1)) if kind in (0, 2) else 0
        dropped = [str(c) for c in rng.choice(FEATURES, size=n_drop, replace=False)]
        n_add = int(rng.integers(1, 5)) if kind in (1, 2) else 0
        extras = [f"extra_{case}_{j}" for j in range(n_add)]

        frame = base.drop(columns=dropped)
        for name in extras:
            frame[name] = rng.normal(size=n)
        frame = frame.loc[:, list(rng.permutation(frame.columns))]

        with pytest.raises(ValueError) as excinfo:
            model.predict(frame)
        message = str(excinfo.value)
        for name in dropped + extras:
            assert name in message, f"{ctx} message does not name {name}: {message}"


def test_load_errors_and_unfitted_predict(trained_models, tmp_path):
    """Feature: dhund-khabar-ai, Property 15: Column mismatch is rejected after load

    **Validates: Requirements 11.11, 18.12**

    Loading a missing file raises FileNotFoundError, a payload with missing
    keys raises ValueError naming them, and an unfitted model cannot predict.
    """
    import joblib

    with pytest.raises(FileNotFoundError):
        SmogPredictorModel().load_model(tmp_path / "does_not_exist.joblib")

    good = tmp_path / "good.joblib"
    next(iter(trained_models.values())).save_model(good)
    payload = joblib.load(good)
    for key in ("estimator", "feature_columns", "model_type"):
        broken = {k: v for k, v in payload.items() if k != key}
        path = tmp_path / f"no_{key}.joblib"
        joblib.dump(broken, path)
        with pytest.raises(ValueError, match=key):
            SmogPredictorModel().load_model(path)

    not_dict = tmp_path / "list.joblib"
    joblib.dump([1, 2, 3], not_dict)
    with pytest.raises(ValueError):
        SmogPredictorModel().load_model(not_dict)

    frame = pd.DataFrame(np.zeros((2, N_FEATURES)), columns=FEATURES)
    with pytest.raises(RuntimeError):
        SmogPredictorModel().predict(frame)


# ---------------------------------------------------------------------------
# Example and edge-case tests (task 13.17)
# ---------------------------------------------------------------------------

def _split_tiny(tiny_model_data):
    """Return (X_train, y_train, X_test, y_test) from the tiny fixture."""
    X = tiny_model_data[FEATURES]
    y = tiny_model_data["target"]
    return X.iloc[:240], y.iloc[:240], X.iloc[240:], y.iloc[240:]


def test_constructor_defaults():
    """Requirements 11.1: defaults are random_forest, 24h, pm2_5, forecast, no city."""
    model = SmogPredictorModel()
    assert model.model_type == "random_forest"
    assert model.horizon == 24
    assert model.target == "pm2_5"
    assert model.mode == "forecast"
    assert model.city is None
    assert model.is_fitted is False
    assert model.feature_columns == []


def test_random_forest_parameters():
    """Requirements 11.2: RF uses n_estimators=100, max_depth=12, random_state=42."""
    from sklearn.ensemble import RandomForestRegressor

    est = SmogPredictorModel("random_forest")._build_estimator()
    assert isinstance(est, RandomForestRegressor)
    assert est.n_estimators == 100
    assert est.max_depth == 12
    assert est.random_state == 42


def test_linear_and_gradient_boosting_selection():
    """Requirements 11.3, 11.4: type selects estimator; GB gets random_state=42."""
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression

    assert isinstance(SmogPredictorModel("linear_regression")._build_estimator(),
                      LinearRegression)
    gb = SmogPredictorModel("gradient_boosting")._build_estimator()
    assert isinstance(gb, GradientBoostingRegressor)
    assert gb.random_state == 42


def test_unsupported_model_type_names_supported_types():
    """Requirements 11.7: ValueError message lists every supported type."""
    with pytest.raises(ValueError) as excinfo:
        SmogPredictorModel("svm")
    message = str(excinfo.value)
    for name in ("random_forest", "linear_regression", "gradient_boosting", "xgboost"):
        assert name in message


def test_xgboost_unavailable_raises_value_error(monkeypatch):
    """Requirements 1.5, 11.7: missing xgboost gives a ValueError, not ImportError."""
    monkeypatch.setattr(utils, "xgboost_available", lambda: False)
    with pytest.raises(ValueError, match="xgboost"):
        SmogPredictorModel("xgboost")
    # Other model types are unaffected.
    assert SmogPredictorModel("random_forest").model_type == "random_forest"


@pytest.mark.parametrize("kwargs", [
    {"target": "pm10"},
    {"mode": "live"},
    {"horizon": 0},
    {"horizon": -24},
    {"horizon": 1.5},
    {"horizon": True},
])
def test_invalid_constructor_settings_raise(kwargs):
    """Invalid target, mode, or horizon raises ValueError."""
    with pytest.raises(ValueError):
        SmogPredictorModel(**kwargs)


def test_predict_and_evaluate_before_fit_raise_runtime_error():
    """Requirements 11.7: unfitted predict and evaluate raise RuntimeError."""
    frame = pd.DataFrame(np.zeros((3, N_FEATURES)), columns=FEATURES)
    model = SmogPredictorModel()
    with pytest.raises(RuntimeError):
        model.predict(frame)
    with pytest.raises(RuntimeError):
        model.evaluate(frame, np.zeros(3))


def test_train_rejects_bad_inputs(tiny_model_data):
    """train raises TypeError for a non-DataFrame X and ValueError on length mismatch."""
    X = tiny_model_data[FEATURES]
    y = tiny_model_data["target"]
    with pytest.raises(TypeError):
        SmogPredictorModel("linear_regression").train(X.to_numpy(), y)
    with pytest.raises(ValueError):
        SmogPredictorModel("linear_regression").train(X, y.iloc[:-1])


def test_train_returns_self_and_stores_feature_columns(tiny_model_data):
    """Requirements 11.8: train returns self and records the column names."""
    X_train, y_train, _, _ = _split_tiny(tiny_model_data)
    model = SmogPredictorModel("linear_regression")
    result = model.train(X_train, y_train)
    assert result is model
    assert model.is_fitted is True
    assert model.feature_columns == FEATURES


def test_evaluate_returns_mae_rmse_r2_matching_reference(tiny_model_data):
    """Requirements 11.5: evaluate returns MAE, RMSE, R2 equal to utils and sklearn."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    X_train, y_train, X_test, y_test = _split_tiny(tiny_model_data)
    model = SmogPredictorModel("linear_regression").train(X_train, y_train)
    metrics = model.evaluate(X_test, y_test)
    pred = model.predict(X_test)

    assert set(metrics) == {"MAE", "RMSE", "R2"}
    assert metrics["MAE"] == pytest.approx(utils.mae(y_test, pred))
    assert metrics["RMSE"] == pytest.approx(utils.rmse(y_test, pred))
    assert metrics["R2"] == pytest.approx(utils.r2(y_test, pred))
    assert metrics["MAE"] == pytest.approx(mean_absolute_error(y_test, pred))
    assert metrics["RMSE"] == pytest.approx(mean_squared_error(y_test, pred) ** 0.5)
    assert metrics["R2"] == pytest.approx(r2_score(y_test, pred))
    assert metrics["R2"] > 0.9


@pytest.mark.parametrize("aqi", [0, 42, 51, 120, 175, 250, 400])
def test_get_risk_advisory_matches_alerts(aqi):
    """Requirements 11.12: the model advisory is exactly alerts.get_aqi_advisory."""
    from src import alerts

    assert SmogPredictorModel().get_risk_advisory(aqi) == alerts.get_aqi_advisory(aqi)


def test_linear_regression_importance_is_sorted_abs_coef(tiny_model_data):
    """Linear regression importance is |coef_| sorted descending."""
    X_train, y_train, _, _ = _split_tiny(tiny_model_data)
    model = SmogPredictorModel("linear_regression").train(X_train, y_train)
    imp = model.get_feature_importance()

    expected = pd.Series(np.abs(model.estimator.coef_), index=FEATURES)
    expected = expected.sort_values(ascending=False, kind="mergesort")
    assert list(imp["feature"]) == list(expected.index)
    assert np.allclose(imp["importance"].to_numpy(), expected.to_numpy())
    assert (np.diff(imp["importance"].to_numpy()) <= 0).all()


def test_random_forest_predict_is_deterministic(tiny_model_data):
    """Repeated predict calls and two models with the same seed agree exactly."""
    X_train, y_train, X_test, _ = _split_tiny(tiny_model_data)
    first = SmogPredictorModel("random_forest").train(X_train, y_train)
    second = SmogPredictorModel("random_forest").train(X_train, y_train)

    assert np.array_equal(first.predict(X_test), first.predict(X_test))
    assert np.array_equal(first.predict(X_test), second.predict(X_test))
