"""Tests for src.utils."""

import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)

from src import utils

N_CASES = 150


class TestProjectMetrics:
    """Project metrics (mae, rmse, r2) against scikit-learn."""

    def test_metrics_equal_sklearn(self):
        """Feature: dhund-khabar-ai, Property 9: Project metrics equal scikit-learn

        **Validates: Requirements 11.5, 18.7**

        Seeded random cases (seed 42): length 1..500, varying scale, some
        constant arrays, and (n,) vectors mixed with (n, 1) columns.
        """
        rng = np.random.default_rng(42)
        for case in range(N_CASES):
            n = int(rng.integers(1, 501))
            scale = 10.0 ** rng.uniform(-3, 4)
            offset = rng.normal(0, scale * 5)
            y_true = rng.normal(offset, scale, size=n)
            y_pred = y_true + rng.normal(0, scale, size=n)
            if case % 10 == 0:
                y_true = np.full(n, offset)  # constant actuals
            if case % 13 == 0:
                y_pred = np.full(n, offset)  # constant predictions
            if case == 0:
                y_true = y_pred = np.full(n, 3.0)  # identical constants

            t_in, p_in = y_true, y_pred
            if case % 3 == 1:
                t_in = y_true.reshape(-1, 1)
            if case % 3 == 2:
                p_in = y_pred.reshape(-1, 1)
            if case % 6 == 5:
                t_in, p_in = y_true.reshape(-1, 1), y_pred.reshape(-1, 1)

            assert np.isclose(
                utils.mae(t_in, p_in), mean_absolute_error(y_true, y_pred),
                rtol=1e-12, atol=1e-12 * scale,
            ), f"mae case {case}"
            assert np.isclose(
                utils.rmse(t_in, p_in), root_mean_squared_error(y_true, y_pred),
                rtol=1e-12, atol=1e-12 * scale,
            ), f"rmse case {case}"
            assert np.isclose(
                utils.rmse(t_in, p_in),
                np.sqrt(np.mean((y_true - y_pred) ** 2)), rtol=1e-12,
            )
            if n >= 2:
                assert np.isclose(
                    utils.r2(t_in, p_in), r2_score(y_true, y_pred),
                    rtol=1e-12, atol=1e-12,
                ), f"r2 case {case}"

    def test_r2_single_sample_is_nan(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert np.isnan(utils.r2([1.0], [2.0]))

    def test_accepts_lists_and_series(self):
        import pandas as pd
        assert utils.mae([1, 2, 3], [2, 2, 5]) == pytest.approx(1.0)
        assert utils.rmse(pd.Series([0.0, 0.0]), pd.Series([3.0, 4.0])) == \
            pytest.approx(np.sqrt(12.5))

    @pytest.mark.parametrize("fn", [utils.mae, utils.rmse, utils.r2])
    def test_length_mismatch_raises(self, fn):
        with pytest.raises(ValueError, match="same length"):
            fn([1.0, 2.0, 3.0], [1.0, 2.0])

    @pytest.mark.parametrize("fn", [utils.mae, utils.rmse, utils.r2])
    def test_empty_input_raises(self, fn):
        with pytest.raises(ValueError, match="empty"):
            fn([], [])

    @pytest.mark.parametrize("fn", [utils.mae, utils.rmse, utils.r2])
    def test_multi_column_raises(self, fn):
        with pytest.raises(ValueError, match="1-D or a single column"):
            fn(np.zeros((4, 2)), np.zeros(4))

    @pytest.mark.parametrize("fn", [utils.mae, utils.rmse])
    def test_nan_gives_nan(self, fn):
        assert np.isnan(fn([1.0, np.nan, 3.0], [1.0, 2.0, 3.0]))
        assert np.isnan(fn([1.0, 2.0, 3.0], [1.0, np.nan, 3.0]))

    def test_r2_nan_raises(self):
        with pytest.raises(ValueError):
            utils.r2([1.0, np.nan, 3.0], [1.0, 2.0, 3.0])


class TestInversionProxy:
    """Inversion proxy formula, monotonicity, and level thresholds."""

    @staticmethod
    def _expected_level(p):
        if p >= 0.50:
            return "Strong"
        if p >= 0.25:
            return "Moderate"
        return "Weak"

    def test_formula_and_monotonicity(self):
        """Feature: dhund-khabar-ai, Property 17: Inversion proxy formula and monotonicity
        **Validates: Requirements 8.8, 17.9**
        Seeded random scalars and arrays (seed 42): temperature -10..50,
        humidity 0..100, positive increments dT and dRH.
        """
        rng = np.random.default_rng(42)
        for case in range(N_CASES):
            t = float(rng.uniform(-10, 50))
            rh = float(rng.uniform(0, 100))
            d_t = float(rng.uniform(0.01, 20))
            d_rh = float(rng.uniform(0.01, 20))

            got = utils.inversion_proxy(t, rh)
            assert isinstance(got, float), case
            assert np.isclose(got, rh / 100 - t / 40), case
            assert utils.inversion_proxy(t - d_t, rh) > got, case
            assert utils.inversion_proxy(t, rh + d_rh) > got, case

            n = int(rng.integers(1, 50))
            ts = rng.uniform(-10, 50, n)
            rhs = rng.uniform(0, 100, n)
            d_ts = rng.uniform(0.01, 20, n)
            d_rhs = rng.uniform(0.01, 20, n)
            arr = utils.inversion_proxy(ts, rhs)
            assert isinstance(arr, np.ndarray) and arr.shape == (n,), case
            assert np.allclose(arr, rhs / 100 - ts / 40), case
            assert np.all(utils.inversion_proxy(ts - d_ts, rhs) > arr), case
            assert np.all(utils.inversion_proxy(ts, rhs + d_rhs) > arr), case

    def test_level_thresholds_scalar_and_array(self):
        """Feature: dhund-khabar-ai, Property 17: Inversion proxy formula and monotonicity
        **Validates: Requirements 8.8, 17.9**
        Levels: Strong >= 0.50, Moderate 0.25 <= p < 0.50, Weak otherwise,
        including the exact boundaries and the float just below each.
        """
        edges = [
            0.25, float(np.nextafter(0.25, 0)), float(np.nextafter(0.25, 1)),
            0.50, float(np.nextafter(0.50, 0)), float(np.nextafter(0.50, 1)),
        ]
        rng = np.random.default_rng(42)
        values = np.concatenate([edges, rng.uniform(-1.5, 1.5, N_CASES)])
        expected = [self._expected_level(float(v)) for v in values]

        assert utils.inversion_level(0.25) == "Moderate"
        assert utils.inversion_level(0.50) == "Strong"
        assert utils.inversion_level(float(np.nextafter(0.25, 0))) == "Weak"
        assert utils.inversion_level(float(np.nextafter(0.50, 0))) == "Moderate"

        for v, e in zip(values, expected):
            got = utils.inversion_level(float(v))
            assert isinstance(got, str)
            assert got == e, v
        arr = utils.inversion_level(values)
        assert arr.shape == values.shape
        assert list(arr) == expected


class TestSliderClampAndScenarioRow:
    """clamp_slider_default and build_scenario_row."""

    def test_clamp_slider_default(self):
        """Feature: dhund-khabar-ai, Property 19: Slider default clamping and scenario rows
        **Validates: Requirements 17.10, 17.11, 17.12**
        """
        rng = np.random.default_rng(42)
        ranges = list(utils.SLIDER_RANGES.values())
        assert ranges == [(0.0, 30.0), (5.0, 35.0), (20.0, 100.0)]
        for case in range(N_CASES):
            if case % 2 == 0:
                low, high = ranges[case % 3]
            else:
                low = float(rng.uniform(-50, 50))
                high = low + float(rng.uniform(0.1, 80))
            value = float(rng.uniform(low - 40, high + 40))
            clipped, was = utils.clamp_slider_default(value, low, high)
            assert clipped == min(max(value, low), high)
            assert was is (value < low or value > high)
            assert low <= clipped <= high
        for low, high in ranges:
            for edge in (low, high):  # boundary values are not clamped
                assert utils.clamp_slider_default(edge, low, high) == (edge, False)
            assert utils.clamp_slider_default(low - 1, low, high) == (low, True)
            assert utils.clamp_slider_default(high + 1, low, high) == (high, True)
        for bad in [(np.nan, 0, 1), (np.inf, 0, 1), (-np.inf, 0, 1), (0.5, np.nan, 1),
                    (0.5, 0, np.inf), (0.5, 2.0, 1.0)]:
            with pytest.raises(ValueError):
                utils.clamp_slider_default(*bad)

    @staticmethod
    def _cols():
        from src.pipeline import AirQualityDataPipeline
        return AirQualityDataPipeline().get_feature_columns("us_aqi", "simulation")

    @staticmethod
    def _random_base(rng, cols):
        base = {c: float(rng.uniform(-100, 500)) for c in cols}
        base["extra_col"] = float(rng.normal())  # not a feature column
        return pd.Series(base)

    def test_build_scenario_row(self):
        """Feature: dhund-khabar-ai, Property 19: Slider default clamping and scenario rows
        **Validates: Requirements 17.10, 17.11, 17.12**
        """
        rng = np.random.default_rng(42)
        cols = self._cols()
        assert len(cols) == 15
        sliders = ("wind_speed_10m", "temperature_2m", "relative_humidity_2m")
        derived = set(sliders) | {"stagnation_flag", "inversion_proxy"}
        for case in range(N_CASES):
            base = self._random_base(rng, cols)
            snapshot = base.copy()
            wind = float(rng.uniform(0, 30))
            temp = float(rng.uniform(5, 35))
            rh = float(rng.uniform(20, 100))
            if case % 10 == 0:
                wind = 5.0  # stagnation boundary: not stagnant
            row = utils.build_scenario_row(base, wind, temp, rh, cols)
            assert isinstance(row, pd.DataFrame)
            assert list(row.columns) == cols
            assert len(row) == 1
            r = row.iloc[0]
            assert r["wind_speed_10m"] == wind
            assert r["temperature_2m"] == temp
            assert r["relative_humidity_2m"] == rh
            assert r["stagnation_flag"] == float(wind < 5.0)
            assert np.isclose(r["inversion_proxy"], rh / 100 - temp / 40)
            for c in cols:
                if c not in derived:
                    assert r[c] == base[c], c
            pd.testing.assert_series_equal(base, snapshot)  # not mutated

            frame = base.to_frame().T
            frame_snapshot = frame.copy()
            row2 = utils.build_scenario_row(frame, wind, temp, rh, cols)
            pd.testing.assert_frame_equal(row, row2)
            pd.testing.assert_frame_equal(frame, frame_snapshot)

    def test_build_scenario_row_stagnation_values(self):
        """Feature: dhund-khabar-ai, Property 19: Slider default clamping and scenario rows
        **Validates: Requirements 17.12**
        """
        cols = self._cols()
        base = self._random_base(np.random.default_rng(42), cols)
        low = utils.build_scenario_row(base, 4.99, 20.0, 50.0, cols).iloc[0]
        edge = utils.build_scenario_row(base, 5.0, 20.0, 50.0, cols).iloc[0]
        assert low["stagnation_flag"] == 1
        assert edge["stagnation_flag"] == 0

    def test_build_scenario_row_invalid_inputs(self):
        """Feature: dhund-khabar-ai, Property 19: Slider default clamping and scenario rows
        **Validates: Requirements 17.10, 17.12**
        """
        rng = np.random.default_rng(42)
        cols = self._cols()
        base = self._random_base(rng, cols)
        for c in ("pm2_5_rolling_mean_24h", "surface_pressure", "hour_sin"):
            with pytest.raises(ValueError):
                utils.build_scenario_row(base.drop(c), 10.0, 20.0, 50.0, cols)
        with pytest.raises(ValueError):
            utils.build_scenario_row(pd.concat([base.to_frame().T] * 2), 10.0, 20.0, 50.0, cols)
        with pytest.raises(ValueError):
            utils.build_scenario_row(base.to_frame().T.iloc[0:0], 10.0, 20.0, 50.0, cols)
        for sliders in [(np.nan, 20.0, 50.0), (10.0, np.inf, 50.0), (10.0, 20.0, -np.inf)]:
            with pytest.raises(ValueError):
                utils.build_scenario_row(base, *sliders, cols)


def _expected_tier(values: np.ndarray) -> np.ndarray:
    """Tier index with hardcoded bounds 50/100/150/200/300, lower < AQI <= upper."""
    bounds = [50, 100, 150, 200, 300]
    return np.array([sum(v > b for b in bounds) for v in values])


class TestTierMetricsProperty:
    """Property 20 for tier_accuracy and unhealthy_precision_recall."""

    def test_tier_metrics_match_definitions(self):
        """Feature: dhund-khabar-ai, Property 20: Tier metrics match their definitions

        **Validates: Requirements 13.5, 13.6**
        """
        from sklearn.metrics import precision_score, recall_score

        rng = np.random.default_rng(42)
        for _ in range(N_CASES):
            n = int(rng.integers(1, 80))
            actual = rng.uniform(0, 600, n)
            pred = rng.uniform(-100, 600, n)
            # Force values onto tier boundaries in some cases.
            if rng.random() < 0.5:
                idx = rng.integers(0, n, size=min(n, 5))
                actual[idx] = rng.choice([0, 50, 100, 150, 200, 300], size=len(idx))
                pred[idx] = rng.choice([0, 50, 100, 150, 150.01, 200, 300], size=len(idx))

            exp_acc = np.mean(
                _expected_tier(actual) == _expected_tier(np.clip(pred, 0, None))
            )
            assert utils.tier_accuracy(actual, pred) == pytest.approx(exp_acc)

            true_ev = actual > 150
            pred_ev = pred > 150
            tp = int(np.sum(true_ev & pred_ev))
            exp_p = tp / pred_ev.sum() if pred_ev.sum() else np.nan
            exp_r = tp / true_ev.sum() if true_ev.sum() else np.nan
            prec, rec = utils.unhealthy_precision_recall(actual, pred)
            if np.isnan(exp_p):
                assert np.isnan(prec)
            else:
                assert prec == pytest.approx(exp_p)
                assert prec == pytest.approx(
                    precision_score(true_ev, pred_ev, zero_division=0)
                )
            if np.isnan(exp_r):
                assert np.isnan(rec)
            else:
                assert rec == pytest.approx(exp_r)
                assert rec == pytest.approx(
                    recall_score(true_ev, pred_ev, zero_division=0)
                )

    def test_150_is_not_an_event_but_150_01_is(self):
        prec, rec = utils.unhealthy_precision_recall([150.0, 150.01], [150.0, 150.01])
        assert prec == 1.0 and rec == 1.0
        prec, rec = utils.unhealthy_precision_recall([150.0], [150.0])
        assert np.isnan(prec) and np.isnan(rec)
        prec, rec = utils.unhealthy_precision_recall([150.01], [150.0])
        assert np.isnan(prec) and rec == 0.0

    def test_all_correct_predictions(self):
        actual = np.array([10.0, 75.0, 160.0, 250.0, 400.0])
        assert utils.tier_accuracy(actual, actual) == 1.0
        assert utils.unhealthy_precision_recall(actual, actual) == (1.0, 1.0)

    def test_no_events_anywhere_gives_nan(self):
        prec, rec = utils.unhealthy_precision_recall([10.0, 80.0], [20.0, 90.0])
        assert np.isnan(prec) and np.isnan(rec)

    def test_negative_predictions_clipped_to_good(self):
        assert utils.tier_accuracy([10.0], [-30.0]) == 1.0

    def test_invalid_inputs_raise(self):
        for fn in (utils.tier_accuracy, utils.unhealthy_precision_recall):
            with pytest.raises(ValueError, match="same length"):
                fn([1.0, 2.0], [1.0])
            with pytest.raises(ValueError, match="empty"):
                fn([], [])
            with pytest.raises(ValueError, match="NaN"):
                fn([1.0, 2.0], [1.0, float("nan")])


class TestUtilsExamples:
    """Plain example tests with literal expected values.

    **Validates: Requirements 5.10, 8.7, 13.6**
    """

    @pytest.mark.parametrize("conc, expected", [
        (0.0, 0.0), (9.0, 50.0), (9.1, 51.0), (35.4, 100.0), (35.5, 101.0),
        (55.4, 150.0), (55.5, 151.0), (125.4, 200.0), (125.5, 201.0),
        (225.4, 300.0), (225.5, 301.0), (325.4, 500.0),
        (400.0, 500.0), (1000.0, 500.0),
    ])
    def test_pm25_sub_index_2024_endpoints(self, conc, expected):
        assert utils.pm25_sub_index(np.array([conc]), "2024")[0] == pytest.approx(expected)

    @pytest.mark.parametrize("conc, expected", [
        (0.0, 0.0), (12.0, 50.0), (12.1, 51.0), (35.4, 100.0), (35.5, 101.0),
        (55.4, 150.0), (55.5, 151.0), (150.4, 200.0), (150.5, 201.0),
        (250.4, 300.0), (250.5, 301.0), (500.4, 500.0), (600.0, 500.0),
    ])
    def test_pm25_sub_index_2012_endpoints(self, conc, expected):
        assert utils.pm25_sub_index(np.array([conc]), "2012")[0] == pytest.approx(expected)

    def test_pm25_sub_index_negative_and_nan_give_nan(self):
        out = utils.pm25_sub_index(np.array([-0.1, np.nan]), "2024")
        assert np.isnan(out).all()

    def test_pm25_sub_index_unknown_table_raises(self):
        with pytest.raises(ValueError):
            utils.pm25_sub_index(np.array([10.0]), "1999")

    def test_precision_recall_nan_without_positives(self):
        p, r = utils.unhealthy_precision_recall([10, 50, 100], [20, 60, 90])
        assert np.isnan(p) and np.isnan(r)

    def test_precision_nan_when_nothing_predicted_positive(self):
        p, r = utils.unhealthy_precision_recall([200, 50], [100, 50])
        assert np.isnan(p)
        assert r == 0.0

    def test_recall_nan_when_nothing_actually_positive(self):
        p, r = utils.unhealthy_precision_recall([50, 100], [200, 50])
        assert p == 0.0
        assert np.isnan(r)

    def test_precision_recall_with_positives(self):
        p, r = utils.unhealthy_precision_recall([200, 200, 50, 50], [200, 100, 200, 50])
        assert p == pytest.approx(0.5)
        assert r == pytest.approx(0.5)

    @pytest.mark.parametrize("wind, expected", [
        (4.99, 1), (5.0, 0), (0.0, 1), (float("nan"), 0), (12.0, 0),
    ])
    def test_stagnation_flag_scalar(self, wind, expected):
        out = utils.stagnation_flag(wind)
        assert isinstance(out, int)
        assert out == expected

    def test_stagnation_flag_array(self):
        out = utils.stagnation_flag(np.array([4.99, 5.0, 0.0, np.nan]))
        assert isinstance(out, np.ndarray)
        assert out.tolist() == [1, 0, 1, 0]

    @pytest.mark.parametrize("proxy, expected", [
        (0.25, "Moderate"), (0.50, "Strong"), (0.2499, "Weak"),
        (0.4999, "Moderate"), (-0.65, "Weak"), (0.70, "Strong"),
    ])
    def test_inversion_level_boundaries(self, proxy, expected):
        assert utils.inversion_level(proxy) == expected

    def test_inversion_level_array(self):
        out = utils.inversion_level(np.array([0.1, 0.25, 0.5]))
        assert out.tolist() == ["Weak", "Moderate", "Strong"]

    def test_model_path_format(self):
        path = utils.model_path("Peshawar", "us_aqi", 24, "forecast", "random_forest")
        assert path.name == "peshawar_us_aqi_h24_forecast_random_forest.joblib"
        assert path.parent == utils.MODELS_DIR

    def test_model_path_lowercases_mixed_case_city(self):
        path = utils.model_path("LaHoRe", "pm2_5", 72, "simulation", "gradient_boosting")
        assert path.name == "lahore_pm2_5_h72_simulation_gradient_boosting.joblib"

    def test_dew_point_example(self):
        assert utils.dew_point_c(20.0, 50.0) == pytest.approx(9.26, abs=0.02)

    def test_dew_point_nan_for_zero_humidity(self):
        assert np.isnan(utils.dew_point_c(20.0, 0.0))

    def test_required_columns(self):
        cols = utils.REQUIRED_COLUMNS
        assert len(cols) == 26
        assert len(set(cols)) == 26
        assert cols.index("pm10") < cols.index("pm2_5")
