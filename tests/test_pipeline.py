"""Property tests for the data pipeline (synthetic data only, seeded cases)."""

import numpy as np
import pandas as pd
import pytest

from src.pipeline import AirQualityDataPipeline
from src.utils import REQUIRED_COLUMNS
from tests.conftest import build_features_frame, make_synthetic_master

N_CASES = 120


def test_load_validation_ordering_and_city_filter(tmp_path):
    """Feature: dhund-khabar-ai, Property 18: Load validation, ordering, and city filter

    **Validates: Requirements 6.3, 6.4, 6.6, 18.1**

    For random column subsets removed, load_data raises ValueError naming every
    removed column in file order. For random row shuffles, load_data returns
    rows sorted by city then datetime, and filter_city returns exactly the rows
    of that city; an unknown city raises ValueError listing available cities.
    """
    rng = np.random.default_rng(42)
    pipe_flags = tmp_path / "no_such_flags.json"
    csv = tmp_path / "case.csv"

    for case in range(N_CASES):
        n_hours = int(rng.integers(3, 9))
        master = make_synthetic_master(n_hours, seed=int(rng.integers(0, 10_000)))
        # Keep at least one column: a CSV with no columns at all is an empty
        # file, which pandas rejects before the column check runs.
        n_missing = int(rng.integers(1, len(REQUIRED_COLUMNS)))
        drop_idx = rng.choice(len(REQUIRED_COLUMNS), size=n_missing, replace=False)
        dropped = [REQUIRED_COLUMNS[i] for i in sorted(drop_idx)]
        shuffled = master.sample(frac=1.0, random_state=int(rng.integers(0, 10_000)))
        shuffled = shuffled.reset_index(drop=True)

        # Part 1: missing columns are all named, in file order.
        shuffled.drop(columns=dropped).to_csv(csv, index=False)
        pipe = AirQualityDataPipeline(flags_path=pipe_flags)
        with pytest.raises(ValueError) as exc:
            pipe.load_data(csv)
        msg = str(exc.value)
        assert all(col in msg for col in dropped), (case, dropped, msg)
        assert msg.endswith(", ".join(dropped)), (case, dropped, msg)

        # Part 2: shuffled valid file loads sorted by city then datetime.
        shuffled.to_csv(csv, index=False)
        pipe = AirQualityDataPipeline(flags_path=pipe_flags)
        loaded = pipe.load_data(csv)
        expected = master.assign(datetime=pd.to_datetime(master["datetime"]))
        expected = expected.sort_values(["city", "datetime"]).reset_index(drop=True)
        assert len(loaded) == len(expected), case
        assert loaded["city"].tolist() == expected["city"].tolist(), case
        assert loaded["datetime"].tolist() == expected["datetime"].tolist(), case
        key = list(zip(loaded["city"], loaded["datetime"]))
        assert key == sorted(key), case

        # Part 3: filter_city returns exactly that city's rows.
        for city in expected["city"].unique():
            sub = pipe.filter_city(city)
            want = loaded[loaded["city"] == city].reset_index(drop=True)
            assert (sub["city"] == city).all(), (case, city)
            assert len(sub) == n_hours, (case, city)
            pd.testing.assert_frame_equal(sub, want)

        with pytest.raises(ValueError) as exc:
            pipe.filter_city("Atlantis")
        for city in expected["city"].unique():
            assert city in str(exc.value), (case, city)


def test_no_cross_city_leakage(tmp_path):
    """Feature: dhund-khabar-ai, Property 1: No cross-city leakage

    **Validates: Requirements 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 18.2**

    For random two-city frames with very different levels, the first k rows of
    the second city are NaN for every k-hour lag, the first w-1 rows of each
    city are NaN for rolling features of window w, and every lag, rolling, and
    change feature on the combined frame equals the same feature computed on
    each city's rows alone.
    """
    from src.pipeline import ROLLING_FEATURES

    rng = np.random.default_rng(42)
    pipe = AirQualityDataPipeline(flags_path=tmp_path / "no_such_flags.json")
    csv = tmp_path / "case.csv"
    lag_cols = {f"pm2_5_lag_{k}": k for k in (1, 3, 6, 12, 24, 48)}
    lag_cols.update({"us_aqi_lag_1": 1, "us_aqi_lag_24": 24})
    feature_cols = (list(lag_cols) + [r[0] for r in ROLLING_FEATURES]
                    + ["pm2_5_change_24h"])

    for case in range(N_CASES):
        n_hours = int(rng.integers(80, 201))
        names = ["Lahore", "Peshawar"]
        rng.shuffle(names)
        offsets = {names[0]: float(rng.uniform(0, 50)),
                   names[1]: float(rng.uniform(300, 600))}
        master = make_synthetic_master(
            n_hours, seed=int(rng.integers(0, 10_000)), city_offsets=offsets)
        master.to_csv(csv, index=False)
        prepared = pipe.generate_time_features(pipe.load_data(csv))
        combined = pipe.create_lag_and_rolling_features(prepared)

        cities = sorted(prepared["city"].unique())  # load_data order
        for pos, city in enumerate(cities):
            rows = combined[combined["city"] == city].reset_index(drop=True)
            assert len(rows) == n_hours, (case, city)

            # Lags: the first k rows of a city are NaN, so the second city
            # never sees the first city's tail.
            for col, k in lag_cols.items():
                assert rows[col].iloc[:k].isna().all(), (case, city, col)
                assert rows[col].iloc[k:].notna().all(), (case, city, col)
            # Rolling windows: first w-1 rows NaN, and the w-th row is valid.
            for col, src, w, _stat in ROLLING_FEATURES:
                assert rows[col].iloc[:w - 1].isna().all(), (case, city, col)
                assert rows[col].iloc[w - 1:].notna().all(), (case, city, col)
            assert rows["pm2_5_change_24h"].iloc[:24].isna().all(), (case, city)

            # Combined frame equals the city computed alone.
            alone_in = prepared[prepared["city"] == city].reset_index(drop=True)
            alone = pipe.create_lag_and_rolling_features(alone_in)
            pd.testing.assert_frame_equal(rows, alone)
            pd.testing.assert_frame_equal(rows[feature_cols], alone[feature_cols])

        # Second city's lag values come from its own (much higher) level.
        second = combined[combined["city"] == cities[1]].reset_index(drop=True)
        lvl_gap = (second["pm2_5"].mean()
                   - combined.loc[combined["city"] == cities[0], "pm2_5"].mean())
        assert lvl_gap != 0, case
        valid = second["pm2_5_lag_1"].dropna()
        assert np.allclose(valid.to_numpy(), second["pm2_5"].iloc[:-1].to_numpy()), case


def test_past_only_lag_and_rolling_features(tmp_path):
    """Feature: dhund-khabar-ai, Property 2: Past-only lag and rolling features

    **Validates: Requirements 8.9, 18.3**

    For random two-city frames and a random cut row t per city, replacing every
    source value after t with new random values, or dropping those rows, leaves
    every lag, rolling, change, stagnation, and inversion feature at rows 0..t
    unchanged. Lags equal the value k rows earlier in the same city, and
    rolling mean/std equal an independent trailing-window computation.
    """
    from src.pipeline import ROLLING_FEATURES

    rng = np.random.default_rng(42)
    pipe = AirQualityDataPipeline(flags_path=tmp_path / "no_such_flags.json")
    csv = tmp_path / "case.csv"
    lag_cols = {f"pm2_5_lag_{k}": ("pm2_5", k) for k in (1, 3, 6, 12, 24, 48)}
    lag_cols.update({"us_aqi_lag_1": ("us_aqi", 1), "us_aqi_lag_24": ("us_aqi", 24)})
    feature_cols = (list(lag_cols) + [r[0] for r in ROLLING_FEATURES]
                    + ["pm2_5_change_24h", "stagnation_flag", "inversion_proxy"])
    source_cols = ["pm2_5", "us_aqi", "wind_speed_10m", "relative_humidity_2m",
                   "temperature_2m"]

    for case in range(N_CASES):
        n_hours = int(rng.integers(80, 201))
        master = make_synthetic_master(n_hours, seed=int(rng.integers(0, 10_000)))
        master.to_csv(csv, index=False)
        prepared = pipe.generate_time_features(pipe.load_data(csv))
        base = pipe.create_lag_and_rolling_features(prepared)

        # Per-city cut row, then perturb or truncate everything after it.
        cuts = {c: int(rng.integers(0, n_hours)) for c in prepared["city"].unique()}
        pos = prepared.groupby("city").cumcount().to_numpy()
        cut_arr = prepared["city"].map(cuts).to_numpy()
        after = pos > cut_arr

        perturbed = prepared.copy()
        for col in source_cols:
            scale = float(prepared[col].abs().mean()) + 1.0
            perturbed.loc[after, col] = rng.uniform(0, 3 * scale, size=int(after.sum()))
        pert = pipe.create_lag_and_rolling_features(perturbed)
        trunc = pipe.create_lag_and_rolling_features(
            prepared.loc[~after].reset_index(drop=True))

        keep = ~after
        want = base.loc[keep, feature_cols].reset_index(drop=True)
        pd.testing.assert_frame_equal(
            pert.loc[keep, feature_cols].reset_index(drop=True), want)
        pd.testing.assert_frame_equal(trunc[feature_cols], want)

        # Independent checks of lag and trailing-window values.
        for city, grp in base.groupby("city"):
            grp = grp.reset_index(drop=True)
            for col, (src, k) in lag_cols.items():
                vals = grp[src].to_numpy(dtype=float)
                exp = np.full(len(grp), np.nan)
                exp[k:] = vals[:-k]
                np.testing.assert_allclose(grp[col].to_numpy(dtype=float), exp,
                                           equal_nan=True, err_msg=f"{case} {city} {col}")
            for col, src, w, stat in ROLLING_FEATURES:
                vals = grp[src].to_numpy(dtype=float)
                exp = np.full(len(grp), np.nan)
                for i in range(w - 1, len(grp)):
                    win = vals[i - w + 1:i + 1]
                    exp[i] = win.mean() if stat == "mean" else win.std(ddof=1)
                np.testing.assert_allclose(grp[col].to_numpy(dtype=float), exp,
                                           rtol=1e-9, atol=1e-9, equal_nan=True,
                                           err_msg=f"{case} {city} {col}")


def test_cyclical_time_features_bounded_and_consistent(tmp_path):
    """Feature: dhund-khabar-ai, Property 4: Cyclical time features are bounded and consistent
    **Validates: Requirements 7.1, 7.2, 18.5**

    For random start dates and lengths, hour_sin/hour_cos/month_sin/month_cos
    lie in [-1, 1], equal the formulas computed independently from the
    timestamps, satisfy sin^2 + cos^2 = 1, contain no NaN, and the input frame
    is not mutated. Fixed points: hour 0 -> (0, 1), month 12 -> (~0, 1),
    month 3 -> sin 1.
    """
    rng = np.random.default_rng(42)
    pipe = AirQualityDataPipeline(flags_path=tmp_path / "no_such_flags.json")
    cols = ["hour_sin", "hour_cos", "month_sin", "month_cos"]
    base = pd.Timestamp("2018-01-01")
    for case in range(N_CASES):
        start = base + pd.Timedelta(hours=int(rng.integers(0, 24 * 365 * 10)))
        n_rows = int(rng.integers(1, 60))
        # Random step so frames cross day and month boundaries.
        step = int(rng.integers(1, 400))
        stamps = pd.date_range(start, periods=n_rows, freq=f"{step}h")
        frame = pd.DataFrame({"datetime": stamps, "pm2_5": rng.random(n_rows)})
        before = frame.copy(deep=True)

        out = pipe.generate_time_features(frame)

        pd.testing.assert_frame_equal(frame, before)
        assert not out[cols].isna().any().any()
        hour = np.array([t.hour for t in stamps], dtype=float)
        month = np.array([t.month for t in stamps], dtype=float)
        expected = {
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "month_sin": np.sin(2 * np.pi * month / 12),
            "month_cos": np.cos(2 * np.pi * month / 12),
        }
        for col in cols:
            vals = out[col].to_numpy()
            assert np.all(vals >= -1.0) and np.all(vals <= 1.0), (case, col)
            np.testing.assert_allclose(vals, expected[col], atol=1e-12, rtol=0)
        assert np.allclose(out["hour_sin"] ** 2 + out["hour_cos"] ** 2, 1.0, atol=1e-12)
        assert np.allclose(out["month_sin"] ** 2 + out["month_cos"] ** 2, 1.0, atol=1e-12)

    # Fixed points: hour 0 (and its 24-equivalent), month 12 (0-equivalent), month 3.
    fixed = pd.DataFrame({"datetime": pd.to_datetime(
        ["2024-12-01 00:00:00", "2024-03-15 06:00:00", "2025-01-01 12:00:00"])})
    out = pipe.generate_time_features(fixed)
    assert out["hour_sin"].iloc[0] == pytest.approx(0.0, abs=1e-12)
    assert out["hour_cos"].iloc[0] == pytest.approx(1.0, abs=1e-12)
    assert out["month_sin"].iloc[0] == pytest.approx(0.0, abs=1e-12)
    assert out["month_cos"].iloc[0] == pytest.approx(1.0, abs=1e-12)
    assert out["month_sin"].iloc[1] == pytest.approx(1.0, abs=1e-12)
    assert out["hour_sin"].iloc[1] == pytest.approx(1.0, abs=1e-12)
    assert out["hour_cos"].iloc[2] == pytest.approx(-1.0, abs=1e-12)


def test_calendar_flags_match_timestamp(tmp_path):
    """Feature: dhund-khabar-ai, Property 5: Calendar flags match the timestamp
    **Validates: Requirements 7.3, 7.4**

    For random start dates and spans that cross month and year boundaries,
    is_weekend is 1 exactly when the weekday is Saturday or Sunday (checked with
    datetime.weekday(), even when the incoming is_weekend column is wrong), and
    is_winter is 1 exactly for months 11, 12, 1, and 2. Both flags are 0/1 ints
    with no NaN.
    """
    rng = np.random.default_rng(42)
    pipe = AirQualityDataPipeline(flags_path=tmp_path / "no_such_flags.json")
    months_seen = set()
    for case in range(N_CASES):
        start = pd.Timestamp(
            year=int(rng.integers(2015, 2031)),
            month=int(rng.integers(1, 13)),
            day=int(rng.integers(1, 29)),
            hour=int(rng.integers(0, 24)),
        )
        # Spans up to ~70 days cross month boundaries; some cross a year end.
        n_rows = int(rng.integers(1, 40))
        step_h = int(rng.integers(1, 100))
        stamps = [start + pd.Timedelta(i * step_h, unit="h") for i in range(n_rows)]
        # The incoming flag is random garbage, as if the file were wrong.
        wrong = rng.integers(0, 2, size=n_rows)
        frame = pd.DataFrame({"datetime": pd.to_datetime(stamps), "is_weekend": wrong})

        out = pipe.generate_time_features(frame)

        for col in ("is_weekend", "is_winter"):
            assert not out[col].isna().any(), f"case {case}: NaN in {col}"
            assert out[col].dtype.kind == "i", f"case {case}: {col} not int"
            assert set(out[col].unique()) <= {0, 1}
        for ts, wk, wn in zip(stamps, out["is_weekend"], out["is_winter"]):
            py = ts.to_pydatetime()
            assert wk == (1 if py.weekday() >= 5 else 0), f"case {case}: {ts}"
            assert wn == (1 if py.month in (11, 12, 1, 2) else 0), f"case {case}: {ts}"
            months_seen.add(py.month)
    assert months_seen == set(range(1, 13))


def _parsed_master(n_hours: int, seed: int) -> pd.DataFrame:
    """Synthetic two-city frame sorted by city, datetime, with parsed datetimes."""
    df = make_synthetic_master(n_hours, seed=seed)
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df.sort_values(["city", "datetime"]).reset_index(drop=True)


def test_target_shift_correctness(tmp_path):
    """Feature: dhund-khabar-ai, Property 3: Target shift correctness
    **Validates: Requirements 9.1, 9.2, 18.4**

    For random horizon lists and frame sizes, every kept row's targets equal
    the same city's value h hours later (checked by an independent join on
    (city, datetime + h)), no target is NaN, the dropped rows are exactly the
    last max(h) rows per city, and non-hourly steps raise ValueError naming
    the city and first bad timestamp.
    """
    rng = np.random.default_rng(42)
    horizon_choices = [[24], [48], [72], [24, 48, 72], [6], [12, 24], [1, 3]]
    pipe = AirQualityDataPipeline(flags_path=tmp_path / "no_such_flags.json")
    for case in range(N_CASES):
        horizons = horizon_choices[int(rng.integers(0, len(horizon_choices)))]
        hmax = max(horizons)
        n_hours = int(rng.integers(hmax + 5, 251))
        df = _parsed_master(n_hours, seed=int(rng.integers(0, 10_000)))
        before = df.copy()
        out = pipe.create_targets(df, horizons)
        pd.testing.assert_frame_equal(df, before)  # input untouched

        target_cols = [f"{t}_t_plus_{h}" for h in horizons for t in ("pm2_5", "us_aqi")]
        assert not out[target_cols].isna().any().any(), case

        # Dropped rows: exactly the last hmax rows per city.
        expected_keep = n_hours - hmax
        counts = out.groupby("city").size().to_dict()
        assert counts == {"Peshawar": expected_keep, "Lahore": expected_keep}, case
        for city, grp in df.groupby("city"):
            kept = out.loc[out["city"] == city, "datetime"]
            assert kept.tolist() == grp["datetime"].iloc[:expected_keep].tolist(), case
        assert pipe.target_drop_counts == {"Peshawar": hmax, "Lahore": hmax}, case

        # Independent join on (city, datetime + h).
        for h in horizons:
            future = df[["city", "datetime", "pm2_5", "us_aqi"]].copy()
            future["datetime"] = future["datetime"] - pd.Timedelta(h, unit="h")
            future = future.rename(columns={"pm2_5": "f_pm", "us_aqi": "f_aqi"})
            merged = out[["city", "datetime", f"pm2_5_t_plus_{h}",
                          f"us_aqi_t_plus_{h}"]].merge(
                future, on=["city", "datetime"], how="left", validate="one_to_one")
            assert merged["f_pm"].notna().all(), case
            np.testing.assert_array_equal(merged[f"pm2_5_t_plus_{h}"], merged["f_pm"])
            np.testing.assert_array_equal(merged[f"us_aqi_t_plus_{h}"], merged["f_aqi"])
            # Cities sit ~100 apart, so a cross-city value would be far off.
            low = out["city"] == "Peshawar"
            assert out.loc[low, f"pm2_5_t_plus_{h}"].max() < \
                out.loc[~low, f"pm2_5_t_plus_{h}"].min(), case

        # Non-hourly steps: a removed row, then a duplicated row.
        city = ["Peshawar", "Lahore"][int(rng.integers(0, 2))]
        pos = np.flatnonzero((df["city"] == city).to_numpy())
        k = int(rng.integers(1, len(pos) - 1))
        removed = df.drop(index=pos[k]).reset_index(drop=True)
        bad_stamp = df["datetime"].iloc[pos[k + 1]]
        with pytest.raises(ValueError) as exc:
            pipe.create_targets(removed, horizons)
        assert city in str(exc.value), case
        assert f"{bad_stamp:%Y-%m-%d %H:%M:%S}" in str(exc.value), case

        dup = pd.concat([df.iloc[:pos[k] + 1], df.iloc[[pos[k]]],
                         df.iloc[pos[k] + 1:]]).reset_index(drop=True)
        dup_stamp = df["datetime"].iloc[pos[k]]
        with pytest.raises(ValueError) as exc:
            pipe.create_targets(dup, horizons)
        assert city in str(exc.value), case
        assert f"{dup_stamp:%Y-%m-%d %H:%M:%S}" in str(exc.value), case


def test_split_ordering_and_period_membership(tmp_path):
    """Feature: dhund-khabar-ai, Property 6: Split ordering and period membership
    **Validates: Requirements 2.5, 9.3, 9.4, 9.5, 18.6**

    For seeded two-city frames that start 6-10 days before 2025-01-01 and end
    3-10 days after it, with horizon 24/48/72/None and the default or a random
    split date inside the frame: train is strictly before the split, test is
    at or after it, the two sets share no (city, datetime) key, neither
    contains new rows or warm-up NaN, test starts at the split when the split
    row is usable, horizon None equals 72, and each city is in each set
    exactly when the input has a row that qualifies for it.
    """
    from src.utils import SPLIT_DATE

    rng = np.random.default_rng(42)
    pipe = AirQualityDataPipeline(flags_path=tmp_path / "no_such_flags.json")
    cache: dict[tuple[int, int, int], pd.DataFrame] = {}
    default_split = pd.Timestamp(SPLIT_DATE)
    default_checked = 0
    both_nonempty = 0

    def key_set(frame: pd.DataFrame) -> set[tuple[str, pd.Timestamp]]:
        return set(zip(frame["city"], frame["datetime"]))

    for case in range(N_CASES):
        pre_days = int(rng.choice([6, 8, 10]))
        post_days = int(rng.choice([3, 6, 10]))
        start_hour = int(rng.choice([0, 11]))
        ck = (pre_days, post_days, start_hour)
        if ck not in cache:
            start = default_split - pd.Timedelta(days=pre_days) + pd.Timedelta(hours=start_hour)
            end = default_split + pd.Timedelta(days=post_days)
            n_hours = int((end - start) / pd.Timedelta(hours=1))
            raw = make_synthetic_master(n_hours, start=str(start), seed=int(rng.integers(0, 10_000)))
            cache[ck] = build_features_frame(raw, tmp_path)
        feats = cache[ck]
        warm_cols = pipe._warmup_columns(feats)
        assert warm_cols, case
        usable = feats.loc[~feats[warm_cols].isna().any(axis=1)]
        lo, hi = usable["datetime"].min(), feats["datetime"].max()

        if rng.random() < 0.4:
            split = default_split
            default_checked += 1
        else:
            span_h = int((hi - lo) / pd.Timedelta(hours=1))
            split = lo + pd.Timedelta(hours=int(rng.integers(1, span_h)))
        horizon = [24, 48, 72, None][int(rng.integers(0, 4))]
        h = 72 if horizon is None else horizon

        before = feats.copy()
        result = pipe.prepare_train_test_split(feats, split, horizon=horizon)
        pd.testing.assert_frame_equal(feats, before)  # input untouched
        assert isinstance(result, tuple) and len(result) == 2, case
        train, test = result
        assert isinstance(train, pd.DataFrame) and isinstance(test, pd.DataFrame), case

        # Membership and ordering.
        assert (train["datetime"] < split).all(), case
        assert (test["datetime"] >= split).all(), case
        if len(train) and len(test):
            both_nonempty += 1
            assert train["datetime"].max() < test["datetime"].min(), case
        assert (train["datetime"] + pd.Timedelta(hours=h) < split).all(), case

        # No overlap, no new rows, no duplicate keys inside a set.
        train_keys, test_keys = key_set(train), key_set(test)
        assert len(train_keys) == len(train) and len(test_keys) == len(test), case
        assert not (train_keys & test_keys), case
        assert (train_keys | test_keys) <= key_set(feats), case

        # Warm-up NaN rows are absent from both sets.
        assert not train[warm_cols].isna().any().any(), case
        assert not test[warm_cols].isna().any().any(), case
        assert (train_keys | test_keys) <= key_set(usable), case

        # Test starts at the split when the split row exists and is usable.
        if (usable["datetime"] == split).any():
            assert test["datetime"].min() == split, case
        if split == default_split and h in (24, 48, 72):
            last_train = default_split - pd.Timedelta(hours=h + 1)
            if (usable["datetime"] == last_train).any():
                assert train["datetime"].max() == last_train, case
            else:
                assert train.empty or train["datetime"].max() < last_train, case

        # Each city appears in each set exactly when a qualifying row exists.
        for city, grp in usable.groupby("city"):
            stamps = grp["datetime"]
            want_train = bool(((stamps + pd.Timedelta(hours=h)) < split).any())
            want_test = bool((stamps >= split).any())
            assert (city in set(train["city"])) == want_train, case
            assert (city in set(test["city"])) == want_test, case

        # horizon=None behaves exactly like 72.
        if horizon is None:
            train72, test72 = pipe.prepare_train_test_split(feats, split, horizon=72)
            pd.testing.assert_frame_equal(train, train72)
            pd.testing.assert_frame_equal(test, test72)

    assert default_checked >= 20
    assert both_nonempty >= N_CASES // 2


def test_per_horizon_embargo(tmp_path):
    """Feature: dhund-khabar-ai, Property 7: Per-horizon embargo
    **Validates: Requirements 9.3, 9.5**

    For seeded two-city frames around 2025-01-01 (created with the default
    horizons plus one custom horizon), a horizon h from 24/48/72 or the custom
    value, and a random split inside the usable range: every training label's
    source timestamp (datetime + h) is before the split, for both
    pm2_5_t_plus_h and us_aqi_t_plus_h, checked by joining labels to the source
    values at t + h; the last training row is split - (h + 1) hours when that
    row is usable; the embargo removes exactly h rows per city when the frame
    covers split - h; and the training size does not grow with h.
    """
    rng = np.random.default_rng(42)
    pipe = AirQualityDataPipeline(flags_path=tmp_path / "no_such_flags.json")
    ref_split = pd.Timestamp("2025-01-01")
    cache: dict[tuple[int, int, int], tuple[pd.DataFrame, pd.DataFrame]] = {}
    custom_choices = [1, 6, 12, 36, 96]
    exact_last_checked = 0
    exact_loss_checked = 0

    for case in range(N_CASES):
        pre_days = int(rng.choice([9, 12]))
        post_days = int(rng.choice([5, 8]))
        custom = int(rng.choice(custom_choices))
        ck = (pre_days, post_days, custom)
        if ck not in cache:
            start = ref_split - pd.Timedelta(days=pre_days)
            end = ref_split + pd.Timedelta(days=post_days)
            n_hours = int((end - start) / pd.Timedelta(hours=1))
            raw = make_synthetic_master(n_hours, start=str(start), seed=int(rng.integers(0, 10_000)))
            path = tmp_path / f"raw_{len(cache)}.csv"
            raw.to_csv(path, index=False)
            loaded = pipe.load_data(path)
            source = loaded[["city", "datetime", "pm2_5", "us_aqi"]].copy()
            feats = pipe.generate_time_features(loaded)
            feats = pipe.create_lag_and_rolling_features(feats)
            feats = pipe.create_targets(feats, horizons=sorted({24, 48, 72, custom}))
            cache[ck] = (source, feats)
        source, feats = cache[ck]
        warm_cols = pipe._warmup_columns(feats)
        usable = feats.loc[~feats[warm_cols].isna().any(axis=1)]
        lo, hi = usable["datetime"].min(), usable["datetime"].max()
        span_h = int((hi - lo) / pd.Timedelta(hours=1))

        if rng.random() < 0.3:
            split = ref_split
        else:
            split = lo + pd.Timedelta(hours=int(rng.integers(1, span_h)))
        h = int([24, 48, 72, custom][int(rng.integers(0, 4))])
        offset = pd.Timedelta(h, unit="h")

        train, test = pipe.prepare_train_test_split(feats, split, horizon=h)

        # Every training label comes from before the split.
        assert (train["datetime"] + offset < split).all(), case
        for target in ("pm2_5", "us_aqi"):
            label_col = f"{target}_t_plus_{h}"
            src = source.rename(columns={target: "src_value"})[["city", "datetime", "src_value"]]
            probe = train[["city", "datetime", label_col]].copy()
            probe["src_ts"] = probe["datetime"] + offset
            joined = probe.merge(src, left_on=["city", "src_ts"],
                                 right_on=["city", "datetime"], how="left",
                                 suffixes=("", "_src"))
            assert len(joined) == len(train), case
            assert joined["src_value"].notna().all(), case
            assert np.allclose(joined[label_col], joined["src_value"]), case
            assert (joined["src_ts"] < split).all(), case

        # The last training row, per city.
        last_wanted = split - pd.Timedelta(hours=h + 1)
        for city, grp in usable.groupby("city"):
            stamps = grp["datetime"]
            tr_city = train.loc[train["city"] == city, "datetime"]
            if (stamps == last_wanted).any():
                exact_last_checked += 1
                assert tr_city.max() == last_wanted, case
            else:
                assert tr_city.empty or tr_city.max() < last_wanted, case

            # Rows lost to the embargo: usable rows in [split - h, split).
            no_embargo = int((stamps < split).sum())
            kept = len(tr_city)
            if stamps.min() <= split - offset and split <= stamps.max():
                exact_loss_checked += 1
                assert no_embargo - kept == h, case

        # The embargo grows with h: training size never increases.
        sizes = [len(pipe.prepare_train_test_split(feats, split, horizon=k)[0])
                 for k in sorted({24, 48, 72, custom})]
        assert sizes == sorted(sizes, reverse=True), case

    assert exact_last_checked >= N_CASES
    assert exact_loss_checked >= N_CASES


@pytest.mark.parametrize("h", [24, 48, 72])
def test_train_rows_never_use_test_period_labels(boundary_frame, tmp_path, h):
    """Embargo: no training label comes from the Test_Period.

    **Validates: Requirements 9.3, 9.5, 18.6**
    """
    split = pd.Timestamp("2025-01-01")
    delta = pd.Timedelta(h, unit="h")
    df = build_features_frame(boundary_frame, tmp_path)
    pipe = AirQualityDataPipeline(flags_path=tmp_path / "no_such_flags.json")
    train, test = pipe.prepare_train_test_split(df, "2025-01-01", horizon=h)

    assert len(train) > 0 and len(test) > 0
    assert (train["datetime"] + delta).max() < split
    assert train["datetime"].max() == split - pd.Timedelta(h + 1, unit="h")
    assert test["datetime"].min() == split

    for target in ("pm2_5", "us_aqi"):
        source = df[["city", "datetime", target]].rename(
            columns={"datetime": "label_time", target: "source_value"})
        keyed = train[["city", "datetime", f"{target}_t_plus_{h}"]].copy()
        keyed["label_time"] = keyed["datetime"] + delta
        joined = keyed.merge(source, on=["city", "label_time"], how="left",
                             validate="one_to_one")
        assert joined["source_value"].notna().all()
        assert (joined[f"{target}_t_plus_{h}"].to_numpy()
                == joined["source_value"].to_numpy()).all()
        assert (joined["label_time"] < split).all()
        assert joined["label_time"].max() == pd.Timestamp("2024-12-31 23:00")


def test_feature_lists_never_leak_targets_or_aqi_family(tmp_path):
    """Feature: dhund-khabar-ai, Property 8: Feature lists never leak targets or the AQI family into PM2.5 models

    **Validates: Requirements 10.3, 10.5, 10.6**
    """
    rng = np.random.default_rng(42)
    pipe = AirQualityDataPipeline(flags_path=tmp_path / "no_such_flags.json")
    frame = build_features_frame(make_synthetic_master(240, seed=3), tmp_path)
    frame_cols = set(frame.columns)

    banned = {"city", "latitude", "longitude", "time", "datetime", "year",
              "month", "day", "hour", "day_of_week", "aqi_category",
              "is_forecast_like"}
    aqi_family = {"us_aqi", "us_aqi_lag_1", "us_aqi_lag_24", "aqi_category"}
    pollutant_tokens = ("pm2_5", "pm10", "carbon_monoxide", "nitrogen_dioxide",
                        "sulphur_dioxide", "ozone", "dust", "us_aqi")
    sim_required = {"pm2_5_rolling_mean_24h", "wind_speed_10m", "temperature_2m",
                    "relative_humidity_2m", "stagnation_flag", "inversion_proxy",
                    "hour_sin", "hour_cos", "month_sin", "month_cos",
                    "is_weekend", "is_winter"}
    targets = ["pm2_5", "us_aqi"]
    modes = ["forecast", "simulation"]
    seen = set()
    n_cases = 200

    for case in range(n_cases):
        target = targets[int(rng.integers(0, 2))]
        mode = modes[int(rng.integers(0, 2))]
        seen.add((target, mode))
        cols = pipe.get_feature_columns(target, mode)
        ctx = (case, target, mode)

        assert isinstance(cols, list) and cols, ctx
        assert len(cols) == len(set(cols)), ctx
        assert not [c for c in cols if "_t_plus_" in c], ctx
        assert not banned & set(cols), ctx
        assert set(cols) <= frame_cols, (ctx, set(cols) - frame_cols)

        if target == "pm2_5":
            assert not aqi_family & set(cols), ctx
        elif mode == "forecast":
            assert {"us_aqi", "us_aqi_lag_1", "us_aqi_lag_24"} <= set(cols), ctx
            assert "aqi_category" not in cols, ctx

        if mode == "simulation":
            other = "us_aqi" if target == "pm2_5" else "pm2_5"
            assert cols == pipe.get_feature_columns(other, mode), ctx
            assert sim_required <= set(cols), ctx
            polluted = [c for c in cols if any(t in c for t in pollutant_tokens)]
            assert polluted == ["pm2_5_rolling_mean_24h"], (ctx, polluted)
            assert not [c for c in cols if "_lag_" in c], ctx

        # A fresh list per call: mutating it must not change later calls.
        original = list(cols)
        cols.append("junk")
        cols.reverse()
        assert pipe.get_feature_columns(target, mode) == original, ctx

    assert len(seen) == 4

    for bad_target, bad_mode in [("pm10", "forecast"), ("PM2_5", "simulation"),
                                 ("", "forecast"), ("pm2_5", "nowcast"),
                                 ("pm2_5", ""), ("bogus_t", "bogus_m")]:
        with pytest.raises(ValueError) as exc:
            pipe.get_feature_columns(bad_target, bad_mode)
        msg = str(exc.value)
        if bad_target not in targets:
            assert repr(bad_target) in msg
        if bad_mode not in modes:
            assert repr(bad_mode) in msg
        for accepted in targets + modes:
            assert accepted in msg


def test_persistence_equals_shifted_target(tmp_path):
    """Feature: dhund-khabar-ai, Property 16: Persistence equals the shifted target
    **Validates: Requirements 13.3, 18.13**

    For random horizons and both targets, the label at row t equals the series
    value at t + h (independent join on (city, datetime + h)), the persistence
    prediction at row t is the series value at t, so the label shifted forward
    by h rows within a city equals persistence wherever it exists. Persistence
    ignores the label columns, never mixes cities, returns a float (n,) copy,
    and rejects a bad target_type or a missing column.
    """
    from src.utils import persistence_forecast

    rng = np.random.default_rng(42)
    horizon_choices = [24, 48, 72, 1, 3, 6, 12, 5]
    pipe = AirQualityDataPipeline(flags_path=tmp_path / "no_such_flags.json")
    for case in range(N_CASES):
        h = int(rng.choice(horizon_choices))
        n_hours = int(rng.integers(h + 5, 201))
        df = _parsed_master(n_hours, seed=int(rng.integers(0, 10_000)))
        out = pipe.create_targets(df, [h])
        for target in ("pm2_5", "us_aqi"):
            label = f"{target}_t_plus_{h}"
            pred = persistence_forecast(out, target)
            assert pred.dtype == np.float64 and pred.shape == (len(out),), case

            # Independent join: label at t is the series value at t + h, and
            # the persistence prediction for it is the series value at t.
            series = df[["city", "datetime", target]].copy()
            series["datetime"] = series["datetime"] - pd.Timedelta(h, unit="h")
            series = series.rename(columns={target: "future"})
            merged = out[["city", "datetime", label]].assign(pred=pred).merge(
                series, on=["city", "datetime"], how="left", validate="one_to_one")
            assert merged["future"].notna().all(), case
            np.testing.assert_array_equal(merged[label], merged["future"])
            now = out[["city", "datetime"]].merge(
                df[["city", "datetime", target]], on=["city", "datetime"], how="left")
            np.testing.assert_array_equal(pred, now[target].to_numpy(dtype=float))
            np.testing.assert_allclose(
                merged["future"] - merged["pred"],
                merged[label] - now[target].to_numpy(dtype=float))

            # Label shifted forward h rows within a city equals persistence.
            for city, grp in out.groupby("city", sort=False):
                p_city = pred[out.index.get_indexer(grp.index)]
                m = len(grp) - h
                if m > 0:
                    # label at row t is the value h rows later = persistence at t + h
                    np.testing.assert_array_equal(
                        grp[label].to_numpy()[:m], p_city[h:])

                # Never mixes cities: the city subset gives the same output.
                np.testing.assert_array_equal(persistence_forecast(grp, target), p_city)

            # Never uses the label: scrambling every target column changes nothing.
            scrambled = out.copy()
            for col in [c for c in out.columns if "_t_plus_" in c]:
                scrambled[col] = rng.normal(size=len(out)) * 1e6
            np.testing.assert_array_equal(persistence_forecast(scrambled, target), pred)

            # Returned array is a copy.
            before = out[target].copy()
            pred[:] = -1.0
            pd.testing.assert_series_equal(out[target], before)

        with pytest.raises(ValueError):
            persistence_forecast(out, "pm10")
        with pytest.raises(ValueError):
            persistence_forecast(out.drop(columns=["pm2_5"]), "pm2_5")


# ---------------------------------------------------------------------------
# Task 13.13: example and edge-case tests (literal expectations)
# ---------------------------------------------------------------------------

LAG_ROLLING_COLUMNS = [
    "pm2_5_lag_1", "pm2_5_lag_3", "pm2_5_lag_6", "pm2_5_lag_12",
    "pm2_5_lag_24", "pm2_5_lag_48", "us_aqi_lag_1", "us_aqi_lag_24",
    "pm2_5_rolling_mean_24h", "pm2_5_rolling_mean_72h", "pm2_5_rolling_std_24h",
    "wind_speed_rolling_mean_6h", "humidity_rolling_mean_6h",
    "temperature_rolling_mean_6h", "pm2_5_change_24h",
]

FORECAST_PM2_5_COLUMNS = [
    "pm2_5", "pm10", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide",
    "ozone", "dust",
    "pm2_5_lag_1", "pm2_5_lag_3", "pm2_5_lag_6", "pm2_5_lag_12",
    "pm2_5_lag_24", "pm2_5_lag_48",
    "pm2_5_rolling_mean_24h", "pm2_5_rolling_mean_72h",
    "pm2_5_rolling_std_24h", "pm2_5_change_24h",
    "temperature_2m", "relative_humidity_2m", "surface_pressure",
    "wind_speed_10m", "wind_direction_10m", "precipitation",
    "wind_speed_rolling_mean_6h", "humidity_rolling_mean_6h",
    "temperature_rolling_mean_6h",
    "stagnation_flag", "inversion_proxy",
    "hour_sin", "hour_cos", "month_sin", "month_cos", "is_weekend", "is_winter",
]
FORECAST_US_AQI_COLUMNS = (
    FORECAST_PM2_5_COLUMNS[:7] + ["us_aqi", "us_aqi_lag_1", "us_aqi_lag_24"]
    + FORECAST_PM2_5_COLUMNS[7:]
)
SIMULATION_COLUMNS = [
    "pm2_5_rolling_mean_24h",
    "temperature_2m", "relative_humidity_2m", "surface_pressure",
    "wind_speed_10m", "wind_direction_10m", "precipitation",
    "stagnation_flag", "inversion_proxy",
    "hour_sin", "hour_cos", "month_sin", "month_cos", "is_weekend", "is_winter",
]


def test_warmup_rows_dropped_and_first_training_row_offset(boundary_frame, tmp_path):
    """Warm-up rows are removed; the first row has a full 72 h window behind it."""
    feats = build_features_frame(boundary_frame, tmp_path)
    pipe = AirQualityDataPipeline()
    train, test = pipe.prepare_train_test_split(feats, horizon=72)

    for frame in (train, test):
        assert not frame[LAG_ROLLING_COLUMNS].isna().any().any()

    # The 72 h rolling mean needs rows 0..71, so the first valid row is index
    # 71: 2024-12-20 00:00 + 71 h = 2024-12-22 23:00. Every city loses 71 rows.
    for city in ("Lahore", "Peshawar"):
        first = train.loc[train["city"] == city, "datetime"].min()
        assert first == pd.Timestamp("2024-12-22 23:00:00")
    assert pipe.warmup_drop_counts == {"Lahore": 71, "Peshawar": 71}


def test_unknown_city_error_lists_available_cities(tmp_path):
    path = tmp_path / "m.csv"
    make_synthetic_master(5, seed=0).to_csv(path, index=False)
    pipe = AirQualityDataPipeline(flags_path=tmp_path / "none.json")
    pipe.load_data(path)
    with pytest.raises(ValueError) as exc:
        pipe.filter_city("Karachi")
    assert "Unknown city 'Karachi'" in str(exc.value)
    assert "['Lahore', 'Peshawar']" in str(exc.value)


def test_filter_city_before_load_data_raises_runtime_error():
    with pytest.raises(RuntimeError, match="Call load_data first"):
        AirQualityDataPipeline().filter_city("Lahore")


def test_create_targets_rejects_removed_row(tmp_path):
    df = _parsed_master(100, seed=3)
    # Lahore sorts before Peshawar; drop its row at 2024-12-01 05:00.
    bad = df[~((df["city"] == "Lahore")
               & (df["datetime"] == pd.Timestamp("2024-12-01 05:00")))]
    with pytest.raises(ValueError) as exc:
        AirQualityDataPipeline().create_targets(bad.reset_index(drop=True))
    msg = str(exc.value)
    assert "city 'Lahore'" in msg
    assert "2024-12-01 06:00:00" in msg
    assert "2 h after the previous row" in msg


def test_create_targets_rejects_duplicated_row():
    df = _parsed_master(100, seed=4)
    row = df[(df["city"] == "Peshawar") & (df["datetime"] == pd.Timestamp("2024-12-01 07:00"))]
    bad = pd.concat([df, row]).sort_values(["city", "datetime"], kind="mergesort")
    with pytest.raises(ValueError) as exc:
        AirQualityDataPipeline().create_targets(bad.reset_index(drop=True))
    msg = str(exc.value)
    assert "city 'Peshawar'" in msg
    assert "2024-12-01 07:00:00" in msg
    assert "0 h after the previous row" in msg


def test_lag_features_reject_unsorted_frames():
    df = _parsed_master(30, seed=5)
    pipe = AirQualityDataPipeline()
    shuffled = df.sample(frac=1.0, random_state=0).reset_index(drop=True)
    with pytest.raises(ValueError, match="Frame must be sorted by city and datetime"):
        pipe.create_lag_and_rolling_features(shuffled)

    reversed_cities = pd.concat([df[df["city"] == "Peshawar"], df[df["city"] == "Lahore"]],
                                ignore_index=True)
    with pytest.raises(ValueError, match="Frame must be sorted by city and datetime"):
        pipe.create_lag_and_rolling_features(reversed_cities)


def test_split_returns_tuple_and_none_horizon_means_72(boundary_frame, tmp_path):
    feats = build_features_frame(boundary_frame, tmp_path)
    pipe = AirQualityDataPipeline()
    result = pipe.prepare_train_test_split(feats)
    assert isinstance(result, tuple) and len(result) == 2
    assert all(isinstance(part, pd.DataFrame) for part in result)

    train72, test72 = pipe.prepare_train_test_split(feats, horizon=72)
    pd.testing.assert_frame_equal(result[0], train72)
    pd.testing.assert_frame_equal(result[1], test72)


def test_exact_feature_lists():
    pipe = AirQualityDataPipeline()
    assert len(FORECAST_PM2_5_COLUMNS) == 34
    assert len(FORECAST_US_AQI_COLUMNS) == 37
    assert len(SIMULATION_COLUMNS) == 15
    assert pipe.get_feature_columns("pm2_5", "forecast") == FORECAST_PM2_5_COLUMNS
    assert pipe.get_feature_columns("us_aqi", "forecast") == FORECAST_US_AQI_COLUMNS
    assert pipe.get_feature_columns("pm2_5", "simulation") == SIMULATION_COLUMNS
    assert pipe.get_feature_columns("us_aqi", "simulation") == SIMULATION_COLUMNS


def test_gap_strategy_on_small_synthetic_csv(tmp_path, capsys):
    base = make_synthetic_master(30, seed=6, cities=["Lahore"], ramp=True)
    base["pm2_5"] = 100.0 + np.arange(30)  # unique, hand-checkable values
    base.loc[2, "ozone"] = -5.0                     # negative -> clipped to 0
    base.loc[[15, 16, 17], "pm2_5"] = np.nan        # 3 h interior gap -> filled
    base.loc[[20, 21, 22, 23], "pm10"] = np.nan     # 4 h interior gap -> left NaN
    base.loc[[0, 1], "carbon_monoxide"] = np.nan    # leading gap -> left NaN
    removed_hour = base.loc[10, "datetime"]
    dup = base.loc[[5]].copy()
    dup["pm2_5"] = 999.0                            # later duplicate must lose
    raw = pd.concat([base.drop(index=10), dup], ignore_index=True)
    path = tmp_path / "gaps.csv"
    raw.to_csv(path, index=False)

    pipe = AirQualityDataPipeline(flags_path=tmp_path / "none.json")
    out = pipe.load_data(path).set_index("datetime")
    printed = capsys.readouterr().out

    assert len(out) == 30                           # grid complete again
    assert out.loc["2024-12-01 05:00", "pm2_5"] == 105.0   # first duplicate kept
    inserted = pd.Timestamp(removed_hour)
    assert inserted in out.index                    # missing hour re-inserted
    assert out.loc[inserted, "pm2_5"] == 109.0      # 1 h gap filled from hour 9
    assert out.loc[inserted, "temperature_2m"] != out.loc[inserted, "temperature_2m"]
    assert out.loc["2024-12-01 02:00", "ozone"] == 0.0
    assert out["pm2_5"].iloc[15:18].tolist() == [114.0, 114.0, 114.0]
    assert out["pm10"].iloc[20:24].isna().all()
    assert out["carbon_monoxide"].iloc[:2].isna().all()

    assert pipe.gap_counts["duplicates_dropped"] == 1
    assert pipe.gap_counts["rows_inserted"] == 1
    assert pipe.gap_counts["values_clipped"] == 1
    assert "negative pollutant values clipped to 0: 1 (ozone 1)" in printed
