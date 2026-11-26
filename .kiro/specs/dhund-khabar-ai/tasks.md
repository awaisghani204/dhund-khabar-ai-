# Implementation Plan: Dhund-Khabar AI (دُھند خبر)

## Overview

The work follows the seven delivery phases from Requirement 4, in order, with a Phase_Checkpoint after each phase. At every checkpoint, implementation stops, the Developer summarizes what was done, shows the real console and file outputs produced on `data/pakistan_smog_master_hourly.csv`, and waits for the user to reply "continue". Phase 5 has two checkpoints: one for the Stitch design review and one after `app.py` is built.

Environment: Windows, PowerShell, global Python 3.14.3 (no virtual environment). All commands run from the project root `d:\5th sem\prog ai lab\dhund khabar ai`, for example `python scripts/audit_data.py`, `python scripts/train_all.py`, and `python -m pytest`. `streamlit run app.py` is a long-running server, so the user starts it manually in their own terminal.

Honesty rule for every task: no metric, count, or timestamp is typed by hand. Every number in a checkpoint summary, `docs/data_audit.md`, `results/`, and `README.md` comes from running the scripts on the real master file. Synthetic data is used only inside `tests/` (Requirement 2.1, 2.2).

Code rules for every task: exactly two classes (AirQualityDataPipeline, SmogPredictorModel), everything else plain functions; every function and method has a docstring and full type hints; comments explain only the reason for a decision; `random_state=42` on every estimator and sampling call (Requirements 1.6, 3.1–3.4).

## Tasks

- [x] 1. Phase 1: Data audit
  - [x] 1.1 Create the package skeleton and the Phase 1 parts of `src/utils.py`
    - Create `src/__init__.py` (empty) so `src` is importable from the project root
    - In `src/utils.py` add the constants the audit needs: `MASTER_PATH`, `PROCESSED_DIR`, `MODELS_DIR`, `RESULTS_DIR`, `FIGURES_DIR`, `REQUIRED_COLUMNS` (26 names in file order, pm10 before pm2_5), `POLLUTANT_COLUMNS`, `WEATHER_COLUMNS`, `CITIES`, `WINTER_MONTHS`, `RANDOM_STATE = 42`
    - Add `ensure_dir(path)` and `pm25_sub_index(pm25_24h, table)` with the 2024 and 2012 EPA breakpoint tables from the design, linear interpolation inside each row, values above the top row capped at AQI 500
    - utils imports only numpy, pandas, and scikit-learn metrics
    - _Requirements: 1.2, 1.6, 2.3, 3.2, 3.3, 5.10_

  - [ ]* 1.2 Write quick unit tests for `pm25_sub_index`
    - In `tests/test_utils.py`: breakpoint endpoints for the 2024 table (9.0 → 50, 35.4 → 100) and the 2012 table (12.0 → 50), cap at 500 above the top row
    - _Requirements: 5.10_

  - [x] 1.3 Implement the structural audit functions in `scripts/audit_data.py`
    - Add the project root to `sys.path` so the script runs as `python scripts/audit_data.py`
    - Implement `audit_shape_and_dtypes`, `audit_date_range_per_city` (with row count per city), `audit_missing_per_city`, `audit_duplicate_timestamps`, `audit_hourly_gaps` (one row per non-1-hour step), `audit_analysis_stats` (describe() per city for pm2_5, us_aqi, temperature_2m, relative_humidity_2m, wind_speed_10m), `audit_impossible_values` (negative pollutants, relative_humidity_2m > 100, us_aqi < 0), `audit_datetime_vs_time`, `audit_monthly_means` (with the winter-peak sentence), `audit_aqi_above_500`, and `write_report`
    - `main()` loads the master file without model code, prints every section, and writes `docs/data_audit.md`
    - The report always includes the two fixed statements: us_aqi may be the maximum of several pollutant sub-indices, and the dataset provenance must be confirmed by the student because it may be modeled or reanalysis data
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 5.11, 5.12, 5.13_

  - [x] 1.4 Add the AQI sub-index comparison and Forecast_Like_Row detection to `scripts/audit_data.py`
    - `audit_aqi_subindex`: sample 5,000 rows with both values present using `random_state=42`; for both tables report mismatch rate (`round(sub_index) != round(us_aqi)`), mean absolute difference, and share where us_aqi > sub-index
    - `audit_forecast_like_rows`: flag rows later than the audit run time; for the last 180 days compute weekly repeated-value share, median decimal places, and lag-1 autocorrelation for pm2_5 and us_aqi, flag weeks against the 2023 baseline, and report the earliest flagged timestamp and the affected range per city
    - Write `data/processed/audit_flags.json` as `{"forecast_like_start", "method", "audit_run"}`; rows are never removed
    - Wire both sections into `main()` and the report
    - _Requirements: 1.6, 2.4, 5.10, 5.11, 5.14_

  - [x] 1.5 Run the audit on the real master file
    - Run `python scripts/audit_data.py` from the project root
    - Confirm `docs/data_audit.md` contains every section from 1.3 and 1.4 and that `data/processed/audit_flags.json` exists
    - Keep the full console output for the checkpoint; fix and rerun if any section errors
    - _Requirements: 2.1, 5.1–5.14_

- [x] 2. Checkpoint: Phase 1 complete. Report audit findings and propose the Gap_Strategy
  - Stop implementation. Summarize the audit and show the real outputs: shape and dtypes, date range and rows per city, missing counts per column per city (Leading_Gap length), duplicate count, hourly gaps, impossible values, datetime-vs-time mismatches, monthly means and whether peaks fall in months 11, 12, 1, 2, sub-index mismatch rates for both tables, us_aqi > 500 count and maximum, and the Forecast_Like_Row range from `audit_flags.json`
  - Propose the Gap_Strategy from the design (reindex to a complete hourly grid, keep first duplicate, leave the Leading_Gap unimputed, forward-fill interior pollutant and us_aqi gaps up to 3 hours within each city, leave weather unchanged, keep us_aqi > 500, keep aqi_category as-is and never as a feature), adjusted to the real audit numbers
  - Raise the us_aqi feature decision if the sub-index audit suggests us_aqi uses future values
  - Wait for the user to approve or change the Gap_Strategy and reply "continue". Record the approved strategy in `docs/data_audit.md`
  - _Requirements: 4.2, 4.3, 4.4_

- [x] 3. Phase 2: AirQualityDataPipeline and processed dataset
  - [x] 3.1 Add the Phase 2 constants and feature helpers to `src/utils.py`
    - Constants: `HORIZONS`, `TARGET_TYPES`, `SPLIT_DATE = "2025-01-01"`, `STAGNATION_WIND_KMH = 5.0`, `INVERSION_LEVELS`, `SLIDER_RANGES`
    - Functions: `inversion_proxy(temperature_2m, relative_humidity_2m)` = `relative_humidity_2m / 100 - temperature_2m / 40` (formula in the docstring), `inversion_level(proxy)` (Strong ≥ 0.50, Moderate ≥ 0.25, else Weak), `stagnation_flag(wind_speed_10m)` (1 where wind < 5 km/h), all NumPy-vectorised
    - _Requirements: 8.7, 8.8, 8.10, 17.9_

  - [x] 3.2 Implement loading, validation, city filter, and the approved Gap_Strategy in `src/pipeline.py`
    - Class `AirQualityDataPipeline` with `DEFAULT_LAG_STEPS`, `US_AQI_LAG_STEPS`, and `__init__(flags_path, ffill_limit=3)`
    - `load_data(filepath)`: read CSV, raise `ValueError` naming every missing Required_Column in file order, parse `datetime` with `%Y-%m-%d %H:%M:%S`, stable sort by city then datetime, call `_apply_gap_strategy`, add `is_forecast_like` from `audit_flags.json` (all 0 with a printed notice if the file is absent), store in `self.data`, return a copy
    - `_apply_gap_strategy(df)`: implement exactly the strategy approved at checkpoint 2, printing inserted-row and filled-value counts
    - `filter_city(city_name)`: `RuntimeError` before `load_data`; `ValueError` listing available cities for an unknown city
    - _Requirements: 2.4, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7_

  - [x] 3.3 Implement time, lag, and rolling features in `src/pipeline.py`
    - `generate_time_features(df)`: hour_sin, hour_cos, month_sin, month_cos with NumPy; is_weekend from the parsed timestamp (Saturday, Sunday = 1); is_winter for months 11, 12, 1, 2
    - `create_lag_and_rolling_features(df, lag_steps=None)`: raise `ValueError("Frame must be sorted by city and datetime")` if unsorted; all features via `groupby("city")`: pm2_5 lags 1, 3, 6, 12, 24, 48; us_aqi lags 1 and 24; trailing rolling means and std with `min_periods` equal to the window; pm2_5_change_24h; stagnation_flag and inversion_proxy from `utils`
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9, 8.10_

  - [x] 3.4 Implement targets and the embargo-aware split in `src/pipeline.py`
    - `create_targets(df, horizons=None)`: check that consecutive timestamps within each city differ by exactly 1 hour (else `ValueError` naming the city and first bad timestamp); add `pm2_5_t_plus_{h}` and `us_aqi_t_plus_{h}` by `groupby("city").shift(-h)`; drop rows with NaN in any created target
    - `prepare_train_test_split(df, split_date=SPLIT_DATE, horizon=None)`: `horizon=None` means the largest horizon present (72); drop warm-up rows with NaN lag or rolling features; train mask `datetime < split` and `datetime + h hours < split` (Embargo: last training feature row is 2024-12-31 23:00 minus h, every training label before 2025-01-01); test mask `datetime >= split`; assert train max < test min and `(train.datetime + h).max() < split`; return `(train, test)`
    - _Requirements: 2.5, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6_

  - [x] 3.5 Implement feature column sets, processed-data saving, and the pipeline summary in `src/pipeline.py`
    - `get_feature_columns(target_type, mode)`: validate arguments (`ValueError` naming accepted values and the bad value); return the forecast and simulation column sets exactly as in the design table; pm2_5 lists never contain us_aqi, us_aqi_lag_1, us_aqi_lag_24, or aqi_category; no target, datetime, city, latitude, longitude, raw calendar, or is_forecast_like columns
    - `save_processed(df, filename)` writes to `data/processed/` with `datetime` as `YYYY-MM-DD HH:MM:SS`
    - `build_features(filepath)`: load → time features → lag and rolling features → save `features_all.csv` → create_targets → save `model_dataset.csv` → return the target frame
    - Add an `if __name__ == "__main__":` block so `python -m src.pipeline` builds both files and prints row and column counts, NaN warm-up counts, and, per city and horizon, train and test sizes and the last training feature and label timestamps
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7_

  - [ ]* 3.6 Write quick unit tests for the pipeline
    - In `tests/test_pipeline.py`: missing-column `ValueError` on a tiny synthetic CSV; per-horizon embargo on a short synthetic frame around 2025-01-01
    - _Requirements: 6.3, 9.3, 9.5_

  - [x] 3.7 Run the pipeline on the real master file
    - Run `python -m src.pipeline` from the project root
    - Confirm `data/processed/features_all.csv` and `data/processed/model_dataset.csv` exist; keep the printed summary, including the real embargo table (last training feature row per horizon and rows dropped per city)
    - _Requirements: 2.1, 10.7_

- [x] 4. Checkpoint: Phase 2 complete
  - Stop implementation. Summarize the pipeline and show the real outputs: Gap_Strategy counts (inserted rows, filled values), processed file shapes, feature column lists for each target and mode, warm-up rows dropped, and per city and horizon the train/test sizes and last training timestamps
  - Wait for the user to reply "continue"
  - _Requirements: 4.2, 4.3_

- [x] 5. Phase 3: Alerts, model class, and training on real data
  - [x] 5.1 Add metric and model helpers to `src/utils.py`
    - `mae` and `rmse` in NumPy, `r2` wrapping `sklearn.metrics.r2_score`, `persistence_forecast(df, target_type)`, `model_path(city, target, horizon, mode, model_type)`, `xgboost_available()` via `importlib.util.find_spec`
    - _Requirements: 1.4, 1.5, 11.5, 13.3_

  - [x] 5.2 Implement `src/alerts.py` with the exact EPA rules
    - `TIERS` with the six names, colours (#00E400, #FFFF00, #FF7E00, #FF0000, #8F3F97, #7E0023), the six verbatim warning texts, and the respiratory_guidance texts from the design table; `TIER_UPPER_BOUNDS = [50, 100, 150, 200, 300]`
    - `validate_aqi` (`ValueError("AQI is NaN")`, `ValueError("AQI cannot be negative")`), vectorised `aqi_to_tier_index` with `np.searchsorted(..., side="left")` and the same validation, `get_aqi_tier`, `get_school_status` (≤ 100 Normal, ≤ 300 Outdoor Restrictions, else Closure Recommended), `get_industrial_advisory` (> 200, that is Very Unhealthy or Hazardous (Emergency): curfew text; else the no-curfew text), `get_aqi_advisory` returning tier, color, warnings, school_status, respiratory_guidance, industrial_advisory
    - Values above 500 map to Hazardous (Emergency); imports numpy only
    - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5, 12.6, 12.7, 12.8, 12.9, 12.10, 12.11_

  - [x] 5.3 Add tier metrics to `src/utils.py`
    - `tier_accuracy(y_true_aqi, y_pred_aqi)` and `unhealthy_precision_recall(y_true_aqi, y_pred_aqi)` (event = AQI > 150, NaN when a denominator is 0); clip predictions at 0 before tier mapping; use `alerts.aqi_to_tier_index` through a function-local import so utils keeps no module-level `src` imports
    - _Requirements: 13.5, 13.6_

  - [ ]* 5.4 Write quick unit tests for the alerts module
    - In `tests/test_alerts.py`: the 13 boundary values with tiers and colours; NaN and negative errors
    - _Requirements: 12.1, 12.2, 12.10, 12.11_

  - [x] 5.5 Implement `SmogPredictorModel` in `src/model.py`
    - `SUPPORTED_MODEL_TYPES`; constructor with `model_type="random_forest", city=None, horizon=24, target="pm2_5", mode="forecast"`; `ValueError` naming supported types for an unknown type and `ValueError("xgboost is not installed")` when xgboost is requested but missing
    - `_build_estimator`: RandomForestRegressor(n_estimators=100, max_depth=12, n_jobs=-1, random_state=42), LinearRegression(), GradientBoostingRegressor(random_state=42), XGBRegressor imported inside the method only
    - `train`, `evaluate` (MAE, RMSE, R2 from utils), `predict` (`RuntimeError` when not fitted, `ValueError` naming missing and unexpected columns, silent reorder, float array of shape (n,)), `get_risk_advisory`, `get_feature_importance` (feature_importances_ or |coef_|, sorted descending), `save_model` (joblib payload with compress=3 and sklearn_version), `load_model` (restores all fields, returns self, warns on version mismatch)
    - _Requirements: 1.5, 1.6, 3.1, 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7, 11.8, 11.9, 11.10, 11.11, 11.12_

  - [ ]* 5.6 Write quick unit tests for the model class
    - In `tests/test_model.py`: one prediction per row on a tiny synthetic frame; `RuntimeError` before fit
    - _Requirements: 11.6, 11.12_

  - [x] 5.7 Implement the forecast training loop in `scripts/train_all.py`
    - Add the project root to `sys.path`; `main()` calls `AirQualityDataPipeline().build_features(MASTER_PATH)`
    - For each city × target × horizon: `prepare_train_test_split(..., horizon=h)`, forecast feature columns, drop NaN rows (count printed), evaluate persistence, linear_regression, gradient_boosting, random_forest, and xgboost when available, on subset "all" and subset "winter" (test months 11, 12, 1, 2)
    - Tier accuracy and Unhealthy-or-worse recall and precision for us_aqi rows (including persistence), NaN for pm2_5 rows
    - Save every fitted model to `utils.model_path(...)` under `models/`
    - Compare Random Forest MAE with persistence MAE; fill `beats_persistence_mae` and a plain `note` (for example "Random Forest did not beat persistence on MAE (RF x vs persistence y)") and print the same verdict
    - Write `results/metrics.csv` with the design schema
    - _Requirements: 1.4, 1.5, 2.1, 2.5, 13.1, 13.3, 13.4, 13.5, 13.6, 13.7, 13.8_

  - [x] 5.8 Add simulation models, sensitivity analysis, and the CLI to `scripts/train_all.py`
    - Train one Simulation_Mode Random Forest per city (h = 24, us_aqi), evaluate on "all" and "winter", add to `metrics.csv`, save to `models/`
    - Sensitivity: test rows with temperature_2m ≤ 15 °C; for the simulation model and the forecast us_aqi h = 24 Random Forest, mean change in predicted AQI for wind_speed_10m + 5 and for relative_humidity_2m + 20 (clipped to 100), recomputing stagnation_flag and inversion_proxy and leaving rolling means unchanged; write `results/sensitivity.csv` (`city, mode, perturbation, delta, n_rows, mean_delta_aqi`) and print results without adjustment
    - Add `argparse` with a `--figures-only` flag (wired to figures in task 7.3)
    - _Requirements: 13.2, 13.4, 13.7, 13.9, 13.10_

  - [x] 5.9 Run training on the real data
    - Run `python scripts/train_all.py` from the project root (long run; use a generous timeout)
    - Confirm 12 forecast Random Forest models, 2 simulation models, and the baseline models exist in `models/`, that `results/metrics.csv` has persistence rows for all 12 combinations and both subsets, and that `results/sensitivity.csv` exists
    - Keep the console output, including every beat-persistence verdict
    - _Requirements: 2.1, 13.1, 13.2, 13.3, 13.7, 13.8_

- [x] 6. Checkpoint: Phase 3 complete
  - Stop implementation. Show the real metrics table read from `results/metrics.csv` (MAE, RMSE, R2 per city, target, horizon, model, for full test and winter-only), tier accuracy and Unhealthy-or-worse recall and precision for us_aqi models, every Random Forest vs persistence verdict stated plainly, the simulation model metrics, and `results/sensitivity.csv`
  - State whether xgboost was trained or skipped
  - Wait for the user to reply "continue"
  - _Requirements: 4.2, 4.3, 13.8_

- [x] 7. Phase 4: Charts and saved figures
  - [x] 7.1 Implement the theme, saving, and actual-vs-predicted chart in `src/charts.py`
    - `matplotlib.use("Agg")` at import; `apply_dark_theme` (figure #0B0F19, axes #161F30, cyan #00F2FE accents, light grey text, #2A3B53 grid); `save_figure` (PNG, 150 dpi, dark facecolor, closes the figure)
    - `plot_actual_vs_predicted(dates, actual, predicted, target_type, city, horizon)` with `axhspan` tier bands at alpha 0.12 (AQI tiers, or 2024 EPA PM2.5 breakpoints for pm2_5); default winter window 2025-11-01 to 2026-01-31
    - Every function returns a Matplotlib Figure
    - _Requirements: 14.1, 14.2_

  - [x] 7.2 Implement the remaining charts in `src/charts.py`
    - `plot_correlation_heatmap(df_city, city)` (rows wind_speed_10m, temperature_2m, relative_humidity_2m, surface_pressure, precipitation; columns pm2_5, pm10, us_aqi), `plot_feature_importance(importance_df, title, top_n=10)`, `plot_predicted_vs_actual_scatter(actual, predicted, title)` with a dashed y = x line, `plot_city_comparison(df)` (monthly mean us_aqi per city, Winter_Months shaded)
    - _Requirements: 14.1, 14.3, 14.4, 14.5, 14.6_

  - [x] 7.3 Wire figure generation into `scripts/train_all.py`
    - After training, and alone under `--figures-only`, load the saved models, rebuild test predictions, and save to `results/figures/`: actual vs predicted and scatter per city and horizon for us_aqi Random Forest, correlation heatmap per city, feature importance per city for the h = 24 us_aqi and simulation models, city comparison
    - _Requirements: 14.7_

  - [x] 7.4 Generate the figures from the real models
    - Run `python scripts/train_all.py --figures-only`; list the PNG files in `results/figures/` and open a sample to check the dark theme and tier bands
    - _Requirements: 2.1, 14.7_

- [x] 8. Checkpoint: Phase 4 complete
  - Stop implementation. List every saved figure and show the key figures (actual vs predicted with tier bands, heatmap, feature importance, scatter, city comparison) produced from the real models
  - Wait for the user to reply "continue"
  - _Requirements: 4.2, 4.3_

- [x] 9. Phase 5a: Stitch dashboard design
  - [x] 9.1 Create the Stitch project and design system
    - Load and call `tool::stitch-trainer::create_project` with title "Dhund-Khabar AI Dashboard"
    - Create the design system (`create_design_system` or `create_design_system_from_design_md`) with background #0B0F19, surface #161F30, 1px border #2A3B53, cyan #00F2FE, amber #FFB300, crimson #FF3366, violet #9933FF, the six EPA tier colours, Outfit (or Inter) for headings and numbers, Plus Jakarta Sans for body, system-ui fallback, 16px rounded cards, glass effect, dark mode; apply it with `apply_design_system`
    - If any Stitch call fails, report the exact error and wait for the user's direction
    - _Requirements: 15.1, 15.4, 16.1, 16.2, 16.3_

  - [x] 9.2 Generate the dashboard screen
    - Call `generate_screen_from_text` (desktop) with a prompt listing the seven layout sections in order: (1) header "Dhund-Khabar AI (دُھند خبر)" with the "Latest Available Data: … | Peshawar vs. Lahore" badge labelled historical; (2) Peshawar / Lahore segmented toggle with two summary cards and the 24 / 48 / 72 horizon selector; (3) semi-gauge hero AQI card tinted by tier with a pulse state; (4) PM2.5, wind (rotated arrow), and thermal inversion cards; (5) What-If Simulation Lab with three sliders, scenario tier, delta, and caption; (6) Visual Analytics tabs (actual vs predicted, heatmap, feature importance, model comparison table); (7) Actionable Health & Policy Terminal checklist; plus the fixed disclaimer footer
    - _Requirements: 15.1, 17.1–17.16_

  - [x] 9.3 Retrieve the generated design
    - Call `get_screen`, then `download_file` for the HTML and the screenshot; save them under `docs/stitch/` for the review and the later adaptation
    - _Requirements: 15.2_

- [x] 10. Checkpoint: Stitch design review
  - Stop implementation. Present the Stitch screenshot and a short summary of how each of the seven sections and the footer appear
  - Apply user feedback with `edit_screens` or `generate_variants`, re-download, and present again until the user approves
  - Write no dashboard code before approval. Wait for the user to approve and reply "continue"
  - _Requirements: 4.2, 4.3, 15.2, 15.4_

- [x] 11. Phase 5b: Streamlit dashboard
  - [x] 11.1 Add the What-If helpers to `src/utils.py`
    - `clamp_slider_default(value, low, high)` returning `(clipped, was_clamped)` and `build_scenario_row(base_row, wind, temp, humidity, feature_cols)` returning a one-row frame with the slider values and recomputed stagnation_flag and inversion_proxy
    - _Requirements: 17.10, 17.11, 17.12_

  - [ ]* 11.2 Write quick unit tests for the What-If helpers
    - In `tests/test_utils.py`: clamping below, inside, and above the range; scenario row values
    - _Requirements: 17.10, 17.11, 17.12_

  - [x] 11.3 Implement the style layer, loaders, and header in `app.py`
    - `inject_css()`: one CSS string adapted from the approved Stitch HTML/CSS, Tailwind classes converted to named classes (`.dk-card`, `.dk-hero`, `.dk-badge`, `.dk-pulse`, …), Google Fonts `@import` for Outfit and Plus Jakarta Sans with system-ui fallback, hidden `#MainMenu`, `header`, `footer`, responsive columns, injected with `st.markdown(unsafe_allow_html=True)`
    - `load_features` and `load_metrics` with `@st.cache_data`; `load_models` with `@st.cache_resource` loading saved joblib files only
    - `render_missing_file(path, script)` naming the missing file and the script that creates it
    - `render_header(last_ts)`: title and badge with the last timestamp read from the data, labelled "Historical dataset", never "live"
    - Escape interpolated strings with `html.escape` and format numbers with fixed precision
    - _Requirements: 15.3, 16.1, 16.2, 16.3, 16.4, 16.5, 17.1, 17.2, 17.17, 17.18, 17.19_

  - [x] 11.4 Implement the city toggle, summary cards, horizon selector, and hero card in `app.py`
    - City toggle with `st.segmented_control` if available in streamlit 1.42, else horizontal `st.radio` styled as a segmented control; horizon selector 24 / 48 / 72
    - `render_city_summary` for both cities side by side
    - Hero prediction from the last row with all us_aqi forecast features present, using the forecast us_aqi model for the selected city and horizon; `render_hero_card` with inline SVG semi-gauge (arc `min(aqi, 500) / 500`, real value printed), tier-tinted background, target time, and the `dk-pulse` class for Very Unhealthy and Hazardous (Emergency)
    - _Requirements: 17.3, 17.4, 17.5, 17.6_

  - [x] 11.5 Implement the condition cards and What-If Simulation Lab in `app.py`
    - `render_condition_cards`: latest pm2_5 with pm2_5_change_24h; wind_speed_10m with an arrow rotated by wind_direction_10m; inversion card from `utils.inversion_proxy` and `utils.inversion_level`, stating it is a heuristic proxy
    - `render_whatif_lab`: sliders wind 0–30 km/h, temperature 5–35 °C, humidity 20–100 % with defaults from `clamp_slider_default` and the "Observed: … (outside slider range, …)" note when clamped; scenario prediction from the simulation model via `build_scenario_row`; new tier, advisory, delta versus the baseline simulation prediction at observed values; the caption "Scenario exploration from a statistical model, not a physical simulation."
    - _Requirements: 17.7, 17.8, 17.9, 17.10, 17.11, 17.12, 17.13_

  - [x] 11.6 Implement analytics tabs, policy terminal, footer, and `main()` in `app.py`
    - `render_analytics_tabs`: actual vs predicted with city, horizon, and date-range controls (charts from `src/charts.py`), correlation heatmap, feature importance, model comparison table (persistence, linear, gradient boosting, random forest) from `results/metrics.csv`
    - `render_policy_terminal(advisory)`: School Operations Status, Respiratory Protective Guidance, Industrial Advisory from `get_risk_advisory`
    - `render_footer()`: "Student learning project. Not an official health warning system. Follow guidance from Pakistan's health and environmental authorities." on every view
    - `main()`: page config, CSS, then the seven sections in order and the footer; missing files show `render_missing_file` while the rest of the page still renders where possible
    - _Requirements: 17.14, 17.15, 17.16, 17.18, 17.19_

  - [x] 11.7 Check the dashboard headlessly
    - Run a short headless check with `streamlit.testing.v1.AppTest.from_file("app.py").run()` (for example from a temporary `python -c` command) and confirm no exceptions, and that the header, badge with the real last timestamp, caption, and footer text are present
    - Tell the user to start the server manually with `streamlit run app.py` (or `python -m streamlit run app.py`) for the visual check
    - _Requirements: 17.1, 17.13, 17.16, 17.19_

- [x] 12. Checkpoint: Phase 5 complete
  - Stop implementation. Summarize how the approved Stitch design was adapted, list the seven sections and the footer as implemented, show the headless check output, and give the exact command to start the dashboard
  - Wait for the user to reply "continue"
  - _Requirements: 4.2, 4.3_

- [x] 13. Phase 6: Complete test suite
  - Extend or replace the optional quick tests from earlier phases so the files below contain the complete required list. Every property test runs at least 100 seeded cases with `np.random.default_rng(42)`, uses only synthetic data, and carries the docstring tag `Feature: dhund-khabar-ai, Property {n}: {title}`. Hypothesis is not used.
  - [x] 13.1 Write the shared fixtures in `tests/conftest.py`
    - `make_synthetic_master(n_hours_per_city, start, seed)` with all 26 Required_Columns in file order, correct `time` and `datetime` string formats, weekday-name day_of_week, non-negative pollutants
    - `boundary_frame` (two cities hourly 2024-12-20 00:00 to 2025-01-10 23:00) and `tiny_model_data` (300 rows × 8 numeric features); a `tmp_path`-based helper to write synthetic CSVs
    - _Requirements: 2.2_

  - [x] 13.2 Write property test for load validation, ordering, and city filter
    - **Property 18: Load validation, ordering, and city filter**
    - **Validates: Requirements 6.3, 6.4, 6.6, 18.1**

  - [x] 13.3 Write property test for cross-city leakage
    - **Property 1: No cross-city leakage**
    - **Validates: Requirements 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 18.2**

  - [x] 13.4 Write property test for past-only features
    - **Property 2: Past-only lag and rolling features**
    - **Validates: Requirements 8.9, 18.3**

  - [x] 13.5 Write property test for cyclical time features
    - **Property 4: Cyclical time features are bounded and consistent**
    - **Validates: Requirements 7.1, 7.2, 18.5**

  - [x] 13.6 Write property test for calendar flags
    - **Property 5: Calendar flags match the timestamp**
    - **Validates: Requirements 7.3, 7.4**

  - [x] 13.7 Write property test for target shifts
    - **Property 3: Target shift correctness**
    - **Validates: Requirements 9.1, 9.2, 18.4**

  - [x] 13.8 Write property test for split ordering and period membership
    - **Property 6: Split ordering and period membership**
    - **Validates: Requirements 2.5, 9.3, 9.4, 9.5, 18.6**

  - [x] 13.9 Write property test for the per-horizon embargo
    - **Property 7: Per-horizon embargo**
    - **Validates: Requirements 9.3, 9.5**

  - [x] 13.10 Write the named embargo test `test_train_rows_never_use_test_period_labels`
    - In `tests/test_pipeline.py`, on `boundary_frame` for h in 24, 48, 72: `(train.datetime + h).max() < 2025-01-01`; `train.datetime.max() == 2025-01-01 − (h + 1) hours`; every training label equals the source value at t + h joined on (city, datetime + h) and that source timestamp is before 2025-01-01; `test.datetime.min() == 2025-01-01 00:00`
    - _Requirements: 9.3, 9.5, 18.6_

  - [x] 13.11 Write property test for feature lists
    - **Property 8: Feature lists never leak targets or the AQI family into PM2.5 models**
    - **Validates: Requirements 10.3, 10.5, 10.6**

  - [x] 13.12 Write property test for the persistence baseline
    - **Property 16: Persistence equals the shifted target**
    - **Validates: Requirements 13.3, 18.13**

  - [x] 13.13 Write the pipeline example and edge-case tests
    - In `tests/test_pipeline.py`: warm-up rows dropped; unknown city `ValueError` listing cities; `filter_city` before `load_data` raises `RuntimeError`; non-hourly step `ValueError` in `create_targets`; unsorted frame `ValueError`; `prepare_train_test_split` returns a tuple; `horizon=None` uses 72; exact forecast and simulation feature lists for both targets
    - _Requirements: 6.7, 9.4, 9.6, 10.1, 10.2, 10.4_

  - [x] 13.14 Write property test for prediction shape and feature importance
    - **Property 13: One prediction per input row**
    - **Validates: Requirements 11.6, 11.8, 18.10**

  - [x] 13.15 Write property test for the save and load round trip
    - **Property 14: Save and load round trip**
    - **Validates: Requirements 11.9, 11.10, 18.11**

  - [x] 13.16 Write property test for column mismatch after load
    - **Property 15: Column mismatch is rejected after load**
    - **Validates: Requirements 11.11, 18.12**

  - [x] 13.17 Write the model example and edge-case tests
    - In `tests/test_model.py`: Random Forest parameters (n_estimators=100, max_depth=12, random_state=42); LinearRegression and GradientBoosting(random_state=42) selection; unsupported model_type `ValueError` naming supported types; `RuntimeError` before fit for `predict` and `evaluate`; `evaluate` returns MAE, RMSE, R2; `get_risk_advisory` equals `alerts.get_aqi_advisory`; xgboost-unavailable path via monkeypatched `utils.xgboost_available`
    - _Requirements: 1.5, 11.1, 11.2, 11.3, 11.4, 11.5, 11.7, 11.12_

  - [x] 13.18 Write property test for EPA tier mapping
    - **Property 10: EPA tier mapping by decimal boundaries**
    - **Validates: Requirements 12.1, 12.2, 12.9, 18.8**

  - [x] 13.19 Write property test for advisory fields
    - **Property 11: Advisory fields follow the thresholds**
    - **Validates: Requirements 12.3, 12.4, 12.5, 12.6, 12.7, 12.8, 11.7**

  - [x] 13.20 Write property test for invalid AQI inputs
    - **Property 12: Invalid AQI inputs are rejected**
    - **Validates: Requirements 12.10, 12.11, 18.9**

  - [x] 13.21 Write the alerts example tests
    - In `tests/test_alerts.py`: the 13 values 0, 50, 51, 100, 101, 150, 151, 200, 201, 300, 301, 500, 600 with tier and colour; all six verbatim warning texts; 200 gives Unhealthy with no curfew and 200.5 gives Very Unhealthy with "Brick Kiln & Heavy Vehicle Curfew Required" (curfew follows the tier boundary)
    - _Requirements: 12.1, 12.2, 12.3, 12.7, 12.8, 18.8_

  - [x] 13.22 Write property test for project metrics
    - **Property 9: Project metrics equal scikit-learn**
    - **Validates: Requirements 11.5, 18.7**

  - [x] 13.23 Write property test for the inversion proxy
    - **Property 17: Inversion proxy formula and monotonicity**
    - **Validates: Requirements 8.8, 17.9**

  - [x] 13.24 Write property test for slider clamping and scenario rows
    - **Property 19: Slider default clamping and scenario rows**
    - **Validates: Requirements 17.10, 17.11, 17.12**

  - [x] 13.25 Write property test for tier metrics
    - **Property 20: Tier metrics match their definitions**
    - **Validates: Requirements 13.5, 13.6**

  - [x] 13.26 Write the utils example tests
    - In `tests/test_utils.py`: `pm25_sub_index` breakpoint endpoints for both tables; precision and recall NaN when there are no positives; `stagnation_flag` at 4.99 and 5.0 km/h; `inversion_level` at 0.25 and 0.50; `model_path` format
    - _Requirements: 5.10, 8.7, 13.6_

  - [x] 13.27 Run the full test suite and fix real failures
    - Run `python -m pytest -v` from the project root
    - Fix failures in the source code when the test reflects the spec; never weaken a test to make it pass; rerun until green or report what still fails and why
    - Keep the real pass/fail output for the checkpoint
    - _Requirements: 18.14_

- [x] 14. Checkpoint: Phase 6 complete
  - Stop implementation. Show the real `python -m pytest -v` output with pass and fail counts, list which test covers each of Requirements 18.1–18.13 and the embargo test, and describe any source fixes made
  - Wait for the user to reply "continue"
  - _Requirements: 4.2, 4.3, 18.14_

- [x] 15. Phase 7: Documentation
  - [x] 15.1 Write `requirements.txt`
    - Exactly: joblib==1.5.3, matplotlib==3.10.9, numpy==2.5.3, pandas==2.3.3, pytest==9.1.1, scikit-learn==1.8.0, seaborn==0.13.2, streamlit==1.42.0; xgboost not pinned
    - _Requirements: 1.2, 1.3_

  - [x] 15.2 Write the README problem, dataset, and methodology sections
    - Problem statement on winter smog in Lahore and Peshawar; objectives; dataset description with real counts from `docs/data_audit.md` and provenance flagged as unconfirmed
    - Methodology: feature groups, exact inversion proxy formula and thresholds, us_aqi feature decision, time-based split with the per-horizon embargo table from the Phase 2 output, why random cross-validation is wrong for this data, models and baselines, EPA alert system with the decimal-boundary rule
    - _Requirements: 19.1, 19.2_

  - [x] 15.3 Write the README results and discussion sections
    - Tables built from `results/metrics.csv` (full test and winter-only, tier accuracy, Unhealthy-or-worse recall and precision) and figures from `results/figures/`
    - Which horizons work, where the model fails, every case where Random Forest did not beat persistence, winter performance
    - Why the Simulation_Mode model exists, with the measured sensitivity of both models from `results/sensitivity.csv`
    - _Requirements: 19.3, 19.4, 19.5_

  - [x] 15.4 Write the README limitations, future work, commands, and structure sections
    - Compute the real EPA tier counts by applying `alerts.aqi_to_tier_index` to the master file's non-NaN us_aqi per city with a short command, and quote the printed output
    - Limitations: possibly modeled data, no emissions, stubble-burning, or traffic data, two cities only, skill drops with horizon, sliders are statistical, not a health tool, the Leading_Gap, tier imbalance with real counts, us_aqi > 500 mapped to Hazardous (Emergency), Forecast_Like_Rows found by the audit
    - Future work; run commands `pip install -r requirements.txt`, `python scripts/audit_data.py`, `python scripts/train_all.py`, `streamlit run app.py`, `pytest` (with the `python -m` forms for the global interpreter); project structure tree
    - _Requirements: 19.6, 19.7, 19.8_

  - [x] 15.5 Cross-check every README number against the result files
    - Compare each number in `README.md` with `results/metrics.csv`, `results/sensitivity.csv`, `docs/data_audit.md`, and the tier-count output; fix any mismatch
    - _Requirements: 2.1, 19.3_

- [x] 16. Checkpoint: Phase 7 complete
  - Stop implementation. Show the README section outline, the results tables as written, the limitations list, and `requirements.txt`
  - Wait for the user to reply "continue"
  - _Requirements: 4.2, 4.3_

- [x] 17. Final deliverable
  - Provide the final file tree (generated from the real workspace) and the exact PowerShell commands, run from the project root, for every script, the tests, and the dashboard: `python -m pip install -r requirements.txt`, `python scripts/audit_data.py`, `python -m src.pipeline`, `python scripts/train_all.py`, `python scripts/train_all.py --figures-only`, `python -m pytest`, `python -m streamlit run app.py`
  - Give a plain-language walkthrough with every technical term explained: the purpose of AirQualityDataPipeline and SmogPredictorModel, the end-to-end prediction flow, why the split is by time with an embargo, and the Persistence_Baseline with the real result of whether Random Forest beat it for each city, target, and horizon
  - Give the five most likely teacher questions with answers based on the real results
  - List every known weakness of the system (data provenance, Forecast_Like_Rows, Leading_Gap, tier imbalance, AQI > 500, horizon skill drop, weak weather sensitivity if measured, heuristic inversion proxy, two cities, no emissions data, and any test or check that did not pass)
  - _Requirements: 20.1, 20.2, 20.3, 20.4_

## Notes

- Tasks marked with `*` are optional quick tests written next to each module. Skipping them is safe because Phase 6 (task 13) delivers the complete required test list, including every listed test, the named embargo test, and all 20 property tests. Task 13 sub-tasks are not optional.
- Every checkpoint is a hard stop. The next phase starts only after the user replies "continue". At checkpoint 2, Phase 2 also waits for Gap_Strategy approval; at checkpoint 10, no dashboard code is written until the Stitch design is approved.
- Use the real Windows/PowerShell environment and the global Python 3.14.3 interpreter. Do not create a virtual environment. Run every command from the project root; the path contains spaces, so quote it when it appears in a command.
- `python scripts/train_all.py` trains many forests on about 37,000 rows per city and may take several minutes; run it with a long timeout rather than in the background.
- Do not start `streamlit run app.py` from the agent. The user runs it manually.
- `python -m src.pipeline` (task 3.5) and the README tier-count command (task 15.4) are both described in the design (pipeline "Module entry point" and Testing Strategy smoke checks).
- Never fabricate metrics. If a script fails, report the error instead of filling in numbers.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["1.2", "1.3"] },
    { "id": 2, "tasks": ["1.4"] },
    { "id": 3, "tasks": ["1.5"] },
    { "id": 4, "tasks": ["2"] },
    { "id": 5, "tasks": ["3.1", "3.2"] },
    { "id": 6, "tasks": ["3.3"] },
    { "id": 7, "tasks": ["3.4"] },
    { "id": 8, "tasks": ["3.5"] },
    { "id": 9, "tasks": ["3.6", "3.7"] },
    { "id": 10, "tasks": ["4"] },
    { "id": 11, "tasks": ["5.1", "5.2"] },
    { "id": 12, "tasks": ["5.3", "5.4", "5.5"] },
    { "id": 13, "tasks": ["5.6", "5.7"] },
    { "id": 14, "tasks": ["5.8"] },
    { "id": 15, "tasks": ["5.9"] },
    { "id": 16, "tasks": ["6"] },
    { "id": 17, "tasks": ["7.1"] },
    { "id": 18, "tasks": ["7.2"] },
    { "id": 19, "tasks": ["7.3"] },
    { "id": 20, "tasks": ["7.4"] },
    { "id": 21, "tasks": ["8"] },
    { "id": 22, "tasks": ["9.1"] },
    { "id": 23, "tasks": ["9.2"] },
    { "id": 24, "tasks": ["9.3"] },
    { "id": 25, "tasks": ["10"] },
    { "id": 26, "tasks": ["11.1", "11.3"] },
    { "id": 27, "tasks": ["11.2", "11.4"] },
    { "id": 28, "tasks": ["11.5"] },
    { "id": 29, "tasks": ["11.6"] },
    { "id": 30, "tasks": ["11.7"] },
    { "id": 31, "tasks": ["12"] },
    { "id": 32, "tasks": ["13.1"] },
    { "id": 33, "tasks": ["13.2", "13.14", "13.18", "13.22"] },
    { "id": 34, "tasks": ["13.3", "13.15", "13.19", "13.23"] },
    { "id": 35, "tasks": ["13.4", "13.16", "13.20", "13.24"] },
    { "id": 36, "tasks": ["13.5", "13.17", "13.21", "13.25"] },
    { "id": 37, "tasks": ["13.6", "13.26"] },
    { "id": 38, "tasks": ["13.7"] },
    { "id": 39, "tasks": ["13.8"] },
    { "id": 40, "tasks": ["13.9"] },
    { "id": 41, "tasks": ["13.10"] },
    { "id": 42, "tasks": ["13.11"] },
    { "id": 43, "tasks": ["13.12"] },
    { "id": 44, "tasks": ["13.13"] },
    { "id": 45, "tasks": ["13.27"] },
    { "id": 46, "tasks": ["14"] },
    { "id": 47, "tasks": ["15.1", "15.2"] },
    { "id": 48, "tasks": ["15.3"] },
    { "id": 49, "tasks": ["15.4"] },
    { "id": 50, "tasks": ["15.5"] },
    { "id": 51, "tasks": ["16"] },
    { "id": 52, "tasks": ["17"] }
  ]
}
```
