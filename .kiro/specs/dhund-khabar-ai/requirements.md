# Requirements Document

## Introduction

Dhund-Khabar AI (دُھند خبر) is a student project that forecasts winter smog in Peshawar and Lahore. It audits the real hourly dataset `data/pakistan_smog_master_hourly.csv`, builds leakage-free features per city, trains Random Forest forecasters for PM2.5 and US AQI at 24, 48, and 72 hour horizons, compares them with persistence, Linear Regression, and Gradient Boosting baselines on a strict time-based split, maps predicted AQI to US EPA tiers with fixed advisory texts, and presents the results in a custom-styled Streamlit dashboard with a What-If Simulation Lab.

Verified data facts that shape these requirements:

- The Master_Dataset holds 75,456 rows and 26 columns, with both cities covering 2022-08-01T00:00 to 2026-11-19T23:00.
- Column order in the file is: city, latitude, longitude, time, pm10, pm2_5, carbon_monoxide, nitrogen_dioxide, sulphur_dioxide, ozone, dust, us_aqi, temperature_2m, relative_humidity_2m, surface_pressure, wind_speed_10m, wind_direction_10m, precipitation, datetime, year, month, day, hour, day_of_week, is_weekend, aqi_category. pm10 comes before pm2_5.
- `time` has the format `YYYY-MM-DDTHH:MM`; `datetime` has the format `YYYY-MM-DD HH:MM:SS`; `day_of_week` is a weekday-name string.
- Pollutant columns are missing for the first 77 hours of each city. us_aqi is missing for about the first 101 hours of each city, where aqi_category is "Unknown". Weather columns have no missing values.
- The maximum us_aqi is 538, which is above the EPA scale top of 500.
- AQI tiers are imbalanced; the Good tier has only 131 rows.

Delivery runs in this order, with a Phase_Checkpoint after each step: Phase 1 data audit, Phase 2 pipeline, Phase 3 model and training, Phase 4 charts, Phase 5 dashboard, Phase 6 tests, Phase 7 documentation, then the final deliverable.

## Glossary

- **System**: The complete Dhund-Khabar AI project: src modules, scripts, app.py, tests, saved models, results, and documentation.
- **Developer**: The agent that builds the System and runs its code on the real data.
- **Master_Dataset**: The file `data/pakistan_smog_master_hourly.csv` with the 26 columns listed in the Introduction.
- **Required_Columns**: The 26 column names of the Master_Dataset, spelled exactly as in the file header.
- **Analysis_Columns**: The columns pm2_5, us_aqi, temperature_2m, relative_humidity_2m, and wind_speed_10m.
- **Leading_Gap**: The block of hours at the start of each city's series where pollutant columns (77 hours) or us_aqi (about 101 hours, aqi_category "Unknown") are missing.
- **Forecast_Like_Row**: A recent row whose values show signs of being model forecast output rather than observation, as detected by the Audit_Script.
- **Gap_Strategy**: The written plan for handling missing values and hourly gaps that the Developer proposes at the end of Phase 1 and the user approves.
- **Audit_Script**: The script `scripts/audit_data.py`.
- **Audit_Report**: The file `docs/data_audit.md` written by the Audit_Script.
- **AirQualityDataPipeline**: The class in `src/pipeline.py` that loads, validates, and feature-engineers the data and produces the time-based split.
- **SmogPredictorModel**: The class in `src/model.py` that trains, evaluates, predicts, explains, saves, and loads one model.
- **Alert_Module**: The plain-function module `src/alerts.py` that maps an AQI value to an EPA_Tier and advisory fields.
- **Charts_Module**: The plain-function module `src/charts.py` that builds Matplotlib and Seaborn figures.
- **Training_Script**: The script `scripts/train_all.py`.
- **Dashboard**: The Streamlit application `app.py`.
- **Test_Suite**: The pytest tests in `tests/`.
- **README**: The file `README.md`.
- **Horizon**: The number of hours between a feature row's timestamp and the predicted value's timestamp; one of 24, 48, or 72.
- **Target_Type**: The predicted variable, either pm2_5 or us_aqi.
- **Forecast_Mode**: The feature set made of autoregressive lags plus weather and pollutant values at time t.
- **Simulation_Mode**: The feature set made of weather and time features plus pm2_5_rolling_mean_24h as the only pollution feature.
- **Train_Period**: Rows with timestamps from 2022-08-01 00:00 through 2024-12-31 23:00.
- **Test_Period**: Rows with timestamps from 2025-01-01 00:00 through 2026-11-19 23:00, the end of the data.
- **Embargo**: The per-Horizon rule that keeps a training row at time t for Horizon h only if t + h hours is earlier than 2025-01-01 00:00, so training feature rows end at 2024-12-31 23:00 minus h hours and every training label timestamp lies in the Train_Period.
- **Winter_Months**: The months 11, 12, 1, and 2.
- **Persistence_Baseline**: A forecast that predicts the Target_Type value at time t+Horizon as the current value at time t for the same city.
- **EPA_Tier**: One of the six tiers Good, Moderate, Unhealthy for Sensitive Groups, Unhealthy, Very Unhealthy, and Hazardous (Emergency) defined in Requirement 12.
- **Unhealthy_Or_Worse_Event**: An hour whose AQI falls in the Unhealthy, Very Unhealthy, or Hazardous (Emergency) EPA_Tier (AQI above 150).
- **Stitch_Design**: The dashboard design generated with the connected Google Stitch MCP tools.
- **Phase_Checkpoint**: The stop point after a phase where the Developer summarizes the work, shows real outputs, and waits for the user to reply "continue".

## Requirements

### Requirement 1: Environment and dependencies

**User Story:** As a student, I want the project to run on my installed Python and pinned libraries, so that results are reproducible on my machine.

#### Acceptance Criteria

1. THE System SHALL run on Python 3.10 or later, using the existing global Python 3.14.3 interpreter without a virtual environment.
2. THE System SHALL use only numpy, pandas, scikit-learn, matplotlib, seaborn, streamlit, joblib, and pytest as third-party libraries in the baseline project.
3. THE System SHALL provide a `requirements.txt` that pins the tested versions joblib==1.5.3, matplotlib==3.10.9, numpy==2.5.3, pandas==2.3.3, pytest==9.1.1, scikit-learn==1.8.0, seaborn==0.13.2, and streamlit==1.42.0.
4. WHERE the xgboost package installs cleanly, THE Training_Script SHALL train an XGBoost model as an additional comparison model.
5. IF the xgboost package is not installed, THEN THE System SHALL run every script, test, and Dashboard view without xgboost.
6. THE System SHALL set random_state=42 on every estimator and sampling call that accepts a random_state parameter.

### Requirement 2: Data integrity and honesty

**User Story:** As a student, I want every reported number to come from real executed code, so that my project is honest and defensible.

#### Acceptance Criteria

1. THE System SHALL compute every number shown in the README, the Dashboard, the Audit_Report, and phase summaries from code run on the Master_Dataset.
2. THE System SHALL use synthetic data only inside the Test_Suite.
3. THE System SHALL reference only Required_Columns and columns created by the AirQualityDataPipeline by name in code and reports.
4. THE System SHALL keep all rows of the Master_Dataset, including Forecast_Like_Rows, and record Forecast_Like_Rows with a flag.
5. THE System SHALL split data by timestamp only, without random train/test splits and without k-fold cross-validation.

### Requirement 3: Code structure and style

**User Story:** As a teacher grading the project, I want a clear structure with documented code, so that I can follow the design.

#### Acceptance Criteria

1. THE System SHALL define exactly two classes: AirQualityDataPipeline in `src/pipeline.py` and SmogPredictorModel in `src/model.py`.
2. THE System SHALL implement the Alert_Module, the Charts_Module, utility code, and the Dashboard as plain functions.
3. THE System SHALL give every function and method a docstring and type hints for all parameters and the return value.
4. THE System SHALL limit code comments to explaining the reason for a decision.

### Requirement 4: Phase order and checkpoints

**User Story:** As a student, I want work delivered one phase at a time, so that I can review real outputs before the next phase starts.

#### Acceptance Criteria

1. THE Developer SHALL deliver the work in the order Phase 1, Phase 2, Phase 3, Phase 4, Phase 5, Phase 6, Phase 7, then the final deliverable.
2. WHEN a phase is complete, THE Developer SHALL stop at a Phase_Checkpoint, summarize the work, and show the real console and file outputs of that phase.
3. WHILE a Phase_Checkpoint is open, THE Developer SHALL wait for the user to reply "continue" before starting the next phase.
4. WHEN Phase 1 is complete, THE Developer SHALL propose a Gap_Strategy for missing values and hourly gaps and wait for the user's approval before Phase 2 applies the Gap_Strategy.

### Requirement 5: Phase 1 data audit

**User Story:** As a student, I want an audit of the real data before any modeling, so that later decisions are based on facts.

#### Acceptance Criteria

1. THE Audit_Script SHALL load the Master_Dataset without notebooks and without model code, print every audit result, and save the results to the Audit_Report.
2. WHEN the Audit_Script runs, THE Audit_Script SHALL report the shape, the column dtypes, the date range per city, and the row count per city.
3. WHEN the Audit_Script runs, THE Audit_Script SHALL report the missing-value count per column per city.
4. WHEN the Audit_Script runs, THE Audit_Script SHALL report the duplicate timestamp count per city.
5. WHEN the Audit_Script runs, THE Audit_Script SHALL report each gap in the hourly sequence per city, where a gap is a step between consecutive timestamps other than 1 hour.
6. WHEN the Audit_Script runs, THE Audit_Script SHALL report the value range and summary statistics per city for each Analysis_Column.
7. WHEN the Audit_Script runs, THE Audit_Script SHALL report the count of impossible values per city: negative pollutant concentrations, relative_humidity_2m above 100, and us_aqi below 0.
8. WHEN the Audit_Script runs, THE Audit_Script SHALL report the count of rows where the `datetime` column differs from the parsed `time` column.
9. WHEN the Audit_Script runs, THE Audit_Script SHALL report the monthly mean pm2_5 and monthly mean us_aqi per city and state whether the peak months fall in the Winter_Months.
10. WHEN the Audit_Script runs, THE Audit_Script SHALL compute the US EPA PM2.5 sub-index for a random_state=42 sample of rows, compare the sub-index with us_aqi, and report the mismatch rate.
11. THE Audit_Report SHALL state that us_aqi may be the maximum of several pollutant sub-indices, so a mismatch with the PM2.5 sub-index alone is expected.
12. THE Audit_Report SHALL state that the provenance of the Master_Dataset must be confirmed by the student, because the data may be modeled or reanalysis data rather than ground sensor data.
13. WHEN the Audit_Script runs, THE Audit_Script SHALL report the count of us_aqi values above 500 and the maximum us_aqi value.
14. WHEN the Audit_Script runs, THE Audit_Script SHALL check recent rows for signs of Forecast_Like_Rows and report the result and the affected timestamp range.

### Requirement 6: Pipeline data loading

**User Story:** As a student, I want data loaded and validated with clear errors, so that schema problems are caught immediately.

#### Acceptance Criteria

1. WHEN `load_data(filepath: str)` is called, THE AirQualityDataPipeline SHALL read the CSV file and return a pandas DataFrame.
2. WHEN `load_data` reads a file, THE AirQualityDataPipeline SHALL parse the `datetime` column with the format `%Y-%m-%d %H:%M:%S`.
3. IF one or more Required_Columns are absent from the file, THEN THE AirQualityDataPipeline SHALL raise a ValueError whose message names every missing column.
4. WHEN `load_data` returns, THE AirQualityDataPipeline SHALL return rows sorted by city and then by datetime.
5. WHEN `load_data` returns, THE AirQualityDataPipeline SHALL apply the user-approved Gap_Strategy to missing values and hourly gaps.
6. WHEN `filter_city(city_name: str)` is called, THE AirQualityDataPipeline SHALL return only the rows whose city equals city_name.
7. IF `filter_city` receives a city name absent from the loaded data, THEN THE AirQualityDataPipeline SHALL raise a ValueError that lists the available cities.

### Requirement 7: Time features

**User Story:** As a student, I want cyclical and seasonal time features, so that the model can learn daily and winter patterns.

#### Acceptance Criteria

1. WHEN `generate_time_features(df)` is called, THE AirQualityDataPipeline SHALL add hour_sin and hour_cos computed as sin and cos of 2*pi*hour/24.
2. WHEN `generate_time_features(df)` is called, THE AirQualityDataPipeline SHALL add month_sin and month_cos computed as sin and cos of 2*pi*month/12.
3. WHEN `generate_time_features(df)` is called, THE AirQualityDataPipeline SHALL add is_weekend with value 1 for Saturday and Sunday and value 0 for other days.
4. WHEN `generate_time_features(df)` is called, THE AirQualityDataPipeline SHALL add is_winter with value 1 for the Winter_Months and value 0 for other months.

### Requirement 8: Lag and rolling features

**User Story:** As a student, I want lag and rolling features built per city from past data only, so that the evaluation has no leakage.

#### Acceptance Criteria

1. WHEN `create_lag_and_rolling_features(df, lag_steps: list)` is called, THE AirQualityDataPipeline SHALL compute every lag and rolling feature within each city separately.
2. WHEN `create_lag_and_rolling_features` is called with the default lag_steps, THE AirQualityDataPipeline SHALL add pm2_5 lags of 1, 3, 6, 12, 24, and 48 hours.
3. WHEN `create_lag_and_rolling_features` is called, THE AirQualityDataPipeline SHALL add us_aqi lags of 1 and 24 hours.
4. WHEN `create_lag_and_rolling_features` is called, THE AirQualityDataPipeline SHALL add pm2_5_rolling_mean_24h, pm2_5_rolling_mean_72h, and pm2_5_rolling_std_24h.
5. WHEN `create_lag_and_rolling_features` is called, THE AirQualityDataPipeline SHALL add wind_speed_rolling_mean_6h, humidity_rolling_mean_6h, and temperature_rolling_mean_6h.
6. WHEN `create_lag_and_rolling_features` is called, THE AirQualityDataPipeline SHALL add pm2_5_change_24h equal to the current pm2_5 minus the pm2_5 value 24 hours earlier.
7. WHEN `create_lag_and_rolling_features` is called, THE AirQualityDataPipeline SHALL add a stagnation flag with value 1 where wind_speed_10m is below 5 km/h and value 0 elsewhere.
8. WHEN `create_lag_and_rolling_features` is called, THE AirQualityDataPipeline SHALL add a numeric inversion proxy that increases as temperature_2m decreases and as relative_humidity_2m increases, with the exact formula stated in the method docstring.
9. THE AirQualityDataPipeline SHALL compute every rolling feature from trailing windows that end at the current row, using values at or before the current timestamp.
10. THE AirQualityDataPipeline SHALL use NumPy operations for feature computations where NumPy expresses the computation directly.

### Requirement 9: Targets and time-based split

**User Story:** As a student, I want correct future targets and a strict chronological split, so that test metrics reflect forecasting of unseen future data.

#### Acceptance Criteria

1. WHEN `create_targets(df, horizons=[24, 48, 72])` is called, THE AirQualityDataPipeline SHALL add target columns pm2_5_t_plus_{h} and us_aqi_t_plus_{h} for each horizon h, created by a negative shift of h rows within each city.
2. WHEN `create_targets` is called, THE AirQualityDataPipeline SHALL drop rows in which a created target column is NaN.
3. WHEN `prepare_train_test_split(df, split_date: str, horizon: int)` is called with split_date "2025-01-01" and Horizon h, THE AirQualityDataPipeline SHALL apply the Embargo: the training set SHALL contain only rows at time t with t + h hours earlier than 2025-01-01 00:00, so the last training feature row is 2024-12-31 23:00 minus h hours and every training label timestamp is earlier than 2025-01-01 00:00, and the test set SHALL contain the Test_Period rows.
4. WHEN `prepare_train_test_split` is called, THE AirQualityDataPipeline SHALL drop warm-up rows that contain NaN lag features.
5. WHEN `prepare_train_test_split` builds the split, THE AirQualityDataPipeline SHALL assert that the maximum training timestamp is earlier than the minimum test timestamp and that the maximum training timestamp plus h hours is earlier than 2025-01-01 00:00.
6. WHEN `prepare_train_test_split` is called, THE AirQualityDataPipeline SHALL return a tuple containing the training rows and the test rows.

### Requirement 10: Feature column selection and processed data

**User Story:** As a student, I want feature lists that exclude leaking columns, so that each model only sees information available at prediction time.

#### Acceptance Criteria

1. WHEN `get_feature_columns(target_type: str, mode: str)` is called with mode "forecast", THE AirQualityDataPipeline SHALL return the autoregressive lag and rolling features plus the weather and pollutant concentration values at time t, subject to the pm2_5 exclusions in criterion 3.
2. WHEN `get_feature_columns` is called with mode "simulation", THE AirQualityDataPipeline SHALL return weather and time features plus pm2_5_rolling_mean_24h as the only pollution feature.
3. WHEN `get_feature_columns` is called with target_type "pm2_5" in either mode, THE AirQualityDataPipeline SHALL return a list without us_aqi, aqi_category, us_aqi_lag_1, and us_aqi_lag_24.
4. WHEN `get_feature_columns` is called with target_type "us_aqi" and mode "forecast", THE AirQualityDataPipeline SHALL include us_aqi at time t and the us_aqi lags of 1 and 24 hours, and SHALL exclude aqi_category.
5. THE AirQualityDataPipeline SHALL return feature lists without target columns and without any column dated after time t.
6. IF `get_feature_columns` receives a target_type other than "pm2_5" or "us_aqi" or a mode other than "forecast" or "simulation", THEN THE AirQualityDataPipeline SHALL raise a ValueError naming the accepted values.
7. WHEN the pipeline finishes feature and target creation, THE AirQualityDataPipeline SHALL save the processed dataset to `data/processed/`.

### Requirement 11: Model class

**User Story:** As a student, I want one model class that trains, evaluates, predicts, explains, and persists, so that every model is handled the same way.

#### Acceptance Criteria

1. THE SmogPredictorModel SHALL accept the constructor arguments model_type="random_forest", city=None, horizon=24, target="pm2_5", and mode="forecast".
2. WHEN `train(X_train, y_train)` is called with model_type "random_forest", THE SmogPredictorModel SHALL fit RandomForestRegressor(n_estimators=100, max_depth=12, n_jobs=-1, random_state=42).
3. WHEN `train` is called with model_type "linear_regression" or "gradient_boosting", THE SmogPredictorModel SHALL fit LinearRegression or GradientBoostingRegressor(random_state=42) respectively.
4. IF the SmogPredictorModel receives an unsupported model_type, THEN THE SmogPredictorModel SHALL raise a ValueError naming the supported types.
5. WHEN `evaluate(X_test, y_test)` is called, THE SmogPredictorModel SHALL return a dict with keys MAE, RMSE, and R2, where MAE and RMSE are computed with NumPy code written in the project.
6. WHEN `predict(X_new)` is called, THE SmogPredictorModel SHALL return a NumPy array with one prediction per input row.
7. WHEN `get_risk_advisory(predicted_aqi: float)` is called, THE SmogPredictorModel SHALL return the dict produced by the Alert_Module for predicted_aqi.
8. WHEN `get_feature_importance()` is called, THE SmogPredictorModel SHALL return a pandas DataFrame of feature names and importance values sorted from highest to lowest.
9. WHEN `save_model(filepath)` is called, THE SmogPredictorModel SHALL save the fitted estimator, city, horizon, target, mode, and feature column names with joblib.
10. WHEN `load_model(filepath)` is called, THE SmogPredictorModel SHALL restore the fitted estimator, city, horizon, target, mode, and feature column names.
11. IF `predict` receives input whose columns differ from the saved feature column names, THEN THE SmogPredictorModel SHALL raise a ValueError naming the missing and unexpected columns.
12. IF `predict` or `evaluate` is called before the SmogPredictorModel is trained or loaded, THEN THE SmogPredictorModel SHALL raise a RuntimeError stating that the model is not fitted.

### Requirement 12: EPA alert rules

**User Story:** As a resident, I want each predicted AQI mapped to an exact EPA tier with clear guidance, so that I know what to do.

#### Acceptance Criteria

1. THE Alert_Module SHALL map AQI values to EPA_Tiers by these ranges, where each tier includes its upper bound and a value between two listed ranges belongs to the higher tier (lower < AQI ≤ upper, with Good starting at 0 inclusive): 0–50 Good, 51–100 Moderate, 101–150 Unhealthy for Sensitive Groups, 151–200 Unhealthy, 201–300 Very Unhealthy, 301 and above Hazardous (Emergency). For example, 50 is Good, 50.4 is Moderate, 200 is Unhealthy, and 200.5 is Very Unhealthy.
2. THE Alert_Module SHALL assign the colors Good #00E400, Moderate #FFFF00, Unhealthy for Sensitive Groups #FF7E00, Unhealthy #FF0000, Very Unhealthy #8F3F97, and Hazardous (Emergency) #7E0023.
3. THE Alert_Module SHALL assign these warning texts verbatim:
   - Good: "Air quality is satisfactory. Ideal for all outdoor sports and physical activities."
   - Moderate: "Acceptable air quality. Unusually sensitive individuals should monitor outdoor exertion."
   - Unhealthy for Sensitive Groups: "Children, elderly, and asthmatic patients should wear masks outdoors and avoid strenuous exercise."
   - Unhealthy: "Everyone may begin to experience health effects. N95 masks strongly advised. Public outdoor events should be minimized."
   - Very Unhealthy: "Health alert: serious risk of respiratory symptoms across the population. Outdoor school sports banned; high-risk individuals stay indoors."
   - Hazardous (Emergency): "Emergency conditions: entire population affected. Primary schools and non-essential outdoor work should close; heavy air purifier use required."
4. WHEN the Alert_Module maps an AQI value, THE Alert_Module SHALL return a dict containing tier, color hex, warnings, school_status, respiratory_guidance, and industrial_advisory.
5. THE Alert_Module SHALL set school_status to "Normal" for AQI up to 100, "Outdoor Restrictions" for AQI above 100 up to 300, and "Closure Recommended" for AQI above 300.
6. THE Alert_Module SHALL set respiratory_guidance to a tier-specific text consistent with the tier's warning text: no protection needed for Good, monitoring of exertion for sensitive individuals for Moderate, masks for children, elderly, and asthmatic patients for Unhealthy for Sensitive Groups, N95 masks for everyone for Unhealthy, N95 masks with high-risk individuals indoors for Very Unhealthy, and staying indoors with heavy air purifier use for Hazardous (Emergency).
7. WHEN the AQI value is above 200, which is exactly when the EPA_Tier is Very Unhealthy or Hazardous (Emergency), THE Alert_Module SHALL set industrial_advisory to "Brick Kiln & Heavy Vehicle Curfew Required".
8. WHEN the AQI value is 200 or below, THE Alert_Module SHALL set industrial_advisory to a text stating that no curfew is required.
9. WHEN the AQI value is above 500, THE Alert_Module SHALL map the value to Hazardous (Emergency).
10. IF the Alert_Module receives a negative AQI value, THEN THE Alert_Module SHALL raise a ValueError stating that AQI cannot be negative.
11. IF the Alert_Module receives a NaN AQI value, THEN THE Alert_Module SHALL raise a ValueError stating that AQI is NaN.

### Requirement 13: Training script and comparisons

**User Story:** As a student, I want all models and baselines trained and compared honestly, so that I can show where the model helps and where it does not.

#### Acceptance Criteria

1. WHEN the Training_Script runs, THE Training_Script SHALL train 12 Forecast_Mode Random Forest models, one for each combination of city (Peshawar, Lahore), Horizon (24, 48, 72), and Target_Type (pm2_5, us_aqi).
2. WHEN the Training_Script runs, THE Training_Script SHALL train one Simulation_Mode model per city for Horizon 24 and Target_Type us_aqi.
3. WHEN the Training_Script runs, THE Training_Script SHALL evaluate the Persistence_Baseline, a LinearRegression model, and a GradientBoostingRegressor model for every combination of city, Horizon, and Target_Type.
4. WHEN the Training_Script evaluates models, THE Training_Script SHALL compute MAE, RMSE, and R2 for every model and baseline on the full Test_Period and on the Test_Period rows in the Winter_Months.
5. WHEN the Training_Script evaluates us_aqi models, THE Training_Script SHALL report the tier accuracy as the share of test rows where the predicted EPA_Tier equals the actual EPA_Tier.
6. WHEN the Training_Script evaluates us_aqi models, THE Training_Script SHALL report the recall and precision of Unhealthy_Or_Worse_Event detection.
7. WHEN the Training_Script completes, THE Training_Script SHALL save all metrics to `results/metrics.csv` and all trained models to `models/*.joblib`.
8. IF a Random Forest model does not beat the Persistence_Baseline on MAE for a city, Horizon, and Target_Type, THEN THE Training_Script SHALL state that result plainly in its printed output and in `results/metrics.csv`.
9. WHEN the Simulation_Mode models are trained, THE Training_Script SHALL measure and report the average change in predicted AQI when wind_speed_10m increases and when relative_humidity_2m increases under cold temperature_2m values.
10. WHEN the Training_Script measures sensitivity, THE Training_Script SHALL also measure the same sensitivity for the Forecast_Mode us_aqi Horizon 24 model and report both results without adjustment.

### Requirement 14: Charts

**User Story:** As a student, I want dark-themed charts matching the dashboard, so that the report and app look consistent.

#### Acceptance Criteria

1. THE Charts_Module SHALL return a Matplotlib Figure from every chart function, with figure background #0B0F19, axes background #161F30, and cyan #00F2FE accents.
2. WHEN the actual-versus-predicted chart is requested for a city and Horizon, THE Charts_Module SHALL plot actual and predicted PM2.5 or AQI over a winter window in the Test_Period with EPA_Tier bands as shaded horizontal regions.
3. WHEN the correlation heatmap is requested for a city, THE Charts_Module SHALL plot correlations between wind_speed_10m, temperature_2m, relative_humidity_2m, surface_pressure, precipitation and pm2_5, pm10, us_aqi.
4. WHEN the feature importance chart is requested, THE Charts_Module SHALL plot a bar chart of the top 10 features by importance.
5. WHEN the scatter chart is requested, THE Charts_Module SHALL plot predicted against actual values with a y = x reference line.
6. WHEN the city comparison chart is requested, THE Charts_Module SHALL plot the monthly mean us_aqi of Peshawar and Lahore.
7. WHEN charts are generated for the report, THE System SHALL save static copies to `results/figures/`.

### Requirement 15: Dashboard design process

**User Story:** As a student, I want to review a designed layout before it is coded, so that the dashboard matches my intent.

#### Acceptance Criteria

1. WHEN Phase 5 starts, THE Developer SHALL generate the Stitch_Design with the connected Google Stitch MCP tools from the design system and layout in Requirements 16 and 17.
2. WHEN the Stitch_Design is generated, THE Developer SHALL present the Stitch_Design to the user and wait for approval before writing Dashboard code.
3. WHEN the user approves the Stitch_Design, THE Developer SHALL adapt the Stitch_Design into custom CSS and HTML injected with st.markdown(unsafe_allow_html=True).
4. IF a Stitch MCP tool call fails, THEN THE Developer SHALL report the error to the user and wait for direction.

### Requirement 16: Dashboard design system

**User Story:** As a viewer, I want a polished custom interface, so that the project does not look like a default Streamlit app.

#### Acceptance Criteria

1. THE Dashboard SHALL use background #0B0F19 and card surfaces #161F30 with a 1px #2A3B53 border, rounded corners, and a glass effect.
2. THE Dashboard SHALL use cyan #00F2FE for telemetry, amber #FFB300 for moderate states, and crimson #FF3366 and violet #9933FF for hazardous states.
3. THE Dashboard SHALL load Outfit or Inter for headings and numbers and Plus Jakarta Sans for body text from Google Fonts, with a system-ui fallback.
4. THE Dashboard SHALL hide the default Streamlit menu, header, and footer.
5. THE Dashboard SHALL use responsive columns with consistent spacing.

### Requirement 17: Dashboard layout and behavior

**User Story:** As a resident, I want to see predicted AQI, conditions, scenarios, analytics, and advisories for my city, so that I can understand and act on smog risk.

#### Acceptance Criteria

1. THE Dashboard SHALL display the header "Dhund-Khabar AI (دُھند خبر)" and a status badge reading "Latest Available Data: <last timestamp in dataset> | Peshawar vs. Lahore", with the timestamp read from the loaded data.
2. THE Dashboard SHALL label the data as historical, without the word "live".
3. THE Dashboard SHALL provide a segmented Peshawar / Lahore city toggle and side-by-side summary cards for both cities.
4. WHEN the user selects a Horizon of 24, 48, or 72 hours, THE Dashboard SHALL show a radial or semi-gauge hero card, rendered with inline SVG or HTML and CSS, with the predicted AQI and EPA_Tier from the Forecast_Mode us_aqi model for the selected city and Horizon.
5. THE Dashboard SHALL tint the hero card background with the predicted EPA_Tier color.
6. WHILE the predicted EPA_Tier is Very Unhealthy or Hazardous (Emergency), THE Dashboard SHALL apply a pulse animation to the hero card.
7. THE Dashboard SHALL display a PM2.5 card with the latest pm2_5 and the change versus 24 hours earlier.
8. THE Dashboard SHALL display a wind card with the latest wind_speed_10m and an arrow rotated by wind_direction_10m.
9. THE Dashboard SHALL display a thermal inversion indicator card computed from temperature_2m and relative_humidity_2m.
10. THE Dashboard SHALL provide What-If Simulation Lab sliders for wind speed (0–30 km/h), temperature (5–35 C), and humidity (20–100%), with defaults equal to the latest observed values for the selected city.
11. IF a latest observed value lies outside its slider range, THEN THE Dashboard SHALL set the slider default to the nearest range bound and display the observed value.
12. WHEN a What-If slider value changes, THE Dashboard SHALL recompute AQI with the Simulation_Mode model and show the new EPA_Tier, the advisory, and the difference from the baseline Simulation_Mode prediction at the latest observed values.
13. THE Dashboard SHALL display the caption "Scenario exploration from a statistical model, not a physical simulation." in the What-If Simulation Lab.
14. THE Dashboard SHALL provide Visual Analytics tabs for the actual-versus-predicted chart with city, Horizon, and date-range controls, the correlation heatmap, the feature importance chart, and a model comparison table of persistence, linear, gradient boosting, and random forest loaded from `results/metrics.csv`.
15. THE Dashboard SHALL display an Actionable Health & Policy Terminal checklist of School Operations Status, Respiratory Protective Guidance, and Industrial Advisory, filled from get_risk_advisory for the predicted AQI.
16. THE Dashboard SHALL display the footer "Student learning project. Not an official health warning system. Follow guidance from Pakistan's health and environmental authorities." on every view.
17. THE Dashboard SHALL cache loaded data with st.cache_data and loaded models with st.cache_resource.
18. IF a data file or model file is missing, THEN THE Dashboard SHALL display a message that names the missing file and the script that creates the file.
19. THE Dashboard SHALL load saved models only, without training any model.

### Requirement 18: Tests

**User Story:** As a student, I want real automated tests that I have run, so that I can prove the core logic works.

#### Acceptance Criteria

1. THE Test_Suite SHALL test that load_data raises a ValueError naming the missing columns when Required_Columns are absent.
2. THE Test_Suite SHALL test that lag features of the first rows of the second city contain no values from the first city.
3. THE Test_Suite SHALL test that rolling features at a row are unchanged when values after that row are modified.
4. THE Test_Suite SHALL test that the value of each target column at time t equals the original value h rows later in the same city.
5. THE Test_Suite SHALL test that hour_sin, hour_cos, month_sin, and month_cos lie within [-1, 1].
6. THE Test_Suite SHALL test, for each Horizon h, that the maximum training timestamp is earlier than the minimum test timestamp, that the sets do not overlap, that the last training feature row is 2024-12-31 23:00 minus h hours, that every training label timestamp is earlier than 2025-01-01 00:00, and that the test set starts at 2025-01-01 00:00.
7. THE Test_Suite SHALL test that the project's MAE and RMSE equal scikit-learn's mean_absolute_error and root mean squared error on random arrays generated with seed 42.
8. THE Test_Suite SHALL test that the AQI values 0, 50, 51, 100, 101, 150, 151, 200, 201, 300, 301, 500, and 600 map to the correct EPA_Tier and color.
9. THE Test_Suite SHALL test that negative and NaN AQI inputs raise errors.
10. THE Test_Suite SHALL test, on a small synthetic fixture, that a trained SmogPredictorModel returns one prediction per input row.
11. THE Test_Suite SHALL test that a save_model and load_model round-trip produces identical predictions.
12. THE Test_Suite SHALL test that predicting with mismatched feature columns after load_model raises a clear error.
13. THE Test_Suite SHALL test that the Persistence_Baseline equals the target series shifted by the Horizon within each city.
14. WHEN Phase 6 is complete, THE Developer SHALL run the Test_Suite and report the real pass and fail results.

### Requirement 19: Documentation

**User Story:** As a student, I want a complete README with real results and honest limitations, so that I can present and defend the project.

#### Acceptance Criteria

1. THE README SHALL contain a problem statement on winter smog in Lahore and Peshawar, the objectives, and a dataset description with provenance that flags the unconfirmed points.
2. THE README SHALL describe the methodology: feature engineering, the time-based split, the reason random cross-validation is wrong for this data, the models, the baselines, and the EPA alert system.
3. THE README SHALL present results as tables and figures taken from `results/`.
4. THE README SHALL discuss which Horizons work, where the model fails, and performance in the Winter_Months.
5. THE README SHALL explain that the Simulation_Mode model exists because the Forecast_Mode model may react weakly to weather slider changes, and report the measured sensitivity of both.
6. THE README SHALL list these limitations: the data may be modeled rather than ground-truth sensor data, no emissions, stubble-burning, or traffic data, two cities only, forecast skill drops with Horizon, sliders are statistical rather than physical, and the project is not a health tool.
7. THE README SHALL also list as limitations the Leading_Gap, the tier imbalance with the real tier counts, the us_aqi values above 500 mapped to Hazardous (Emergency), and any Forecast_Like_Rows found by the Audit_Script.
8. THE README SHALL contain future work, the run commands `pip install -r requirements.txt`, `python scripts/audit_data.py`, `python scripts/train_all.py`, `streamlit run app.py`, and `pytest`, and the project structure tree.

### Requirement 20: Final deliverable

**User Story:** As a student, I want a final handover with explanations and weaknesses, so that I am prepared for my presentation.

#### Acceptance Criteria

1. WHEN all phases are complete, THE Developer SHALL provide the final file tree and the exact commands to run every script, the tests, and the Dashboard.
2. WHEN all phases are complete, THE Developer SHALL provide a plain-language walkthrough, with every technical term explained, of the purpose of each of the two classes, the end-to-end prediction flow, the reason for the time-based split, and the Persistence_Baseline with the real result of whether the model beat the Persistence_Baseline.
3. WHEN all phases are complete, THE Developer SHALL provide the five most likely teacher questions with answers based on the real results.
4. WHEN all phases are complete, THE Developer SHALL provide a list of every known weakness of the System.
