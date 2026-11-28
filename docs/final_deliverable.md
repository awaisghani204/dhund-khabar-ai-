# Dhund-Khabar AI: final deliverable

Student learning project. Not an official health warning system. All numbers below were read from `results/metrics.csv`, `results/sensitivity.csv`, `results/train_log.txt`, `docs/data_audit.md` and `README.md`, and the 12-row table in section 2.4 was computed by a script.

---

## 1. File tree and commands

### 1.1 Final file tree (real workspace)

Excluded: `__pycache__/`, `.pytest_cache/`, `.git/`. `.kiro/specs/dhund-khabar-ai/` holds the spec (requirements, design, tasks) and is not part of the program.

```
dhund khabar ai/
├── .gitignore
├── .kiro/specs/dhund-khabar-ai/    spec files (requirements.md, design.md, tasks.md)
├── app.py                          Streamlit dashboard (loads files only, never retrains)
├── pytest.ini
├── README.md                       Full report
├── requirements.txt
├── data/
│   ├── lahore_smog_hourly.csv
│   ├── peshawar_smog_hourly.csv
│   ├── pakistan_smog_daily.csv
│   ├── pakistan_smog_master_hourly.csv      used by the pipeline
│   └── processed/                           NOT in git (rebuilt)
│       ├── audit_flags.json
│       ├── features_all.csv
│       └── model_dataset.csv
├── docs/
│   ├── data_audit.md
│   ├── data_source_check.md                 Open-Meteo source check (supersedes the audit's provenance caveat)
│   ├── data_source_check_numbers.json       raw numbers and request URLs of the source check
│   ├── final_deliverable.md                 this file
│   └── stitch/                              dashboard design files (html/png/md/json)
├── models/                                  NOT in git (rebuilt): 38 .joblib files, about 84 MB
├── results/
│   ├── metrics.csv
│   ├── sensitivity.csv
│   ├── train_log.txt
│   └── figures/                             21 PNG files
├── scripts/
│   ├── audit_data.py
│   └── train_all.py
├── src/
│   ├── __init__.py
│   ├── alerts.py                            EPA tiers and advisories
│   ├── charts.py
│   ├── model.py                             SmogPredictorModel
│   ├── pipeline.py                          AirQualityDataPipeline
│   └── utils.py
└── tests/
    ├── __init__.py
    ├── conftest.py
    ├── test_alerts.py
    ├── test_conftest_smoke.py
    ├── test_model.py
    ├── test_pipeline.py
    └── test_utils.py
```

### 1.2 Commands (PowerShell, from the project root)

The path contains spaces, so quote it:

```powershell
cd "d:\5th sem\prog ai lab\dhund khabar ai"
```

| Step | Command | What it does |
|---|---|---|
| 1 | `python -m pip install -r requirements.txt` | Installs the dependencies. |
| 2 | `python scripts/audit_data.py` | Audits the raw data. Writes `docs/data_audit.md` and `data/processed/audit_flags.json`. |
| 3 | `python -m src.pipeline` | Cleans the data, builds features. Writes `data/processed/features_all.csv` and `model_dataset.csv`. |
| 4 | `python scripts/train_all.py` | Rebuilds features, trains all models (about 8 minutes), writes `models/*.joblib`, `results/metrics.csv`, `results/sensitivity.csv`, `results/train_log.txt`, and the figures. |
| 5 | `python scripts/train_all.py --figures-only` | Redraws the figures from saved models without retraining. |
| 6 | `python -m pytest` | Runs the tests (last run: 166 passed in 116.58 s). |
| 7 | `python -m streamlit run app.py` | Starts the dashboard. Open the address it prints. |

**Fresh clone.** `.gitignore` excludes `models/` and `data/processed/`, so a fresh clone does not have them. Run steps 1, 2, 3, 4 (in that order) before step 7. Step 4 also rebuilds the features and the figures, so step 5 is only needed to redraw figures later. Git history still contains those files, because they were committed in the first four commits before being ignored.

---

## 2. Plain-language walkthrough

### 2.1 Terms used in this project

| Term | Meaning |
|---|---|
| PM2.5 | Tiny particles in the air (under 2.5 micrometres), measured in µg/m³. The main smog pollutant here. |
| US AQI | The Air Quality Index, a 0 to 500 scale made from pollutant levels. Higher is worse. |
| EPA tier | One of six bands of AQI (Good, Moderate, Unhealthy for Sensitive Groups, Unhealthy, Very Unhealthy, Hazardous) defined by the US Environmental Protection Agency. |
| Feature | One input number given to a model (for example "PM2.5 24 hours ago"). |
| Lag | The value of a column some hours earlier. `pm2_5_lag_24` is PM2.5 24 hours ago. |
| Rolling mean | The average of the last N hours, ending at the current hour. |
| Horizon | How far ahead the forecast looks: 24, 48 or 72 hours. |
| Target | The number the model must predict (PM2.5 or US AQI, h hours ahead). |
| Leakage | Information from the future (or from the test data) reaching the model during training. It makes scores look better than they are. |
| Train / test split | Training data teaches the model. Test data is kept unseen to measure it. |
| Embargo | A gap at the boundary between training and test data (explained in 2.3). |
| Random Forest | Many decision trees that each give an answer; the forest averages them. |
| Linear Regression | The simplest model: a weighted sum of the inputs. |
| Gradient Boosting | Trees built one after another, each correcting the previous one's mistakes. |
| MAE | Mean absolute error: the average size of the miss. Lower is better. |
| RMSE | Like MAE, but large misses count more. |
| R² | How much better the model is than always guessing the average. 1 is perfect, 0 is no better than the average, negative is worse. |
| Recall (Unhealthy or worse) | Of the hours that really had AQI above 150, the share the model also flagged. |
| Precision (Unhealthy or worse) | Of the hours the model flagged, the share that really were above 150. |
| Winter | November, December, January, February. |
| Persistence_Baseline | "Tomorrow will be like now": the forecast for t+h is the value at t. No training. |
| Open-Meteo / CAMS | Open-Meteo is a free weather and air-quality data service. Its air-quality values come from CAMS (the Copernicus Atmosphere Monitoring Service), an atmospheric computer model. The data file matches Open-Meteo exactly, so the values are model estimates, not sensor readings. |
| Inversion proxy | A formula (`humidity/100 - temperature/40`) standing in for a thermal inversion, a warm air layer that traps cold polluted air near the ground. It is a heuristic, not a measurement. |

### 2.2 The two classes

**`AirQualityDataPipeline` (`src/pipeline.py`)** turns the raw CSV into model-ready tables. It does not train anything.

- `load_data(filepath)`: reads the CSV, checks that all 26 required columns exist, sorts by city and time.
- `filter_city(city_name)`: returns one city's rows, so cities are never mixed.
- `generate_time_features(df)`: adds hour and month as sine/cosine (so hour 23 is next to hour 0), weekend flag, winter flag.
- `create_lag_and_rolling_features(df)`: adds lags and rolling averages, always computed inside one city and from past values only.
- `create_targets(df, horizon)`: adds the future value (t+h) that the model must learn to predict.
- `prepare_train_test_split(df, horizon)`: splits by time with the embargo (2.3).
- `get_feature_columns(target_type, mode)`: lists which columns each model may use.
- `build_features(filepath)` and `save_processed(...)`: run the whole chain and write the files in `data/processed/`.

**`SmogPredictorModel` (`src/model.py`)** wraps one trained model (Random Forest, Linear Regression, Gradient Boosting, or XGBoost if installed).

- `train(X_train, y_train)`: fits the model.
- `predict(X_new)`: returns predicted values.
- `evaluate(X_test, y_test)`: returns MAE, RMSE, R².
- `get_risk_advisory(predicted_aqi)`: returns the EPA tier, colour, warning text, school status, respiratory guidance and industrial advisory (through `src/alerts.py`).
- `get_feature_importance()`: returns which inputs the model relied on most.
- `save_model(path)` and `load_model(path)`: write and read the `.joblib` file.

### 2.3 End-to-end prediction flow

1. **Data file.** `data/pakistan_smog_master_hourly.csv`: 75,456 rows, one row per hour per city, 2022-08-01 to 2026-11-19.
2. **Audit.** `scripts/audit_data.py` checks gaps, missing values, impossible values, and rows that look like forecasts.
3. **Cleaning.** The pipeline drops duplicates (0 found), clips 61 negative pollutant values to 0, leaves the first missing hours missing, and keeps AQI values above 500.
4. **Features.** Per city, from the current and past hours only: pollutant values, PM2.5 lags (1 to 48 h), rolling means, weather, stagnation flag, inversion proxy, time features (34 features for PM2.5 models, 37 for AQI models, which add `us_aqi` and two lags).
5. **Model.** A saved model for the chosen city, target and horizon predicts the value h hours ahead.
6. **AQI.** The dashboard uses the US AQI forecast model, so the output is a predicted AQI.
7. **EPA tier and advisory.** `alerts.py` maps the AQI to a tier (50 is Good, 50.4 is Moderate, 200 is Unhealthy, 200.5 is Very Unhealthy, above 300 is Hazardous) and picks school status and respiratory guidance. Above AQI 200 the industrial advisory is "Brick Kiln & Heavy Vehicle Curfew Required".
8. **Dashboard.** `app.py` shows the prediction from the last row of the historical file (not live data), the cards, the What-If lab, the charts, and the policy checklist, with the disclaimer in the footer.

**Why a time split with an embargo (analogy).** Imagine predicting tomorrow's weather for an exam. A random split is like giving the student last week's newspapers with Tuesday's weather printed in them and then asking about Monday and Wednesday: they can read the answer from the days next to it. Hourly air quality is the same, because 14:00 looks almost exactly like 13:00 and 15:00. A time split is the real situation: learn from the past, then be tested on a later period that is fully unseen. The embargo is a safety gap. A training row at 2024-12-31 20:00 with a 24-hour target would learn a value from 2025-01-01, which belongs to the test period. So for horizon h, any training row whose target falls on or after 2025-01-01 is removed (24, 48 or 72 rows per city). Training is 2022-08 to 2024-12-31, testing is 2025-01-01 onward (16,440 scored rows per city, 4,680 in winter).

### 2.4 Persistence_Baseline: did Random Forest beat it?

Rule: RF beats persistence when its MAE is lower. Forecast mode, test period 2025-01-01 onward. Generated by a script from `results/metrics.csv`.

| City | Target | Horizon | RF MAE (full) | Persistence MAE (full) | RF MAE (winter) | Persistence MAE (winter) | RF beat persistence? |
|---|---|---|---|---|---|---|---|
| Peshawar | pm2_5 | 24 h | 10.16 | 10.94 | 12.15 | 13.51 | full: yes, winter: yes |
| Peshawar | pm2_5 | 48 h | 11.36 | 12.54 | 12.74 | 14.20 | full: yes, winter: yes |
| Peshawar | pm2_5 | 72 h | 11.79 | 13.44 | 13.46 | 15.67 | full: yes, winter: yes |
| Peshawar | us_aqi | 24 h | 13.61 | 15.18 | 14.33 | 16.53 | full: yes, winter: yes |
| Peshawar | us_aqi | 48 h | 17.82 | 19.15 | 17.44 | 19.06 | full: yes, winter: yes |
| Peshawar | us_aqi | 72 h | 18.90 | 20.93 | 17.90 | 21.03 | full: yes, winter: yes |
| Lahore | pm2_5 | 24 h | 19.99 | 20.71 | 30.58 | 30.84 | full: yes, winter: yes |
| Lahore | pm2_5 | 48 h | 23.05 | 25.96 | 36.92 | 40.53 | full: yes, winter: yes |
| Lahore | pm2_5 | 72 h | 24.37 | 28.38 | 39.71 | 45.54 | full: yes, winter: yes |
| Lahore | us_aqi | 24 h | 17.93 | 21.10 | 18.83 | 22.07 | full: yes, winter: yes |
| Lahore | us_aqi | 48 h | 25.02 | 28.26 | 28.72 | 31.66 | full: yes, winter: yes |
| Lahore | us_aqi | 72 h | 28.42 | 31.38 | 33.81 | 37.08 | full: yes, winter: yes |

Result: **yes in 12 of 12 on the full test and 12 of 12 in winter (24 of 24 comparisons).** The margin ranges from 0.8 % (Lahore PM2.5, 24 h, winter: 30.58 vs 30.84) to 15.0 % (Lahore US AQI, 24 h, full test). The Lahore PM2.5 24 h margin is only 3.5 % on the full test and 0.8 % in winter, so that case is close to a tie.

Two honest qualifications:
- MAE is not the only measure. For Unhealthy-or-worse recall, persistence is higher than Random Forest in 9 of 12 forecast US AQI cases (for example Peshawar, 24 h, full test: 0.587 vs 0.402). Random Forest has higher precision in all 12 (fewer false alarms, more misses).
- The simulation model (What-If lab) did not beat persistence in one case: Peshawar US AQI 24 h, full test (15.75 vs 15.18).

---

## 3. Five most likely teacher questions

**Q1. Why not a random `train_test_split`, and what is the embargo?**
Hourly air quality changes slowly, so with a random split the hours next to every test hour are in the training set. The model only copies its neighbours and the score looks too good. It can also train on later days and be tested on earlier days, which a real forecast never can. This project trains on 2022-08 to 2024-12-31 and tests on 2025-01-01 onward. The embargo removes training rows whose target date falls in the test period (24, 48 or 72 rows per city for the three horizons), so every training label is from 2024 or earlier. The pipeline also asserts this, and `test_per_horizon_embargo` tests it.

**Q2. Does your model beat the naive baseline, and by how much?**
Yes. Random Forest has a lower MAE than persistence ("tomorrow like now") in all 24 comparisons (2 cities x 2 targets x 3 horizons x full/winter). The gain ranges from 0.8 % to 15.0 %. The best case is Lahore US AQI at 24 h: MAE 17.93 vs 21.10, R² 0.670. The weakest is Lahore PM2.5 at 24 h in winter: 30.58 vs 30.84. The gain is modest, and persistence catches more Unhealthy-or-worse hours than the model in 9 of 12 cases (the model has higher precision).

**Q3. Why is Random Forest your default when Linear Regression or Gradient Boosting had lower MAE?**
Honest answer: Random Forest was chosen as the default at the start of the project, not because it scored best. In the results, Linear Regression or Gradient Boosting had a lower MAE in 24 of 24 comparisons. The gap is a median of 3.6 % of Random Forest's MAE (range 0.3 % to 13.4 %), largest for Lahore at 72 h in winter (33.81 vs 29.28). Random Forest still beats persistence in all 24, so it is a valid forecaster, but I do not claim it is the best model. A likely reason, not tested, is that hourly pollution is so strongly tied to its recent values that a linear model uses them almost as well, while forest averaging flattens extremes. Switching the default to Gradient Boosting or Linear Regression is a one-line change and is listed as future work.

**Q4. How do you know there is no data leakage?**
Four checks, with named tests in `tests/test_pipeline.py`: `test_no_cross_city_leakage` (features of one city never use rows from the other), `test_past_only_lag_and_rolling_features` (lags and rolling windows use only values at or before the current hour), `test_per_horizon_embargo` (last training label is before 2025-01-01 for 24, 48 and 72 h), and `test_feature_lists_never_leak_targets_or_aqi_family` (PM2.5 models never get `us_aqi` or `aqi_category`; no feature list contains a target). The split is by time, and `us_aqi` is allowed only in the AQI models because it is known at prediction time. Limit: these tests run on synthetic data, so they prove the code, not the real data; the pipeline also asserts the embargo when it runs on the real file.

**Q5. How reliable is the data, what are the limits, and can this be used for real warnings?**
No, it must not be used for real warnings. The data is Open-Meteo output: an exact-match check against the Open-Meteo APIs (5 windows x 2 cities, 72 hours each) found every tested value identical. Open-Meteo's air-quality values are CAMS atmospheric-model estimates, so the data are model estimates and not sensor readings (details in `docs/data_source_check.md`). Who prepared the file is unknown. `us_aqi` does not match the EPA PM2.5 formula (73.18 % of a 5,000-row sample differ). The model is useful at 24 h (R² 0.63 Peshawar, 0.67 Lahore for US AQI), moderate at 48 h (0.40, 0.43), weak at 72 h (0.32, 0.29), and it under-predicts peaks: Peshawar Unhealthy-or-worse recall is 0.402 at 24 h. Lahore is better (0.796). It was tested on one split of about 22 months, and it has no emissions or stubble-burning data. The dashboard says this in its footer.

### Bonus questions

**B4. Where is the dataset from?**
The file matches Open-Meteo exactly. I downloaded the same hours from the Open-Meteo Air Quality API and weather series for Lahore and Peshawar in 5 three-day windows (2023-01 to 2026-10) and every compared value was identical, with timestamps in Asia/Karachi time, wind in km/h and concentrations in µg/m³. So the data is Open-Meteo output. Its air-quality values are CAMS atmospheric-model estimates, not ground-sensor readings, and the weather is Open-Meteo's default model series. What I do not know: who prepared the file, when, and with which script; the earliest rows (from 2022-08) were not tested; and an exact match today does not prove what the API returned when the file was made. The last two hours of the file (2026-11-19 22:00 and 23:00) are later than the file's creation time (21:18), so they are likely forecast values (likely, not tested). Full numbers: `docs/data_source_check.md`.

**B1. What do the What-If sliders mean, and are they causal?**
They change wind, temperature or humidity for the last row and show how the simulation model's predicted AQI changes. They are not causal and not a physical simulation. The model learned associations from past data. Wind moved the prediction the expected way (lower AQI) in 4 of 4 city/model cases (simulation: Peshawar -1.9, Lahore -11.1 AQI points for +5 km/h). Humidity matched the expected direction in only 2 of 4 cases, with effects of at most 0.18 points, so treat humidity as no evidence of a real effect.

**B2. What is the inversion level on the dashboard?**
A heuristic: `humidity/100 - temperature/40`, Strong at 0.50 or more, Moderate at 0.25 or more. It is not a measured inversion and was not validated.

**B3. Why does the model miss the highest peaks?**
Lahore hours above about 350 AQI are predicted at roughly 130 to 200 (see the scatter figure). Tree models average toward values they saw often and cannot go beyond the range of training values. This explanation is likely but was not tested separately.

---

## 4. Known weaknesses and what to say if asked

| # | Weakness | What to say if asked |
|---|---|---|
| 1 | **Data are model estimates, not sensors.** The file matches Open-Meteo exactly in 5 windows x 2 cities, and Open-Meteo's air-quality values are CAMS atmospheric-model output. Who prepared the file, when and how is unknown; the earliest rows were not tested. | "The data is Open-Meteo output (checked by exact match), so my results describe a model product, not ground sensors. I do not know who prepared the file." |
| 2 | **Forecast_Like_Rows unresolved.** The audit flagged none (0 of 27 weeks per city), but 2 rows per city (2026-11-19 22:00 and 23:00) are later than the file's creation time (21:18). They were kept. They are likely forecast values (inference from timestamps, not tested by the API comparison). Recent days before them may also be forecast rather than analysis values (untested). | "The audit found no forecast-like pattern, but the last 2 hours per city are later than the file was created, so they are probably forecasts. I did not test that." |
| 3 | **Leading_Gap.** The first 77 hours (pollutants) and 101 hours (`us_aqi`) are missing in both cities and left missing. | "The start of the record has no pollution values; those rows are not used. Weather is complete." |
| 4 | **Tier imbalance.** Good has 131 rows of 75,254 (0.17 %). Hazardous has 230 rows, all in Lahore. Peshawar has only 139 Very Unhealthy and 0 Hazardous hours. | "Rare tiers have too few examples to score reliably, so I report Unhealthy-or-worse recall and precision, not per-tier scores." |
| 5 | **AQI above 500.** 9 rows in Lahore, maximum 538, above the EPA scale. Kept and mapped to Hazardous (Emergency). | "The scale ends at 500; I kept the rows and treat them as the top tier." |
| 6 | **Skill drops with horizon.** US AQI R² (RF, full test): Peshawar 0.632, 0.403, 0.319; Lahore 0.670, 0.425, 0.290 at 24, 48, 72 h. Lahore winter 72 h is 0.080. | "24 h is usable, 48 h moderate, 72 h weak, and I say so." |
| 7 | **Low Unhealthy-or-worse recall, especially Peshawar.** RF recall 0.402 (24 h) to 0.199 (72 h) on the full test. Persistence is higher in 9 of 12 cases. | "The model raises fewer false alarms but misses more real ones in Peshawar; it should not be relied on for emergencies." |
| 8 | **Peaks under-predicted.** Hours above about 350 in Lahore are predicted at 130 to 200. | "Trees average toward the middle; it is a known limit, and I have not tested the cause." |
| 9 | **Weak humidity sensitivity.** 2 of 4 cases in the expected direction, effects at most 0.18 AQI points. Wind: 4 of 4. | "Humidity has almost no effect in the model, so do not read it as physics." |
| 10 | **Heuristic inversion proxy.** Built from temperature and humidity, never validated. | "It is a rough indicator, labelled as a proxy on the dashboard." |
| 11 | **Two cities only.** Lahore and Peshawar. | "I have no evidence it works elsewhere." |
| 12 | **No emissions or stubble-burning data.** The model sees only weather and past pollution. | "It cannot anticipate a change in sources such as crop burning or kilns." |
| 13 | **Single train/test split.** About 22 months of test data, no rolling-origin validation. | "One split is a limit; walk-forward validation is listed as future work." |
| 14 | **`us_aqi` does not match the EPA PM2.5 formula.** 73.18 % mismatch on a 5,000-row sample (2024 table). | "The models predict the file's `us_aqi`, whose exact formula is unknown, not an AQI I computed." |
| 15 | **Dashboard deviates from the original spec.** Near-black Stitch palette instead of navy #0B0F19 / #161F30; confidence band dropped; duplicate missing-file messages can appear; the actual-vs-predicted tab shows AQI only; the agent did not check the dashboard in a real browser. | "The look follows the approved Stitch design. I have run it only headlessly, so check it in your browser before presenting." |
| 16 | **Tests use synthetic data.** About 7,000 harmless DeprecationWarnings from joblib / NumPy 2.5 appear in pytest output. | "The tests check the code logic; the warnings are library deprecations, not failures (166 passed)." |
| 17 | **XGBoost not installed, so not tested.** | "It is optional; the code skips it when the package is missing." |
| 18 | **Git history still contains `models/` and processed data** (committed in the first four commits before `.gitignore`). | "Current commits ignore them; old commits are larger because of it." |
| 19 | **Simulation model is worse than persistence for Peshawar on the full test** (MAE 15.75 vs 15.18), and always worse than the forecast RF (1.55 to 2.14 points). | "The simulation model is for exploring inputs, not for forecasting." |
| 20 | **Random Forest is not the most accurate model.** Linear Regression or Gradient Boosting had lower MAE in 24 of 24 comparisons. | "It was the default by decision; the results favour the simpler models by a few percent." |
| 21 | **Historical replay, not live.** Last data point is 2026-11-19 23:00. | "The badge says 'Historical dataset' on purpose." |
