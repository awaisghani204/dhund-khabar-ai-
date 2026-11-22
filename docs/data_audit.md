# Dhund-Khabar AI: Phase 1 data audit

## Run information

- Source file: `data/pakistan_smog_master_hourly.csv`
- Audit run: 2026-11-21 23:11:48

## Shape and dtypes

Rows: 75456, columns: 26

| position | column | dtype |
|---|---|---|
| 1 | city | object |
| 2 | latitude | float64 |
| 3 | longitude | float64 |
| 4 | time | object |
| 5 | pm10 | float64 |
| 6 | pm2_5 | float64 |
| 7 | carbon_monoxide | float64 |
| 8 | nitrogen_dioxide | float64 |
| 9 | sulphur_dioxide | float64 |
| 10 | ozone | float64 |
| 11 | dust | float64 |
| 12 | us_aqi | float64 |
| 13 | temperature_2m | float64 |
| 14 | relative_humidity_2m | int64 |
| 15 | surface_pressure | float64 |
| 16 | wind_speed_10m | float64 |
| 17 | wind_direction_10m | int64 |
| 18 | precipitation | float64 |
| 19 | datetime | object |
| 20 | year | int64 |
| 21 | month | int64 |
| 22 | day | int64 |
| 23 | hour | int64 |
| 24 | day_of_week | object |
| 25 | is_weekend | int64 |
| 26 | aqi_category | object |

## Date range and rows per city

| city | first_timestamp | last_timestamp | rows | expected_hourly_rows |
|---|---|---|---|---|
| Peshawar | 2022-08-01 00:00:00 | 2026-11-19 23:00:00 | 37728 | 37728 |
| Lahore | 2022-08-01 00:00:00 | 2026-11-19 23:00:00 | 37728 | 37728 |

## Missing values per column per city

| column | Peshawar missing | Peshawar leading run | Lahore missing | Lahore leading run |
|---|---|---|---|---|
| city | 0 | 0 | 0 | 0 |
| latitude | 0 | 0 | 0 | 0 |
| longitude | 0 | 0 | 0 | 0 |
| time | 0 | 0 | 0 | 0 |
| pm10 | 77 | 77 | 77 | 77 |
| pm2_5 | 77 | 77 | 77 | 77 |
| carbon_monoxide | 77 | 77 | 77 | 77 |
| nitrogen_dioxide | 77 | 77 | 77 | 77 |
| sulphur_dioxide | 77 | 77 | 77 | 77 |
| ozone | 77 | 77 | 77 | 77 |
| dust | 77 | 77 | 77 | 77 |
| us_aqi | 101 | 101 | 101 | 101 |
| temperature_2m | 0 | 0 | 0 | 0 |
| relative_humidity_2m | 0 | 0 | 0 | 0 |
| surface_pressure | 0 | 0 | 0 | 0 |
| wind_speed_10m | 0 | 0 | 0 | 0 |
| wind_direction_10m | 0 | 0 | 0 | 0 |
| precipitation | 0 | 0 | 0 | 0 |
| datetime | 0 | 0 | 0 | 0 |
| year | 0 | 0 | 0 | 0 |
| month | 0 | 0 | 0 | 0 |
| day | 0 | 0 | 0 | 0 |
| hour | 0 | 0 | 0 | 0 |
| day_of_week | 0 | 0 | 0 | 0 |
| is_weekend | 0 | 0 | 0 | 0 |
| aqi_category | 0 | 0 | 0 | 0 |

## Duplicate timestamps per city

| city | duplicate_timestamps |
|---|---|
| Peshawar | 0 |
| Lahore | 0 |

## Hourly gaps

Steps other than 1 hour: 0.

(no rows)

## Analysis column statistics per city

| city / column | count | mean | std | min | 25% | 50% | 75% | max |
|---|---|---|---|---|---|---|---|---|
| Peshawar / pm2_5 | 37651 | 37.39 | 20.31 | 0.40 | 22.60 | 33.90 | 47.50 | 156.50 |
| Peshawar / us_aqi | 37627 | 111.23 | 30.38 | 12 | 87 | 106 | 134 | 228 |
| Peshawar / temperature_2m | 37728 | 22.45 | 9.03 | -1.40 | 15.50 | 23.50 | 29.30 | 46.10 |
| Peshawar / relative_humidity_2m | 37728 | 59.60 | 21.51 | 4 | 43 | 61 | 77 | 100 |
| Peshawar / wind_speed_10m | 37728 | 4.89 | 3.71 | 0 | 2.20 | 4 | 6.70 | 40.10 |
| Lahore / pm2_5 | 37651 | 70.26 | 48.65 | 0.50 | 37.40 | 55.20 | 87.30 | 365.10 |
| Lahore / us_aqi | 37627 | 153.98 | 46.64 | 12 | 121 | 154 | 177 | 538 |
| Lahore / temperature_2m | 37728 | 24.00 | 8.41 | 1.10 | 17.70 | 25.50 | 30.20 | 46.50 |
| Lahore / relative_humidity_2m | 37728 | 65.21 | 22.18 | 5 | 49 | 68 | 84 | 100 |
| Lahore / wind_speed_10m | 37728 | 6.76 | 3.92 | 0 | 4 | 6.20 | 8.70 | 37.20 |

## Impossible values per city

| check | Peshawar | Lahore |
|---|---|---|
| pm10 < 0 | 0 | 0 |
| pm2_5 < 0 | 0 | 0 |
| carbon_monoxide < 0 | 0 | 0 |
| nitrogen_dioxide < 0 | 2 | 0 |
| sulphur_dioxide < 0 | 0 | 0 |
| ozone < 0 | 22 | 37 |
| dust < 0 | 0 | 0 |
| relative_humidity_2m > 100 | 0 | 0 |
| us_aqi < 0 | 0 | 0 |

## datetime vs time

Rows where `datetime` differs from the parsed `time`: 0.

## Monthly means of pm2_5 and us_aqi

| month | Peshawar pm2_5 | Peshawar us_aqi | Lahore pm2_5 | Lahore us_aqi |
|---|---|---|---|---|
| 1 | 54.53 | 138.69 | 140.14 | 219.57 |
| 2 | 37.00 | 106.25 | 72.84 | 156.73 |
| 3 | 30.01 | 92.52 | 47.72 | 127.02 |
| 4 | 24.61 | 85.80 | 35.50 | 110.69 |
| 5 | 27.89 | 98.49 | 45.80 | 135.13 |
| 6 | 32.97 | 113.73 | 50.77 | 142.53 |
| 7 | 40.31 | 121.87 | 55.28 | 143.51 |
| 8 | 37.91 | 114.79 | 49.63 | 134.75 |
| 9 | 38.57 | 117.20 | 53.74 | 140.52 |
| 10 | 34.48 | 103.08 | 73.29 | 158.10 |
| 11 | 41.37 | 114.37 | 102.87 | 184.56 |
| 12 | 48.27 | 126.39 | 117.42 | 195.38 |

- Peshawar pm2_5: peak month 1 (mean 54.53); top 3 months 1, 12, 11, all in the Winter_Months (11, 12, 1, 2).
- Peshawar us_aqi: peak month 1 (mean 138.69); top 3 months 1, 12, 7, 2 of 3 in the Winter_Months (11, 12, 1, 2).
- Lahore pm2_5: peak month 1 (mean 140.14); top 3 months 1, 12, 11, all in the Winter_Months (11, 12, 1, 2).
- Lahore us_aqi: peak month 1 (mean 219.57); top 3 months 1, 12, 11, all in the Winter_Months (11, 12, 1, 2).

## us_aqi vs EPA PM2.5 sub-index

- PM2.5 averaging: trailing 24-hour mean of pm2_5 per city (time-based window, all 24 hourly values required). The hourly pm2_5 value is not used, because the EPA PM2.5 index is defined on a 24-hour average.
- Rows with both the 24-hour mean and us_aqi present: 75254. Sample: 5000 rows drawn with random_state=42.
- Mismatch means round(sub-index) != round(us_aqi), both rounded half up. Mean abs diff uses the unrounded sub-index. Sub-index values above the top breakpoint row are capped at 500.

| table / group | rows | mismatch % | mean abs diff | us_aqi > sub-index % | us_aqi < sub-index % |
|---|---|---|---|---|---|
| 2024 / All | 5000 | 73.18 | 5.84 | 30.04 | 43.14 |
| 2024 / Peshawar | 2480 | 68.47 | 6.19 | 29.03 | 39.44 |
| 2024 / Lahore | 2520 | 77.82 | 5.49 | 31.03 | 46.79 |
| 2012 / All | 5000 | 84.76 | 8.12 | 64.90 | 19.86 |
| 2012 / Peshawar | 2480 | 81.33 | 6.99 | 60.60 | 20.73 |
| 2012 / Lahore | 2520 | 88.13 | 9.23 | 69.13 | 19.01 |

First non-missing pm2_5 and us_aqi per city (the sub-index window is 24 hours):

- Peshawar: first pm2_5 at 2022-08-04 05:00:00, first us_aqi at 2022-08-05 05:00:00, 24 hours later.
- Lahore: first pm2_5 at 2022-08-04 05:00:00, first us_aqi at 2022-08-05 05:00:00, 24 hours later.

us_aqi may be the maximum of several pollutant sub-indices (see Interpretation notes), so us_aqi above the PM2.5 sub-index is expected. us_aqi below the PM2.5 sub-index is not explained by that and points to a different averaging or breakpoint table.

## us_aqi above 500

Rows with us_aqi above 500: 9. Maximum us_aqi: 538.

| city | us_aqi > 500 | max us_aqi | timestamp of max |
|---|---|---|---|
| Peshawar | 0 | 228 | 2023-06-24 18:00:00 |
| Lahore | 9 | 538 | 2025-05-23 04:00:00 |

## Forecast-like rows

- Audit run time: 2026-11-21 23:11:48
- Master file last modified: 2026-11-19 21:18:50 (rows after this time are counted for review only and are not part of the flag rule)
- forecast_like_start (earliest flagged timestamp, both cities): none, no row was flagged

Method: Two checks per city. (1) Rows with a timestamp later than the audit run time are flagged because they cannot be observations. (2) For the last 180 days of each city, weekly statistics (Monday-Sunday weeks) are computed for pm2_5 and us_aqi: the share of hours equal to the previous hour, the median number of decimal places, and the lag-1 autocorrelation. A week is flagged when its repeated-value share exceeds 3 times the 2023 baseline of the same city and column, or when its median decimal places fall below the 2023 baseline. The lag-1 autocorrelation is reported for review but does not flag a week. forecast_like_start is the earliest flagged timestamp over both checks and both cities. Rows are flagged only, never removed.

### Summary per city

| city | last timestamp | rows after run time | first row after run time | flagged weeks | earliest flagged week row | forecast-like start | affected range | affected rows | rows after file modified time |
|---|---|---|---|---|---|---|---|---|---|
| Peshawar | 2026-11-19 23:00:00 | 0 | none | 0 of 27 | none | none | none | 0 | 2 |
| Lahore | 2026-11-19 23:00:00 | 0 | none | 0 of 27 | none | none | none | 0 | 2 |

### 2023 baseline per city and column

| city / column | hours | repeated % | median decimals | lag-1 autocorr | repeat flag threshold % |
|---|---|---|---|---|---|
| Peshawar / pm2_5 | 8760 | 1.59 | 1 | 0.95 | 4.76 |
| Peshawar / us_aqi | 8760 | 36.71 | 0 | 0.96 | 110.14 |
| Lahore / pm2_5 | 8760 | 1.61 | 1 | 0.98 | 4.83 |
| Lahore / us_aqi | 8760 | 31.35 | 0 | 0.99 | 94.04 |

### Weekly statistics, last 180 days, Peshawar

| week_start | hours | pm2_5 repeated % | pm2_5 median decimals | pm2_5 lag-1 autocorr | pm2_5 flagged | us_aqi repeated % | us_aqi median decimals | us_aqi lag-1 autocorr | us_aqi flagged |
|---|---|---|---|---|---|---|---|---|---|
| 2026-05-18 00:00:00 | 24 | 4.17 | 1 | 0.94 | no | 37.50 | 0 | 0.79 | no |
| 2026-05-25 00:00:00 | 168 | 1.19 | 1 | 0.96 | no | 29.76 | 0 | 0.97 | no |
| 2026-06-01 00:00:00 | 168 | 1.79 | 1 | 0.96 | no | 40.48 | 0 | 0.98 | no |
| 2026-06-08 00:00:00 | 168 | 2.38 | 1 | 0.95 | no | 35.71 | 0 | 0.97 | no |
| 2026-06-15 00:00:00 | 168 | 1.19 | 1 | 0.94 | no | 25 | 0 | 0.93 | no |
| 2026-06-22 00:00:00 | 168 | 0.60 | 1 | 0.95 | no | 26.19 | 0 | 0.94 | no |
| 2026-06-29 00:00:00 | 168 | 1.19 | 1 | 0.92 | no | 16.07 | 0 | 0.93 | no |
| 2026-07-06 00:00:00 | 168 | 0.60 | 1 | 0.94 | no | 19.05 | 0 | 0.93 | no |
| 2026-07-13 00:00:00 | 168 | 2.38 | 1 | 0.95 | no | 27.38 | 0 | 0.94 | no |
| 2026-07-20 00:00:00 | 168 | 2.98 | 1 | 0.93 | no | 26.19 | 0 | 0.95 | no |
| 2026-07-27 00:00:00 | 168 | 1.79 | 1 | 0.96 | no | 24.40 | 0 | 0.93 | no |
| 2026-08-03 00:00:00 | 168 | 2.98 | 1 | 0.93 | no | 32.74 | 0 | 0.94 | no |
| 2026-08-10 00:00:00 | 168 | 1.79 | 1 | 0.90 | no | 16.67 | 0 | 0.89 | no |
| 2026-08-17 00:00:00 | 168 | 1.19 | 1 | 0.93 | no | 16.67 | 0 | 0.93 | no |
| 2026-08-24 00:00:00 | 168 | 0 | 1 | 0.94 | no | 33.93 | 0 | 0.91 | no |
| 2026-08-31 00:00:00 | 168 | 0.60 | 1 | 0.94 | no | 14.88 | 0 | 0.93 | no |
| 2026-09-07 00:00:00 | 168 | 0 | 1 | 0.92 | no | 39.29 | 0 | 0.84 | no |
| 2026-09-14 00:00:00 | 168 | 0.60 | 1 | 0.95 | no | 23.81 | 0 | 0.91 | no |
| 2026-09-21 00:00:00 | 168 | 0.60 | 1 | 0.94 | no | 35.71 | 0 | 0.94 | no |
| 2026-09-28 00:00:00 | 168 | 1.19 | 1 | 0.92 | no | 41.07 | 0 | 0.88 | no |
| 2026-10-05 00:00:00 | 168 | 0.60 | 1 | 0.94 | no | 33.33 | 0 | 0.97 | no |
| 2026-10-12 00:00:00 | 168 | 1.79 | 1 | 0.93 | no | 38.10 | 0 | 0.98 | no |
| 2026-10-19 00:00:00 | 168 | 0.60 | 1 | 0.94 | no | 39.29 | 0 | 1.00 | no |
| 2026-10-26 00:00:00 | 168 | 3.57 | 1 | 0.94 | no | 51.19 | 0 | 1.00 | no |
| 2026-11-02 00:00:00 | 168 | 0.60 | 1 | 0.93 | no | 48.21 | 0 | 0.99 | no |
| 2026-11-09 00:00:00 | 168 | 0.60 | 1 | 0.92 | no | 26.19 | 0 | 0.99 | no |
| 2026-11-16 00:00:00 | 96 | 1.04 | 1 | 0.92 | no | 27.08 | 0 | 0.99 | no |

### Weekly statistics, last 180 days, Lahore

| week_start | hours | pm2_5 repeated % | pm2_5 median decimals | pm2_5 lag-1 autocorr | pm2_5 flagged | us_aqi repeated % | us_aqi median decimals | us_aqi lag-1 autocorr | us_aqi flagged |
|---|---|---|---|---|---|---|---|---|---|
| 2026-05-18 00:00:00 | 24 | 0 | 1 | 0.85 | no | 8.33 | 0 | 0.81 | no |
| 2026-05-25 00:00:00 | 168 | 1.79 | 1 | 0.96 | no | 14.88 | 0 | 0.98 | no |
| 2026-06-01 00:00:00 | 168 | 0.60 | 1 | 0.93 | no | 13.69 | 0 | 0.92 | no |
| 2026-06-08 00:00:00 | 168 | 0.60 | 1 | 0.96 | no | 15.48 | 0 | 0.98 | no |
| 2026-06-15 00:00:00 | 168 | 1.19 | 1 | 0.95 | no | 17.86 | 0 | 0.93 | no |
| 2026-06-22 00:00:00 | 168 | 2.38 | 1 | 0.96 | no | 19.64 | 0 | 0.94 | no |
| 2026-06-29 00:00:00 | 168 | 2.98 | 1 | 0.98 | no | 8.93 | 0 | 0.99 | no |
| 2026-07-06 00:00:00 | 168 | 2.98 | 1 | 0.93 | no | 26.79 | 0 | 0.93 | no |
| 2026-07-13 00:00:00 | 168 | 1.79 | 1 | 0.94 | no | 50 | 0 | 0.91 | no |
| 2026-07-20 00:00:00 | 168 | 0.60 | 1 | 0.97 | no | 27.98 | 0 | 0.99 | no |
| 2026-07-27 00:00:00 | 168 | 2.98 | 1 | 0.96 | no | 17.86 | 0 | 0.96 | no |
| 2026-08-03 00:00:00 | 168 | 2.38 | 1 | 0.94 | no | 20.24 | 0 | 0.93 | no |
| 2026-08-10 00:00:00 | 168 | 0 | 1 | 0.94 | no | 20.83 | 0 | 0.93 | no |
| 2026-08-17 00:00:00 | 168 | 1.79 | 1 | 0.91 | no | 37.50 | 0 | 0.94 | no |
| 2026-08-24 00:00:00 | 168 | 1.19 | 1 | 0.91 | no | 48.21 | 0 | 0.87 | no |
| 2026-08-31 00:00:00 | 168 | 2.38 | 1 | 0.93 | no | 36.31 | 0 | 0.92 | no |
| 2026-09-07 00:00:00 | 168 | 0.60 | 1 | 0.96 | no | 32.74 | 0 | 0.97 | no |
| 2026-09-14 00:00:00 | 168 | 2.38 | 1 | 0.96 | no | 17.26 | 0 | 0.95 | no |
| 2026-09-21 00:00:00 | 168 | 1.79 | 1 | 0.91 | no | 39.88 | 0 | 0.87 | no |
| 2026-09-28 00:00:00 | 168 | 1.79 | 1 | 0.95 | no | 26.79 | 0 | 0.93 | no |
| 2026-10-05 00:00:00 | 168 | 1.79 | 1 | 0.91 | no | 32.14 | 0 | 0.96 | no |
| 2026-10-12 00:00:00 | 168 | 0 | 1 | 0.92 | no | 30.95 | 0 | 0.98 | no |
| 2026-10-19 00:00:00 | 168 | 2.38 | 1 | 0.95 | no | 39.29 | 0 | 1.00 | no |
| 2026-10-26 00:00:00 | 168 | 1.19 | 1 | 0.94 | no | 45.24 | 0 | 1.00 | no |
| 2026-11-02 00:00:00 | 168 | 0.60 | 1 | 0.95 | no | 31.55 | 0 | 1.00 | no |
| 2026-11-09 00:00:00 | 168 | 0.60 | 1 | 0.95 | no | 22.02 | 0 | 1.00 | no |
| 2026-11-16 00:00:00 | 96 | 0 | 1 | 0.95 | no | 30.21 | 0 | 1.00 | no |

Result saved to `data/processed/audit_flags.json`. Rows are flagged only and never removed.

## Approved Gap_Strategy

Approved by the user at the Phase 1 checkpoint on 2026-11-21. Phase 2 applies it in `AirQualityDataPipeline._apply_gap_strategy` (`src/pipeline.py`).

1. Reindex each city to a complete hourly grid and drop duplicate (city, datetime) timestamps, keeping the first row. This is a safety net: the audit's counts are in "Duplicate timestamps per city" and "Hourly gaps", and the pipeline prints how many rows it inserts and drops.
2. Leave the Leading_Gap (the leading missing pollutant and us_aqi hours per city, shown as "leading run" in "Missing values per column per city") unimputed. Those rows are removed as warm-up rows by the split.
3. Forward-fill interior pollutant and us_aqi gaps of up to 3 consecutive hours, within each city only. A gap longer than 3 hours is left entirely missing, not partly filled. Weather columns are unchanged.
4. Clip negative pollutant concentrations to 0, treated as model artifacts. At approval they were found in ozone and nitrogen_dioxide (see "Impossible values per city"). The pipeline prints how many values it changed.
5. Keep us_aqi values above 500. They map to Hazardous (Emergency) and are listed in the README limitations.
6. Keep all rows. At approval no Forecast_Like_Rows were flagged (see "Forecast-like rows"). The rows after the master file's last-modified time (column "rows after file modified time" in that section) are kept and mentioned in the README limitations.
7. aqi_category is kept as-is and is never used as a feature.

## Interpretation notes

- us_aqi may be the maximum of several pollutant sub-indices (for example PM2.5, PM10, ozone, nitrogen dioxide, sulphur dioxide, and carbon monoxide), not the PM2.5 sub-index alone. A mismatch between us_aqi and a PM2.5-only sub-index is therefore expected and is not by itself a data error.
- The provenance of the Master_Dataset must be confirmed by the student. The values may be modeled or reanalysis data (for example from an air-quality forecast service) rather than readings from ground sensors, and the project conclusions must be worded accordingly.
