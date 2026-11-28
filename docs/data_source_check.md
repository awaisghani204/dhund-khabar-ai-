# Data source check: where does `pakistan_smog_master_hourly.csv` come from?

Task 18 of the spec. This document supersedes the provenance caveat in `docs/data_audit.md`. That file keeps its original wording (the Phase 1 audit script writes it, and the sentence there is a required Phase 1 statement). Read the two together: the audit said "not confirmed", this check tested the question.

All numbers below were read from `docs/data_source_check_numbers.json` (110 request URLs, 24 failed requests, 470 result rows of which 446 are comparisons and 24 are error markers). The table in section 4 was built from that file by a throwaway script.

---

## 1. Purpose

The README and the audit said the data source was unconfirmed and that the column names look like Open-Meteo output. This check tests that guess by downloading the same cities, hours and variables from Open-Meteo and comparing them with the CSV, value by value.

## 2. What was searched in the project, and what was found

Searched: README, documentation, file metadata, notebooks, download scripts, `Zone.Identifier` streams, git commit messages, and a OneNote folder in the parent directory.

Found:
- No file documents the source. The only provenance statements in the project are the builder's own guesses from column names (README, `docs/final_deliverable.md`, a comment in `app.py`, the audit text).
- The four CSVs in `data/` were all created on 2026-11-19 between 21:18:35 and 21:18:50. The master file was added to git in the first commit (2026-11-23).
- The OneNote folder had no hits. OneNote files are binary, so this is not conclusive.
- Not found: who prepared the files, with what script, under what licence, and the exact download date.

## 3. Method

**Locations.** Peshawar `latitude=34.0151, longitude=71.5249`. Lahore `latitude=31.5204, longitude=74.3587`. These are the coordinates in the CSV's `latitude` and `longitude` columns and the ones used in every URL.

**Windows.** Five 3-day windows starting on 2023-01-10, 2024-12-10, 2025-12-10, 2026-01-10 and 2026-10-10 (hours 2023-01-10 00:00 to 2023-01-12 23:00, and so on), n = 72 hours per window, per city, per variable. Each request asked for a 5-day range (the day before the window to the day after) so that the timezone-shift test had data on both sides; the comparison used the 72 hours of the window.

**APIs and variables.**

| Short name | Endpoint | Variables |
|---|---|---|
| aq | `air-quality-api.open-meteo.com/v1/air-quality` | pm10, pm2_5, carbon_monoxide, nitrogen_dioxide, sulphur_dioxide, ozone, dust, us_aqi |
| archive | `archive-api.open-meteo.com/v1/archive` | temperature_2m, relative_humidity_2m, surface_pressure, wind_speed_10m, wind_direction_10m, precipitation |
| histfc | `historical-forecast-api.open-meteo.com/v1/forecast` | same six weather variables |
| forecast | `api.open-meteo.com/v1/forecast` | same six weather variables |

**Timezone variants.** `timezone=Asia/Karachi` (written PKT in the table) and `timezone=GMT`. For each variable the script also shifted the API series by -5 to +5 hours against the CSV and recorded the mean absolute difference per shift (`mad_by_shift` in the JSON) and the best shift.

**Unit variant.** Weather requests were repeated with `wind_speed_unit=ms` (written `PKT+ms`) to find the unit of the CSV wind column from the ratio CSV / API.

**Matching.** For each city, window, source, variant and variable, the API series and the CSV series were aligned by timestamp. The script recorded: `exact` (share of hours where the two values are equal), `mad` (mean absolute difference), `maxad` (largest absolute difference), `corr` (Pearson correlation), `bias`, `ratio` and `best_shift`. A correlation is undefined when one series is constant (see the note under the table).

The request URLs are in the JSON (`urls`) and failed requests are in `errors`.

## 4. Results

Built from the 446 comparison rows in the JSON. A "row" is one city x window x variable. All rows have n = 72.

| Source | Variant | Rows | Variables | Windows | Exact share (min / mean) | Mean abs diff (mean / max) | Max abs diff | Correlation (min) | Best shift (count of rows) |
|---|---|---|---|---|---|---|---|---|---|
| aq | PKT | 80 | 8 | 5 | 1 / 1 | 0 / 0 | 0 | 1 | 0 (80) |
| archive | PKT | 60 | 6 | 5 | 1 / 1 | 0 / 0 | 0 | 1 (see note) | 0 (54), -5 (6, see note) |
| histfc | PKT | 60 | 6 | 5 | 1 / 1 | 0 / 0 | 0 | 1 (see note) | 0 (54), -5 (6, see note) |
| forecast | PKT | 12 | 6 | 1 (2026-10-10) | 1 / 1 | 0 / 0 | 0 | 1 | 0 (12) |
| aq | GMT | 80 | 8 | 5 | 0 / 0.062 | 171.1 / 1810 | 5857 | -0.39 | +5 (80) |
| archive | GMT | 60 | 6 | 5 | 0 / 0.164 | 17.87 / 112.2 | 335 | -0.08 | +5 (54), -5 (6, see note) |
| histfc | GMT | 60 | 6 | 5 | 0 / 0.164 | 17.87 / 112.2 | 335 | -0.08 | +5 (54), -5 (6, see note) |
| forecast | GMT | 12 | 6 | 1 | 0 / 0.117 | 17.84 / 108 | 318 | -0.05 | +5 (12) |
| archive | PKT+ms (wind only) | 10 | 1 | 5 | 0 / 0.003 | 2.914 / 5.541 | 12.48 | 0.9995 | no clear shift |
| histfc | PKT+ms (wind only) | 10 | 1 | 5 | 0 / 0.003 | 2.914 / 5.541 | 12.48 | 0.9995 | no clear shift |
| forecast | PKT+ms (wind only) | 2 | 1 | 1 | 0 / 0.007 | 4.633 / 5.541 | 12.48 | 0.9999 | no clear shift |

Notes on the table:
- **Exact match.** With `timezone=Asia/Karachi`, every one of the 212 comparison rows (80 aq + 60 archive + 60 histfc + 12 forecast) has exact share 1, mean absolute difference 0 and maximum absolute difference 0. In the 5 windows x 2 cities, every tested value is identical to the CSV. For the weather variables this holds for the archive and historical-forecast APIs in all 5 windows, and for the regular forecast API in the 2026-10-10 window only (the only window that API accepts, see section 7).
- **Precipitation.** In 6 rows of the archive and of the historical-forecast API the correlation is undefined (JSON `null`), because the series is constant (no rain in the window, all zeros). Those rows still match exactly (exact share 1). The "best shift" of -5 in those rows is a tie of equal zero differences, not a finding.
- **Wind in m/s.** The `PKT+ms` rows have a mean ratio CSV / API of about 3.60 (the printed ratios run from 3.599 to 3.61). One m/s is 3.6 km/h, so the CSV wind column is in km/h. Rows with `PKT+ms` are not matches by design, and the best shift there is noise (the series differ by a constant factor).
- **JSON units field.** The JSON field `hourly_units` shows `m/s` for the weather sources. That is an artifact of the `wind_speed_unit=ms` requests that overwrote the stored units. Rely on the ratios, not on that field.
- **Returned grid cell.** The API returns the centre of the grid cell it used (`grid_coords_returned` in the JSON): the air-quality cell is `34.0, 71.5` for Peshawar and `31.5, 74.40001` for Lahore, and the weather cell is `34.059753, 71.549995` and `31.528997, 74.38995`. These are slightly different from the requested point. The CSV matches what the API returns for the requested coordinates.

## 5. Timezone and units

**Timezone.** The CSV timestamps are in Asia/Karachi (UTC+5). With `timezone=Asia/Karachi` the best shift is 0 for all rows that have a defined best shift. With `timezone=GMT` the best shift is +5 hours (all 80 aq rows, 54 of 60 archive rows, and so on), which is the same 5-hour offset seen from the other side. The 6 rows with shift -5 are the constant precipitation series described above.

**Units** (from the exact match with the API, which states these units, and from the wind ratio):

| Column | Unit |
|---|---|
| pm10, pm2_5, nitrogen_dioxide, sulphur_dioxide, ozone, dust, carbon_monoxide | µg/m³ (the API reports µg/m³ for all of them, including carbon monoxide) |
| us_aqi | US AQI index |
| temperature_2m | °C |
| relative_humidity_2m | % |
| surface_pressure | hPa |
| wind_speed_10m | km/h (ratio 3.6 against m/s) |
| wind_direction_10m | degrees |
| precipitation | mm |

## 6. The ERA5 side test (run once, not saved)

A one-off test requested the archive API with `models=era5`. That series differs from the CSV (up to 6.6 °C in temperature, 2.0 hPa in pressure and 9.3 km/h in wind speed). **This test was run once by hand and is not stored in `docs/data_source_check_numbers.json`, so it cannot be reproduced from the file.** If it matters, repeat it with the URL pattern in section 10 plus `&models=era5`.

Reading: the CSV weather is Open-Meteo's default weather series (the "best match" blend of models), not pure ERA5 reanalysis.

## 7. Failed requests

24 of the 110 requests failed. All 24 returned HTTP 400, and all 24 went to `api.open-meteo.com/v1/forecast` (the regular forecast API). The JSON error text for the ones I read says the start date is out of the allowed range from 2026-08-26 to 2026-12-12. The failed requests cover the four windows before that range (2023-01-10, 2024-12-10, 2025-12-10, 2026-01-10) for 2 cities and 3 variants (8 per variant). The other 86 requests returned data. (No HTTP 503 errors appear in the stored `errors` list.)

So the regular forecast API was compared only for 2026-10-10, and the three weather endpoints (archive, historical-forecast, forecast) return identical values in that window. The comparison therefore cannot tell which of the three the file came from.

## 8. Verdict

**The CSV values match Open-Meteo's Air Quality API and Open-Meteo's default weather series exactly in every tested window (5 windows x 2 cities, 72 hours each, 8 air-quality and 6 weather variables). This is strong evidence that the file was produced from Open-Meteo.** The chance of identical values to the last decimal in 212 city/window/variable series by coincidence is not a realistic explanation.

**What the Open-Meteo values are.** According to the [Open-Meteo Air Quality API documentation](https://open-meteo.com/en/docs/air-quality-api), the air-quality forecasts are based on CAMS (Copernicus Atmosphere Monitoring Service) atmospheric-composition model data: the 11 km European domain and the 45 km global domain. Pakistan is outside the European domain, so the values for these two cities come from the coarse global domain (the page states that it has a 3-hourly step, so the hourly series is derived from it; I did not test how). The documentation calls its output a "forecast" and describes archived forecasts for past data. In short: **the air-quality columns (including `us_aqi`) are atmospheric-model output, not ground-sensor readings.** The weather columns are Open-Meteo's default weather-model series, also not station measurements. Content was rephrased for compliance with licensing restrictions.

The returned grid cell for air quality (34.0, 71.5 and 31.5, 74.4) is consistent with a coarse model grid and not with a monitoring station in the city.

## 9. What remains unknown

- **Who prepared the file, when, and with which script.** Nothing in the project says.
- **Whether the CSV was downloaded on 2026-11-19.** The file times (2026-11-19 21:18) show when the file was created on this machine, not when the data was downloaded.
- **Rows from 2022-08-01 onward** (the earliest rows) were not tested. The earliest tested window is 2023-01-10.
- **The regular forecast API for dates before 2026-08-26.** It refuses those dates, so it could not be compared.
- **What the API returned at download time.** An exact match today does not prove the API returned the same values when the file was made. Open-Meteo's archived and reanalysis-based series can be revised. The match shows the file is consistent with what the API returns now.
- **Licence and attribution.** Open-Meteo asks users to credit Open-Meteo and the CAMS data provider (see the documentation above). The dashboard footer now links to Open-Meteo and states non-commercial use.

## 10. Implication for this project

1. **Results describe model output.** Every accuracy figure in the README is about forecasting a model product (CAMS-based air-quality values and Open-Meteo weather), not about real measured air. Hourly model fields are smooth, which may make them easier for simple models to predict (this is an inference, not tested). The project cannot claim the dashboard predicts what a sensor would read.
2. **The last hours of the file are probably not observations (inference, not tested by the API comparison).** The file's last timestamp is 2026-11-19 23:00, but the file was created at 21:18 on 2026-11-19. The last 2 hours per city (22:00 and 23:00) are after the file's creation time, so they could not have been observed when the file was made. Assuming the timestamps and the file clock are both Pakistan local time, those hours are likely model forecast values. The same may hold for recent days before that, because model "analysis" and "forecast" values are not distinguished in the file. This is an inference from timestamps. It was not tested.
3. **The test period (2025-01-01 to 2026-11-16) contains recent days that may be forecast rather than analysis values.** Untested.

## 11. Reproducing the check

The throwaway script lived in `%TEMP%` and is gone. It must be rewritten to rerun the full comparison; it is not part of the project. Steps:

1. Request the URLs below (or any in `docs/data_source_check_numbers.json`, key `urls`). The API needs no key for light use.
2. Read the CSV `data/pakistan_smog_master_hourly.csv`, keep one city (`latitude`, `longitude`), set `datetime` as the index.
3. Compare each variable for the 72 hours of the window.

Example URLs (copied from the JSON; Lahore, 2023-01-10 window and 2026-10-10 window):

```
https://air-quality-api.open-meteo.com/v1/air-quality?latitude=31.5204&longitude=74.3587&hourly=pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,dust,us_aqi&start_date=2023-01-09&end_date=2023-01-13&timezone=Asia%2FKarachi
https://archive-api.open-meteo.com/v1/archive?latitude=31.5204&longitude=74.3587&hourly=temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,wind_direction_10m,precipitation&start_date=2023-01-09&end_date=2023-01-13&timezone=Asia%2FKarachi
https://historical-forecast-api.open-meteo.com/v1/forecast?latitude=31.5204&longitude=74.3587&hourly=temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,wind_direction_10m,precipitation&start_date=2023-01-09&end_date=2023-01-13&timezone=Asia%2FKarachi
https://api.open-meteo.com/v1/forecast?latitude=31.5204&longitude=74.3587&hourly=temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,wind_direction_10m,precipitation&start_date=2026-10-09&end_date=2026-10-13&timezone=Asia%2FKarachi
```

The last URL works only for dates inside the allowed range (at the time of the check, 2026-08-26 to 2026-12-12). The first three URLs are for 2023-01-10 and the Peshawar equivalents use `latitude=34.0151&longitude=71.5249`.

Short standalone snippet for one series (Lahore pm2_5, 2023-01-10 to 2023-01-12):

```python
import json, urllib.request, pandas as pd
url = ("https://air-quality-api.open-meteo.com/v1/air-quality?latitude=31.5204&longitude=74.3587"
       "&hourly=pm2_5&start_date=2023-01-10&end_date=2023-01-12&timezone=Asia%2FKarachi")
h = json.load(urllib.request.urlopen(url))["hourly"]
api = pd.Series(h["pm2_5"], index=pd.to_datetime(h["time"]))
csv = pd.read_csv("data/pakistan_smog_master_hourly.csv", parse_dates=["datetime"])
csv = csv[csv["city"] == "Lahore"].set_index("datetime")["pm2_5"]
print((csv.reindex(api.index) == api).mean())   # 1.0 means every hour is identical
```

(The snippet was written for this document and has not been run as written; the check itself used a different script. If the `city` column spelling or the `datetime` column differs, adjust it.)

## 12. Follow-ups

- **No `src/live_data.py` exists and the project has no live-data module.** This check did not create one. If a live feed is added later, it must cite this document for its source: Open-Meteo Air Quality API plus Weather API, timezone `Asia/Karachi`, wind in km/h, concentrations in µg/m³, the same variable names, and the same coordinates (Peshawar 34.0151, 71.5249; Lahore 31.5204, 74.3587).
- A live feed would also need the same default weather series ("best match", no `models=` parameter), and a rule for how forecast hours are treated (for example whether the model is scored on forecast hours at all), because the file's last hours are probably forecast values.
- Add the Open-Meteo and CAMS attribution to the dashboard footer if the project is published.
- Optional: test the earliest rows (2022-08) and keep a copy of the comparison script in the project if the check needs to be repeated.

## 13. Live feed check before trusting recent hours

On 2026-11-28 a call to `https://api.open-meteo.com/v1/forecast` with `past_days=1` and `forecast_days=0` for Lahore returned HTTP 200, timezone Asia/Karachi, temperature in degrees C, and 24 hourly rows for 2026-11-27 (the last completed day; today was not included). `src.live_data.load_recent("Lahore", 1)` returned the same window and the notice `Forecast is based on 2026-11-27 23:00 (Open-Meteo, Asia/Karachi).`

The overlapping historical comparison is section 8: exact match on every tested hour that exists in both the CSV and the API. There is no systematic difference, so this project does not retrain on a new download. Retraining stays a manual step (`python scripts/train_all.py`). The dashboard only loads the saved models.

`forecast_days=0` returns completed past days only. Hours later than the Pakistan clock are dropped if they appear.
