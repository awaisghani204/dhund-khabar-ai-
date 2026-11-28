"""Dhund-Khabar AI Streamlit dashboard.

Run from the project root with ``streamlit run app.py``. The dashboard only
reads files that ``python scripts/train_all.py`` has already written:
``data/processed/features_all.csv``, ``results/metrics.csv`` and the saved
``models/*.joblib`` files. It never trains a model (Requirement 17.19).

The visual style is adapted from the approved Stitch screen "Apple Inspired
Atmospheric Terminal" (``docs/stitch/dashboard_approved.html``): a near-black
page, frosted glass cards, compact uppercase mono labels and Urdu labels next
to the English ones. Content the data cannot support (confidence bands, live
clocks, invented statuses) was dropped at the design checkpoint; see
``docs/stitch/design_system.md``.

The file is organised as plain functions: a style layer (``inject_css``),
cached loaders, small formatting helpers, and one ``render_*`` function per
page section.
"""

from __future__ import annotations

import html
import math
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# src.charts selects the Agg backend before pyplot is imported, so it must be
# imported before matplotlib.pyplot.
from src import alerts, charts, live_data, utils
from src.model import SmogPredictorModel
from src.pipeline import DATETIME_FORMAT, MODEL_DATASET_FILENAME, AirQualityDataPipeline

import matplotlib.pyplot as plt  # noqa: E402  (after src.charts, see above)

# --------------------------------------------------------------------------
# Paths and constants
# --------------------------------------------------------------------------

FEATURES_PATH: Path = utils.PROCESSED_DIR / "features_all.csv"
METRICS_PATH: Path = utils.RESULTS_DIR / "metrics.csv"
# Every file the dashboard reads is produced by this one script, so the
# missing-file messages can always point at it.
TRAIN_SCRIPT: str = "python scripts/train_all.py"

# The dashboard default is the Forecast_Mode Random Forest (Phase 3 decision,
# option a) plus the Simulation_Mode Random Forest used by the What-If lab.
DEFAULT_MODEL_TYPE: str = "random_forest"
SIMULATION_TARGET: str = "us_aqi"
SIMULATION_HORIZON: int = 24

# Format of the "Latest Available Data" timestamp in the header badge.
TIMESTAMP_FORMAT: str = "%Y-%m-%d %H:%M"

# Urdu city names shown next to the English ones, as in the approved design.
CITY_URDU: dict[str, str] = {"Peshawar": "پشاور", "Lahore": "لاہور"}

# Target the hero card and summary cards predict (Requirement 17.4).
HERO_TARGET: str = "us_aqi"
# Human-readable name of the default model, printed on the hero card.
MODEL_LABEL: str = "Random Forest, forecast mode"

# The gauge and the tier bar map AQI 0-500 linearly. Values above 500 put the
# marker at the end of the scale, while the printed number stays the real one.
GAUGE_MAX_AQI: float = 500.0
# Tier segments on the 0-500 scale, (lower, upper, colour), taken from
# alerts.TIERS so the gauge can never disagree with the tier rules. The last
# tier is open-ended in alerts, so it is drawn up to GAUGE_MAX_AQI.
GAUGE_SEGMENTS: list[tuple[float, float, str]] = [
    (float(t["lower"]), min(float(t["upper"]), GAUGE_MAX_AQI), t["color"])
    for t in alerts.TIERS
]
# AQI values printed under the gauge and the tier bar (tier boundaries).
GAUGE_TICKS: list[int] = [0, 50, 100, 150, 200, 300, 500]
# Tier index from alerts.aqi_to_tier_index at and above which the hero pulses:
# 4 = Very Unhealthy, 5 = Hazardous (Emergency) (Requirement 17.6).
PULSE_TIER_INDEX: int = 4

# Pollutants shown in the hero mini row: (column, label). The file matches the
# Open-Meteo Air Quality API exactly, which reports all four in µg/m³, including
# carbon monoxide (values around 1000 in the data). See docs/data_source_check.md.
HERO_POLLUTANTS: list[tuple[str, str]] = [
    ("pm2_5", "PM2.5"),
    ("pm10", "PM10"),
    ("nitrogen_dioxide", "NO₂"),
    ("carbon_monoxide", "CO"),
]
POLLUTANT_UNIT: str = "µg/m³"

# Measured model sensitivities written by the Training_Script (Requirement
# 13.9-13.10). The What-If lab quotes them so users see how strongly the
# simulation model actually reacts to the sliders.
SENSITIVITY_PATH: Path = utils.RESULTS_DIR / "sensitivity.csv"
# "Cold" in sensitivity.csv means Test_Period hours at or below this
# temperature; it mirrors COLD_TEMPERATURE_C in scripts/train_all.py, which is
# not imported because that module pulls in the whole training stack.
SENSITIVITY_COLD_C: float = 15.0

# What-If slider steps; the ranges themselves come from utils.SLIDER_RANGES.
SLIDER_STEPS: dict[str, float] = {
    "wind_speed_10m": 0.5,
    "temperature_2m": 0.5,
    "relative_humidity_2m": 1.0,
}
# The Simulation_Mode model exists only for us_aqi at 24 h, so the lab names
# its horizon explicitly instead of following the forecast horizon selector.
SIMULATION_LABEL: str = "Simulation model, 24 h horizon"
# Caption required verbatim in the What-If lab (Requirement 17.13).
WHATIF_CAPTION: str = "Scenario exploration from a statistical model, not a physical simulation."

# 16-point compass names, clockwise from north, 22.5 degrees apart.
COMPASS_POINTS: list[str] = [
    "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
    "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW",
]

# Visual Analytics (Requirement 17.14). The actual-vs-predicted tab rebuilds
# the Test_Period predictions from the model dataset, exactly as training did.
MODEL_DATASET_PATH: Path = utils.PROCESSED_DIR / MODEL_DATASET_FILENAME
ANALYTICS_TARGET: str = "us_aqi"
# Default chart window: the 2025-26 winter, where the smog episodes are. The
# picker itself is limited to the Test_Period.
WINDOW_DEFAULT_START: date = pd.Timestamp(charts.DEFAULT_WINTER_START).date()
WINDOW_DEFAULT_END: date = pd.Timestamp(charts.DEFAULT_WINTER_END).date()
TEST_START: date = pd.Timestamp(utils.SPLIT_DATE).date()
IMPORTANCE_TOP_N: int = 10
# Display names and row order of the model comparison table.
MODEL_NAMES: dict[str, str] = {
    "persistence": "Persistence",
    "linear_regression": "Linear Regression",
    "gradient_boosting": "Gradient Boosting",
    "random_forest": "Random Forest",
}
TARGET_LABELS: dict[str, str] = {"us_aqi": "US AQI", "pm2_5": "PM2.5"}
SUBSET_LABELS: dict[str, str] = {"all": "All test hours", "winter": "Winter (Nov-Feb)"}
# Row highlight for the lowest MAE in the comparison table (green accent).
BEST_ROW_STYLE: str = "background-color: rgba(48, 209, 88, 0.18); font-weight: 600"

# Policy terminal rows: (advisory key, title, icon). The texts themselves come
# from get_risk_advisory, so the terminal can never disagree with the tiers.
POLICY_ROWS: list[tuple[str, str, str]] = [
    ("school_status", "School Operations Status", "🏫"),
    ("respiratory_guidance", "Respiratory Protective Guidance", "😷"),
    ("industrial_advisory", "Industrial Advisory", "🏭"),
]

# Footer text required verbatim on every view (Requirement 17.16).
FOOTER_TEXT: str = (
    "Student learning project. Not an official health warning system. "
    "Follow guidance from Pakistan's health and environmental authorities."
)

# --------------------------------------------------------------------------
# Style layer
# --------------------------------------------------------------------------

# One CSS string adapted from the approved Stitch HTML. Tailwind utility
# classes are replaced by named ``dk-*`` classes so the HTML templates in the
# render functions stay short. The Google Fonts @import must be the first rule.
#
# Two deliberate notes:
# * The rules that hide Streamlit's chrome use the element selectors
#   ``header`` and ``footer`` (Requirement 16.4). Our own header and footer are
#   therefore plain ``<div>`` elements, otherwise they would be hidden too.
# * The six EPA tier colours are exact and must not be shifted. The tier bar
#   segments are sized in proportion to their AQI ranges on a 0-500 scale:
#   0-50, 51-100, 101-150, 151-200 are 10 % each, 201-300 is 20 %,
#   301-500 is 40 %.
DK_CSS: str = """
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600&family=Noto+Nastaliq+Urdu:wght@500;600&display=swap');
:root {
  color-scheme: dark;
  --dk-bg: #000000;
  --dk-canvas: #09090b;
  --dk-card: rgba(255, 255, 255, 0.035);
  --dk-card-subtle: rgba(255, 255, 255, 0.024);
  --dk-border: rgba(255, 255, 255, 0.08);
  --dk-border-hover: rgba(255, 255, 255, 0.14);
  --dk-text: #f5f5f7;
  --dk-muted: #86868b;
  --dk-secondary: #a1a1a6;
  --dk-tertiary: #636366;
  --dk-green: #30d158;
  --dk-amber: #FFB300;
  --dk-crimson: #FF3366;
  --dk-violet: #9933FF;
  --dk-tier-good: #00E400;
  --dk-tier-moderate: #FFFF00;
  --dk-tier-usg: #FF7E00;
  --dk-tier-unhealthy: #FF0000;
  --dk-tier-very-unhealthy: #8F3F97;
  --dk-tier-hazardous: #7E0023;
  --dk-font-head: 'Outfit', system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif;
  --dk-font-body: 'Plus Jakarta Sans', system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif;
  --dk-font-mono: 'JetBrains Mono', ui-monospace, 'SF Mono', Consolas, monospace;
  --dk-font-urdu: 'Noto Nastaliq Urdu', 'Jameel Noori Nastaleeq', 'Urdu Typesetting', serif;
  --dk-radius: 24px;
  --dk-gap: 16px;
}
/* Streamlit chrome: hidden so the page does not look like a default app. */
#MainMenu, header, footer,
[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stStatusWidget"], [data-testid="stHeaderActionElements"] {
  visibility: hidden; display: none !important;
}
html, body, .stApp, [data-testid="stAppViewContainer"] {
  background-color: var(--dk-bg) !important;
  color: var(--dk-text);
  font-family: var(--dk-font-body);
  -webkit-font-smoothing: antialiased;
  letter-spacing: -0.01em;
}
.stApp p, .stApp li, .stApp label, .stApp span { font-family: var(--dk-font-body); }
.stApp h1, .stApp h2, .stApp h3, .stApp h4 { font-family: var(--dk-font-head); color: var(--dk-text); letter-spacing: -0.02em; }
[data-testid="stMainBlockContainer"], .block-container {
  max-width: 1200px; padding-top: 1.5rem; padding-bottom: 5rem;
}
/* Responsive columns: consistent gap, stack on narrow screens. */
[data-testid="stHorizontalBlock"] { gap: var(--dk-gap); }
@media (max-width: 900px) {
  [data-testid="stHorizontalBlock"] { flex-wrap: wrap; }
  [data-testid="stHorizontalBlock"] > [data-testid="stColumn"],
  [data-testid="stHorizontalBlock"] > [data-testid="column"] {
    width: 100% !important; flex: 1 1 100% !important; min-width: 100% !important;
  }
}
/* Generic grid used inside HTML cards. */
.dk-grid { display: grid; gap: var(--dk-gap); grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); }
/* Typography helpers. */
.dk-num { font-family: var(--dk-font-head); font-variant-numeric: tabular-nums; font-weight: 600; letter-spacing: -0.03em; color: var(--dk-text); }
.dk-label { font-family: var(--dk-font-mono); font-size: 11px; font-weight: 500; text-transform: uppercase; letter-spacing: 0.08em; color: var(--dk-muted); }
.dk-muted { color: var(--dk-muted); }
.dk-urdu { font-family: var(--dk-font-urdu); font-weight: 500; line-height: 1.9; }
.dk-dot { display: inline-block; width: 6px; height: 6px; border-radius: 50%; background: var(--dk-green); margin-right: 6px; vertical-align: middle; }
/* Header. */
.dk-header {
  display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between;
  gap: 12px 24px; padding: 12px 0 20px 0; margin-bottom: 8px;
  border-bottom: 1px solid var(--dk-border);
}
.dk-brand { display: flex; align-items: center; gap: 12px; }
.dk-logo {
  width: 36px; height: 36px; border-radius: 10px; display: flex; align-items: center; justify-content: center;
  background: rgba(255, 255, 255, 0.06); border: 1px solid rgba(255, 255, 255, 0.1);
  font-family: var(--dk-font-head); font-weight: 700; color: var(--dk-text);
}
.stApp h1.dk-title { font-family: var(--dk-font-head); font-size: 28px; font-weight: 600; margin: 0; padding: 0; line-height: 1.3; }
.dk-title .dk-urdu { font-size: 22px; }
.dk-subtitle { font-family: var(--dk-font-mono); font-size: 11px; color: var(--dk-muted); text-transform: uppercase; letter-spacing: 0.08em; margin-top: 2px; }
.dk-header-meta { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; }
/* Pills and badges. */
.dk-badge {
  display: inline-flex; align-items: center; padding: 6px 14px; border-radius: 9999px;
  background: rgba(255, 255, 255, 0.06); border: 1px solid var(--dk-border);
  font-family: var(--dk-font-mono); font-size: 12px; color: rgba(255, 255, 255, 0.9);
  font-variant-numeric: tabular-nums;
}
.dk-pill {
  display: inline-flex; align-items: center; padding: 3px 10px; border-radius: 9999px;
  font-family: var(--dk-font-mono); font-size: 10px; text-transform: uppercase; letter-spacing: 0.08em;
  border: 1px solid var(--dk-border); color: var(--dk-secondary); background: var(--dk-card-subtle);
}
.dk-pill-historical { color: var(--dk-green); border-color: rgba(48, 209, 88, 0.35); background: rgba(48, 209, 88, 0.08); }
/* Glass cards. */
.dk-card {
  background: var(--dk-card); border: 1px solid var(--dk-border); border-radius: var(--dk-radius);
  backdrop-filter: blur(40px) saturate(180%); -webkit-backdrop-filter: blur(40px) saturate(180%);
  box-shadow: inset 0 1px 0 0 rgba(255, 255, 255, 0.08), 0 8px 32px 0 rgba(0, 0, 0, 0.4);
  padding: 24px; margin-bottom: var(--dk-gap); transition: border-color 0.2s ease;
}
.dk-card:hover { border-color: var(--dk-border-hover); }
.dk-card-subtle {
  background: var(--dk-card-subtle); border: 1px solid rgba(255, 255, 255, 0.06); border-radius: 16px;
  box-shadow: inset 0 1px 0 0 rgba(255, 255, 255, 0.05); padding: 14px 16px;
}
.dk-card-head { display: flex; align-items: center; justify-content: space-between; gap: 8px; margin-bottom: 12px; }
.dk-section-title { font-family: var(--dk-font-head); font-size: 20px; font-weight: 600; color: var(--dk-text); margin: 24px 0 12px 0; }
/* Hero card: background tinted by the tier colour through --dk-tier. */
.dk-hero {
  position: relative; overflow: hidden;
  background: linear-gradient(160deg, color-mix(in srgb, var(--dk-tier, #30d158) 16%, transparent) 0%, var(--dk-card) 70%);
  border-color: color-mix(in srgb, var(--dk-tier, #30d158) 35%, transparent);
}
.dk-hero-aqi { font-family: var(--dk-font-head); font-size: 88px; line-height: 1; font-weight: 600; letter-spacing: -0.04em; font-variant-numeric: tabular-nums; }
.dk-tier-pill {
  display: inline-flex; align-items: center; gap: 6px; padding: 3px 12px; border-radius: 9999px; font-size: 12px; font-weight: 500;
  color: var(--dk-text); border: 1px solid color-mix(in srgb, var(--dk-tier, #30d158) 45%, transparent);
  background: color-mix(in srgb, var(--dk-tier, #30d158) 18%, transparent);
}
/* Pulse for Emergency tiers (Very Unhealthy, Hazardous). */
@keyframes dk-pulse-glow {
  0%, 100% { box-shadow: inset 0 1px 0 0 rgba(255,255,255,0.08), 0 0 0 0 rgba(255, 51, 102, 0.0), 0 8px 32px rgba(0,0,0,0.4); }
  50% { box-shadow: inset 0 1px 0 0 rgba(255,255,255,0.08), 0 0 36px 4px rgba(153, 51, 255, 0.45), 0 8px 32px rgba(0,0,0,0.4); }
}
.dk-pulse { animation: dk-pulse-glow 2.6s ease-in-out infinite; }
@media (prefers-reduced-motion: reduce) { .dk-pulse { animation: none; box-shadow: 0 0 0 2px var(--dk-crimson); } }
/* Semi-circle gauge (inline SVG). */
.dk-gauge { width: 100%; max-width: 360px; margin: 8px auto 0 auto; display: block; }
.dk-gauge text { font-family: var(--dk-font-mono); fill: var(--dk-muted); font-size: 10px; }
/* Straight tier bar under the gauge, segments proportional to AQI ranges. */
.dk-spectrum { position: relative; margin: 18px 0 6px 0; }
.dk-spectrum-track { display: flex; height: 6px; border-radius: 9999px; overflow: hidden; }
.dk-spectrum-track span { display: block; height: 100%; }
.dk-seg-good { width: 10%; background: #00E400; }
.dk-seg-moderate { width: 10%; background: #FFFF00; }
.dk-seg-usg { width: 10%; background: #FF7E00; }
.dk-seg-unhealthy { width: 10%; background: #FF0000; }
.dk-seg-very-unhealthy { width: 20%; background: #8F3F97; }
.dk-seg-hazardous { width: 40%; background: #7E0023; }
.dk-spectrum-thumb {
  position: absolute; top: -5px; width: 16px; height: 16px; margin-left: -8px; border-radius: 50%;
  background: #ffffff; box-shadow: 0 3px 10px rgba(0, 0, 0, 0.5), 0 0 0 1px rgba(0, 0, 0, 0.1);
}
.dk-spectrum-scale { position: relative; height: 14px; margin-top: 6px; font-family: var(--dk-font-mono); font-size: 10px; color: var(--dk-tertiary); }
.dk-spectrum-scale span { position: absolute; transform: translateX(-50%); }
/* PM2.5 / PM10 / NO2 / CO mini row in the hero. */
.dk-mini-row { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 8px; margin-top: 18px; }
.dk-mini { background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.06); border-radius: 14px; padding: 10px 12px; }
.dk-mini .dk-num { font-size: 18px; }
@media (max-width: 600px) { .dk-mini-row { grid-template-columns: repeat(2, minmax(0, 1fr)); } .dk-hero-aqi { font-size: 64px; } }
/* Hero layout: number and labels on the left, gauge on the right. */
.dk-hero-grid { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 24px; align-items: center; }
@media (max-width: 800px) { .dk-hero-grid { grid-template-columns: 1fr; } }
.dk-hero-meta { font-size: 13px; color: var(--dk-secondary); line-height: 1.6; margin-top: 12px; }
.dk-hero-meta b { color: var(--dk-text); font-weight: 500; }
/* City summary cards (both cities side by side). */
.dk-summary-active { border-color: rgba(255, 255, 255, 0.28); }
.dk-summary-city { font-family: var(--dk-font-head); font-size: 20px; font-weight: 600; color: var(--dk-text); }
.dk-summary-row { display: flex; flex-wrap: wrap; gap: 20px; align-items: baseline; margin-top: 6px; }
.dk-summary-row .dk-num { font-size: 32px; }
.dk-summary-forecast { margin-top: 14px; padding-top: 12px; border-top: 1px solid var(--dk-border); }
/* Condition cards. */
.dk-metric { font-family: var(--dk-font-head); font-size: 40px; font-weight: 600; line-height: 1.1; font-variant-numeric: tabular-nums; }
.dk-unit { font-family: var(--dk-font-mono); font-size: 12px; color: var(--dk-muted); margin-left: 4px; }
.dk-delta-up { color: #ff453a; }
.dk-delta-down { color: var(--dk-green); }
.dk-compass {
  position: relative; width: 120px; height: 120px; margin: 8px auto; border-radius: 50%;
  border: 1px solid var(--dk-border); background: radial-gradient(circle, rgba(255,255,255,0.05) 0%, transparent 70%);
}
.dk-compass-label { position: absolute; font-family: var(--dk-font-mono); font-size: 10px; color: var(--dk-tertiary); }
.dk-compass-n { top: 4px; left: 50%; transform: translateX(-50%); }
.dk-compass-s { bottom: 4px; left: 50%; transform: translateX(-50%); }
.dk-compass-e { right: 6px; top: 50%; transform: translateY(-50%); }
.dk-compass-w { left: 6px; top: 50%; transform: translateY(-50%); }
.dk-arrow { position: absolute; inset: 0; display: flex; align-items: center; justify-content: center; transition: transform 0.4s ease; }
.dk-arrow svg { width: 64px; height: 64px; }
.dk-note { font-size: 12px; color: var(--dk-muted); margin-top: 8px; line-height: 1.5; }
.dk-cond-grid { display: grid; gap: var(--dk-gap); grid-template-columns: repeat(4, minmax(0, 1fr)); }
@media (max-width: 1000px) { .dk-cond-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
@media (max-width: 600px) { .dk-cond-grid { grid-template-columns: 1fr; } }
.dk-cond-grid .dk-card { margin-bottom: 0; height: 100%; }
.dk-warn-pill { color: #ff9f0a; border-color: rgba(255, 159, 10, 0.4); background: rgba(255, 159, 10, 0.08); }
.dk-wx-row { display: flex; justify-content: space-between; align-items: baseline; padding: 8px 0; border-top: 1px solid var(--dk-border); }
.dk-wx-row:first-of-type { border-top: none; }
.dk-wx-row .dk-num { font-size: 22px; }
.dk-scenario-aqi { font-family: var(--dk-font-head); font-size: 64px; line-height: 1; font-weight: 600; letter-spacing: -0.04em; font-variant-numeric: tabular-nums; }
/* Policy terminal. */
.dk-terminal { font-family: var(--dk-font-body); }
.dk-terminal-row {
  display: grid; grid-template-columns: 28px minmax(160px, 1fr) 2fr; gap: 12px; align-items: start;
  padding: 14px 0; border-top: 1px solid var(--dk-border);
}
.dk-terminal-row:first-child { border-top: none; }
.dk-check { width: 20px; height: 20px; border-radius: 6px; border: 1px solid var(--dk-green); color: var(--dk-green); display: flex; align-items: center; justify-content: center; font-size: 13px; }
@media (max-width: 700px) { .dk-terminal-row { grid-template-columns: 28px 1fr; } .dk-terminal-row > :last-child { grid-column: 2; } }
/* Missing-file message. */
.dk-missing { border-color: rgba(255, 159, 10, 0.4); background: rgba(255, 159, 10, 0.06); }
.dk-missing code { font-family: var(--dk-font-mono); font-size: 12px; color: #ff9f0a; background: transparent; }
/* Footer: a div, because the element selector `footer` is hidden above. */
.dk-footer {
  margin-top: 40px; padding: 20px 0 8px 0; border-top: 1px solid var(--dk-border);
  font-size: 12px; color: var(--dk-muted); text-align: center; line-height: 1.6;
}
/* Native Streamlit widgets styled to match the glass look. */
[data-testid="stButtonGroup"] {
  background: rgba(255, 255, 255, 0.06); border: 1px solid rgba(255, 255, 255, 0.06);
  border-radius: 9999px; padding: 3px; width: fit-content;
}
[data-testid="stBaseButton-segmented_control"], [data-testid="stBaseButton-segmented_controlActive"] {
  border: none !important; border-radius: 9999px !important; background: transparent !important;
  color: var(--dk-muted) !important; font-family: var(--dk-font-body); font-size: 13px; padding: 4px 16px !important;
}
[data-testid="stBaseButton-segmented_controlActive"] {
  background: rgba(255, 255, 255, 0.18) !important; color: #ffffff !important; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.35);
}
[data-testid="stSlider"] [role="slider"] { background: #ffffff !important; box-shadow: 0 3px 10px rgba(0,0,0,0.5) !important; }
[data-testid="stSlider"] [data-testid="stThumbValue"] { font-family: var(--dk-font-mono); color: var(--dk-text); }
.stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid var(--dk-border); }
.stTabs [data-baseweb="tab"] { font-family: var(--dk-font-body); color: var(--dk-muted); background: transparent; }
.stTabs [aria-selected="true"] { color: var(--dk-text) !important; }
.stTabs [data-baseweb="tab-highlight"] { background-color: var(--dk-green) !important; }
[data-testid="stCaptionContainer"] { color: var(--dk-muted); }
[data-testid="stDataFrame"] { border: 1px solid var(--dk-border); border-radius: 16px; overflow: hidden; }
"""


def inject_css() -> None:
    """Inject the dashboard design system CSS into the page.

    Loads Outfit (headings, numbers) and Plus Jakarta Sans (body) from Google
    Fonts with a system-ui fallback, plus JetBrains Mono for small labels and
    Noto Nastaliq Urdu for Urdu text, hides Streamlit's menu, header and
    footer, and defines the ``dk-*`` classes used by the render functions
    (Requirements 15.3, 16.1-16.5).

    Returns:
        None.
    """
    st.markdown(f"<style>{DK_CSS}</style>", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Formatting helpers (HTML safety)
# --------------------------------------------------------------------------


def esc(value: object) -> str:
    """Return ``value`` as HTML-escaped text.

    Every string interpolated into ``st.markdown(..., unsafe_allow_html=True)``
    goes through this, so a malformed value in a CSV cannot inject markup.

    Args:
        value: Any value; it is converted with ``str`` first.

    Returns:
        The escaped string, with quotes escaped as well.
    """
    return html.escape(str(value), quote=True)


def fmt_num(value: float | int | None, decimals: int = 0, signed: bool = False) -> str:
    """Format a number with fixed precision for display in HTML.

    Args:
        value: The number to format. None, NaN and infinities give a dash.
        decimals: Digits after the decimal point.
        signed: Prefix positive values with "+", for changes and deltas.

    Returns:
        The formatted number, e.g. ``fmt_num(12.345, 1) == "12.3"``, or
        ``"–"`` when the value is missing.
    """
    if value is None:
        return "–"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "–"
    if not math.isfinite(number):
        return "–"
    text = f"{number:{'+' if signed else ''}.{int(decimals)}f}"
    # "-0.0" or "+0.0" would suggest a change that rounding has removed.
    return text.lstrip("+-") if float(text) == 0 else text


def fmt_aqi(aqi: float | None) -> str:
    """Format an AQI for display so the number never contradicts its tier.

    The value is rounded half up to a whole number (Python's ``round`` and
    f-string formatting round half to even, which reads oddly for AQI). The
    tier rules use the unrounded value, so 200.4 is Very Unhealthy while a
    printed "200" would read as Unhealthy. When the rounded integer falls in
    a different tier than the real value, one decimal is shown instead.

    Args:
        aqi: AQI value, 0 or more. None, NaN or a negative value gives a dash
            or the plain number, because the tier is undefined there.

    Returns:
        For example ``"181"``, ``"200.4"`` or ``"–"``.
    """
    if aqi is None:
        return "–"
    try:
        value = float(aqi)
    except (TypeError, ValueError):
        return "–"
    if not math.isfinite(value):
        return "–"
    if value < 0:
        return fmt_num(value, 1)
    rounded = math.floor(value + 0.5)
    if int(alerts.aqi_to_tier_index(float(rounded))) != int(alerts.aqi_to_tier_index(value)):
        return fmt_num(value, 1)
    return str(rounded)


def fmt_timestamp(ts: pd.Timestamp) -> str:
    """Format a timestamp as ``YYYY-MM-DD HH:MM`` for the header badge.

    Args:
        ts: Timestamp from the loaded data.

    Returns:
        The formatted timestamp, or ``"unknown"`` when ``ts`` is missing.
    """
    if ts is None or pd.isna(ts):
        return "unknown"
    return pd.Timestamp(ts).strftime(TIMESTAMP_FORMAT)


def urdu(text: str) -> str:
    """Wrap Urdu text in a right-to-left ``<bdi>`` so brackets do not flip.

    Args:
        text: Urdu text (escaped here).

    Returns:
        HTML snippet ``<bdi dir="rtl" lang="ur" class="dk-urdu">…</bdi>``.
    """
    return f'<bdi dir="rtl" lang="ur" class="dk-urdu">{esc(text)}</bdi>'


def html_block(*parts: str) -> str:
    """Join HTML fragments into one line.

    Markdown treats lines indented by four spaces as code blocks, so the
    templates are built from fragments and joined without newlines. The
    fragments are not stripped, because their edge spaces separate words.

    Args:
        *parts: HTML fragments, written without newlines or indentation.

    Returns:
        The joined HTML string.
    """
    return "".join(parts)


# --------------------------------------------------------------------------
# Loaders (cached; files are only read, never created)
# --------------------------------------------------------------------------


@st.cache_data(show_spinner="Loading the processed dataset…")
def load_features(path: str = str(FEATURES_PATH)) -> pd.DataFrame:
    """Read ``data/processed/features_all.csv`` with a parsed ``datetime``.

    Cached with ``st.cache_data`` so the 30 MB file is read once per session
    (Requirement 17.17). Call it only after checking that the file exists,
    so a missing file is reported instead of being cached as an error.

    Args:
        path: CSV path as a string (strings hash cheaply for the cache key).

    Returns:
        The feature frame sorted by city and datetime.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    df = pd.read_csv(path, parse_dates=["datetime"])
    # Sorting here means "last row per city" is always the latest hour.
    return df.sort_values(["city", "datetime"], kind="mergesort").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_metrics(path: str = str(METRICS_PATH)) -> pd.DataFrame:
    """Read ``results/metrics.csv`` written by the Training_Script.

    Args:
        path: CSV path as a string.

    Returns:
        The metrics frame exactly as saved.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_sensitivity(path: str = str(SENSITIVITY_PATH)) -> pd.DataFrame:
    """Read ``results/sensitivity.csv`` written by the Training_Script.

    Args:
        path: CSV path as a string.

    Returns:
        Frame with columns ``city, mode, perturbation, delta, n_rows,
        mean_delta_aqi``, exactly as saved.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    return pd.read_csv(path)


def expected_model_keys() -> list[tuple[str, str, int, str]]:
    """List the models the dashboard needs, as ``(city, target, horizon, mode)``.

    These are the 12 Forecast_Mode Random Forests (2 cities x 2 targets x 3
    horizons) and the 2 Simulation_Mode us_aqi 24 h Random Forests.

    Returns:
        The 14 model keys in a fixed order.
    """
    keys: list[tuple[str, str, int, str]] = []
    for city in utils.CITIES:
        for target in utils.TARGET_TYPES:
            for horizon in utils.HORIZONS:
                keys.append((city, target, int(horizon), "forecast"))
        keys.append((city, SIMULATION_TARGET, SIMULATION_HORIZON, "simulation"))
    return keys


def model_file(key: tuple[str, str, int, str]) -> Path:
    """Return the saved joblib path of one dashboard model.

    Args:
        key: ``(city, target, horizon, mode)``.

    Returns:
        Path built by ``utils.model_path`` for the default Random Forest.
    """
    city, target, horizon, mode = key
    return utils.model_path(city, target, horizon, mode, DEFAULT_MODEL_TYPE)


@st.cache_resource(show_spinner="Loading saved models…")
def load_models(available: tuple[str, ...]) -> dict[tuple[str, str, int, str], SmogPredictorModel]:
    """Load the saved dashboard models from ``models/*.joblib``.

    Only saved files are loaded; nothing is trained (Requirements 17.17,
    17.19). Files that are missing or cannot be read are skipped, so the
    caller can report them and still render the rest of the page.

    Args:
        available: Paths of the model files that exist right now. Passing them
            as the argument makes them part of the cache key, so a model file
            created later is picked up without restarting the server.

    Returns:
        A dict from ``(city, target, horizon, mode)`` to the loaded model.
    """
    available_set = set(available)
    models: dict[tuple[str, str, int, str], SmogPredictorModel] = {}
    for key in expected_model_keys():
        path = model_file(key)
        if str(path) not in available_set:
            continue
        try:
            models[key] = SmogPredictorModel().load_model(path)
        except (OSError, ValueError, EOFError) as exc:
            # A truncated or foreign file must not stop the dashboard; the
            # caller reports the key as unavailable.
            print(f"Could not load {path}: {exc}")
    return models


def available_model_files() -> tuple[str, ...]:
    """Return the paths of the expected model files that exist on disk.

    Returns:
        Tuple of path strings, in ``expected_model_keys`` order.
    """
    return tuple(str(model_file(key)) for key in expected_model_keys() if model_file(key).is_file())


# --------------------------------------------------------------------------
# Render functions
# --------------------------------------------------------------------------


def _relative(path: Path) -> str:
    """Show ``path`` relative to the project root when possible.

    Args:
        path: Absolute or relative path.

    Returns:
        A short POSIX-style path such as ``models/x.joblib``.
    """
    try:
        return Path(path).resolve().relative_to(utils.PROJECT_ROOT).as_posix()
    except ValueError:
        return Path(path).as_posix()


def render_missing_file(path: Path, script: str) -> None:
    """Show a message naming a missing file and the script that creates it.

    Example: "Missing models/lahore_us_aqi_h24_forecast_random_forest.joblib.
    Run python scripts/train_all.py to create it." (Requirement 17.18).

    Args:
        path: The file that could not be found or read.
        script: Command that creates the file.

    Returns:
        None.
    """
    path = Path(path)
    status = "Could not read" if path.is_file() else "Missing"
    st.markdown(
        html_block(
            '<div class="dk-card dk-missing" role="alert">',
            '<div class="dk-label">File not available</div>',
            f'<div style="margin-top:6px">{status} <code>{esc(_relative(path))}</code>. ',
            f'Run <code>{esc(script)}</code> to create it.</div>',
            "</div>",
        ),
        unsafe_allow_html=True,
    )


def render_header(last_ts: pd.Timestamp, source_label: str = 'Historical dataset') -> None:
    """Render the title, the data badge and the "Historical dataset" label.

    The badge reads "Latest Available Data: <last timestamp> | Peshawar vs.
    Lahore" with the timestamp taken from the loaded data, and the data is
    labelled historical, never live (Requirements 17.1, 17.2).

    Args:
        last_ts: Latest ``datetime`` in the loaded feature data.

    Returns:
        None.
    """
    badge = f"Latest Available Data: {fmt_timestamp(last_ts)} | Peshawar vs. Lahore"
    st.markdown(
        html_block(
            '<div class="dk-header">',
            '<div class="dk-brand">',
            '<div class="dk-logo" aria-hidden="true">DK</div>',
            "<div>",
            f'<h1 class="dk-title">Dhund-Khabar AI ({urdu("دُھند خبر")})</h1>',
            '<div class="dk-subtitle">Smog forecast · Peshawar ',
            f'{urdu(CITY_URDU["Peshawar"])} · Lahore {urdu(CITY_URDU["Lahore"])}</div>',
            "</div>",
            "</div>",
            '<div class="dk-header-meta">',
            f'<span class="dk-badge"><span class="dk-dot"></span>{esc(badge)}</span>',
            '<span class="dk-pill dk-pill-historical">{esc(source_label)}</span>',
            "</div>",
            "</div>",
        ),
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# Controls: city toggle and horizon selector
# --------------------------------------------------------------------------


def city_label(city: str) -> str:
    """Return the plain-text widget label for a city, English plus Urdu.

    Widget labels are plain text, so the Urdu name is added as text here; in
    HTML cards it is wrapped with ``urdu()`` instead.

    Args:
        city: ``"Peshawar"`` or ``"Lahore"``.

    Returns:
        For example ``"Peshawar · پشاور"``.
    """
    return f"{city} · {CITY_URDU.get(city, '')}".strip(" ·")


def segmented(label: str, options: list, default: object, key: str,
              format_func=str) -> object:
    """Render a single-choice segmented control and return the selection.

    Uses ``st.segmented_control`` when this Streamlit version has it (1.42
    does) and falls back to a horizontal ``st.radio`` otherwise. A segmented
    control can be clicked off, which returns None; the default is used then,
    so the rest of the page always has a valid choice.

    Args:
        label: Accessible label (hidden visually).
        options: The choices.
        default: The choice selected on first render.
        key: Widget key, also used to set the value from tests.
        format_func: Turns an option into its display text.

    Returns:
        The selected option, never None.
    """
    if hasattr(st, "segmented_control"):
        choice = st.segmented_control(label, options, default=default, key=key,
                                      format_func=format_func, label_visibility="collapsed")
    else:
        choice = st.radio(label, options, index=options.index(default), key=key,
                          format_func=format_func, horizontal=True,
                          label_visibility="collapsed")
    return default if choice is None else choice


def render_controls() -> tuple[str, int]:
    """Render the Peshawar / Lahore toggle and the 24 / 48 / 72 h selector.

    Requirement 17.3 (city toggle) and 17.4 (horizon). Both controls sit in
    one row, as in the approved design.

    Returns:
        ``(city, horizon)`` chosen by the user.
    """
    left, right = st.columns([3, 2])
    with left:
        city = segmented("City", list(utils.CITIES), utils.CITIES[0], "dk_city",
                         format_func=city_label)
    with right:
        horizon = segmented("Forecast horizon", [int(h) for h in utils.HORIZONS],
                            int(utils.HORIZONS[0]), "dk_horizon",
                            format_func=lambda h: f"+{h} h")
    return str(city), int(horizon)


# --------------------------------------------------------------------------
# Forecast computation (no rendering)
# --------------------------------------------------------------------------


def latest_observed(features: pd.DataFrame, city: str) -> pd.Series | None:
    """Return the latest row of ``city`` with observed us_aqi and pm2_5.

    Args:
        features: Output of ``load_features``, sorted by city and datetime.
        city: City name.

    Returns:
        The row, or None if the city has no such row.
    """
    rows = features[(features["city"] == city)
                    & features["us_aqi"].notna() & features["pm2_5"].notna()]
    return None if rows.empty else rows.iloc[-1]


def latest_complete_row(features: pd.DataFrame, city: str,
                        feature_cols: list[str]) -> pd.Series | None:
    """Return the last row of ``city`` with every model feature present.

    The newest hours can miss a pollutant value, and a model cannot predict
    from NaN, so the hero uses the newest row that is complete for the model's
    own feature list (design: "Hero prediction input").

    Args:
        features: Output of ``load_features``.
        city: City name.
        feature_cols: ``model.feature_columns`` of the model to use.

    Returns:
        The row, or None if no row is complete.
    """
    rows = features[features["city"] == city]
    rows = rows[rows[feature_cols].notna().all(axis=1)]
    return None if rows.empty else rows.iloc[-1]


def compute_forecast(features: pd.DataFrame,
                     models: dict[tuple[str, str, int, str], SmogPredictorModel],
                     city: str, horizon: int) -> dict | None:
    """Predict us_aqi for ``city`` at ``horizon`` hours with the saved model.

    Uses the Forecast_Mode us_aqi Random Forest. The result is returned, not
    rendered, so the summary cards, the hero card, and later sections (policy
    terminal, What-If lab) share one prediction and one advisory.

    Args:
        features: Output of ``load_features``.
        models: Output of ``load_models``.
        city: City name.
        horizon: 24, 48, or 72.

    Returns:
        A dict with ``city``, ``horizon``, ``base_row`` (pd.Series),
        ``base_ts``, ``target_ts``, ``pred_aqi`` (float, clipped at 0) and
        ``advisory`` (from ``get_risk_advisory``), or None when the model is
        missing or no complete input row exists.
    """
    model = models.get((city, HERO_TARGET, int(horizon), "forecast"))
    if model is None:
        return None
    base_row = latest_complete_row(features, city, model.feature_columns)
    if base_row is None:
        return None
    X = base_row[model.feature_columns].to_frame().T.astype(float)
    raw = float(model.predict(X)[0])
    # Random Forest never predicts below 0, but get_risk_advisory raises on
    # negative input, so the clip keeps the page safe for any model type.
    pred_aqi = max(raw, 0.0)
    base_ts = pd.Timestamp(base_row["datetime"])
    return {
        "city": city,
        "horizon": int(horizon),
        "base_row": base_row,
        "base_ts": base_ts,
        # Timedelta(value, unit=...) avoids the NumPy 2.5 "generic unit"
        # DeprecationWarning that Timedelta(hours=...) raises in pandas 2.3.
        "target_ts": base_ts + pd.Timedelta(int(horizon), unit="h"),
        "pred_aqi": pred_aqi,
        "advisory": model.get_risk_advisory(pred_aqi),
    }


# --------------------------------------------------------------------------
# Gauge and tier bar (inline SVG / HTML, no JavaScript)
# --------------------------------------------------------------------------


def gauge_fraction(aqi: float) -> float:
    """Map an AQI to its position on the 0-500 gauge.

    Args:
        aqi: AQI value, 0 or more.

    Returns:
        ``min(aqi, 500) / 500``, limited to [0, 1].
    """
    return max(0.0, min(float(aqi), GAUGE_MAX_AQI)) / GAUGE_MAX_AQI


def _arc_point(fraction: float, cx: float, cy: float, r: float) -> tuple[float, float]:
    """Return the point at ``fraction`` along a left-to-right upper semicircle.

    Fraction 0 is the left end, 0.5 the top, 1 the right end. SVG's y axis
    points down, so the y offset is subtracted.

    Args:
        fraction: Position in [0, 1].
        cx: Centre x.
        cy: Centre y.
        r: Radius.

    Returns:
        ``(x, y)`` in SVG coordinates.
    """
    angle = math.pi * (1.0 - fraction)
    return cx + r * math.cos(angle), cy - r * math.sin(angle)


def _arc_path(f0: float, f1: float, cx: float, cy: float, r: float) -> str:
    """Return an SVG path for the semicircle arc from ``f0`` to ``f1``.

    Each tier spans at most 40 % of the half circle (72 degrees), so the
    small-arc flag is always correct; sweep flag 1 draws clockwise on screen,
    which goes over the top from left to right.

    Args:
        f0: Start fraction.
        f1: End fraction, greater than ``f0``.
        cx: Centre x.
        cy: Centre y.
        r: Radius.

    Returns:
        The ``d`` attribute value.
    """
    x0, y0 = _arc_point(f0, cx, cy, r)
    x1, y1 = _arc_point(f1, cx, cy, r)
    return f"M {x0:.2f} {y0:.2f} A {r} {r} 0 0 1 {x1:.2f} {y1:.2f}"


def gauge_svg(aqi: float) -> str:
    """Build the inline SVG semi-gauge for one AQI value.

    Six tier segments sit on a linear 0-500 scale, each with an arc length in
    proportion to its AQI range. The full scale is drawn faint, and the part
    up to the value is drawn in full colour, ending at a white marker at
    ``min(aqi, 500) / 500`` (adaptation note 3 in design_system.md).

    Args:
        aqi: Predicted AQI, 0 or more.

    Returns:
        An ``<svg>`` string on one line.
    """
    cx, cy, r = 160.0, 160.0, 125.0
    fraction = gauge_fraction(aqi)
    parts = [
        '<svg class="dk-gauge" viewBox="0 0 320 185" role="img" ',
        f'aria-label="Gauge: AQI {esc(fmt_aqi(aqi))} on a 0 to 500 scale">',
    ]
    for lower, upper, color in GAUGE_SEGMENTS:
        f0, f1 = lower / GAUGE_MAX_AQI, upper / GAUGE_MAX_AQI
        parts.append(f'<path d="{_arc_path(f0, f1, cx, cy, r)}" stroke="{esc(color)}" '
                     'stroke-width="16" fill="none" stroke-opacity="0.22"/>')
        filled_end = min(f1, fraction)
        if filled_end > f0:
            parts.append(f'<path d="{_arc_path(f0, filled_end, cx, cy, r)}" stroke="{esc(color)}" '
                         'stroke-width="16" fill="none"/>')
    for tick in GAUGE_TICKS:
        tx, ty = _arc_point(tick / GAUGE_MAX_AQI, cx, cy, r + 20)
        parts.append(f'<text x="{tx:.1f}" y="{ty + 3:.1f}" text-anchor="middle">{tick}</text>')
    # Needle from the centre to the marker, so the reading is clear even when
    # the filled arc is short (low AQI).
    mx, my = _arc_point(fraction, cx, cy, r)
    nx, ny = _arc_point(fraction, cx, cy, r - 30)
    parts.append(f'<line x1="{cx}" y1="{cy}" x2="{nx:.2f}" y2="{ny:.2f}" '
                 'stroke="rgba(255,255,255,0.55)" stroke-width="2" stroke-linecap="round"/>')
    parts.append(f'<circle cx="{cx}" cy="{cy}" r="4" fill="rgba(255,255,255,0.7)"/>')
    parts.append(f'<circle cx="{mx:.2f}" cy="{my:.2f}" r="9" fill="#ffffff" '
                 'stroke="rgba(0,0,0,0.35)" stroke-width="2"/>')
    parts.append("</svg>")
    return html_block(*parts)


def tier_bar_html(aqi: float) -> str:
    """Build the straight tier bar with a marker at the gauge position.

    The segment widths come from the ``dk-seg-*`` CSS classes (10/10/10/10/
    20/40 %), the same proportions as the gauge.

    Args:
        aqi: Predicted AQI, 0 or more.

    Returns:
        HTML string on one line.
    """
    left = gauge_fraction(aqi) * 100.0
    segments = "".join(f'<span class="dk-seg-{name}"></span>' for name in
                       ("good", "moderate", "usg", "unhealthy", "very-unhealthy", "hazardous"))
    ticks = "".join(f'<span style="left:{tick / GAUGE_MAX_AQI * 100:.1f}%">{tick}</span>'
                    for tick in GAUGE_TICKS)
    return html_block(
        '<div class="dk-spectrum" aria-hidden="true">',
        f'<div class="dk-spectrum-track">{segments}</div>',
        f'<div class="dk-spectrum-thumb" style="left:{left:.2f}%"></div>',
        f'<div class="dk-spectrum-scale">{ticks}</div>',
        "</div>",
    )


# --------------------------------------------------------------------------
# Summary cards and hero card
# --------------------------------------------------------------------------


def _tier_pill(advisory: dict) -> str:
    """Return the tier pill HTML, tinted with the tier colour.

    Args:
        advisory: Dict from ``get_risk_advisory``.

    Returns:
        HTML ``<span>``.
    """
    return (f'<span class="dk-tier-pill" style="--dk-tier:{esc(advisory["color"])}">'
            f'<span class="dk-dot" style="background:{esc(advisory["color"])}"></span>'
            f'{esc(advisory["tier"])}</span>')


def render_city_summary(latest_by_city: dict[str, dict], selected_city: str,
                        horizon: int) -> None:
    """Render one summary card per city, side by side (Requirement 17.3).

    Each card shows the latest observed us_aqi and pm2_5 with their timestamp
    and the predicted AQI and tier for the selected horizon.

    Args:
        latest_by_city: City name to ``{"observed": pd.Series | None,
            "forecast": dict | None, "model_path": Path}``.
        selected_city: City picked in the toggle; its card is outlined.
        horizon: Selected horizon in hours.

    Returns:
        None.
    """
    columns = st.columns(len(latest_by_city))
    for column, (city, info) in zip(columns, latest_by_city.items()):
        observed: pd.Series | None = info["observed"]
        forecast: dict | None = info["forecast"]
        active = " dk-summary-active" if city == selected_city else ""
        if observed is None:
            observed_html = '<div class="dk-note">No observed AQI in the data for this city.</div>'
        else:
            observed_html = html_block(
                f'<div class="dk-label" style="margin-top:8px">Latest observed · {esc(fmt_timestamp(observed["datetime"]))}</div>',
                '<div class="dk-summary-row">',
                f'<div><span class="dk-num">{esc(fmt_aqi(observed["us_aqi"]))}</span><span class="dk-unit">AQI</span></div>',
                f'<div><span class="dk-num">{esc(fmt_num(observed["pm2_5"], 1))}</span><span class="dk-unit">PM2.5 {POLLUTANT_UNIT}</span></div>',
                "</div>",
            )
        if forecast is None:
            forecast_html = html_block(
                '<div class="dk-summary-forecast"><div class="dk-note">Forecast unavailable: ',
                f'<code>{esc(_relative(info["model_path"]))}</code> is missing or has no complete input row. ',
                f'Run <code>{esc(TRAIN_SCRIPT)}</code>.</div></div>',
            )
        else:
            forecast_html = html_block(
                '<div class="dk-summary-forecast">',
                f'<div class="dk-label">Forecast +{int(horizon)} h · {esc(fmt_timestamp(forecast["target_ts"]))}</div>',
                '<div class="dk-summary-row">',
                f'<div><span class="dk-num">{esc(fmt_aqi(forecast["pred_aqi"]))}</span><span class="dk-unit">AQI</span></div>',
                f"<div>{_tier_pill(forecast['advisory'])}</div>",
                "</div></div>",
            )
        with column:
            st.markdown(
                html_block(
                    f'<div class="dk-card{active}">',
                    f'<div class="dk-summary-city">{esc(city)} {urdu(CITY_URDU.get(city, ""))}</div>',
                    observed_html,
                    forecast_html,
                    "</div>",
                ),
                unsafe_allow_html=True,
            )


def render_hero_card(pred_aqi: float, advisory: dict, horizon: int,
                     forecast: dict) -> None:
    """Render the hero card with the predicted AQI for the selected city.

    Shows the real predicted value (also above 500), the tier name and
    colour, an inline SVG semi-gauge and the straight tier bar (both with the
    marker at ``min(aqi, 500) / 500``), the target time, the model used, and
    the observed PM2.5, PM10, NO2 and CO at the base time. The background is
    tinted with the tier colour, and Very Unhealthy and Hazardous (Emergency)
    add the ``dk-pulse`` class (Requirements 17.4-17.6).

    Args:
        pred_aqi: Predicted AQI, already clipped at 0.
        advisory: Dict from ``get_risk_advisory(pred_aqi)``.
        horizon: Forecast horizon in hours.
        forecast: Dict from ``compute_forecast`` (city, base_row, base_ts,
            target_ts).

    Returns:
        None.
    """
    city = forecast["city"]
    base_row: pd.Series = forecast["base_row"]
    # The tier index comes from alerts, so the pulse rule uses the same
    # boundaries as the tier name.
    tier_index = int(alerts.aqi_to_tier_index(pred_aqi))
    pulse = " dk-pulse" if tier_index >= PULSE_TIER_INDEX else ""
    over_scale = (
        f'<div class="dk-note">The gauge ends at {int(GAUGE_MAX_AQI)}; the value above is the real prediction.</div>'
        if pred_aqi > GAUGE_MAX_AQI else ""
    )
    minis = "".join(
        html_block(
            '<div class="dk-mini">',
            f'<div class="dk-label">{esc(label)}</div>',
            f'<span class="dk-num">{esc(fmt_num(base_row.get(column), 1))}</span>',
            f'<span class="dk-unit">{POLLUTANT_UNIT}</span>',
            "</div>",
        )
        for column, label in HERO_POLLUTANTS
    )
    st.markdown(
        html_block(
            f'<div class="dk-card dk-hero{pulse}" style="--dk-tier:{esc(advisory["color"])}" ',
            f'role="region" aria-label="Predicted AQI for {esc(city)}">',
            '<div class="dk-card-head">',
            f'<span class="dk-label">Predicted AQI · {esc(city)} {urdu(CITY_URDU.get(city, ""))}</span>',
            f'<span class="dk-pill">+{int(horizon)} h</span>',
            "</div>",
            '<div class="dk-hero-grid">',
            "<div>",
            f'<div class="dk-hero-aqi" style="color:{esc(advisory["color"])}">{esc(fmt_aqi(pred_aqi))}</div>',
            f'<div style="margin-top:10px">{_tier_pill(advisory)}</div>',
            '<div class="dk-hero-meta">',
            f'Forecast for <b>{esc(fmt_timestamp(forecast["target_ts"]))}</b> (+{int(horizon)} h) ',
            f'from data at <b>{esc(fmt_timestamp(forecast["base_ts"]))}</b><br>',
            f'Model: {esc(MODEL_LABEL)}',
            "</div>",
            over_scale,
            "</div>",
            f"<div>{gauge_svg(pred_aqi)}</div>",
            "</div>",
            tier_bar_html(pred_aqi),
            f'<div class="dk-label" style="margin-top:14px">Observed at {esc(fmt_timestamp(forecast["base_ts"]))}</div>',
            f'<div class="dk-mini-row" style="margin-top:8px">{minis}</div>',
            "</div>",
        ),
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# Condition cards
# --------------------------------------------------------------------------


def compass_label(direction_deg: float) -> str:
    """Return the 16-point compass name of a wind direction.

    Args:
        direction_deg: Direction in degrees clockwise from north.

    Returns:
        For example ``"NE"`` for 45 degrees, or ``"–"`` when missing.
    """
    try:
        value = float(direction_deg)
    except (TypeError, ValueError):
        return "–"
    if not math.isfinite(value):
        return "–"
    # Each point covers 22.5 degrees centred on its own direction, hence +0.5.
    return COMPASS_POINTS[int(math.floor((value % 360.0) / 22.5 + 0.5)) % 16]


def _finite_or_none(value: object) -> float | None:
    """Return ``value`` as a float, or None when it is missing or not finite.

    Args:
        value: A cell from a pandas row.

    Returns:
        The float, or None.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _pm25_card(latest: pd.Series) -> str:
    """Build the PM2.5 card: latest value and change versus 24 h earlier.

    Args:
        latest: The hero base row.

    Returns:
        Card HTML on one line.
    """
    change = _finite_or_none(latest.get("pm2_5_change_24h"))
    if change is None:
        change_html = '<div class="dk-note">No value 24 h earlier to compare with.</div>'
    else:
        # Rising PM2.5 is bad news, so "up" is red and "down" is green.
        cls, arrow = ("dk-delta-up", "▲") if change > 0 else ("dk-delta-down", "▼") if change < 0 else ("dk-muted", "■")
        change_html = html_block(
            f'<div style="margin-top:8px" class="{cls}">{arrow} {esc(fmt_num(change, 1, signed=True))} ',
            f'<span class="dk-unit">{POLLUTANT_UNIT}</span></div>',
            '<div class="dk-note">Change versus 24 h earlier (pm2_5_change_24h).</div>',
        )
    return html_block(
        '<div class="dk-card">',
        '<div class="dk-card-head"><span class="dk-label">PM2.5</span>',
        '<span class="dk-pill">Fine particles</span></div>',
        f'<div><span class="dk-metric">{esc(fmt_num(latest.get("pm2_5"), 1))}</span>',
        f'<span class="dk-unit">{POLLUTANT_UNIT}</span></div>',
        change_html,
        "</div>",
    )


def _wind_card(latest: pd.Series) -> str:
    """Build the wind card with a compass and an arrow rotated by the direction.

    ``wind_direction_10m`` follows the meteorological convention: it is the
    direction the wind comes FROM. The arrow is drawn pointing down (south) at
    0 degrees and the whole arrow is rotated by ``wind_direction_10m``, so it
    shows where the air flows to, for example a north wind (0 degrees) points
    south. The label states both, so the reading is not ambiguous.

    Args:
        latest: The hero base row.

    Returns:
        Card HTML on one line.
    """
    speed = _finite_or_none(latest.get("wind_speed_10m"))
    direction = _finite_or_none(latest.get("wind_direction_10m"))
    rotation = 0.0 if direction is None else direction
    stagnant = speed is not None and utils.stagnation_flag(speed) == 1
    badge = (f'<span class="dk-pill dk-warn-pill" title="Wind below {fmt_num(utils.STAGNATION_WIND_KMH)} km/h">Stagnant air</span>'
             if stagnant else '<span class="dk-pill">10 m</span>')
    tooltip = ("Meteorological convention: the direction the wind comes from. "
               "The arrow shows where the air flows to.")
    arrow_svg = html_block(
        '<svg viewBox="0 0 64 64" aria-hidden="true">',
        '<line x1="32" y1="10" x2="32" y2="50" stroke="#f5f5f7" stroke-width="3" stroke-linecap="round"/>',
        '<polygon points="32,58 24,44 40,44" fill="#30d158"/>',
        "</svg>",
    )
    direction_text = ("Direction unknown" if direction is None
                      else f"From {compass_label(direction)} ({fmt_num(direction)}°)")
    return html_block(
        '<div class="dk-card">',
        f'<div class="dk-card-head"><span class="dk-label">Wind</span>{badge}</div>',
        f'<div><span class="dk-metric">{esc(fmt_num(speed, 1))}</span><span class="dk-unit">km/h</span></div>',
        f'<div class="dk-compass" title="{esc(tooltip)}" role="img" aria-label="{esc(direction_text)}">',
        '<span class="dk-compass-label dk-compass-n">N</span><span class="dk-compass-label dk-compass-e">E</span>',
        '<span class="dk-compass-label dk-compass-s">S</span><span class="dk-compass-label dk-compass-w">W</span>',
        f'<div class="dk-arrow" style="transform:rotate({rotation:.1f}deg)">{arrow_svg}</div>',
        "</div>",
        f'<div class="dk-note" title="{esc(tooltip)}">{esc(direction_text)}. ',
        "Wind direction is where the wind comes from; the arrow shows where it blows to.</div>",
        "</div>",
    )


def _inversion_card(latest: pd.Series) -> str:
    """Build the thermal inversion card from the heuristic proxy.

    The proxy is ``relative_humidity_2m / 100 - temperature_2m / 40`` from
    ``utils.inversion_proxy``; the level comes from ``utils.inversion_level``.
    The data has no vertical temperature profile, so the card says it is a
    heuristic and shows no inversion altitude (Requirement 17.9).

    Args:
        latest: The hero base row.

    Returns:
        Card HTML on one line.
    """
    temp = _finite_or_none(latest.get("temperature_2m"))
    rh = _finite_or_none(latest.get("relative_humidity_2m"))
    if temp is None or rh is None:
        # inversion_level maps NaN to "Weak", which would be a false reading.
        body = '<div class="dk-note">Temperature or humidity missing, so no proxy can be computed.</div>'
    else:
        proxy = float(utils.inversion_proxy(temp, rh))
        level = str(utils.inversion_level(proxy))
        body = html_block(
            f'<div><span class="dk-metric">{esc(fmt_num(proxy, 2))}</span><span class="dk-unit">proxy</span></div>',
            f'<div style="margin-top:8px"><span class="dk-pill">{esc(level)}</span></div>',
        )
    return html_block(
        '<div class="dk-card">',
        '<div class="dk-card-head"><span class="dk-label">Thermal inversion</span>',
        '<span class="dk-pill">Heuristic</span></div>',
        body,
        '<div class="dk-note">Heuristic proxy, not a measured inversion: RH/100 − T/40 from 2 m ',
        "temperature and humidity. Colder, more humid air gives a higher value. ",
        "Levels: Strong ≥ 0.50, Moderate ≥ 0.25, else Weak.</div>",
        "</div>",
    )


def _weather_card(latest: pd.Series) -> str:
    """Build the temperature / humidity / dew point card.

    The dew point is not in the data; it is derived from temperature and
    humidity with ``utils.dew_point_c`` (Magnus formula) and labelled so.

    Args:
        latest: The hero base row.

    Returns:
        Card HTML on one line.
    """
    temp = _finite_or_none(latest.get("temperature_2m"))
    rh = _finite_or_none(latest.get("relative_humidity_2m"))
    dew = None if temp is None or rh is None else _finite_or_none(utils.dew_point_c(temp, rh))
    rows = [
        ("Temperature", fmt_num(temp, 1), "°C"),
        ("Humidity", fmt_num(rh), "%"),
        ("Dew point (derived)", fmt_num(dew, 1), "°C"),
    ]
    rows_html = "".join(
        f'<div class="dk-wx-row"><span class="dk-label">{esc(name)}</span>'
        f'<span><span class="dk-num">{esc(value)}</span><span class="dk-unit">{esc(unit)}</span></span></div>'
        for name, value, unit in rows
    )
    return html_block(
        '<div class="dk-card">',
        '<div class="dk-card-head"><span class="dk-label">Air at 2 m</span><span class="dk-pill">Weather</span></div>',
        rows_html,
        '<div class="dk-note">Dew point is computed from temperature and humidity (Magnus formula), not measured.</div>',
        "</div>",
    )


def render_condition_cards(latest: pd.Series) -> None:
    """Render the PM2.5, wind, thermal inversion and weather cards.

    All four cards use the hero base row, so they describe the same hour as
    the prediction, and that hour is printed above them (Requirements
    17.7-17.9).

    Args:
        latest: The hero base row (``compute_forecast(...)["base_row"]``).

    Returns:
        None.
    """
    city = str(latest.get("city", ""))
    st.markdown(
        html_block(
            f'<div class="dk-section-title">Conditions · {esc(city)} {urdu(CITY_URDU.get(city, ""))}</div>',
            f'<div class="dk-label" style="margin-bottom:10px">Observed at {esc(fmt_timestamp(latest.get("datetime")))}</div>',
            '<div class="dk-cond-grid">',
            _pm25_card(latest),
            _wind_card(latest),
            _inversion_card(latest),
            _weather_card(latest),
            "</div>",
        ),
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# What-If Simulation Lab
# --------------------------------------------------------------------------


def predict_simulation(sim_model: SmogPredictorModel, base_row: pd.Series,
                       wind: float, temp: float, humidity: float) -> float:
    """Predict us_aqi 24 h ahead with the Simulation_Mode model for one scenario.

    Args:
        sim_model: Loaded Simulation_Mode us_aqi 24 h model.
        base_row: The hero base row (observed features).
        wind: Wind speed at 10 m in km/h.
        temp: Temperature at 2 m in degrees C.
        humidity: Relative humidity at 2 m in %.

    Returns:
        Predicted AQI, clipped at 0 because the advisory rejects negatives.
    """
    row = utils.build_scenario_row(base_row, wind, temp, humidity, sim_model.feature_columns)
    return max(float(sim_model.predict(row.astype(float))[0]), 0.0)


def sensitivity_note(sensitivity: pd.DataFrame, city: str) -> str | None:
    """Phrase the measured simulation-model sensitivity for ``city``.

    The numbers are read from ``results/sensitivity.csv``, never typed in,
    so the note always matches the last training run.

    Args:
        sensitivity: Output of ``load_sensitivity``.
        city: City name.

    Returns:
        A plain sentence, or None when the file has no simulation rows for
        the city.
    """
    rows = sensitivity[(sensitivity["city"] == city) & (sensitivity["mode"] == "simulation")]
    parts: list[str] = []
    n_rows: int | None = None
    for column, label, unit in (("wind_speed_10m", "wind", " km/h"),
                                ("relative_humidity_2m", "humidity", " %")):
        match = rows[rows["perturbation"] == column]
        if match.empty or pd.isna(match.iloc[0]["mean_delta_aqi"]):
            continue
        rec = match.iloc[0]
        n_rows = int(rec["n_rows"])
        parts.append(f"{fmt_num(rec['delta'], 0, signed=True)}{unit} {label} changed this model's "
                     f"AQI by {fmt_num(rec['mean_delta_aqi'], 1, signed=True)} on average")
    if not parts:
        return None
    return (f"Measured on {n_rows} cold test hours in {city} (2 m temperature at or below "
            f"{fmt_num(SENSITIVITY_COLD_C)} °C): " + "; ".join(parts) + ".")


def render_whatif_lab(latest: pd.Series, sim_model: SmogPredictorModel,
                      sensitivity: pd.DataFrame | None) -> None:
    """Render the What-If Simulation Lab for the selected city.

    Three sliders (wind 0-30 km/h, temperature 5-35 °C, humidity 20-100 %)
    start at the observed values of the hero base row, clipped into range by
    ``utils.clamp_slider_default``; a clipped slider shows the real observed
    value. The baseline is the Simulation_Mode prediction at the unmodified
    observed values, and the scenario is the prediction for
    ``utils.build_scenario_row`` with the slider values. The lab shows the
    scenario AQI, tier, advisory warning, and the signed difference from the
    baseline, plus the required caption (Requirements 17.10-17.13). Streamlit
    reruns the script on every slider change, so the result is always current.

    Args:
        latest: The hero base row.
        sim_model: Loaded Simulation_Mode us_aqi 24 h model of the city.
        sensitivity: Output of ``load_sensitivity``, or None if the file is
            missing.

    Returns:
        None.
    """
    city = str(latest.get("city", ""))
    base_ts = pd.Timestamp(latest["datetime"])
    st.markdown(
        html_block(
            f'<div class="dk-section-title">What-If Simulation Lab {urdu("تجربہ گاہ")}</div>',
            f'<div class="dk-label" style="margin-bottom:10px">{esc(SIMULATION_LABEL)} · {esc(city)} · ',
            f'starts from data at {esc(fmt_timestamp(base_ts))}</div>',
        ),
        unsafe_allow_html=True,
    )

    observed = {col: float(latest[col]) for col in utils.SLIDER_RANGES}
    # The baseline uses the real observed values, even when they lie outside
    # the slider range, as the design defines it.
    baseline = predict_simulation(sim_model, latest, observed["wind_speed_10m"],
                                  observed["temperature_2m"], observed["relative_humidity_2m"])

    slider_specs = [
        ("wind_speed_10m", "Wind speed (km/h)", "km/h"),
        ("temperature_2m", "Temperature (°C)", "°C"),
        ("relative_humidity_2m", "Relative humidity (%)", "%"),
    ]
    left, right = st.columns([3, 2])
    values: dict[str, float] = {}
    any_clamped = False
    with left:
        for col, label, unit in slider_specs:
            low, high = utils.SLIDER_RANGES[col]
            default, clamped = utils.clamp_slider_default(observed[col], low, high)
            # The key includes city and base time, so switching city (or new
            # data) starts the sliders again at that city's observed values.
            values[col] = float(st.slider(
                label, min_value=float(low), max_value=float(high), value=float(default),
                step=SLIDER_STEPS[col], key=f"dk_whatif_{col}_{city}_{base_ts:%Y%m%d%H}",
            ))
            if clamped:
                any_clamped = True
                st.markdown(
                    f'<div class="dk-note" style="margin-top:-8px">Observed: {esc(fmt_num(observed[col], 1))} {esc(unit)} '
                    f'(outside slider range, slider starts at {esc(fmt_num(default, 1))} {esc(unit)})</div>',
                    unsafe_allow_html=True,
                )

    scenario = predict_simulation(sim_model, latest, values["wind_speed_10m"],
                                  values["temperature_2m"], values["relative_humidity_2m"])
    advisory = sim_model.get_risk_advisory(scenario)
    delta = scenario - baseline
    delta_cls = "dk-delta-up" if delta > 0.05 else "dk-delta-down" if delta < -0.05 else "dk-muted"
    clamp_note = (
        '<div class="dk-note">A slider starts at its range bound, while the baseline uses the real '
        "observed value, so the difference is not zero at the starting position.</div>"
        if any_clamped else ""
    )
    with right:
        st.markdown(
            html_block(
                f'<div class="dk-card dk-hero" style="--dk-tier:{esc(advisory["color"])}" ',
                f'role="region" aria-label="Scenario AQI for {esc(city)}">',
                f'<div class="dk-card-head"><span class="dk-label">Scenario AQI</span><span class="dk-pill">+{SIMULATION_HORIZON} h</span></div>',
                f'<div class="dk-scenario-aqi" style="color:{esc(advisory["color"])}">{esc(fmt_aqi(scenario))}</div>',
                f'<div style="margin-top:10px">{_tier_pill(advisory)}</div>',
                f'<div class="dk-hero-meta"><span class="{delta_cls}">{esc(fmt_num(delta, 1, signed=True))}</span> ',
                f'versus baseline <b>{esc(fmt_aqi(baseline))}</b> at the observed values<br>',
                f'For {esc(fmt_timestamp(base_ts + pd.Timedelta(SIMULATION_HORIZON, unit="h")))}</div>',
                clamp_note,
                f'<div class="dk-note" style="color:var(--dk-secondary)">{esc(advisory["warnings"])}</div>',
                "</div>",
            ),
            unsafe_allow_html=True,
        )
    if sensitivity is not None:
        note = sensitivity_note(sensitivity, city)
        if note:
            st.markdown(f'<div class="dk-note">{esc(note)}</div>', unsafe_allow_html=True)
    st.caption(WHATIF_CAPTION)


# --------------------------------------------------------------------------
# Visual Analytics tabs
# --------------------------------------------------------------------------


@st.cache_data(show_spinner="Rebuilding test predictions…")
def test_predictions(city: str, horizon: int, model_mtime: float,
                     _model: SmogPredictorModel,
                     path: str = str(MODEL_DATASET_PATH)) -> pd.DataFrame:
    """Recreate the Test_Period predictions of the forecast us_aqi Random Forest.

    Uses the same embargo split for ``horizon`` and the same NaN drop on the
    model's stored ``feature_columns`` plus label as the Training_Script, so
    the rows match the ``n_test`` rows behind ``metrics.csv``. The result is
    cached per city and horizon, so moving the date-range picker only
    filters this frame and never predicts again.

    Args:
        city: City name (cache key).
        horizon: 24, 48 or 72 (cache key).
        model_mtime: Modification time of the model file; part of the cache
            key so a retrained model is not shown with old predictions.
        _model: The loaded model. The leading underscore tells Streamlit not
            to hash it.
        path: Path of ``data/processed/model_dataset.csv``.

    Returns:
        Frame with ``target_time`` (t + h, when the value was measured),
        ``actual`` and ``predicted``, sorted by ``target_time``.
    """
    label_col = f"{ANALYTICS_TARGET}_t_plus_{int(horizon)}"
    df = pd.read_csv(path)
    # The split compares timestamps, so the column is parsed with the exact
    # format the pipeline wrote.
    df["datetime"] = pd.to_datetime(df["datetime"], format=DATETIME_FORMAT)
    city_df = df[df["city"] == city].reset_index(drop=True)
    del df
    # flags_path=None: the split does not need the audit flags.
    _, test = AirQualityDataPipeline(flags_path=None).prepare_train_test_split(
        city_df, horizon=int(horizon))
    columns = list(_model.feature_columns) + [label_col]
    test = test[test[columns].notna().all(axis=1)].reset_index(drop=True)
    predicted = _model.predict(test[_model.feature_columns])
    out = pd.DataFrame({
        # A row at t carries the label measured at t + h, so the chart puts it
        # on the calendar hour it refers to.
        "target_time": test["datetime"] + pd.Timedelta(int(horizon), unit="h"),
        "actual": test[label_col].to_numpy(dtype=float),
        "predicted": np.asarray(predicted, dtype=float),
    })
    return out.sort_values("target_time", kind="mergesort").reset_index(drop=True)


def _select_city(label: str, default: str, key: str) -> str:
    """Render a city selectbox that starts at the page's city.

    The key includes the page selection, so changing the page toggle resets
    the tab's control to the new city instead of keeping an old choice.

    Args:
        label: Widget label.
        default: The page's selected city.
        key: Base widget key.

    Returns:
        The chosen city.
    """
    cities = list(utils.CITIES)
    return str(st.selectbox(label, cities, index=cities.index(default),
                            format_func=city_label, key=f"{key}_{default}"))


def _select_horizon(label: str, default: int, key: str) -> int:
    """Render a horizon selectbox that starts at the page's horizon.

    Args:
        label: Widget label.
        default: The page's selected horizon.
        key: Base widget key (the page horizon is appended, see
            ``_select_city``).

    Returns:
        The chosen horizon in hours.
    """
    horizons = [int(h) for h in utils.HORIZONS]
    return int(st.selectbox(label, horizons, index=horizons.index(int(default)),
                            format_func=lambda h: f"+{h} h", key=f"{key}_{default}"))


def _show_figure(fig) -> None:
    """Show a Matplotlib figure and close it to free its memory.

    Streamlit reruns the script on every interaction; without the close,
    pyplot would keep every figure ever drawn.

    Args:
        fig: The figure to show.

    Returns:
        None.
    """
    try:
        st.pyplot(fig)
    finally:
        plt.close(fig)


def _info_card(text: str) -> None:
    """Show a plain explanatory card, e.g. when a chart has no data.

    Args:
        text: Message (escaped here).

    Returns:
        None.
    """
    st.markdown(html_block('<div class="dk-card dk-missing" role="status">', esc(text), "</div>"),
                unsafe_allow_html=True)


def _render_actual_vs_predicted(models: dict, city: str, horizon: int) -> None:
    """Render the actual-vs-predicted tab with its own city, horizon and dates.

    Args:
        models: Output of ``load_models``.
        city: Page city (default of the tab's city control).
        horizon: Page horizon (default of the tab's horizon control).

    Returns:
        None.
    """
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        tab_city = _select_city("City", city, "dk_avp_city")
    with c2:
        tab_h = _select_horizon("Horizon", horizon, "dk_avp_h")
    key = (tab_city, ANALYTICS_TARGET, tab_h, "forecast")
    model = models.get(key)
    if model is None:
        render_missing_file(model_file(key), TRAIN_SCRIPT)
        return
    if not MODEL_DATASET_PATH.is_file():
        render_missing_file(MODEL_DATASET_PATH, TRAIN_SCRIPT)
        return
    preds = test_predictions(tab_city, tab_h, model_file(key).stat().st_mtime, model,
                             str(MODEL_DATASET_PATH))
    if preds.empty:
        _info_card(f"No complete Test_Period rows for {tab_city} at +{tab_h} h.")
        return

    last_date = preds["target_time"].max().date()
    # The default window is clipped to the data, so a shorter dataset still
    # opens with a valid range.
    default_start = min(max(WINDOW_DEFAULT_START, TEST_START), last_date)
    default_end = max(min(WINDOW_DEFAULT_END, last_date), default_start)
    with c3:
        picked = st.date_input("Date range (test period)", value=(default_start, default_end),
                               min_value=TEST_START, max_value=last_date,
                               key=f"dk_avp_dates_{tab_city}_{tab_h}")
    # While the user is choosing, the picker returns only the start date.
    if not isinstance(picked, (tuple, list)) or len(picked) != 2:
        _info_card("Pick an end date to show the chart.")
        return
    start, end = picked
    stamps = preds["target_time"]
    in_window = ((stamps >= pd.Timestamp(start))
                 & (stamps < pd.Timestamp(end) + pd.Timedelta(1, unit="D"))).to_numpy()
    window = preds.loc[in_window]
    if len(window) < 2:
        _info_card(f"The window {start} to {end} has fewer than 2 test hours. Choose a wider range.")
        return

    # Metrics are computed live on the shown hours, so they are labelled as
    # the selected window and are not the full-test numbers in metrics.csv.
    win_mae = utils.mae(window["actual"], window["predicted"])
    win_r2 = utils.r2(window["actual"], window["predicted"])
    st.markdown(
        html_block(
            '<div class="dk-header-meta" style="margin:4px 0 8px 0">',
            f'<span class="dk-badge">MAE {esc(fmt_num(win_mae, 2))} (selected window)</span>',
            f'<span class="dk-badge">R² {esc(fmt_num(win_r2, 3))} (selected window)</span>',
            f'<span class="dk-pill">{len(window):,} test hours · {esc(MODEL_LABEL)}</span>',
            "</div>",
        ),
        unsafe_allow_html=True,
    )
    fig = charts.plot_actual_vs_predicted(window["target_time"], window["actual"],
                                          window["predicted"], ANALYTICS_TARGET, tab_city,
                                          tab_h, start=None, end=None)
    _show_figure(fig)
    with st.expander("Predicted vs actual scatter for the selected window"):
        fig = charts.plot_predicted_vs_actual_scatter(
            window["actual"], window["predicted"],
            f"{tab_city}: US AQI, {tab_h} h Random Forest, {start} to {end}")
        _show_figure(fig)


def _render_heatmap(features: pd.DataFrame | None, city: str) -> None:
    """Render the weather-vs-pollution correlation heatmap tab for one city.

    Args:
        features: Output of ``load_features``, or None if the file is missing.
        city: Page city (default of the tab's city control).

    Returns:
        None.
    """
    if features is None:
        render_missing_file(FEATURES_PATH, TRAIN_SCRIPT)
        return
    tab_city = _select_city("City", city, "dk_heat_city")
    try:
        fig = charts.plot_correlation_heatmap(features[features["city"] == tab_city], tab_city)
    except ValueError as exc:
        _info_card(f"Cannot draw the heatmap for {tab_city}: {exc}")
        return
    _show_figure(fig)


def _render_importance(models: dict, city: str, horizon: int) -> None:
    """Render forecast and simulation feature importances side by side.

    Args:
        models: Output of ``load_models``.
        city: Page city (default of the tab's city control).
        horizon: Page horizon (default for the forecast model).

    Returns:
        None.
    """
    c1, c2 = st.columns(2)
    with c1:
        tab_city = _select_city("City", city, "dk_imp_city")
    with c2:
        tab_h = _select_horizon("Forecast horizon", horizon, "dk_imp_h")
    panels = [
        ((tab_city, ANALYTICS_TARGET, tab_h, "forecast"),
         f"{tab_city}: US AQI {tab_h} h Random Forest (forecast)"),
        ((tab_city, SIMULATION_TARGET, SIMULATION_HORIZON, "simulation"),
         f"{tab_city}: US AQI {SIMULATION_HORIZON} h Random Forest (simulation)"),
    ]
    for column, (key, title) in zip(st.columns(2), panels):
        with column:
            model = models.get(key)
            if model is None:
                render_missing_file(model_file(key), TRAIN_SCRIPT)
                continue
            _show_figure(charts.plot_feature_importance(model.get_feature_importance(), title,
                                                        top_n=IMPORTANCE_TOP_N))
    st.caption("The simulation model leaves out pollutant lags, so weather features carry "
               "more of its importance than in the forecast model.")


def _yes_no(value: object) -> str:
    """Turn a ``beats_persistence_mae`` cell into "yes", "no" or blank.

    Args:
        value: Cell value read from CSV (bool, text, or NaN).

    Returns:
        ``"yes"``, ``"no"`` or ``""`` when the cell is empty.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return "yes" if str(value).strip().lower() in ("true", "1", "yes") else "no"


def comparison_table(metrics: pd.DataFrame, city: str, target: str, horizon: int,
                     subset: str) -> tuple[pd.DataFrame, str]:
    """Build the model comparison table and its summary sentence.

    Only forecast-mode rows are used, in the order persistence, linear
    regression, gradient boosting, random forest. Every number comes from
    ``metrics.csv``; nothing is typed in.

    Args:
        metrics: Output of ``load_metrics``.
        city: City name.
        target: ``"us_aqi"`` or ``"pm2_5"``.
        horizon: 24, 48 or 72.
        subset: ``"all"`` or ``"winter"``.

    Returns:
        ``(table, sentence)``. The table has the display columns; the
        sentence names the lowest-MAE model and whether the Random Forest
        beat persistence. Both are empty when no rows match.
    """
    rows = metrics[(metrics["city"] == city) & (metrics["target"] == target)
                   & (metrics["horizon"].astype(int) == int(horizon))
                   & (metrics["mode"] == "forecast") & (metrics["subset"] == subset)]
    rows = rows[rows["model"].isin(MODEL_NAMES)].copy()
    if rows.empty:
        return pd.DataFrame(), ""
    rows["_order"] = rows["model"].map(list(MODEL_NAMES).index)
    rows = rows.sort_values("_order").reset_index(drop=True)
    is_aqi = target == "us_aqi"
    table = pd.DataFrame({
        "Model": rows["model"].map(MODEL_NAMES),
        "MAE": rows["MAE"].astype(float),
        "RMSE": rows["RMSE"].astype(float),
        "R2": rows["R2"].astype(float),
        "Tier accuracy": rows["tier_accuracy"].astype(float),
        # Tiers are AQI concepts, so the unhealthy scores are blank for pm2_5.
        "Unhealthy precision": rows["unhealthy_precision"].astype(float) if is_aqi else np.nan,
        "Unhealthy recall": rows["unhealthy_recall"].astype(float) if is_aqi else np.nan,
        "Beats persistence (MAE)": rows["beats_persistence_mae"].map(_yes_no),
        "Note": rows["note"].fillna("").astype(str),
    })

    finite = table[np.isfinite(table["MAE"])]
    best = finite.loc[finite["MAE"].idxmin(), "Model"] if not finite.empty else None
    mae_of = dict(zip(table["Model"], table["MAE"]))
    rf, pers = mae_of.get(MODEL_NAMES["random_forest"]), mae_of.get(MODEL_NAMES["persistence"])
    if rf is not None and pers is not None and np.isfinite(rf) and np.isfinite(pers):
        beat = (f"{'yes' if rf < pers else 'no'} (MAE {fmt_num(rf, 2)} vs "
                f"{fmt_num(pers, 2)})")
    else:
        beat = "not available"
    sentence = (f"Lowest MAE here: {best or 'not available'}. "
                f"Random Forest (the dashboard model) beat persistence: {beat}.")
    return table, sentence


def _render_comparison(metrics: pd.DataFrame | None, city: str, horizon: int) -> None:
    """Render the model comparison tab from ``results/metrics.csv``.

    Args:
        metrics: Output of ``load_metrics``, or None if the file is missing.
        city: Page city (default of the city selector).
        horizon: Page horizon (default of the horizon selector).

    Returns:
        None.
    """
    if metrics is None:
        render_missing_file(METRICS_PATH, TRAIN_SCRIPT)
        return
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        tab_city = _select_city("City", city, "dk_cmp_city")
    with c2:
        target = str(st.selectbox("Target", list(TARGET_LABELS), format_func=TARGET_LABELS.get,
                                  key="dk_cmp_target"))
    with c3:
        tab_h = _select_horizon("Horizon", horizon, "dk_cmp_h")
    with c4:
        subset = str(st.selectbox("Test subset", list(SUBSET_LABELS),
                                  format_func=SUBSET_LABELS.get, key="dk_cmp_subset"))
    table, sentence = comparison_table(metrics, tab_city, target, tab_h, subset)
    if table.empty:
        _info_card(f"metrics.csv has no forecast rows for {tab_city} {target} +{tab_h} h ({subset}).")
        return
    st.markdown(f'<div class="dk-note" style="font-size:14px;color:var(--dk-text)">{esc(sentence)}</div>',
                unsafe_allow_html=True)

    finite_mae = table["MAE"].where(np.isfinite(table["MAE"]))
    best_idx = finite_mae.idxmin() if finite_mae.notna().any() else None
    styler = (table.style
              .apply(lambda r: [BEST_ROW_STYLE if r.name == best_idx else ""] * len(r), axis=1)
              .format({"MAE": "{:.2f}", "RMSE": "{:.2f}", "R2": "{:.3f}", "Tier accuracy": "{:.3f}",
                       "Unhealthy precision": "{:.3f}", "Unhealthy recall": "{:.3f}"}, na_rep=""))
    st.dataframe(styler, hide_index=True, use_container_width=True)
    st.caption(f"Scores on the 2025-26 test period ({SUBSET_LABELS[subset].lower()}). "
               "The highlighted row has the lowest MAE. Unhealthy precision and recall use "
               f"AQI above {fmt_num(utils.UNHEALTHY_AQI_THRESHOLD)}.")


def render_analytics_tabs(features: pd.DataFrame | None, metrics: pd.DataFrame | None,
                          models: dict, city: str, horizon: int) -> None:
    """Render the four Visual Analytics tabs (Requirement 17.14).

    Each tab handles its own missing files, so one missing input does not
    hide the other tabs.

    Args:
        features: Output of ``load_features``, or None.
        metrics: Output of ``load_metrics``, or None.
        models: Output of ``load_models``.
        city: Page city, the default of every tab's city control.
        horizon: Page horizon, the default of the horizon controls.

    Returns:
        None.
    """
    st.markdown('<div class="dk-section-title">Visual Analytics</div>', unsafe_allow_html=True)
    tabs = st.tabs(["Actual vs predicted", "Correlation heatmap", "Feature importance",
                    "Model comparison"])
    with tabs[0]:
        _render_actual_vs_predicted(models, city, horizon)
    with tabs[1]:
        _render_heatmap(features, city)
    with tabs[2]:
        _render_importance(models, city, horizon)
    with tabs[3]:
        _render_comparison(metrics, city, horizon)


# --------------------------------------------------------------------------
# Policy terminal and footer
# --------------------------------------------------------------------------


def render_policy_terminal(advisory: dict | None, context: str = "") -> None:
    """Render the Actionable Health & Policy Terminal checklist (Requirement 17.15).

    Three rows (School Operations Status, Respiratory Protective Guidance,
    Industrial Advisory) filled from the hero's ``get_risk_advisory`` result,
    under the tier warning, with the tier colour as accent.

    Args:
        advisory: The hero advisory dict, or None when no forecast exists.
        context: Short line naming city, horizon and predicted AQI.

    Returns:
        None.
    """
    st.markdown('<div class="dk-section-title">Actionable Health &amp; Policy Terminal</div>',
                unsafe_allow_html=True)
    if advisory is None:
        _info_card("No forecast is available, so no advisory can be shown.")
        return
    color = esc(advisory["color"])
    rows = "".join(
        html_block(
            '<div class="dk-terminal-row">',
            f'<div class="dk-check" style="border-color:{color};color:{color}" aria-hidden="true">✓</div>',
            f'<div><span aria-hidden="true">{icon}</span> <b>{esc(title)}</b></div>',
            f'<div style="color:var(--dk-secondary)">{esc(advisory[key])}</div>',
            "</div>",
        )
        for key, title, icon in POLICY_ROWS
    )
    st.markdown(
        html_block(
            f'<div class="dk-card dk-terminal" style="--dk-tier:{color};border-left:3px solid {color}" ',
            'role="region" aria-label="Health and policy advisory">',
            f'<div class="dk-card-head"><span class="dk-label">{esc(context)}</span>{_tier_pill(advisory)}</div>',
            f'<div class="dk-note" style="color:var(--dk-text);font-size:13px;margin-bottom:6px">{esc(advisory["warnings"])}</div>',
            rows,
            "</div>",
        ),
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    """Render the required disclaimer footer (Requirement 17.16).

    A ``div.dk-footer``, never a ``<footer>`` element, because the CSS hides
    ``footer`` elements to remove Streamlit's own footer.

    Returns:
        None.
    """
    attribution = ('<a href="https://open-meteo.com/">Weather data by Open-Meteo.com</a>. Non-commercial use.')
    st.markdown(f'<div class="dk-footer">{esc(FOOTER_TEXT)}<br>{attribution}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------


def main() -> None:
    """Build the dashboard page.

    Order: page config, CSS, then header, controls, summary cards, hero,
    condition cards, What-If lab, analytics tabs, policy terminal, and the
    footer. A missing file shows ``render_missing_file`` and the sections
    that do not need it still render. The footer is rendered in a
    ``finally`` block, so it appears on every view, also when a section
    fails (Requirement 17.16).

    Returns:
        None.
    """
    st.set_page_config(
        page_title="Dhund-Khabar AI (دُھند خبر)",
        page_icon="🌫️",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    inject_css()
    try:
        render_sections()
    finally:
        render_footer()



def _recent_feature_frame():
    """Build features from recent Open-Meteo hours. Does not train a model."""
    notices = []
    frames = []
    try:
        for name in utils.CITIES:
            frame, status = live_data.load_recent(name)
            frames.append(frame)
            notices.append(status["notice"])
        raw = pd.concat(frames, ignore_index=True)
        raw["datetime"] = pd.to_datetime(raw["datetime"])
        raw = raw.sort_values(["city", "datetime"]).reset_index(drop=True)
        pipe = AirQualityDataPipeline(flags_path=utils.PROCESSED_DIR / "audit_flags.json")
        featured = pipe.generate_time_features(raw)
        featured = pipe.create_lag_and_rolling_features(featured)
        return featured, notices
    except Exception as exc:
        return None, [
            "Recent Open-Meteo data could not be prepared (" + str(exc) + "). Forecast uses the historical file."
        ]

def render_sections() -> None:
    """Load the inputs and render every page section except the footer.

    Returns:
        None.
    """
    features: pd.DataFrame | None = None
    if FEATURES_PATH.is_file():
        features = load_features(str(FEATURES_PATH))
    # The badge timestamp always comes from the data; without data it says
    # "unknown" instead of showing a made-up time.
    recent, notices = _recent_feature_frame()
    for text in notices:
        st.markdown(
            html_block('<div class="dk-card dk-missing" role="status">', esc(text), "</div>"),
            unsafe_allow_html=True,
        )
    forecast_features = recent if recent is not None else features
    if recent is not None and not recent.empty:
        last_ts = pd.to_datetime(recent["datetime"]).max()
        joined = " ".join(notices)
        if "historical file" in joined and "Open-Meteo, " not in joined:
            source_label = "Historical dataset"
        elif "last saved fetch" in joined:
            source_label = "Saved Open-Meteo fetch"
        else:
            source_label = "Open-Meteo dataset"
    else:
        last_ts = features["datetime"].max() if features is not None and not features.empty else None
        source_label = "Historical dataset"
    render_header(last_ts, source_label)
    if features is None:
        render_missing_file(FEATURES_PATH, TRAIN_SCRIPT)

    metrics: pd.DataFrame | None = None
    if METRICS_PATH.is_file():
        metrics = load_metrics(str(METRICS_PATH))
    else:
        render_missing_file(METRICS_PATH, TRAIN_SCRIPT)

    models = load_models(available_model_files())
    for key in expected_model_keys():
        if key not in models:
            render_missing_file(model_file(key), TRAIN_SCRIPT)

    city, horizon = render_controls()
    selected_forecast: dict | None = None
    if forecast_features is not None:
        selected_forecast = render_forecast_sections(forecast_features, models, city, horizon)

    # Analytics and the policy terminal still render without features: the
    # comparison table needs only metrics.csv, and the terminal explains that
    # no advisory exists.
    render_analytics_tabs(features, metrics, models, city, horizon)
    context = ""
    if selected_forecast is not None:
        context = (f"{city} · +{horizon} h · predicted AQI {fmt_aqi(selected_forecast['pred_aqi'])} "
                   f"for {fmt_timestamp(selected_forecast['target_ts'])}")
    render_policy_terminal(selected_forecast["advisory"] if selected_forecast else None, context)


def render_forecast_sections(features: pd.DataFrame,
                             models: dict[tuple[str, str, int, str], SmogPredictorModel],
                             city: str, horizon: int) -> dict | None:
    """Render summary cards, hero, condition cards and the What-If lab.

    Args:
        features: Output of ``load_features``.
        models: Output of ``load_models``.
        city: Selected city.
        horizon: Selected horizon.

    Returns:
        The selected city's forecast dict from ``compute_forecast`` (its
        ``advisory`` feeds the policy terminal), or None.
    """
    # Both cities are predicted, because the summary cards show both side by
    # side; the selected city's result is reused for the hero card.
    latest_by_city: dict[str, dict] = {}
    for name in utils.CITIES:
        latest_by_city[name] = {
            "observed": latest_observed(features, name),
            "forecast": compute_forecast(features, models, name, horizon),
            "model_path": model_file((name, HERO_TARGET, horizon, "forecast")),
        }
    render_city_summary(latest_by_city, city, horizon)

    # `selected_forecast` (base_row, pred_aqi, advisory) feeds the What-If
    # baseline row and the policy terminal advisory.
    selected_forecast: dict | None = latest_by_city[city]["forecast"]
    if selected_forecast is None and (city, HERO_TARGET, horizon, "forecast") not in models:
        render_missing_file(latest_by_city[city]["model_path"], TRAIN_SCRIPT)
    elif selected_forecast is None:
        # The model exists, but every row misses at least one feature.
        st.markdown(
            html_block('<div class="dk-card dk-missing" role="alert">',
                       f"No row of {esc(city)} has all model features, so no forecast can be made.",
                       "</div>"),
            unsafe_allow_html=True,
        )
    else:
        render_hero_card(selected_forecast["pred_aqi"], selected_forecast["advisory"],
                         horizon, selected_forecast)

    # Condition cards and the What-If lab use the hero base row, so every
    # number on the page refers to the same hour. Without a forecast there is
    # no base row; the message above already explains why.
    if selected_forecast is not None:
        base_row: pd.Series = selected_forecast["base_row"]
        render_condition_cards(base_row)
        sensitivity: pd.DataFrame | None = None
        if SENSITIVITY_PATH.is_file():
            sensitivity = load_sensitivity(str(SENSITIVITY_PATH))
        else:
            render_missing_file(SENSITIVITY_PATH, TRAIN_SCRIPT)
        sim_key = (city, SIMULATION_TARGET, SIMULATION_HORIZON, "simulation")
        sim_model = models.get(sim_key)
        if sim_model is None:
            render_missing_file(model_file(sim_key), TRAIN_SCRIPT)
        else:
            render_whatif_lab(base_row, sim_model, sensitivity)
    return selected_forecast


if __name__ == "__main__":
    main()
