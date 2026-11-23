# Stitch dashboard screen prompt (task 9.2)

- Tool: `generate_screen_from_text`
- projectId: `2753126111242902468`
- deviceType: `DESKTOP`
- designSystem: `assets/5576147302249970983`
- After generation: `apply_design_system` with assetId `5576147302249970983` on the new screen instance (decision from task 9.1).

## Prompt sent (verbatim)

```text
Desktop dashboard for "Dhund-Khabar AI", a student smog (air quality) forecasting project for Peshawar and Lahore, Pakistan. Dark mode only. One long scrolling page, content max width about 1280px, responsive 12-column grid, 16px grid gap, 24px card padding, 8px spacing scale (4, 8, 16, 24, 32px).

DESIGN TOKENS (use these exact values):
- Page background #0B0F19.
- Cards: surface #161F30 at about 75% opacity with glass effect (backdrop-filter: blur(12px)), 1px solid border #2A3B53, border-radius 16px, shadow 0 8px 32px rgba(0,0,0,0.35), subtle inset top highlight rgba(255,255,255,0.04).
- Accents: telemetry cyan #00F2FE (telemetry values, active toggles, slider fill and thumb, focus rings); moderate amber #FFB300; hazardous crimson #FF3366 and violet #9933FF (crimson-to-violet gradient only for hazard glow). Text #E6EDF7, muted text #8A9BB5.
- US EPA AQI tier colours, exact: Good 0-50 #00E400; Moderate 51-100 #FFFF00; Unhealthy for Sensitive Groups 101-150 #FF7E00; Unhealthy 151-200 #FF0000; Very Unhealthy 201-300 #8F3F97; Hazardous 301+ #7E0023.
- Fonts from Google Fonts: Outfit for ALL headings and ALL numbers (weights 500-700, tabular figures); Plus Jakarta Sans for body text, captions and advisories (weights 400-500); fallback stack system-ui, -apple-system, "Segoe UI", Roboto, sans-serif.
- Type scale: hero AQI number Outfit 64px/700; page title Outfit 32px/600; section headings Outfit 22px/600 with a small cyan accent bar; card titles Outfit 16px/600; labels Outfit 12px/500 uppercase with letter-spacing; body Plus Jakarta Sans 16px and 14px/400.
- Pills and badges fully rounded (999px) with a 1px border. Thin line icons. No photos, no stock images.

DATA RULES: This is a layout mock. All values are placeholders that a model fills at runtime. Show values as "---" or "—". Do not invent real-looking results. If a sample is needed to show a visual state, mark it with a small muted "example" tag. Never use the word "live" anywhere (no "live", "real-time", "live feed", "streaming").

SECTIONS, IN THIS ORDER FROM TOP TO BOTTOM:

1. HEADER: Left: title "Dhund-Khabar AI (دُھند خبر)" in Outfit 32px with a short muted subtitle "Smog forecasting for Peshawar and Lahore". Right: a pill status badge with exactly this text: "Latest Available Data: 2026-11-19 23:00 | Peshawar vs. Lahore" (cyan text, small database or clock icon), and next to it a small muted pill label "Historical dataset".

2. CITY AND HORIZON CONTROLS: A segmented pill toggle with two segments "Peshawar" and "Lahore" (Peshawar active: cyan #00F2FE low-opacity fill with cyan text). Below it, two side-by-side summary cards, one for Peshawar and one for Lahore, each showing the city name, "US AQI ---", "PM2.5 --- µg/m³" and a small tier pill "Tier —". Then a horizon selector as a segmented pill with "24 h", "48 h", "72 h" (24 h active), labelled "Forecast horizon".

3. HERO AQI CARD (full width, the largest card): a large semi-circular gauge (180-degree arc) drawn as SVG, track #2A3B53, arc split into the six EPA tier colour segments, with a marker on the arc. Centre: label "Predicted US AQI" and the value "---" in Outfit 64px, tier name below it. Show this mock in the Very Unhealthy example state: card background tinted about 15% with #8F3F97, tier pill "Very Unhealthy" with a small muted "example state" tag, and a slow pulsing glow ring around the card in crimson #FF3366 and violet #9933FF (the pulse is only used for Very Unhealthy and Hazardous tiers). Under the gauge, muted text: "Target time: — (+24 h)" and "Forecast model output".

4. CONDITIONS ROW (three equal cards side by side):
   a. PM2.5 card: latest PM2.5 "--- µg/m³" and "Change vs 24 h earlier: —" with a small up/down arrow icon.
   b. Wind card: wind speed "--- km/h" and a compass circle with an arrow rotated by the wind direction (show the arrow rotated about 45 degrees as an example), direction label "— °".
   c. Thermal Inversion card: status pill "Inversion risk: —", temperature "— °C" and relative humidity "— %", plus a muted note "Heuristic proxy from temperature and humidity, not a measured inversion."

5. WHAT-IF SIMULATION LAB (wide card): heading "What-If Simulation Lab". Left side: three labelled sliders with cyan fill and thumb on a #2A3B53 track: "Wind speed" 0-30 km/h, "Temperature" 5-35 °C, "Humidity" 20-100 %, each showing its value "—" and a small muted note "Default: latest observed value". Right side: result panel with "Scenario AQI ---", a scenario tier pill "Tier —", "Δ vs baseline: —" in amber #FFB300, and a one-line advisory placeholder "—". Caption at the bottom of the card in muted italic, exactly: "Scenario exploration from a statistical model, not a physical simulation."

6. VISUAL ANALYTICS (wide card with tabs): heading "Visual Analytics" and a tab bar with four tabs: "Actual vs Predicted", "Correlation Heatmap", "Feature Importance", "Model Comparison". The active tab "Actual vs Predicted" shows small controls (city select, horizon select, date range picker) and an empty chart placeholder area with legend items "Actual" (muted white line) and "Predicted" (cyan line). Below the chart, show the Model Comparison table preview with columns Model, MAE, RMSE, R², Tier accuracy, Unhealthy recall and rows Persistence, Linear Regression, Gradient Boosting, Random Forest, with every cell "—".

7. ACTIONABLE HEALTH & POLICY TERMINAL (wide card styled like a calm console): heading "Actionable Health & Policy Terminal". A checklist of exactly three rows, each with a check/status icon, a label in Outfit and an advisory placeholder "—" in Plus Jakarta Sans: "School Operations Status", "Respiratory Protective Guidance", "Industrial Advisory". Each row has a 3px left accent border in a neutral placeholder colour #2A3B53 (at runtime it takes the tier colour).

FOOTER: fixed to the bottom of the viewport, thin full-width bar, background #0B0F19 with a 1px top border #2A3B53, muted text #8A9BB5 in Plus Jakarta Sans 12px, centred, exactly: "Student learning project. Not an official health warning system. Follow guidance from Pakistan's health and environmental authorities."

Tone: calm, scientific telemetry console. Minimal gradients except the crimson-to-violet hazard glow and the tier-tinted hero card.
```

## Call results

1. `generate_screen_from_text` (projectId 2753126111242902468, DESKTOP, designSystem `assets/5576147302249970983`, prompt above): ok. Session 12543625459390074721. Screen `c0b225a4172e454891555481093ca978` ("Dhund-Khabar AI Dashboard", 2560 x 5082, status COMPLETE). Screenshot file `2938a26aa1444e0a9d50c4640ae79eba`, HTML file `0a80bec6176b44dca61ab39cee1ee341`.
2. `get_project`: no screen instances listed; `designTheme` still `{}`.
3. `list_screens`: returned `{}`. Polled again after 30 s: same result for both calls.
4. `get_screen` on `c0b225a4172e454891555481093ca978`: screen exists, same files.
5. `apply_design_system` (assetId 5576147302249970983, instance id = screen id `c0b225a4172e454891555481093ca978`, sourceScreen `projects/2753126111242902468/screens/c0b225a4172e454891555481093ca978`): ok. Session 14584859446730461384. New themed screen `848dc84695fd4e349433d61c947b4f09` (generatedBy `polish_edit_theme_agent`, 2560 x 5080, status COMPLETE). Screenshot file `796be0974c0f4ce0a72e2a4a97c9bb3a`, HTML file `7fc572af6edf457488cc114971759e3c`. No text output.
6. `get_project` after apply: `designTheme` now holds the design system theme (DARK, ROUND_TWELVE, #00F2FE / #FFB300 / #FF3366 / #0B0F19, Plus Jakarta Sans body, designMd). `headlineFont` is still not echoed, so Outfit must be checked in the HTML in task 9.3.

Screen to review in task 9.3: `848dc84695fd4e349433d61c947b4f09` (themed). The original `c0b225a4172e454891555481093ca978` is kept for comparison.

## Text returned by `generate_screen_from_text` (verbatim)

> I have designed and delivered the desktop telemetry dashboard for **Dhund-Khabar AI (دُھند خبر)**, adhering to your exact design tokens, structural requirements, and data constraints:
>
> ### Key Design & Architecture Highlights:
> 1. **Typography & Styling Tokens**:
>    - Integrated **Outfit** (500–700, tabular figures) for all numerical telemetry readouts, titles, headings, and labels.
>    - Applied **Plus Jakarta Sans** (400–500) for body copy, technical captions, and policy notes.
>    - Exact color specifications implemented: Page background `#0B0F19`, glass cards `#161F30` at 75% opacity with `backdrop-filter: blur(12px)`, 1px border `#2A3B53`, 16px border-radius, and inset top specular highlight.
>    - Clean accents with telemetry cyan `#00F2FE`, moderate amber `#FFB300`, and a soft hazard crimson/violet pulsing glow ring for high-severity states.
>
> 2. **Ordered Sections**:
>    - **Header**: Bilingual brand title with Urdu typography, dataset status pill (`Latest Available Data: 2026-11-19 23:00 | Peshawar vs. Lahore`), and muted `Historical dataset` badge.
>    - **City & Horizon Controls**: Segmented toggles for Peshawar & Lahore with active cyan illumination, paired side-by-side comparison cards, and a 24h / 48h / 72h horizon selector.
>    - **Hero AQI Card**: 180° semi-circular SVG arc divided into the six exact US EPA AQI tier colors (Good `#00E400` through Hazardous `#7E0023`), positioned with an active glow marker on the Very Unhealthy segment, showing 64px `---` placeholder value and an `example state` tag.
>    - **Conditions Row**: Three 12-column grid cards covering PM2.5 mass concentration, wind vector with a 45° directional compass dial, and thermal inversion heuristic risk indicators.
>    - **What-If Simulation Lab**: Interactive slider controls (Wind speed, Temperature, Humidity) on `#2A3B53` tracks with cyan thumbs, alongside counterfactual delta projections and counterfactual disclaimer.
>    - **Visual Analytics**: Multi-tab analytical terminal featuring tab switching, telemetry line chart container with legend, and a complete Model Comparison baseline matrix.
>    - **Actionable Health & Policy Terminal**: Console-styled checklist covering School Operations, Respiratory Protective Guidance, and Industrial Advisories with neutral `#2A3B53` accent borders ready for runtime tier-tinting.
>    - **Fixed Footer**: Pinned bottom disclaimer across viewports. All data strictly represented as placeholders (`---` or `—`), with zero mention of live/real-time feeds.

Suggestions returned (not acted on):
- "Toggle city selection view to Lahore"
- "Switch the Hero AQI card to Hazardous tier or Good tier"
- "Add interactive slider value drag interactions with JavaScript"
