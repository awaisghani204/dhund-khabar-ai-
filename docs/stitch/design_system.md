# Stitch design system record (task 9.1)

- Project: `projects/2753126111242902468` ("Dhund-Khabar AI Dashboard")
- Design system asset: `assets/5576147302249970983` ("Dhund-Khabar Smog Telemetry (Dark Glass)", version 1)
- Device type for screens: DESKTOP. Colour mode: DARK.
- IDs and call status are also in `docs/stitch/stitch_ids.json`.

## Theme fields sent to `create_design_system`

| Field | Value sent | Note |
|---|---|---|
| colorMode | `DARK` | |
| customColor | `#00F2FE` | telemetry cyan as seed |
| overridePrimaryColor | `#00F2FE` | cyan |
| overrideSecondaryColor | `#FFB300` | amber |
| overrideTertiaryColor | `#FF3366` | crimson |
| overrideNeutralColor | `#0B0F19` | background |
| colorVariant | `FIDELITY` | keeps generated tones close to the seed colours |
| headlineFont | `OUTFIT` | supported directly |
| bodyFont | `PLUS_JAKARTA_SANS` | supported directly |
| labelFont | `OUTFIT` | |
| roundness | `ROUND_TWELVE` | closest enum to 16px; the enum has no 16px value, so 16px cards are specified in `designMd` and `spacing.card-radius` |
| spacing | xs 4px, sm 8px, md 16px, lg 24px, xl 32px, card-padding 24px, grid-gap 16px, card-radius 16px | |
| typography | display-lg Outfit 64px/700; headline-lg Outfit 32px/600; headline-md Outfit 22px/600; title-md Outfit 16px/600; label-md Outfit 12px/500; body-lg Plus Jakarta Sans 16px/400; body-md Plus Jakarta Sans 14px/400 | |

Violet `#9933FF`, surface `#161F30`, border `#2A3B53` and the six EPA tier colours have no dedicated theme field, so they are defined in `designMd` below.

## Theme stored by Stitch (returned by `create_design_system` and `list_design_systems`)

- colorMode `DARK`, roundness `ROUND_TWELVE`, customColor `#00F2FE`, bodyFont `PLUS_JAKARTA_SANS`, colorVariant `FIDELITY`
- overridePrimaryColor `#00F2FE`, overrideSecondaryColor `#FFB300`, overrideTertiaryColor `#FF3366`, overrideNeutralColor `#0B0F19`
- designMd stored unchanged (text below)
- Manifest tokens: colours neutral `#0B0F19`, primary `#00F2FE`, secondary `#FFB300`, tertiary `#FF3366`; type body `Plus Jakarta Sans`; radius DEFAULT `12px`
- Not echoed back: `headlineFont` (OUTFIT), `labelFont`, `typography`, `spacing`. Outfit for headings and numbers is therefore carried only by `designMd` and must be repeated in the 9.2 screen prompt.

## designMd sent (stored verbatim by Stitch)

```markdown
# Dhund-Khabar AI Design System (Dark Glass Telemetry)

Smog / air-quality forecasting dashboard for Peshawar and Lahore. Desktop first, dark mode only. Data is historical, never labelled "live".

## Colour tokens (exact hex)

| Token | Hex | Use |
|---|---|---|
| background | #0B0F19 | page background |
| surface | #161F30 | card surfaces (glass) |
| border | #2A3B53 | 1px solid border on every card and input |
| telemetry-cyan | #00F2FE | telemetry values, primary accents, active toggle, focus rings |
| moderate-amber | #FFB300 | moderate / caution states |
| hazard-crimson | #FF3366 | hazardous states, alerts |
| hazard-violet | #9933FF | hazardous / emergency states, gradients with crimson |
| text-primary | #E6EDF7 | main text on dark |
| text-muted | #8A9BB5 | captions, secondary labels |

### US EPA AQI tier colours (exact, do not shift)

| Tier | AQI range | Hex |
|---|---|---|
| Good | 0-50 | #00E400 |
| Moderate | 51-100 | #FFFF00 |
| Unhealthy for Sensitive Groups (USG) | 101-150 | #FF7E00 |
| Unhealthy | 151-200 | #FF0000 |
| Very Unhealthy | 201-300 | #8F3F97 |
| Hazardous | 301+ | #7E0023 |

Tier colours are used for gauge arcs, tier badges, and a low-opacity tint (about 12-18%) of the hero card background. Very Unhealthy and Hazardous are Emergency tiers: the hero card gets a slow pulse animation (glow in crimson #FF3366 / violet #9933FF).

## Typography

- Headings and all numbers (AQI, PM2.5, wind, metrics): **Outfit** (Inter acceptable as alternate), weights 500-700, tabular figures.
- Body text, captions, advisories: **Plus Jakarta Sans**, weights 400-500.
- Fallback stack: `system-ui, -apple-system, "Segoe UI", Roboto, sans-serif`.
- Fonts loaded from Google Fonts.
- Urdu title text (دُھند خبر) may fall back to system Urdu fonts.

## Shape and surface

- Cards: border-radius **16px**, background #161F30 at ~70-80% opacity, `backdrop-filter: blur(12px)`, 1px solid #2A3B53 border, soft shadow `0 8px 32px rgba(0,0,0,0.35)`.
- Glass effect: semi-transparent surface over the #0B0F19 background, subtle top highlight `inset 0 1px 0 rgba(255,255,255,0.04)`.
- Pills and badges: fully rounded (999px), 1px border, small uppercase Outfit labels with letter-spacing.
- Segmented city toggle (Peshawar / Lahore): pill container with #2A3B53 border, active segment filled with cyan #00F2FE at low opacity and cyan text.
- Inputs and sliders: track #2A3B53, filled portion and thumb cyan #00F2FE.

## Layout and spacing

- 8px spacing scale (4, 8, 16, 24, 32). Card padding 24px, grid gap 16px.
- Responsive column grid; cards align on a consistent baseline.
- Section headings in Outfit with a small cyan accent bar or icon.

## Tone

- Scientific, calm telemetry console. No stock photos. Thin line icons. Minimal gradients except the crimson-to-violet hazard gradient and the tier-tinted hero card.
- Footer text in muted colour: "Student learning project. Not an official health warning system. Follow guidance from Pakistan's health and environmental authorities."
```

## Call log

1. `create_project` `{"title": "Dhund-Khabar AI Dashboard"}`: ok, `projects/2753126111242902468`.
2. `create_design_system` (projectId 2753126111242902468, theme above): ok, `assets/5576147302249970983`.
3. `update_design_system` (name `assets/5576147302249970983`, same payload, as the create tool's instructions direct): **failed** with `MCP tool error: Request contains an invalid argument.`
4. `list_design_systems` (projectId 2753126111242902468): returns the asset above.
5. `get_project`: `designTheme` is `{}` and there are no screen instances, so `apply_design_system` (which requires `selectedScreenInstances`) cannot run until a screen exists (task 9.2).

## User decision after the failure

"Keep as is, apply in 9.2". No retry of `update_design_system`. In task 9.2, after `generate_screen_from_text` (DESKTOP), call `apply_design_system` with assetId `5576147302249970983`, projectId `2753126111242902468` and the new screen instance from `get_project`. The 9.2 prompt must restate Outfit for headings and numbers, Plus Jakarta Sans for body, 16px glass cards, and all hex values, because Stitch did not store `headlineFont`, `typography` or `spacing`.

## Task 9.2 follow-up

- The dashboard screen prompt and full call results are in `docs/stitch/screen_prompt.md`.
- `apply_design_system` (asset 5576147302249970983) succeeded on screen `c0b225a4172e454891555481093ca978` and produced the themed screen `848dc84695fd4e349433d61c947b4f09`. After this call the project `designTheme` holds the design system theme.

## Approved design (task 10, 2026-11-23)

- User reply at the checkpoint: "continue", plus "I have changed the design for you in Stitch".
- Approved screen: `projects/2753126111242902468/screens/14f8d14f71a040d7849bb47166ab7f41` ("Dhund-Khabar AI - Apple Inspired Atmospheric Terminal", 2560 x 4958). Files: `docs/stitch/dashboard_approved.html`, `dashboard_approved.png`, overview `dashboard_approved_preview.png`.
- The user made three versions of one redesign in the same project. By the timestamp prefix in the screenshot file ids the order is: "Smog Intelligence Console" `49559a98c1184a47b050192ce1fe6c26` (`dashboard_alt_console.*`), then "Modern Atmospheric Intelligence" `670548a476db43a6bbf42247a337716b` (`dashboard_alt_modern.*`), then the approved Apple screen. The earlier themed screen `848dc84695fd4e349433d61c947b4f09` is unchanged.

### What the user changed compared with the themed design (`dashboard.html`)

- Style: Apple-like black page (`bg-black`, text `#f5f5f7`, muted `#86868b`, green `#30d158`) instead of navy `#0B0F19`; fonts are an SF Pro system stack + Inter, JetBrains Mono for numbers, Noto Nastaliq Urdu for Urdu. No Outfit, no Plus Jakarta Sans.
- Header: "Dhund-Khabar 2.4 دُھند خبر" wordmark, "EPD & KP-EPA Satellite Mesh", station "PK-PWR-01 (Khyber Pass Baseline)" and a "Sync" button. The exact title only remains in `<title>`.
- Control row: Peshawar پشاور / Lahore لاہور pill toggle, inline city stat line (Peshawar 318 AQI Hazardous, Lahore 422 AQI "Severe Critical"), +24h / +48h / +72h pills. The two summary cards are gone.
- Hero: big AQI number, "Hazardous (خطرناک)" pill, a linear gradient spectrum bar with a thumb (0 Good / 100 Mod / 200 Unhealthy / 300+ Hazardous) instead of the semi-gauge, plus an aerosol row (PM2.5, PM10, NO2, CO).
- Condition tiles: PM2.5 with hourly bars, Wind with compass dial, Inversion Ceiling ("180 m AGL", dT/dz), and a new Psychrometric Matrix tile (temp, humidity, dew point).
- What-If: renamed "Counterfactual Simulation Lab تجربہ گاہ", preset buttons (Rain / Front, Traffic Curfew, Reset), sliders wind 1-35 km/h, boundary temperature 5-38 °C, anthropogenic source throttling 0 to -80 %; outcome card with tier, delta and advisory text.
- Analytics: tabs replaced by one observed-vs-forecast trajectory chart with a 95 % CI band and four model benchmark cards (Gradient Boost v2, Random Forest, Ridge ElasticNet, Persistence) with MAE / RMSE / Recall.
- Policy: "Actionable Health & Civic Policy Board" with four cards (School Operations, Respiratory Health, Industrial Kilns, Vehicular Rationing).
- Footer: "Dhund-Khabar Atmospheric Intelligence (دُھند خبر) | Model 0x7F2B8D | US EPA 40 CFR 58 | Provincial EPA Validated". The disclaimer is gone.

### Adaptation notes agreed at the checkpoint (apply on top of the approved design)

None of these are already resolved by the user's design.

1. Fonts: load Outfit and Plus Jakarta Sans through a Google Fonts `@import`, with the `system-ui, -apple-system, "Segoe UI", Roboto, sans-serif` fallback. The approved HTML loads Inter, JetBrains Mono and Noto Nastaliq Urdu only.
2. Remove invented status text that looks like real data. In this design that includes "Calibrated 05:42 PKT", "Sync" / station id, "Live Model", "Severe Critical", "Protocol Level III", "Next Statutory Review 06:00 PKT", "Section 144", "Code Amber", "Active Patrol", "Rerouted", "Provincial EPA Validated", "Model 0x7F2B8D", "Critical Dawn Window", "Model Loss ... R²". No "Standard Protocol" or "Last 30 Days" text appears in it.
3. Gauge: map AQI 0-500 linearly across the gauge, with tier segments sized in proportion to their AQI ranges (0-50, 51-100, 101-150, 151-200, 201-300, 301-500). The approved spectrum bar has evenly spaced labels at 0/100/200/300+, so this note applies to it as well.
4. Urdu: wrap Urdu text in a `dir="rtl"` element (`<bdi>`) so brackets do not flip. The approved HTML has no `bdi` or `dir="rtl"`.
5. Never label data as live. The approved HTML has "Live Model", a "Live Sync" button, a JS clock that writes the current time, and the caption "... in real-time.".

### Requirement items the approved design does not cover (from tasks 9.2 / 11, Requirement 17)

The orchestrator should confirm with the user which side wins for each item. These are listed only and have not been decided here.

- Exact header "Dhund-Khabar AI (دُھند خبر)" is not visible on the page.
- No "Latest Available Data: … | Peshawar vs. Lahore" badge and no "Historical dataset" label.
- No two side-by-side city summary cards (only an inline stat line).
- Hero uses a linear bar, while Requirement 17.4 and the design call for a semi-gauge. Tier tint and pulse are not visible in the static screen.
- Inversion card shows a measured-looking "180 m AGL" cap altitude. The design says the card must state it is a heuristic proxy.
- What-If sliders differ: required wind 0-30 km/h, temperature 5-35 °C, humidity 20-100 %. The design has no humidity slider and adds an emissions slider that is not a model feature. The exact caption "Scenario exploration from a statistical model, not a physical simulation." is missing.
- Visual Analytics tabs (actual vs predicted, heatmap, feature importance, model comparison table) are replaced by one chart with a 95 % CI band the models do not produce, and benchmark cards for Ridge ElasticNet and Persistence models the project does not train (it trains Linear Regression, Random Forest, Gradient Boosting).
- Policy section has four cards. The required three rows are School Operations Status, Respiratory Protective Guidance and Industrial Advisory.
- The exact footer "Student learning project. Not an official health warning system. Follow guidance from Pakistan's health and environmental authorities." is missing.
- All numbers are hard-coded mock values (318, 422, 268.4, dates Nov 12-22). Treat them as placeholders only.

### User decision at checkpoint 10: "Your look, required content"

The approved screen is "Apple Inspired Atmospheric Terminal" (`docs/stitch/dashboard_approved.html` / `.png`). The user chose to adopt the redesign's visual look and keep the content the spec requires. Where the two conflict, the required content wins and anything the data cannot support is dropped.

**Adopted from the redesign (visual look)**

- Near-black page (`#000000`, canvas `#09090b`), text `#f5f5f7`, muted `#86868b`, rounded frosted glass cards in a card grid (`rgba(255,255,255,0.035)` surface, `rgba(255,255,255,0.08)` border, 40px blur), green `#30d158` accent where the redesign uses it.
- Urdu labels next to English ones (پشاور / لاہور), compact uppercase mono labels.
- PM2.5 / PM10 / NO2 / CO mini row in the hero, wind compass card, a temperature / humidity / dew point card.

**Required content kept (from the spec)**

- Visible exact header "Dhund-Khabar AI (دُھند خبر)", with the Urdu wrapped in `<bdi dir="rtl">`.
- Badge "Latest Available Data: <last timestamp> | Peshawar vs. Lahore" plus a "Historical dataset" label. The timestamp is read from the loaded data.
- Side-by-side summary cards for both cities.
- Hero shows both a semi-circle SVG gauge (AQI 0-500 linear, tier segments proportional to their ranges) and the redesign's straight colour tier bar under it (the user selected both).
- PM2.5 card with the 24 h change; wind card with the arrow rotated by `wind_direction_10m`; thermal inversion card labelled as a heuristic proxy (no altitude).
- What-If sliders: wind 0-30 km/h, temperature 5-35 °C, humidity 20-100 %, with the exact caption "Scenario exploration from a statistical model, not a physical simulation."
- Visual Analytics tabs: actual vs predicted, heatmap, feature importance, model comparison table from `results/metrics.csv`.
- Policy terminal with exactly three rows (School Operations Status, Respiratory Protective Guidance, Industrial Advisory) from `get_risk_advisory`.
- Exact footer "Student learning project. Not an official health warning system. Follow guidance from Pakistan's health and environmental authorities." on every view.

**Dropped (not supported by the data, or misleading)**

- 95 % confidence band, emission-throttling slider, inversion altitude "180 m AGL", Ridge model.
- "Live Model", "real-time", "Sync" and the clock script.
- Invented statuses: "Calibrated 05:42 PKT", "Severe Critical", "Protocol Level III", "Section 144", "Next Statutory Review", "Provincial EPA Validated", "Model 0x7F2B8D", station names, "Satellite Mesh".
- The Vehicular Rationing card.

**Fonts and colours**

- Outfit (headings and numbers) and Plus Jakarta Sans (body) loaded through a Google Fonts `@import` with the `system-ui` fallback, as the spec requires. JetBrains Mono is used for small labels and Noto Nastaliq Urdu for Urdu text, as in the redesign.
- EPA tier colours stay exact: `#00E400`, `#FFFF00`, `#FF7E00`, `#FF0000`, `#8F3F97`, `#7E0023`.
- A pulse animation class (`.dk-pulse`) marks Very Unhealthy and Hazardous.
- This replaces the navy `#0B0F19` / `#161F30` / `#2A3B53` palette of Requirement 16.1 with the redesign's near-black glass palette, by the user's decision. Cyan `#00F2FE`, amber `#FFB300`, crimson `#FF3366` and violet `#9933FF` remain available as CSS variables (crimson and violet drive the pulse glow).
