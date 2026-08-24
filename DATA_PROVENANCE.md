# Data Provenance

Every dataset used in this project, where it came from, how it was retrieved,
its coverage, and what transformations were applied. Reproducibility matters:
each row here should be traceable to the code in `src/`.

| Dataset | Original source | Retrieval method | Coverage | Transformation |
|---|---|---|---|---|
| Snow water content | CDEC (California Dept. of Water Resources), snow-course sensor 3 | CSV download via CNRA open-data mirror; UA-header retry, decade-chunked pulls | 1980–2025, 259 courses | April 1 SWC → % of each course's 1991–2020 normal → winsorized median across courses → `snowpack_pct` |
| Day-ahead LMP (modern) | CAISO OASIS | `gridstatus` wrapper, `Markets.DAY_AHEAD_HOURLY`, 3 trading hubs (NP15/SP15/ZP26) | 2023–2025, summer (Jun–Sep) | hourly LMP per hub → daily mean per hub → mean across hubs → summer `price_mean/peak/vol` |
| Day-ahead LMP (historical) | CAISO OASIS (archived) | Public GitHub mirror of archived OASIS data (`manukalia/CA_Electricty_Price_Prediction_Neural_Net`), Bayshore node `BAYSHOR2_1_N001` | 2016–2018, summer (Jun–Sep) | single node hourly `dam_price_per_mwh` → same daily-aggregation pipeline |
| Hydro generation | CAISO fuel mix | `gridstatus` `get_fuel_mix`, Large + Small Hydro columns | 2018–2025 | 5-min MW × (5/60) h → MWh → summer GWh → `hydro_gwh` |
| Hydro generation (cross-check) | EIA-930, CISO region, WAT data type | EIA v2 API (`electricity/rto/fuel-type-data`), hourly + daily; revision dedup | hourly 2019–2025, daily 2020–2025 | MW/MWh → summer GWh → `hydro_gwh_eia` |
| Temperature | Open-Meteo archive (Fresno, CA) | `archive-api.open-meteo.com`, daily max temp | 2016–2025, summer | daily max °C → summer mean + heat-wave days (≥38 °C) → `temp_mean_c`, `heat_days_38c` |
| Demand | EIA-930, CISO region | EIA v2 API (`electricity/rto/region-data`, data-type D), hourly | 2018–2025 (requires key) | hourly MW → summer mean/peak → `demand_mean_mw`, `demand_peak_mw` |
| Natural gas | EIA, Henry Hub | EIA v2 API (`natural-gas/pri/sum`, series `NG.RNGWHHD.D`), daily | 2001–2025 (requires key) | daily $/MMBtu → summer mean → `gas_mean` |

## Important: two price measurement regimes

The six-year price window is **not** a single homogeneous series:

- **2016–2018** — one CAISO *node* (Bayshore, `BAYSHOR2_1_N001`) from an archived
  third-party mirror of OASIS data.
- **2023–2025** — three CAISO *trading hubs* (NP15, SP15, ZP26) fetched live via
  `gridstatus`.

These are different measurement objects (a nodal price vs. a hub aggregate) from
different retrieval paths. Measured across the three hubs in 2023–2025, the
cross-hub dispersion of summer volatility is ~10–17% of the mean — an upper bound
on the regime difference the single-node 2016–18 data introduces. Year-to-year
volatility swings (5 → 28 $/MWh) are several times larger than this dispersion,
so the volatility comparison is meaningful, but **all cross-regime model
comparisons are treated as exploratory**, and the 2019–2022 gap means the two
regimes never overlap in time for a direct calibration.

## Known gaps (never imputed)

| Gap | Detail |
|---|---|
| 2019–2022 prices | CAISO's public OASIS surface no longer serves pre-2023 LMP; the 2016+ archive sits behind an authenticated bulk downloader |
| 2020 EIA hourly hydro | 893 of 2,928 summer hours present — flagged missing, partial sums would be misleading |
| 2015 snowpack | Index = 0 (extreme drought year); winsorization guards outliers |
| Demand / gas columns | Empty unless `EIA_API_KEY` is set (fetchers are ready, key required) |

## Reproducibility

- `data/raw/*.csv` — cached, committed (~15 MB) so the analysis runs without
  refetching.
- `data/sample/sample_panel.csv` — 6-year demo slice; run the analysis with no
  downloads and no API keys.
- `scripts/make_sample_data.py` regenerates the sample from the full panel.
- Fetchers live in `src/fetch_*.py` and are invoked by `src/run_pipeline.py --fetch`.
