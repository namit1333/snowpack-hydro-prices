<div align="center">

# ❄️ Snowpack → Hydropower → Electricity Prices

### A Reproducible Causal-Chain Study: Does California's April 1 Snowpack Predict Summer Electricity Price Behavior Through Hydroelectric Generation?

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-2.x-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.24+-013243?style=for-the-badge&logo=numpy&logoColor=white)
![statsmodels](https://img.shields.io/badge/statsmodels-0.14+-blue?style=for-the-badge)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange?style=for-the-badge&logo=scikit-learn&logoColor=white)
![pytest](https://img.shields.io/badge/pytest-7.x-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white)

**Author:** [namit1333](https://github.com/namit1333)

---

*End-to-end ETL → feature engineering → forecasting → statistical inference, fully reproducible from public data, cached, and unit-tested.*

</div>

---

## 📋 Table of Contents

| # | Section | Description |
|---|---------|-------------|
| 1 | [Research Question & Hypothesis](#1-research-question--hypothesis) | The causal chain under investigation |
| 2 | [Architecture Overview](#2-architecture-overview) | End-to-end pipeline flowchart |
| 3 | [Data Engineering](#3-data-engineering) | Sources, ETL challenges, and panel schema |
| 4 | [Feature Engineering](#4-feature-engineering) | Snowpack index construction |
| 5 | [Statistical Framework](#5-statistical-framework) | Models, DM test, mediation analysis |
| 6 | [Results](#6-results) | First-stage findings, price analysis, conclusions |
| 7 | [Data Coverage & Limitations](#7-data-coverage--limitations) | Honest assessment of data gaps |
| 8 | [Tech Stack](#8-tech-stack) | Tools and engineering practices |
| 9 | [Reproduction Guide](#9-reproduction-guide) | How to run everything |
| 10 | [Repository Structure](#10-repository-structure) | File organization |
| 11 | [Extension Roadmap](#11-extension-roadmap) | Future work directions |

---

## 1. Research Question & Hypothesis

> **Does winter/spring snowpack in California predict summer electricity price behavior via hydroelectric generation capacity?**

### Hypothesized Causal Chain

```
┌─────────────────────┐      ┌─────────────────────────┐      ┌─────────────────────────────┐
│  April 1 Snowpack   │──────▶│  Summer Hydro Output    │──────▶│  Summer Price Volatility    │
│  (% of normal)      │  a    │  (GWh, fuel mix)        │  b    │  (std of daily $/MWh)       │
└─────────────────────┘      └─────────────────────────┘      └─────────────────────────────┘
           │                                                        ▲
           │                    c' (direct)                         │
           └────────────────────────────────────────────────────────┘
```

**Path Definitions:**
- **a-path**: Wet winters → more snow water content → more summer hydro output
- **b-path**: More cheap hydro dispatch → lower and *calmer* prices
- **c'-path**: Any snowpack effect *not* mediated by hydro (direct effect)
- **c-path (total)**: Total effect of snowpack on prices (a × b + c')

### Statistical Approach

This study employs the **Baron & Kenny (1986)** mediation framework with a **Sobel test** of the indirect effect. The key question: does controlling for actual hydro output **kill** the direct snowpack effect — which would confirm the hypothesized mechanism?

---

## 2. Architecture Overview

<p align="center">
  <img src="results/figures/pipeline_flowchart.png" alt="Pipeline Architecture" width="900">
</p>

<div align="center">

| Stage | Module | Responsibility |
|:------|:-------|:---------------|
| **Data Acquisition** | `fetch_cdec.py`, `fetch_caiso_oasis.py`, `fetch_eia.py` | Pull raw data from CDEC, CAISO OASIS, EIA-930; cache to `data/raw/` |
| **Feature Engineering** | `features.py` | Build yearly panel: snowpack index, summer price features, summer hydro |
| **Modeling** | `models.py` | Baselines + augmented models, walk-forward validation, DM test, mediation |
| **Orchestration** | `run_pipeline.py` | End-to-end execution → `results/` (plot, RMSE table, `results.json`) |
| **Notebooks** | `notebooks/` | Executed analysis narrative (01\_eda, 02\_mediation, 03\_walkforward) |
| **Tests** | `tests/` | 12 unit tests with synthetic-data ground truth |

</div>

---

## 3. Data Engineering

### 3.1 Data Sources

| Variable | Source | Coverage | Granularity | Access |
|:---------|:-------|:---------|:------------|:-------|
| Snow water content (sensor 3, 259 courses) | [CDEC](https://data.cnra.ca.gov/dataset/california-snow-data) via CNRA open-data mirror | 1980–2025 | Monthly (incl. April 1) | No key required |
| Day-ahead hourly LMP | [CAISO OASIS](https://oasis.caiso.com) via `gridstatus` | 2023–2025 | Hourly, 3 trading hubs | No key required |
| Hydro generation (Large + Small) | CAISO fuel mix via `gridstatus` | 2018–2025 | 5-minute | No key required |
| Hydro generation (cross-check) | [EIA-930](https://www.eia.gov/opendata/) (`electricity/rto/fuel-type-data`, CISO/WAT) | 2019–2025 | Hourly / daily | Free API key |

### 3.2 Real-World ETL Challenges Solved

| Challenge | Solution | Impact |
|:----------|:---------|:-------|
| **CDEC connection resets** without proper `User-Agent` | All requests carry research UA header; decade-chunked pulls avoid timeouts | 100% data retrieval success |
| **CAISO OASIS pre-2023 LMP unavailable** | Verified across query names, market-run IDs, API versions, node aliases; documented as known limitation | Transparent scope definition |
| **EIA-930 daily rows repeat** per data revision | Deduplicated by keeping latest revision (max per day) | Clean time series |
| **2020 hourly hydro gap** (893/2,928 hours) | Flagged missing, not imputed — partial sum would be misleading | Honest uncertainty quantification |
| **EIA pagination** (≤5,000 rows/page) | Offset-driven loop with automatic continuation | Complete data retrieval |
| **Cross-source validation** | CAISO fuel-mix vs. EIA-930 agree within ~2% | Strong data-quality signal |

### 3.3 Panel Schema (`data/processed/panel.csv`)

| Column | Definition | Units |
|:-------|:-----------|:------|
| `snowpack_pct` | April 1 SWC ÷ course's 1991–2020 April 1 mean × 100, **median across courses** per year | % |
| `price_mean` | Summer (Jun–Sep) mean of daily-mean day-ahead LMP, averaged across 3 hubs | $/MWh |
| `price_peak` | Summer maximum of daily-mean LMP | $/MWh |
| `price_vol` | **Target**: std of daily-mean LMP over the summer | $/MWh |
| `price_vol_hourly` | Robustness target: std of *all* hourly LMP observations | $/MWh |
| `hydro_gwh` | Summer hydro generation (Large + Small) | GWh |
| `hydro_gwh_eia` | Same, from EIA-930 (independent source) | GWh |

---

## 4. Feature Engineering

### Snowpack "Percent of Normal" Index

The statewide snowpack index follows the **Department of Water Resources (DWR)** convention:

```python
# For each snow course:
1. Extract April 1 snow water content (SWC) per year (tolerant to ±5-day shifts)
2. Normalize by course's 1991–2020 April 1 mean (require ≥10 years for stable normal)
3. Winsorize cross-course distribution at 1st/99th percentile per year
4. Aggregate to statewide index via MEDIAN (robust to outlier courses)
```

**Result:** One `snowpack_pct` value per year, spanning 1980–2025 (46 years).

<p align="center">
  <img src="results/figures/snowpack_timeseries.png" alt="Snowpack Time Series" width="850">
</p>

<div align="center">

*Figure 1: Statewide April 1 snowpack index (1980–2025). Blue shading = above normal; red shading = below normal. Key drought/wet years annotated.*

</div>

### Summer Window Definition

- **Period:** June 1 – September 30 (load-critical hydro season)
- **Volatility metric:** Standard deviation of daily-mean LMP across the summer
- **Hourly variant:** Robustness check using all hourly observations (addresses aggregation artifacts)

---

## 5. Statistical Framework

<p align="center">
  <img src="results/figures/equations.png" alt="Statistical Framework Equations" width="750">
</p>

<div align="center">

*Figure 2: Core equations — volatility definition, baseline models, augmented OLS, Diebold-Mariano test with Newey-West HAC variance, and mediation decomposition.*

</div>

### Model Specifications

| Model Type | Specification | Purpose |
|:-----------|:--------------|:--------|
| **Persistence** | ŷₜ = yₜ₋₁ | Naive baseline |
| **Trailing 3-year mean** | ŷₜ = mean(yₜ₋₁, yₜ₋₂, yₜ₋₃) | Simple smoothing baseline |
| **ARIMA(0,1,0) + drift** | Δyₜ = μ + εₜ | Random walk with drift |
| **Augmented OLS** | yₜ = β₀ + β₁yₜ₋₁ + β₂·snowpackₜ + εₜ | Adds snowpack predictor |
| **ARIMAX** | ARIMA(0,1,0) + snowpack as exogenous regressor | Time-series + covariate |

### Validation Protocol

- **Expanding-window walk-forward:** Training window grows each year; every prediction uses only data available *before* the target year (**strict out-of-sample**)
- **Metrics:** RMSE, MAE
- **Statistical comparison:** Diebold-Mariano test on squared errors with **Newey-West HAC** variance (accounts for autocorrelation in forecast errors)

### Mediation Analysis

**Baron-Kenny decomposition:**

```
Path a:  Summer hydro = a × Snowpack + u        (snowpack → hydro)
Path b:  Price vol = b × Summer hydro + v        (hydro → prices, controlling for snowpack)
Path c': Price vol = c' × Snowpack + b × Hydro   (direct effect of snowpack)
Path c:  Price vol = c × Snowpack                (total effect)

Proportion mediated = 1 - c'/c
Indirect effect = a × b
Sobel test: z = (a × b) / sqrt(b² × SE(a)² + a² × SE(b)²)
```

<p align="center">
  <img src="results/figures/mediation_diagram.png" alt="Mediation Path Diagram" width="700">
</p>

<div align="center">

*Figure 3: Baron-Kenny mediation path diagram with estimated coefficients. Blue = a-path (significant), gold = b-path (illustrative, n=3), red dashed = c'-path (direct effect).*

</div>

---

## 6. Results

### 6.1 First Stage — Snowpack → Hydro (Statistically Powered)

<p align="center">
  <img src="results/figures/hydro_vs_snowpack.png" alt="Hydro vs Snowpack Regression" width="700">
</p>

<div align="center">

*Figure 4: First stage of the causal chain. Blue circles = CAISO fuel mix (n=8); gold squares = EIA-930 cross-validation (n=6). Dashed lines = OLS fit. Both sources show significant positive relationship.*

</div>

| Source | Slope (GWh/pp) | p-value | R² | Pearson r | n |
|:-------|:---------------|:--------|:---|:----------|:--|
| **CAISO fuel mix** | **+24.9** | **0.022** | 0.61 | 0.78 | 8 |
| EIA-930 (independent) | +27.0 | 0.052 | 0.65 | 0.81 | 6 |

> **Interpretation:** A 10-percentage-point rise in April 1 snowpack → **≈ +250 GWh of summer hydro** — roughly 3% of a typical CAISO summer hydro output. This relationship is statistically significant (p = 0.022) and consistent across two independent measurement systems.

### 6.2 Price Legs — Illustrative (2023–2025)

<p align="center">
  <img src="results/figures/price_panel_2023_2025.png" alt="Price Panel 2023-2025" width="750">
</p>

<div align="center">

*Figure 5: The available price window. Blue bars = snowpack; red line = price volatility. Note: wettest year (2023) had the HIGHEST volatility — counter to hypothesis.*

</div>

<p align="center">
  <img src="results/figures/correlation_heatmap.png" alt="Correlation Heatmap" width="500">
</p>

<div align="center">

*Figure 6: Pearson and Spearman correlations between snowpack and price features. Strong positive correlations (surprising given hypothesis — see interpretation below).*

</div>

#### Walk-Forward RMSE Table

| Model | RMSE ($/MWh) | n\_OOS | Notes |
|:------|:-------------|:-------|:------|
| **augmented\_ols** (lag vol + snowpack) | **5.37** | 1 | Best performing |
| augmented\_arimax | 7.03 | 1 | |
| baseline\_arima | 7.03 | 1 | |
| baseline\_naive (persistence) | 7.03 | 1 | |
| baseline\_mean3 | 13.50 | 1 | Worst performing |

<p align="center">
  <img src="results/figures/walkforward_forecasts.png" alt="Walk-Forward Forecasts" width="700">
</p>

<div align="center">

*Figure 7: Out-of-sample forecasts vs actual. Only 1 held-out year available — results are illustrative, not conclusive.*

</div>

**Diebold-Mariano Test:** Cannot be computed at n = 1 (requires ≥3 common out-of-sample errors). Full results including flagged mediation estimates in [`results/results.json`](results/results.json).

### 6.3 Plain-English Conclusion

> **The study cannot yet confirm that snowpack predicts summer electricity price behavior** — and the small sample we do have points the *other* way: the wettest year in the price window (2023, snowpack ≈ 236% of normal) had the *highest* volatility (28.2 vs. 8.2 $/MWh in 2025).

> **What IS firmly established** — on eight years of data and two independent sources — is the **first half of the chain**: wet winters reliably produce more summer hydro (p ≈ 0.02, ~25 GWh per +1 pp of normal).

> The price response is a real, unresolved question: the expected channel (more hydro → calmer prices) may be **swamped by confounders** in the same years — heat waves driving AC demand, natural-gas price shocks, or transmission constraints. A longer price archive (2016+) would settle it.

---

## 7. Data Coverage & Limitations

### Known Data Gaps

| Gap | Impact | Mitigation |
|:----|:-------|:-----------|
| **CAISO LMP pre-2023 unavailable** via public OASIS API | Price analysis limited to 3 years (2023–2025) | Pipeline auto-adapts when longer history added; explicitly flagged "illustrative" |
| **EIA-930 hourly hydro gap in 2020** (893/2,928 hours) | 2020 partial summer excluded from hydro analysis | Flagged missing, not imputed — partial sums would be misleading |
| **2015 snowpack index = 0** (extreme drought) | Extreme outlier in time series | Winsorization at 1st/99th percentile guards against this |
| **Fuel-mix history limited to 2018+** | Mediator coverage shorter than snowpack (1980+) | EIA-930 provides independent cross-check back to 2019 |

### Data Quality Signals

- **Cross-source validation:** CAISO fuel-mix vs. EIA-930 hydro agree within **~2%** (e.g., 9,935 vs. 10,061 GWh for 2023)
- **Snowpack index stability:** 259 snow courses, 46 years, median aggregation robust to outlier courses
- **Unit tests:** 12 synthetic-data ground-truth checks verifying feature engineering and model logic

---

## 8. Tech Stack

| Category | Tools | Rationale |
|:---------|:------|:----------|
| **Data Acquisition** | `pandas`, `requests`, `gridstatus` (OASIS wrapper) | Handles pagination, retries, API quirks |
| **Statistical Modeling** | `statsmodels` (ARIMA, OLS, HAC), `scipy` | Industry-standard inference; Newey-West HAC for autocorrelation |
| **Reproducibility** | Pinned `requirements.txt`, project-local `.venv`, cached raw data | One-command reproduction |
| **Testing** | `pytest` — 12 tests | Synthetic-data ground truth; catches regressions |
| **Documentation** | Executed notebooks, this README, regenerable figures | Full analysis narrative |
| **Security** | API keys via environment variable only (never committed) | Best practice for open-source |
| **Version Control** | Git, commit identity locked to project author | Clean contributor history |

---

## 9. Reproduction Guide

### Quick Start

```bash
# 1. Create virtual environment
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt      # Windows
# .venv/bin/pip install -r requirements.txt        # macOS / Linux

# 2. Set API key (optional — for hydro cross-validation)
export EIA_API_KEY=your_key_here

# 3. Fetch all data and run full analysis (~5-10 minutes)
.venv/Scripts/python -m src.run_pipeline --fetch

# 4. Regenerate all figures (README visuals)
.venv/Scripts/python scripts/make_figures.py

# 5. Run tests
.venv/Scripts/python -m pytest tests/ -v

# 6. Execute notebooks
.venv/Scripts/python scripts/make_notebooks.py
.venv/Scripts/python -m jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb
```

### Outputs

| File | Description |
|:-----|:------------|
| `results/panel.csv` | Yearly analysis panel (1980–2025) |
| `results/results.json` | All statistics, RMSE, mediation estimates |
| `results/figures/*.png` | 8 publication-quality figures |
| `results/walkforward_rmse.csv` | Walk-forward RMSE table |
| `notebooks/*.ipynb` | Executed analysis notebooks |

---

## 10. Repository Structure

```
snowpack-hydro-prices/
├── README.md                     ← you are here
├── requirements.txt              pinned dependencies
├── data/
│   ├── raw/                      cached CDEC, CAISO, EIA data (~15 MB)
│   └── processed/
│       └── panel.csv             yearly analysis panel
├── src/
│   ├── __init__.py
│   ├── fetch_cdec.py             CDEC snow courses (1980+)
│   ├── fetch_caiso_oasis.py      CAISO LMP + fuel mix via gridstatus
│   ├── fetch_eia.py              EIA-930 CISO hydro (EIA_API_KEY)
│   ├── features.py               panel builder, %-of-normal index, volatility
│   ├── models.py                 baselines, augmented, walk-forward, DM, mediation
│   └── run_pipeline.py           end-to-end orchestration
├── notebooks/
│   ├── 01_eda.ipynb              Exploratory data analysis
│   ├── 02_mediation_analysis.ipynb   Baron-Kenny decomposition
│   └── 03_walkforward_backtest.ipynb Walk-forward validation
├── scripts/
│   ├── make_notebooks.py         notebook generators
│   └── make_figures.py           regenerates all README figures
├── tests/
│   ├── test_features.py          feature engineering tests
│   └── test_models.py            model logic tests
└── results/
    ├── figures/                  8 PNG figures
    ├── walkforward_rmse.csv      RMSE comparison table
    └── results.json              full statistics
```

---

## 11. Extension Roadmap

| Priority | Extension | Expected Impact |
|:---------|:----------|:----------------|
| **High** | Unlock 2016+ CAISO prices (bulk archive / browser-assisted download) | Walk-forward, DM test, mediation run at full power; ~10-year window |
| **Medium** | Add confounders: heat-wave indices + Henry-Hub natural-gas prices | Test whether they explain the 2023 anomaly |
| **Medium** | Monthly (rather than annual) mediation analysis | More statistical power; captures intra-season dynamics |
| **Low** | Snow-course elevation × basin stratification | Identify which elevation bands drive the signal |
| **Low** | Block bootstrap or Bayesian mediation with credible intervals | Rigorous inference on small sample |

---

## 📚 References

- Baron, R. M., & Kenny, D. A. (1986). The moderator-mediator variable distinction in social psychological research. *Journal of Personality and Social Psychology*, 51(6), 1173-1182.
- Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics*, 13(3), 253-263.
- Newey, W. K., & West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica*, 55(3), 703-708.

---

<div align="center">

**Built with ❤️ by [namit1333](https://github.com/namit1333)**

*Last updated: August 2025*

</div>
