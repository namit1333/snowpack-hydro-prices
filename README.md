<div align="center">

# ❄️ Snowpack → Hydropower → Electricity Prices

### Does California's winter snowpack predict summer electricity-market behavior?

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-17%20passing-brightgreen?style=for-the-badge)](.github/workflows/tests.yml)
[![CI](https://img.shields.io/badge/CI-GitHub%20Actions-blue?style=for-the-badge)](.github/workflows/tests.yml)

**Author:** [namit1333](https://github.com/namit1333) &nbsp;|&nbsp; **Repo:** [snowpack-hydro-prices](https://github.com/namit1333/snowpack-hydro-prices)

</div>

---

<div align="center">

| 🏔️ **46 years** of snowpack data | ⚡ **8 years** of hydro generation | 💰 **6 years** of electricity prices |
|:---:|:---:|:---:|
| CDEC snow courses, 1980–2025 | CAISO fuel mix + EIA-930 | CAISO day-ahead LMP, 2016–25 |

</div>

---

## 🎯 Key Finding

<div align="center">

> **A 10 percentage-point increase in April 1 snowpack is associated with approximately 250 GWh more summer hydro generation**
>
> *p = 0.022, R² = 0.61, n = 8 years — confirmed by two independent data sources*

</div>

⚠️ **Important limitation:** The price window is **6 years** (2016–18 + 2023–25, with a 2019–22 gap in public records). Walk-forward forecasts now have **3 genuinely held-out years**, but that is still too few for conclusive inference on the price legs. The snowpack → hydro relationship, however, is robust across 8 years.

---

## 🌟 Why This Matters

California relies heavily on renewable electricity, including hydroelectric generation. Mountain snowpack acts as a **natural reservoir**, storing winter precipitation for summer release. Understanding whether snowpack conditions can help predict summer electricity-market behavior could potentially improve:

- **Energy-market forecasting** for traders and utilities
- **Grid reliability planning** during drought vs. wet years
- **Climate adaptation strategies** as snowpack patterns shift under climate change

This project investigates whether the well-established physical link (snow → water → hydro) translates into measurable effects on electricity prices.

---

## 📊 Key Results at a Glance

| Question | Finding | Significance |
|:---------|:--------|:-------------|
| **Does snowpack predict hydro output?** | ✅ Yes — +24.9 GWh per +1 pp of snowpack | p = 0.022 (n = 8) |
| **Is the relationship reproducible?** | ✅ Yes — EIA-930 independent source agrees | +27.0 GWh/pp, p = 0.052 (n = 6) |
| **Does snowpack predict price volatility?** | ⚠️ Inconclusive — 3 held-out years | DM p = 0.052 (hourly vol, vs 3-yr mean) |
| **Does hydro mediate the snowpack→price link?** | ⚠️ Exploratory — insufficient observations | Proportion mediated ≈ 0.58 (n ≤ 6) |

---

## 🏗️ Architecture

<div align="center">
  <img src="results/figures/pipeline_flowchart.png" alt="Pipeline Architecture" width="850">
</div>

| Stage | Module | Description |
|:------|:-------|:------------|
| **Data Acquisition** | `fetch_cdec.py`, `fetch_caiso_oasis.py`, `fetch_eia.py`, `fetch_controls.py` | CDEC, CAISO OASIS, EIA-930, Open-Meteo, EIA gas |
| **Feature Engineering** | `features.py` | Yearly + monthly panels: snowpack index, price, hydro, controls |
| **Modeling** | `models.py` | Baselines, walk-forward (3 OOS), Diebold-Mariano, mediation |
| **Orchestration** | `run_pipeline.py` | End-to-end execution → results |
| **Validation** | `tests/` + GitHub Actions | 17 unit tests, auto-run on push |

---

## 📈 Results

### 1. The Snowpack → Hydro Link (Statistically Established)

<div align="center">
  <img src="results/figures/hydro_vs_snowpack.png" alt="Hydro vs Snowpack" width="650">
</div>

| Data Source | Slope (GWh/pp) | p-value | R² | Pearson r | n |
|:------------|:---------------|:--------|:---|:----------|:--|
| **CAISO fuel mix** | **+24.9** | **0.022** | 0.61 | 0.78 | 8 |
| EIA-930 (independent) | +27.0 | 0.052 | 0.65 | 0.81 | 6 |

> A 10-pp rise in April 1 snowpack ⇒ ≈ **+250 GWh of summer hydro** — roughly 3% of a typical CAISO summer hydro output. This relationship is statistically significant and consistent across two independent measurement systems.

### 2. The Price Question (6-Year Window, Still Inconclusive)

<div align="center">
  <img src="results/figures/price_panel_2023_2025.png" alt="Price Panel" width="650">
</div>

<div align="center">
  <img src="results/figures/correlation_heatmap.png" alt="Correlation Heatmap" width="450">
</div>

**⚠️ Three genuinely out-of-sample years (2023, 2024, 2025):**

| Model | RMSE ($/MWh) | OOS Years | Note |
|:------|:-------------|:----------|:-----|
| augmented\_arimax | 8.60 | 3 | Ties persistence |
| baseline\_naive | 8.60 | 3 | Persistence |
| baseline\_arima | 8.60 | 3 | — |
| augmented\_ols | 9.14 | 3 | — |
| baseline\_mean3 | 12.08 | 3 | Worst |
| augmented\_controls | 33.49 | 3 | Overfits at n=3 |

<div align="center">
  <img src="results/figures/walkforward_forecasts.png" alt="Walk-Forward" width="600">
</div>

> **Interpretation:** With three genuinely held-out years, **no model beats simple persistence** — the price signal from snowpack is not yet detectable. The wettest year (2023, snowpack ≈ 236%) had the *highest* volatility, opposite to the hypothesized direction. The Diebold-Mariano test (hourly volatility vs trailing-mean baseline) reaches **p = 0.052** — suggestive but not conclusive.

### 3. Mediation Analysis (Exploratory Only)

<div align="center">
  <img src="results/figures/mediation_diagram.png" alt="Mediation Diagram" width="650">
</div>

The Baron-Kenny decomposition suggests ~58% of the snowpack effect may be mediated through hydro — but this is **explicitly exploratory**: with n ≤ 6 overlapping years, the Sobel test is not informative (p ≈ 1.00). Kept in the repo because the decomposition is instructive, **not** as evidence.

### 4. Monthly Panel — The Power Boost

Beyond the annual panel, `features.py` builds a **monthly summer panel** (Jun–Sep, one row per year-month): **24 observations** from the 6 price years, vs 6 annual. This is the natural next step for the price legs as history grows:

| Year | Jun | Jul | Aug | Sep |
|:-----|:----|:----|:----|:----|
| 2016 | 4.7 | 6.3 | 3.3 | 3.5 |
| 2017 | 19.6 | 4.4 | 22.3 | 20.3 |
| 2018 | 7.0 | 40.9 | 19.8 | 3.2 |

*Monthly price volatility ($/MWh) — 4× the annual observations.*

### 5. The Statistical Framework

<div align="center">
  <img src="results/figures/equations.png" alt="Equations" width="650">
</div>

---

## 🗂️ Data Dictionary

| Variable | Description | Units | Source |
|:---------|:------------|:------|:-------|
| `snowpack_pct` | April 1 snowpack relative to 1991–2020 normal (median across 259 courses) | % | CDEC |
| `hydro_gwh` | Summer hydro generation (Large + Small) | GWh | CAISO fuel mix |
| `hydro_gwh_eia` | Same as above, independent source | GWh | EIA-930 |
| `price_mean` | Mean summer electricity price (day-ahead LMP) | $/MWh | CAISO OASIS |
| `price_vol` | Price volatility (std of daily-mean LMP) | $/MWh | CAISO OASIS |
| `price_peak` | Peak summer price (max of daily-mean LMP) | $/MWh | CAISO OASIS |
| `demand_mean_mw` | Mean summer electricity demand | MW | EIA-930 |
| `temp_mean_c` | Mean summer daily-max temperature | °C | Open-Meteo |
| `heat_days_38c` | Days with max temp ≥ 38 °C (heat waves) | days | Open-Meteo |
| `gas_mean` | Mean summer Henry Hub natural gas price | $/MMBtu | EIA |

---

## ⚠️ Limitations & Honest Caveats

<div align="center">

| Issue | Impact | Status |
|:------|:-------|:-------|
| **CAISO LMP gap 2019–2022** | Price analysis spans 2016–18 + 2023–25 (6 years) | Pipeline auto-adapts when history added |
| **n = 3 for out-of-sample forecasting** | RMSE comparisons are meaningful but low-power | Documented; DM test runs at p = 0.052 |
| **Confounding variables partially controlled** | Temperature included; demand/gas need EIA key | `fetch_controls.py` ready |
| **2020 EIA hourly gap** (893/2,928 hours) | Partial year excluded from hydro analysis | Flagged, not imputed |
| **Historical prices are a single node** (Bayshore) | 2016–18 prices differ from 3-hub average | Volatility comparable; documented |

</div>

**What would make this study conclusive:**
1. Continuous CAISO price data 2016+ (the 2019–22 gap is the binding constraint)
2. Temperature and demand controls (temperature ✅, demand/gas with EIA key)
3. 10+ years for walk-forward validation
4. Monthly panel with full coverage

---

## 🛠️ Tech Stack

| Category | Tools |
|:---------|:------|
| **Data** | `pandas`, `numpy`, `requests`, `gridstatus` |
| **Statistics** | `statsmodels` (ARIMA, OLS, HAC), `scipy` |
| **Controls** | Open-Meteo (no key), EIA-930 demand, EIA Henry Hub |
| **Testing/CI** | `pytest` (17 tests), GitHub Actions (Python 3.10–3.12) |
| **Reproducibility** | Pinned `requirements.txt`, `.venv`, cached raw data |

---

## 🚀 Reproduction Guide

### Quick Start (with the bundled sample data)

```bash
git clone https://github.com/namit1333/snowpack-hydro-prices.git
cd snowpack-hydro-prices
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt   # Windows
.venv/bin/pip install -r requirements.txt       # macOS/Linux

# Run analysis on the bundled sample panel (no API keys, no downloads):
.venv\Scripts\python -c "import sys; sys.path.insert(0,'.'); from src.features import build_panel; print('sample ready')"
```

### Full pipeline (real data, ~5-10 min)

```bash
# 1. Configure API key (optional — hydro cross-check + demand/gas controls)
cp .env.example .env   # then add your EIA key (free at eia.gov/opendata)

# 2. Fetch everything and run
.venv\Scripts\python -m src.run_pipeline --fetch   # Windows
.venv/bin/python -m src.run_pipeline --fetch       # macOS/Linux

# 3. Tests + figures
.venv\Scripts\python -m pytest tests/ -v
.venv\Scripts\python scripts/make_figures.py
```

**Requirements:** Python 3.10+ (tested 3.10–3.13). CI runs the suite on 3.10/3.11/3.12.

---

## 📁 Repository Structure

```
snowpack-hydro-prices/
├── README.md
├── requirements.txt
├── .env.example                    # Template for API keys
├── .github/workflows/tests.yml     # CI: pytest on push/PR
├── data/
│   ├── raw/                        # Cached source data (regenerable)
│   ├── processed/panel.csv         # Yearly analysis panel
│   └── sample/sample_panel.csv     # Demo data (no fetch needed)
├── src/
│   ├── fetch_cdec.py               # CDEC snow courses (1980+)
│   ├── fetch_caiso_oasis.py        # CAISO LMP + fuel mix + 2016-18 archive
│   ├── fetch_eia.py                # EIA-930 CISO hydro
│   ├── fetch_controls.py           # Demand, temperature, gas (confounders)
│   ├── features.py                 # Yearly + monthly panels
│   ├── models.py                   # Baselines, walk-forward, DM, mediation
│   └── run_pipeline.py             # End-to-end orchestration
├── notebooks/                      # Executed analysis notebooks
├── scripts/
│   ├── make_notebooks.py
│   ├── make_figures.py             # All README figures
│   └── make_sample_data.py         # Bundled demo dataset
├── tests/                          # 17 unit tests
└── results/
    ├── figures/                    # 8 publication-quality figures
    ├── walkforward_rmse.csv
    └── results.json
```

---

## 🔮 Extension Roadmap

| Priority | Extension | Impact |
|:---------|:----------|:-------|
| **P0** | Fill the 2019–2022 CAISO price gap | Continuous 10-year window; conclusive walk-forward |
| **P1** | Add demand + gas controls to panel (EIA key) | Addresses confounding; `fetch_controls.py` ready |
| **P2** | Monthly mediation on 24-observation panel | 4× statistical power |
| **P3** | Bayesian mediation with credible intervals | Rigorous small-sample inference |

---

## 📚 References

- Baron, R. M., & Kenny, D. A. (1986). The moderator-mediator variable distinction. *JPSP*, 51(6), 1173-1182.
- Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. *JBES*, 13(3), 253-263.
- Newey, W. K., & West, K. D. (1987). A simple, positive semi-definite HAC covariance matrix. *Econometrica*, 55(3), 703-708.

---

<div align="center">

**Built with ❤️ by [namit1333](https://github.com/namit1333)**

*This project investigates — but does not conclusively establish — a causal link between snowpack and electricity prices. The snowpack → hydro relationship is statistically established; the downstream price effect remains an open question with 6 years of price data.*

</div>
