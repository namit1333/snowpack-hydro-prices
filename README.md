<div align="center">

# ❄️ Snowpack → Hydropower → Electricity Prices

### Does California's winter snowpack predict summer electricity-market behavior?

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-12%20passing-brightgreen?style=for-the-badge)](tests/)

**Author:** [namit1333](https://github.com/namit1333) &nbsp;|&nbsp; **Repo:** [snowpack-hydro-prices](https://github.com/namit1333/snowpack-hydro-prices)

</div>

---

<div align="center">

| 🏔️ **46 years** of snowpack data | ⚡ **8 years** of hydro generation | 💰 **3 years** of electricity prices |
|:---:|:---:|:---:|
| CDEC snow courses, 1980–2025 | CAISO fuel mix + EIA-930 | CAISO day-ahead LMP |

</div>

---

## 🎯 Key Finding

<div align="center">

> **A 10 percentage-point increase in April 1 snowpack is associated with approximately 250 GWh more summer hydro generation**
>
> *p = 0.022, R² = 0.61, n = 8 years — confirmed by two independent data sources*

</div>

⚠️ **Important limitation:** Public CAISO price data currently provides only a **2023–2025 window** for this analysis. Consequently, the price-volatility and mediation results are **exploratory rather than statistically conclusive**. The snowpack → hydro relationship, however, is robust across 8 years of data.

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
| **Does snowpack predict price volatility?** | ⚠️ Inconclusive — only 3 years of price data | Illustrative result (n = 1 OOS) |
| **Does hydro mediate the snowpack→price link?** | ⚠️ Inconclusive — proportion mediated = 0.58 | Flagged illustrative (n = 3) |

---

## 🏗️ Architecture

<div align="center">
  <img src="results/figures/pipeline_flowchart.png" alt="Pipeline Architecture" width="850">
</div>

| Stage | Module | Description |
|:------|:-------|:------------|
| **Data Acquisition** | `fetch_cdec.py`, `fetch_caiso_oasis.py`, `fetch_eia.py` | Pull raw data from CDEC, CAISO OASIS, EIA-930 |
| **Feature Engineering** | `features.py` | Build yearly panel: snowpack index, price features, hydro output |
| **Modeling** | `models.py` | Baselines, walk-forward validation, Diebold-Mariano test, mediation |
| **Orchestration** | `run_pipeline.py` | End-to-end execution → results |
| **Validation** | `tests/` | 12 unit tests with synthetic ground truth |

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

### 2. The Price Question (Exploratory — Limited Data)

<div align="center">
  <img src="results/figures/price_panel_2023_2025.png" alt="Price Panel" width="650">
</div>

<div align="center">
  <img src="results/figures/correlation_heatmap.png" alt="Correlation Heatmap" width="450">
</div>

**⚠️ Illustrative results (n = 3 years):**

| Model | RMSE ($/MWh) | OOS Observations | Note |
|:------|:-------------|:-----------------|:-----|
| augmented_ols | 5.37 | 1 | Single held-out year — not conclusive |
| baseline_arima | 7.03 | 1 | — |
| baseline_naive | 7.03 | 1 | — |

<div align="center">
  <img src="results/figures/walkforward_forecasts.png" alt="Walk-Forward" width="600">
</div>

> **Interpretation:** The wettest year (2023, snowpack ≈ 236%) had the *highest* price volatility — opposite to the hypothesized direction. With only 3 years of price data, this neither confirms nor refutes the hypothesis. A longer price archive (2016+) is needed.

### 3. Mediation Analysis (Exploratory)

<div align="center">
  <img src="results/figures/mediation_diagram.png" alt="Mediation Diagram" width="650">
</div>

The Baron-Kenny decomposition suggests 58% of the snowpack effect may be mediated through hydro — but this is flagged **illustrative** (n = 3, Sobel p = 1.00).

### 4. The Statistical Framework

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
| `price_mean` | Mean summer electricity price (day-ahead LMP, 3 hubs) | $/MWh | CAISO OASIS |
| `price_vol` | Price volatility (std of daily-mean LMP) | $/MWh | CAISO OASIS |
| `price_peak` | Peak summer price (max of daily-mean LMP) | $/MWh | CAISO OASIS |

---

## ⚠️ Limitations & Honest Caveats

<div align="center">

| Issue | Impact | Status |
|:------|:-------|:-------|
| **CAISO LMP only available 2023–2025** | Price/mediation results are exploratory (n ≤ 3) | Pipeline auto-adapts when history added |
| **n = 1 for out-of-sample forecasting** | RMSE comparisons are illustrative, not conclusive | Documented; awaiting more data |
| **Confounding variables not controlled** | Heat waves, gas prices, demand may drive prices | Future extension |
| **2020 EIA hourly gap** (893/2,928 hours) | Partial year excluded from hydro analysis | Flagged, not imputed |

</div>

**What would make this study conclusive:**
1. CAISO price data from 2016+ (bulk archive access)
2. Temperature and demand controls
3. Natural-gas price covariates
4. 10+ years for walk-forward validation

---

## 🛠️ Tech Stack

| Category | Tools |
|:---------|:------|
| **Data** | `pandas`, `numpy`, `requests`, `gridstatus` |
| **Statistics** | `statsmodels` (ARIMA, OLS, HAC), `scipy` |
| **Testing** | `pytest` — 12 tests with synthetic ground truth |
| **Reproducibility** | Pinned `requirements.txt`, `.venv`, cached raw data |

---

## 🚀 Reproduction Guide

```bash
# 1. Clone and set up environment
git clone https://github.com/namit1333/snowpack-hydro-prices.git
cd snowpack-hydro-prices
python -m venv .venv

# Windows:
.venv\Scripts\pip install -r requirements.txt
# macOS/Linux:
.venv/bin/pip install -r requirements.txt

# 2. Configure API key (optional — for hydro cross-validation)
cp .env.example .env
# Edit .env with your EIA API key (get one free at eia.gov/opendata)

# 3. Run full pipeline (~5-10 minutes)
# Windows:
.venv\Scripts\python -m src.run_pipeline --fetch
# macOS/Linux:
.venv/bin/python -m src.run_pipeline --fetch

# 4. Run tests
# Windows:
.venv\Scripts\python -m pytest tests/ -v
# macOS/Linux:
.venv/bin/python -m pytest tests/ -v

# 5. Regenerate figures
# Windows:
.venv\Scripts\python scripts/make_figures.py
# macOS/Linux:
.venv/bin/python scripts/make_figures.py
```

**Requirements:** Python 3.10+ (tested on 3.13)

---

## 📁 Repository Structure

```
snowpack-hydro-prices/
├── README.md
├── requirements.txt
├── .env.example                    # Template for API keys
├── data/
│   ├── raw/                        # Cached source data (regenerable)
│   └── processed/
│       └── panel.csv               # Yearly analysis panel
├── src/
│   ├── fetch_cdec.py               # CDEC snow courses (1980+)
│   ├── fetch_caiso_oasis.py        # CAISO LMP + fuel mix
│   ├── fetch_eia.py                # EIA-930 CISO hydro
│   ├── features.py                 # Panel builder, snowpack index
│   ├── models.py                   # Baselines, walk-forward, mediation
│   └── run_pipeline.py             # End-to-end orchestration
├── notebooks/                      # Executed analysis notebooks
├── scripts/
│   ├── make_notebooks.py           # Notebook generator
│   └── make_figures.py             # Figure generator
├── tests/                          # 12 unit tests
└── results/
    ├── figures/                    # 8 publication-quality figures
    ├── walkforward_rmse.csv
    └── results.json
```

---

## 🔮 Extension Roadmap

| Priority | Extension | Impact |
|:---------|:----------|:-------|
| **P0** | Unlock 2016+ CAISO prices | Enables full walk-forward + mediation |
| **P1** | Add temperature, demand, gas-price controls | Addresses confounding |
| **P2** | Monthly (not annual) mediation | More statistical power |
| **P3** | Bayesian mediation with credible intervals | Rigorous small-sample inference |

---

## 📚 References

- Baron, R. M., & Kenny, D. A. (1986). The moderator-mediator variable distinction. *JPSP*, 51(6), 1173-1182.
- Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. *JBES*, 13(3), 253-263.
- Newey, W. K., & West, K. D. (1987). A simple, positive semi-definite HAC covariance matrix. *Econometrica*, 55(3), 703-708.

---

<div align="center">

**Built with ❤️ by [namit1333](https://github.com/namit1333)**

*This project investigates — but does not conclusively establish — a causal link between snowpack and electricity prices. The snowpack → hydro relationship is statistically established; the downstream price effect remains an open question.*

</div>
