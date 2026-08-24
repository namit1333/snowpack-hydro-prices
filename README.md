<div align="center">

# Snowpack → Hydropower → Electricity Prices

**Does California's winter snowpack predict summer electricity-market behavior?**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![CI](https://github.com/namit1333/snowpack-hydro-prices/actions/workflows/tests.yml/badge.svg)](https://github.com/namit1333/snowpack-hydro-prices/actions/workflows/tests.yml)

Author: [namit1333](https://github.com/namit1333)

</div>

---

A reproducible causal-chain study: does snowpack predict summer electricity
price behavior through hydroelectric generation? End-to-end ETL → feature
engineering → forecasting → statistical inference, from public data, cached,
unit-tested, and run in CI.

Coverage: **46 years** of snowpack (CDEC, 1980–2025) · **8 years** of hydro
generation (CAISO fuel mix + EIA-930 cross-check) · **6 years** of price
history (2016–18 + 2023–25, **two measurement regimes** — see below).

---

## Key finding

> **The physical mechanism is strongly supported; the market effect is
> inconclusive with current data.**
>
> Wet winters reliably produce more summer hydro: **+24.9 GWh per +1
> percentage point of April 1 snowpack** (95% CI [5.0, 44.8], p = 0.022,
> n = 8), stable under leave-one-out, confirmed by an independent EIA-930
> series (+27.0 GWh/pp). The tradeable downstream signal — snowpack →
> electricity prices — is **not established**: with 6 years of price data
> across two measurement regimes, no snowpack-augmented model beats a
> persistence baseline out-of-sample.

**Important limitation:** the price window is 6 years (2016–18 + 2023–25, with
a 2019–22 gap in public records) from **two different measurement regimes**.
Walk-forward forecasts have 3 strictly out-of-sample years (2023–25), which is
too few for conclusive price-side inference. Everything on the price side is
labeled exploratory; the hydro-side result is the robust finding.

---

## Two price measurement regimes

The 6-year price window is **not** a single homogeneous series:

- **2016–2018:** one CAISO node (Bayshore, `BAYSHOR2_1_N001`) from an archived
  third-party mirror of OASIS data.
- **2023–2025:** three CAISO trading hubs (NP15, SP15, ZP26) fetched live.

Measured across the three hubs in 2023–25, cross-hub dispersion of summer
volatility is ~10–17% of the mean — an upper bound on the regime difference the
single-node data introduces. Year-to-year volatility swings (5 → 28 $/MWh) are
several times larger, so the volatility comparison is meaningful, but **all
cross-regime model comparisons are treated as exploratory**. Full detail in
[DATA_PROVENANCE.md](DATA_PROVENANCE.md).

---

## Results

### 1. Snowpack → hydro (the supported leg)

![hydro_vs_snowpack](results/figures/hydro_vs_snowpack.png)

| Data source | Slope (GWh/pp) | 95% CI (t, n−2) | p | R² | r | n |
|---|---|---|---|---|---|---|
| **CAISO fuel mix** | **+24.9** | **[5.0, 44.8]** | **0.022** | 0.61 | 0.78 | 8 |
| EIA-930 (independent) | +27.0 | [-0.4, 54.3] | 0.052 | 0.65 | 0.81 | 6 |

A 10-pp rise in April 1 snowpack ⇒ ≈ +250 GWh of summer hydro — roughly 3% of
a typical CAISO summer. **Leave-one-out robustness:** dropping any single year
keeps the slope positive in all 8 fits (+19.8 to +40.8 GWh/pp, all p < 0.07),
so the result is not driven by one year. CIs use the t distribution at n − 2
degrees of freedom (see [METHODS.md](METHODS.md)).

### 2. The price question (exploratory)

![price_panel](results/figures/price_panel_2023_2025.png)

![correlation_heatmap](results/figures/correlation_heatmap.png)

Walk-forward, strictly out-of-sample, **3 held-out years (2023, 2024, 2025)**:

| Model | RMSE ($/MWh) | n_OOS | Note |
|---|---|---|---|
| baseline_naive | 8.60 | 3 | Persistence |
| baseline_arima | 8.60 | 3 | Random walk + drift |
| augmented_arimax | 8.60 | 3 | Snowpack exog. |
| augmented_ols | 9.14 | 3 | Snowpack regressor |
| baseline_mean3 | 12.08 | 3 | Trailing 3-yr mean |

![walkforward_forecasts](results/figures/walkforward_forecasts.png)

**No model beats persistence.** The wettest year in the window (2023, snowpack
≈ 236%) had the *highest* volatility — opposite to the hypothesis. The
Diebold–Mariano statistic is reported for completeness only: with three
out-of-sample years it is far too weak for predictive-accuracy inference (the
n = 3 OOS sample is insufficient; DM is descriptive, not evidence).

**Note on multiple comparisons:** the near-threshold p-values (0.022, 0.052,
0.052) come from a small set of *pre-specified* tests, not from searching many
models. The two 0.052 values were independently recomputed and are distinct
computations (an OLS t-test on n = 6 vs. a Newey–West HAC DM test on n = 3)
that happen to round to the same figure. At n ≤ 8 these are exploratory
evidence, not confirmatory.

**Failed hypothesis, stated plainly:** snowpack-augmented price forecasting did
not outperform persistence out-of-sample, and conditioning on observed
temperature/demand (a what-if robustness check, kept out of the leaderboard)
did not rescue it. The physical snowpack → hydro relationship does not
automatically translate into a short-horizon price-volatility signal with the
available data. That is the finding.

### 3. Mediation — exploratory, not inference

Baron–Kenny mediation is implemented and kept in the repo
(`notebooks/02_mediation_analysis.ipynb`) to demonstrate the method and the
decomposition, but it is **intentionally not used for inference**: with only
3–6 overlapping price years the Sobel test is uninformative (p ≈ 1.00), so no
point estimate is reported. It will be re-run at full power when the price
archive is complete.

### 4. Monthly panel

`build_monthly_panel()` produces one row per (year, month) of Jun–Sep — **24
observations** from the 6 price years vs. 6 annual — the natural next step for
the price legs as history grows.

---

## Methodology

Why April 1? Why the summer window? Why volatility? Why persistence and
ARIMA baselines? Why walk-forward over random splits? Why Diebold–Mariano over
RMSE? Why leave-one-out over k-fold at n = 8? Why small-sample t-intervals?
Why these controls, and why the controls model is kept out of the forecast
leaderboard?

Each choice is explained in the author's own words in **[METHODS.md](METHODS.md)**.

---

## Data & provenance

Every dataset, its original source, retrieval path, coverage, and
transformation is documented in **[DATA_PROVENANCE.md](DATA_PROVENANCE.md)** —
including the single-node vs. three-hub price regimes above.

| Variable | Description | Units | Source |
|---|---|---|---|
| `snowpack_pct` | April 1 SWC, % of 1991–2020 normal, median of 259 courses | % | CDEC |
| `hydro_gwh` | Summer hydro generation (Large + Small) | GWh | CAISO fuel mix |
| `hydro_gwh_eia` | Same, independent source | GWh | EIA-930 |
| `price_mean` | Mean summer day-ahead LMP | $/MWh | CAISO OASIS |
| `price_vol` | Volatility: std of daily-mean LMP | $/MWh | CAISO OASIS |
| `price_peak` | Max summer daily-mean LMP | $/MWh | CAISO OASIS |
| `demand_mean_mw` | Mean summer demand | MW | EIA-930 (key) |
| `temp_mean_c` | Mean summer daily-max temperature | °C | Open-Meteo |
| `heat_days_38c` | Days ≥ 38 °C (heat waves) | days | Open-Meteo |
| `gas_mean` | Mean summer Henry Hub price | $/MMBtu | EIA (key) |

Known gaps — flagged, never imputed: 2019–22 prices (OASIS paywall), 2020 EIA
hourly hydro (893/2,928 h), 2015 snowpack (drought → index 0). Demand/gas
columns are empty without `EIA_API_KEY`.

---

## Architecture

![pipeline_flowchart](results/figures/pipeline_flowchart.png)

| Stage | Module |
|---|---|
| Data acquisition | `fetch_cdec.py`, `fetch_caiso_oasis.py`, `fetch_eia.py`, `fetch_controls.py` |
| Feature engineering | `features.py` (yearly + monthly panels) |
| Modeling | `models.py` (baselines, walk-forward, DM, mediation, LOO) |
| Orchestration | `run_pipeline.py` |
| Validation | `tests/` (20 tests) + GitHub Actions (Python 3.10–3.12) |

---

## Reproduce

```bash
git clone https://github.com/namit1333/snowpack-hydro-prices.git
cd snowpack-hydro-prices
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt      # Windows
# .venv/bin/pip install -r requirements.txt        # macOS/Linux

# Sample panel: no downloads, no keys
.venv/Scripts/python scripts/make_sample_data.py

# Full pipeline (real data, ~5-10 min; EIA key optional for hydro cross-check + demand/gas)
cp .env.example .env   # add EIA_API_KEY
.venv/Scripts/python -m src.run_pipeline --fetch

# Tests + figures + notebooks
.venv/Scripts/python -m pytest tests/ -v
.venv/Scripts/python scripts/make_figures.py
.venv/Scripts/python scripts/make_notebooks.py
```

`requirements.txt` declares **minimum versions** for human readability;
`requirements-lock.txt` is the exact frozen environment used for development.
Requires Python 3.10+.

---

## Repository structure

```
snowpack-hydro-prices/
├── README.md  METHODS.md  DATA_PROVENANCE.md
├── requirements.txt  requirements-lock.txt  .env.example
├── .github/workflows/tests.yml        # CI: pytest on push/PR
├── data/
│   ├── raw/                           # cached source data (regenerable)
│   ├── processed/panel.csv            # yearly panel
│   └── sample/sample_panel.csv        # demo data (no fetch)
├── src/
│   ├── fetch_cdec.py                  # CDEC snow (1980+)
│   ├── fetch_caiso_oasis.py           # CAISO LMP + fuel mix + 2016-18 archive
│   ├── fetch_eia.py                   # EIA-930 hydro
│   ├── fetch_controls.py              # demand, temperature, gas
│   ├── features.py                    # yearly + monthly panels
│   ├── models.py                      # baselines, walk-forward, DM, mediation, LOO
│   └── run_pipeline.py                # end-to-end
├── notebooks/                         # executed analysis notebooks
├── scripts/                           # figure/notebook/sample generators
├── tests/                             # 20 unit tests
└── results/                           # figures, RMSE tables, results.json
```

---

## Extension roadmap

| Priority | Extension |
|---|---|
| P0 | Fill the 2019–2022 price gap (continuous 10-yr window) |
| P1 | Demand + gas controls into the panel (EIA key) |
| P2 | Monthly mediation on the 24-observation panel |
| P3 | Bayesian mediation with credible intervals |

---

## References

- Baron, R. M., & Kenny, D. A. (1986). The moderator-mediator variable
  distinction in social psychological research. *JPSP*, 51(6), 1173-1182.
- Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy.
  *JBES*, 13(3), 253-263.
- Newey, W. K., & West, K. D. (1987). A simple, positive semi-definite,
  heteroskedasticity and autocorrelation consistent covariance matrix.
  *Econometrica*, 55(3), 703-708.

---

*This project investigates — but does not conclusively establish — a causal
link between snowpack and electricity prices. The snowpack → hydro relationship
is strongly supported; the downstream price effect remains an open question
with six years of price data.*
