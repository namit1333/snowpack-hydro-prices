<div align="center">

# Snowpack → Hydropower → Electricity Prices

### Testing a Proposed Relationship

**Can California's winter snowpack help predict summer hydropower generation and electricity prices?**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![CI](https://github.com/namit1333/snowpack-hydro-prices/actions/workflows/tests.yml/badge.svg)](https://github.com/namit1333/snowpack-hydro-prices/actions/workflows/tests.yml)

Author: [namit1333](https://github.com/namit1333)

</div>

---

I built a Python data pipeline to investigate this question using public data
from CDEC (California's snow survey agency), CAISO (the state's electricity
grid operator), and EIA (the U.S. Energy Information Administration).

I first tested whether years with more snowpack produce more hydropower during
the summer. I then tested whether that relationship could also help predict
electricity-price volatility.

**The first relationship was strong:** years with more snowpack consistently
produced more summer hydropower, and that result held up under every check I
threw at it.

**The second did not work:** I could not show that snowpack improved
electricity-price forecasts. A simple model that predicts prices will stay
similar to last year beat every snowpack-based model I built on data the
models had never seen.

Because I only had 6–8 yearly observations, I focused on checking whether my
results were reliable instead of using complicated machine-learning models.

Coverage: **46 years** of snowpack (1980–2025) · **8 years** of hydro
generation · **6 years** of price history (2016–18 + 2023–25, from **two
different data sources** — see below).

**Design principle:** with only n = 8 years, my priority was making sure the
results were real — not adding model complexity. Every test in this repo
exists to try to *disprove* my own headline finding; the checks that failed
are reported next to the ones that held.

**If you only have 2 minutes:**

1. Read the key finding below
2. Look at [the one-figure summary](#the-project-in-one-figure)
3. Skim [METHODS.md](METHODS.md) — why I made each choice
4. Run it yourself: `python scripts/make_sample_data.py && python scripts/check_reproducibility.py`

## The project in one figure

![research_summary](results/figures/research_summary.png)

---

## Hypotheses and how I tested them

I decided on these tests *before* looking at the results, so this is a genuine
hypothesis test — not a search through many models until something looks
significant.

| # | Hypothesis | Test | Verdict |
|---|---|---|---|
| **H1** | More April 1 snowpack → more summer hydro generation | Regression, plus four independent reliability checks (below) | **Supported** (every check agrees) |
| **H2** | Snowpack affects summer price volatility *through* hydro generation | One-year-at-a-time forecasting vs. a simple baseline + a power analysis | **Inconclusive** (no improvement found; too few price years to settle it) |

**Key finding:** snowpack → hydro is a **strong, reliable relationship**
(positive under random shuffling, resampling, dropping each year, and 12
different model setups). Snowpack → price shows **no detectable improvement on
data the models never saw**. That asymmetry is the result.

**Main limitation:** I only have 6 years of prices (with a 2019–22 gap in
public records, from two different data sources). My power analysis (below)
says I would need **~12 years** to reliably detect even an effect as strong as
the hydro link. So "no result" on prices means "not enough data yet," not
"there is definitely no effect."

---

## Two price data sources

The 6-year price record is **not** one consistent dataset:

- **2016–2018:** prices from a single pricing location (Bayshore), from a
  public archive of CAISO data.
- **2023–2025:** prices averaged across three CAISO trading hubs, fetched
  live.

I measured how much the three hubs disagree with each other in 2023–25
(~10–17% of the average volatility). Year-to-year volatility swings (5 → 28
$/MWh) are several times larger than that disagreement, so comparing
volatility across the two eras is meaningful — but I treat every model
comparison that spans both eras as exploratory. Full detail in
[DATA_PROVENANCE.md](DATA_PROVENANCE.md).

---

## Results

### 1. Snowpack → hydro: a strong, reliable relationship

![hydro_vs_snowpack](results/figures/hydro_vs_snowpack.png)

| Data source | Slope (GWh/pp) | 95% CI (t, n−2) | p | R² | n |
|---|---|---|---|---|---|
| **CAISO fuel mix** | **+24.9** | **[5.0, 44.8]** | **0.022** | 0.61 | 8 |
| EIA-930 (independent) | +27.0 | [-0.4, 54.3] | 0.052 | 0.65 | 6 |

In words: each extra percentage point of April 1 snowpack comes with about
+25 GWh of summer hydro generation, so a wet winter at +10 pp means roughly
+250 GWh — about 3% of a typical CAISO summer. The two rows are the same
relationship measured by two independent sources, and they agree. Then I
checked the result four separate ways:

**Random shuffling (permutation test).** I shuffled the snowpack years against
the hydro years 10,000 times and re-ran the regression each time — if the
relationship were coincidence, random pairings would often look this strong.
They almost never did: **shuffled p = 0.018**, which matches the standard
regression p-value of 0.022.

![permutation_null](results/figures/permutation_null.png)

**Resampling (bootstrap).** I re-drew the 8 years at random, with replacement,
10,000 times and re-fit the slope each time. The middle 95% of those slopes
was **[14.2, 51.8]**, close to the regression-based interval [5.0, 44.8] — so
the standard formula isn't producing a misleading range.

**Dropping one year at a time (leave-one-out).** With only 8 years, one weird
year could drive everything. I re-fit the regression 8 times, leaving out a
different year each time: the slope stayed positive in **all 8 fits** (+19.8
to +40.8, all p < 0.07). No single year is carrying the result.

**Different model setups (sensitivity).** Does the answer depend on my
arbitrary choices? I re-ran the regression with four different snowpack
measurement dates (Mar 1, Apr 1, May 1, Jun 1 — the archive only measures on
the 1st) crossed with four definitions of "summer":

![sensitivity_grid](results/figures/sensitivity_grid.png)

**12 of the 16 setups had enough data to test; the slope was positive in all
12 (+15.0 to +25.5 GWh/pp), and 8 of 12 were individually significant.** The
relationship stayed positive across every setup I could test. (June 1
snowpack lacks enough years to fit; May–Sep results equal Jun–Sep because the
cached grid data starts in June.)

### 2. Hydro → price, tested on its own

The full chain has two links, so I also tested the second link by itself:
does hydro generation *itself* relate to price volatility in the years where
both exist?

| Slope ($/MWh per GWh) | 95% CI | p | R² | n |
|---|---|---|---|---|
| +0.007 | [-0.016, 0.030] | 0.314 | 0.47 | 4 |

Positive on average, but indistinguishable from zero with only 4 overlapping
years. I report it for completeness; I draw no conclusion from it.

### 3. The price question — an honest negative result

![price_panel](results/figures/price_panel_2023_2025.png)

![correlation_heatmap](results/figures/correlation_heatmap.png)

**How I tested fairly:** each year is predicted using only data from *before*
it ("out-of-sample" — data the model has never seen). An automated test
(`tests/test_no_future_leakage.py`) verifies this: it deliberately corrupts
all future years' values and checks that earlier predictions don't change at
all.

```
TRAIN (past years only)   PREDICT
2016-2018             →   2023
2016-2023             →   2024
2016-2024             →   2025
```

The benchmark philosophy: my snowpack model has to beat the simple baselines
on unseen data to be considered useful. A fancy model that loses to a simple
one has demonstrated nothing.

| Model | Purpose |
|---|---|
| Persistence (naive) | Predicts next year ≈ this year — the bar to clear |
| Trailing 3-yr mean | Simple average benchmark |
| ARIMA(0,1,0)+drift | Standard time-series benchmark |
| OLS + snowpack | **My hypothesis-driven model** |
| ARIMAX + snowpack | Time-series variant with snowpack input |

| Model | RMSE | MAE | Dir. acc. | Note |
|---|---|---|---|---|
| **baseline_naive** | **8.59** | **7.38** | 0.00 | Persistence — predicts "no change" |
| augmented_ols | 9.12 | 7.79 | 0.50 | Snowpack model |
| baseline_mean3 | 12.09 | 11.84 | 0.50 | Trailing 3-yr mean |
| baseline_arima | 13.93 | 12.77 | 0.00 | Random walk + drift |
| augmented_arimax | 16.09 | 15.05 | 0.00 | ARIMAX w/ snowpack |

(RMSE/MAE = average forecast error, lower is better; Dir. acc. = how often it
predicts the *direction* of the change correctly, 0.5 is a coin flip.)

![walkforward_forecasts](results/figures/walkforward_forecasts.png)

> **Result: snowpack did not improve electricity-price volatility forecasts.**
> Across 3 held-out years, my snowpack model lost to a baseline that just
> predicts "next year ≈ this year." **I kept this negative result instead of
> hiding it** — deleting models that perform badly would bias my own research.

The wettest year in the window (2023, snowpack ≈ 236%) had the *highest*
volatility — the opposite of what the hypothesis predicted. The
Diebold–Mariano test (a formal comparison of two models' forecast errors) is
reported in the results file but is meaningless with only 3 predictions, so I
don't use it to claim anything.

**Why the ARIMA rows look the way they do:** a statistics-library quirk
(versions ≥ 0.14 reject this particular ARIMA setup) meant an earlier version
of my pipeline silently caught the error and quietly substituted the
persistence answer — which made ARIMA *look* identical to the naive baseline.
I removed the silent fallback; both ARIMA models are now implemented directly
and fail loudly if something goes wrong. Their true (worse) performance is
the table above.

**Is the snowpack coefficient in the price model stable?** No — and saying so
is part of the result:

| Training years used | Coefficient ($/MWh per pp) | p |
|---|---|---|
| 2016–2018 | −0.016 | 0.94 |
| 2016–2023 | +0.053 | 0.60 |
| 2016–2024 | +0.056 | 0.47 |
| 2016–2025 | +0.070 | 0.30 |

The sign flips with the smallest sample and is never distinguishable from
zero — exactly what the forecast table would predict.

**Why did adding temperature make things worse?** A side experiment that also
used the *observed* summer temperature (a what-if check, kept out of the main
table because in reality you can't know next summer's temperature) had RMSE
41.0 vs. 5.6 for snowpack alone on the same folds. I dug into why rather than
just reporting the number: the inputs weren't strongly related to each other
(condition number 2.0), but with 3 fitted values against only 4 training rows,
the extra input mostly added noise. Removing temperature *improved* RMSE from
41.0 back to 5.6. With this few years, extra model settings hurt — which is
also why I didn't add a bigger model.

**A note on p-values near 0.05:** the p-values 0.022, 0.052, and 0.052 come
from a small set of tests I chose in advance — not from trying many models
until one worked. The two 0.052 values are genuinely different computations
that happen to round the same (I re-derived both independently). At n ≤ 8,
**I do not treat p = 0.052 as evidence of anything** — at this sample size it
is indistinguishable from noise.

**The failed hypothesis, stated plainly:** snowpack did not help predict
price volatility out-of-sample. The real snowpack → hydro relationship does
not automatically become a usable price signal with the data available. That
is the finding.

### 4. Power analysis — how much data would this take?

Instead of just saying "I need more data," I estimated how much more. Using
the noise in the price series, I simulated how often I would detect an effect
as strong as the validated hydro link, at each sample size:

| Years of data | 4 | 6 | 8 | **10–12** | 14 |
|---|---|---|---|---|---|
| Chance of detecting it | 0.21 | 0.48 | 0.69 | **0.80–0.88** | 0.93 |

**About 12 years.** I have 6. Filling the 2019–22 gap is the single most
valuable next step — this number says exactly why.

### 5. Mediation analysis — kept, but not used for conclusions

Mediation analysis asks whether the snowpack effect on prices *works through*
hydro generation. I implemented it (`notebooks/02_mediation_analysis.ipynb`)
to show the decomposition, but with only 3–6 overlapping price years its
significance test is uninformative (p ≈ 1.00), so I report no number from it.

### 6. Monthly data — a higher-frequency extension

`build_monthly_panel()` builds one row per summer *month* (Jun–Sep) — 24 rows
instead of 6 annual ones. **This is 4× the number of rows, not 4× the
statistical power**: months within the same summer are related to each other,
so the effective sample grows much less than the row count suggests. Any
inference on it would need to account for that (clustered standard errors).
It's the natural next step as price history grows.

---

## Validation — every check in one place

| Check | What it protects against | Result |
|---|---|---|
| Leave-one-out (n = 8) | One weird year driving the result | Positive slope in **all 8 fits** (+19.8 to +40.8, all p < 0.07) |
| Independent EIA-930 source | Errors in one data source | Same relationship (+27.0 vs +24.9 GWh/pp) |
| Permutation test (10,000 shuffles) | Coincidence at n = 8 | shuffled p = 0.018 ≈ standard 0.022 |
| Bootstrap (10,000 resamples) | Standard formula being misleading | [14.2, 51.8] ≈ [5.0, 44.8] |
| 12 model setups | My arbitrary choices driving the result | **12/12 slopes positive** (+15.0 to +25.5) |
| Forecasting vs. persistence | Fooling myself on data I trained on | **No model beats persistence** |
| Diebold–Mariano | Chance differences in forecast errors | Inconclusive at 3 predictions — not used |
| Future-leakage test (automated) | Future data sneaking into training | Verified: corrupting future years leaves earlier predictions unchanged |
| Reproducibility check (CI) | Results that can't be regenerated | Analysis run twice per push; outputs must match exactly |

---

## Research decisions

### Why not a bigger machine-learning model?

I only have 6–8 yearly observations. A bigger model would mostly add
overfitting risk, not accuracy. The roadmap is more data, not more machinery.

### Why is persistence the benchmark?

Price volatility is very persistent — this year tends to look like last year.
Any proposed signal has to beat that simple baseline on unseen data to show
it adds anything. None here does.

### Why don't I claim causation?

I studied existing data rather than running an experiment, so I cannot prove
that snowpack *causes* the changes I see. Electricity demand, natural-gas
prices, temperature, transmission constraints, reservoir levels, and other
generation sources could all explain parts of the pattern. Snowpack is set
months before the summer and driven by winter weather, which makes the first
link unusually trustworthy — but "plausibly independent" is not proof.

### Why keep the failed models in the table?

Deleting models that performed badly would bias my own research process — I'd
only be showing you the survivors. The ARIMA/ARIMAX rows that lose to
persistence stay.

---

## Why this is not a causal claim

This project does **not** show that snowpack causes electricity prices.

What it does show: snowpack is strongly associated with later hydro
generation, and that association survives every reliability check I could
design. Snowpack is set months before the summer and driven by winter
weather, so it is mostly independent of summer market conditions — which makes
the first link about as clean as observational data allows. But the price
analysis could be confounded by factors I did not include: demand, gas prices,
heat waves, transmission constraints, reservoir decisions, and other
generation. Demand and temperature are partially in place; gas needs an API
key (roadmap). Proving causation would require the full control set, more
price history, and ideally a design that isolates supply shocks.

---

## What I learned

- **A real physical relationship doesn't guarantee a usable market signal.**
  The snowpack → hydro link survived everything; one step downstream, it
  produced nothing.
- **Simple baselines are hard to beat.** "Next year ≈ this year" is a genuinely
  strong forecast for volatility, and every fancier model lost to it.
- **Small samples make significance fragile.** That's why I ran shuffling,
  resampling, leave-one-out, and 12 model setups before believing one number.
- **Extra model settings hurt when data is tiny.** The temperature experiment
  made things worse; the failure analysis shows exactly why.
- **Checking my own result was worth more than adding complexity.** The
  sensitivity grid changed how much I trust the headline number more than any
  new model would have.
- **Silent error handling can corrupt research.** A swallowed exception once
  made ARIMA look identical to persistence; automated tests now guard against
  that class of bug.

---

## Methodology

Why April 1? Why summer? Why volatility? Why these baselines? Why predict one
year at a time? Why leave-one-out instead of k-fold at n = 8? Why shuffling
and resampling? Why these controls, and why the temperature experiment stays
out of the main table?

Each choice is explained in my own words in **[METHODS.md](METHODS.md)**.

---

## Data & provenance

Every dataset, its original source, how I retrieved it, its coverage, and how
I transformed it is documented in
**[DATA_PROVENANCE.md](DATA_PROVENANCE.md)** — including the two price data
sources above.

| Variable | Description | Units | Source |
|---|---|---|---|
| `snowpack_pct` | April 1 snow water content, % of 1991–2020 normal, median of 259 courses | % | CDEC |
| `hydro_gwh` | Summer hydro generation (Large + Small) | GWh | CAISO fuel mix |
| `hydro_gwh_eia` | Same, independent source | GWh | EIA-930 |
| `price_mean` | Mean summer day-ahead price | $/MWh | CAISO OASIS |
| `price_vol` | Volatility: std of daily prices | $/MWh | CAISO OASIS |
| `price_peak` | Max summer daily price | $/MWh | CAISO OASIS |
| `demand_mean_mw` | Mean summer demand | MW | EIA-930 (key) |
| `temp_mean_c` | Mean summer daily-max temperature | °C | Open-Meteo |
| `heat_days_38c` | Days ≥ 38 °C (heat waves) | days | Open-Meteo |
| `gas_mean` | Mean summer Henry Hub gas price | $/MMBtu | EIA (key) |

Known gaps — flagged, never filled in with guesses: 2019–22 prices (OASIS
paywall), 2020 EIA hourly hydro (893/2,928 h), 2015 snowpack (drought → index
0). Demand/gas columns are empty without `EIA_API_KEY`.

---

## Architecture

![pipeline_flowchart](results/figures/pipeline_flowchart.png)

| Stage | Module |
|---|---|
| Data download | `fetch_cdec.py`, `fetch_caiso_oasis.py`, `fetch_eia.py`, `fetch_controls.py` |
| Data preparation | `features.py` (yearly + monthly panels, configurable windows) |
| Modeling | `models.py` (baselines, one-year-at-a-time forecasting, DM, mediation, stability, failure analysis) |
| Reliability checks | `inference.py` (shuffling, resampling, power analysis, link test) |
| Alternative setups | `sensitivity.py` (the 4×4 grid) |
| Orchestration | `run_pipeline.py` |
| Testing | `tests/` (31 tests incl. future-leakage) + CI: pytest + reproducibility check |

---

## Reproduce

```bash
git clone https://github.com/namit1333/snowpack-hydro-prices.git
cd snowpack-hydro-prices
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt      # Windows
# .venv/bin/pip install -r requirements.txt        # macOS/Linux

# Sample data: no downloads, no keys
.venv/Scripts/python scripts/make_sample_data.py

# Full pipeline (real data, ~5-10 min; EIA key optional)
cp .env.example .env   # add EIA_API_KEY
.venv/Scripts/python -m src.run_pipeline --fetch

# Tests + figures + notebooks
.venv/Scripts/python -m pytest tests/ -v
.venv/Scripts/python scripts/make_figures.py
.venv/Scripts/python scripts/make_notebooks.py

# Reproducibility verification (what CI runs)
.venv/Scripts/python scripts/check_reproducibility.py
```

`requirements.txt` lists **minimum versions** for readability;
`requirements-lock.txt` is the exact environment I developed with. Requires
Python 3.10+. CI runs the tests on 3.10–3.12 **and** re-runs the whole
analysis twice to confirm the outputs are identical.

---

## Repository structure

```
snowpack-hydro-prices/
├── README.md  METHODS.md  DATA_PROVENANCE.md
├── requirements.txt  requirements-lock.txt  .env.example
├── .github/workflows/tests.yml        # CI: pytest 3.10-3.12 + reproducibility
├── data/
│   ├── raw/                           # cached source data (regenerable)
│   ├── processed/panel.csv            # yearly panel
│   └── sample/sample_panel.csv        # demo data (no download needed)
├── src/
│   ├── fetch_cdec.py                  # CDEC snow (1980+)
│   ├── fetch_caiso_oasis.py           # CAISO prices + fuel mix + 2016-18 archive
│   ├── fetch_eia.py                   # EIA-930 hydro
│   ├── fetch_controls.py              # demand, temperature, gas
│   ├── features.py                    # yearly + monthly panels
│   ├── models.py                      # baselines, forecasting, mediation, checks
│   ├── inference.py                   # shuffling, resampling, power, link test
│   ├── sensitivity.py                 # the 4x4 setup grid
│   └── run_pipeline.py                # runs everything end to end
├── notebooks/                         # executed analysis notebooks
├── scripts/                           # figure/notebook/sample/reproducibility
├── tests/                             # 31 unit tests incl. future-leakage
└── results/                           # figures, tables, results.json
```

---

## Extension roadmap

| Priority | Extension |
|---|---|
| P0 | Fill the 2019–2022 price gap → reaches the ~12 years my power analysis says I need |
| P1 | Add demand + gas data to the panel (needs an EIA key) |
| P2 | Monthly evaluation on the 24-row panel, accounting for months within a summer being related |
| P3 | Re-evaluate with more price history — more data, not more machinery |

---

## References

- Baron, R. M., & Kenny, D. A. (1986). The moderator-mediator variable
  distinction in social psychological research. *JPSP*, 51(6), 1173-1182.
- Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy.
  *JBES*, 13(3), 253-263.
- Efron, B. (1979). Bootstrap methods: another look at the jackknife.
  *Annals of Statistics*, 7(1), 1-26.
- Newey, W. K., & West, K. D. (1987). A simple, positive semi-definite,
  heteroskedasticity and autocorrelation consistent covariance matrix.
  *Econometrica*, 55(3), 703-708.

---

*This project tests a proposed relationship — it does not claim to prove one.
The snowpack → hydro relationship is strong and held up under every check I
could design; the price effect remains an open question that my 6-year price
window doesn't have enough data to answer.*
