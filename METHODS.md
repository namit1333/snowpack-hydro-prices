# Methods & Design Choices

This document explains **why** each methodological choice was made, in my own
words. The README shows what the pipeline does; this explains the
reasoning behind it — the part that matters in an interview.

## Why April 1 snowpack?

April 1 is the standard hydrological benchmark in California water management:
by then, essentially all of the seasonal snowpack has accumulated, and the
state's own snow surveys are scheduled around it. It is the natural "state
variable" summarizing the winter that just ended, and it is what water
managers actually use to plan summer reservoir releases. Using it means the
feature is both physically meaningful and directly comparable to how the
industry measures snow.

## Why the summer window (June–September)?

Hydro generation matters most when it can offset summer cooling demand — the
load-critical season when prices are highest and most volatile. Winter hydro is
driven by rain and reservoir releases rather than snowmelt, so restricting to
Jun–Sep isolates the part of the year where the snowpack → hydro → price chain
is supposed to operate.

## Why volatility, not price level?

The hypothesis is about *market behavior* under different hydro conditions.
Price levels are dominated by fuel costs and demand; volatility is where a
supply-side shift (cheap, dispatchable hydro) should show up first — and it is
also the quantity traders actually care about for risk. Using the standard
deviation of daily-mean day-ahead LMP keeps the target clean and comparable
across years.

## Why these baselines?

Forecasting claims mean nothing without strong baselines. Persistence
(ŷₜ = yₜ₋₁) is the hardest baseline to beat in electricity markets, where
prices are heavily autocorrelated; a trailing 3-year mean is the natural naive
alternative; ARIMA(0,1,0) with drift is the standard "random walk with drift"
null model. If an augmented model cannot beat persistence out-of-sample, the
feature has no forecasting value — that is exactly the result we got, and it is
a real finding.

## Why walk-forward (expanding window) rather than a random train/test split?

Price years are autocorrelated and the data are annual; a random split would
leak future information into the training set and misstate out-of-sample
performance. Walk-forward trains only on years strictly before the target year
and expands the window each step — the closest thing to how a model would
actually be used live. Strictness here is the whole point: the predictions in
`results/walkforward_rmse.csv` used no information from the predicted year.

## Why Diebold–Mariano rather than just comparing RMSE?

RMSE differences can be noise — with n = 3 out-of-sample years, one bad year
swings everything. The Diebold–Mariano test compares *squared forecast errors*
between two models on the same observations and accounts for the
autocorrelation in the error differential via Newey–West HAC variance, giving a
proper test of equal predictive accuracy. We deliberately compute it only when
the two models share common out-of-sample years, and we flag it as descriptive
at n = 3 — the test is executable, but with three observations it is far too
weak to be confirmatory.

## Why leave-one-out rather than k-fold?

n = 8 hydro years is far too small for k-fold (folds would be a year or two
each). Leave-one-out is the honest small-sample robustness check: re-fit
dropping each year in turn and see whether the slope and p-value survive every
single deletion. It directly answers "is this result driven by one lucky
year?" — and in our case the slope stayed positive across all eight fits, which
is the strongest claim n = 8 can support.

## Why small-sample t-intervals rather than the normal 1.96?

The 95% CI on the hydro slope is computed with the t distribution at n − 2
degrees of freedom (≈ 2.45 at n = 8), not the asymptotic 1.96. With eight
observations the normal approximation understates uncertainty; the t-interval
[5.0, 44.8] GWh/pp is the honest statement that this is a real but noisy
effect.

## Why these controls?

Prices respond to more than snow. We added summer temperature and heat-wave
days (Open-Meteo), electricity demand (EIA-930), and Henry Hub gas prices
(EIA) as confounders. The controls model is kept **separate from the forecast
leaderboard** because it conditions on *observed* temperature/demand for the
held-out year — it is a what-if robustness check ("given we knew the weather,
does snowpack still matter?"), not a pure forecast. That distinction is
deliberate and is documented in `models.py`.

## Why a permutation test on top of the OLS p-value?

At n = 8 the t-test's p-value leans on a normality assumption the sample
cannot verify. The permutation test makes no distributional assumption: it
shuffles the snowpack-year pairing 10,000 times and asks how often chance
alone produces a slope at least as extreme as the observed one. If the
parametric and permutation p-values agree (they do: 0.022 vs 0.018), the
conclusion is not an artifact of the distributional assumption.

## Why a bootstrap confidence interval?

Same logic, different angle: resample the 8 years with replacement 10,000
times and read the 2.5th/97.5th percentiles of the slope distribution. If the
bootstrap interval broadly agrees with the t-based interval, the t-interval is
not doing special work. They agree (bootstrap [14.2, 51.8] vs t [5.0, 44.8]).

## Why a specification sensitivity grid?

"Does your conclusion depend on one arbitrary modeling choice?" is the first
question a reviewer asks about a single headline regression. The grid re-runs
the snowpack -> hydro regression across every benchmark date the CDEC archive
supports (courses are measured on the 1st, so Mar 1 / Apr 1 / May 1 / Jun 1)
crossed with four definitions of "summer". 12 of 16 specifications are
estimable; all 12 slopes are positive (+15.0 to +25.5 GWh/pp). The headline
number is a property of the data, not of one specification.

## Why test hydro -> price directly?

Mediation entangles the two links of the chain. Regressing price volatility on
hydro generation alone gives the second link its own test, independent of the
first. It is null at n = 4 (p = 0.31) -- consistent with the chain breaking
(not being resolvable) at the market link, not the physical one.

## Why a power analysis?

"We don't have enough data" is weak; "detecting an effect this size at 80%
power would need ~12 years; we have 6" is a falsifiable claim. The power
curve is simulated from the observed predictor distribution and residual
noise. It converts the roadmap's P0 item (fill the 2019-22 price gap) from a
vague aspiration into a quantified requirement.

## Why a leakage test?

"Strictly out-of-sample" is usually a claim; here it is a verified property.
`tests/test_no_future_leakage.py` corrupts every target value from year T
onward and asserts that predictions for years before T are bit-identical. If
any future information ever leaks into a training window, the test fails
before any result is trusted.

## Why mediation, and why it is explicitly exploratory?

Baron–Kenny mediation is the standard way to test whether hydro output *carries*
the snowpack effect on prices. But with only 3–6 overlapping price years the
Sobel test is uninformative (p ≈ 1.00), so the proportion-mediated figure
carries no statistical weight. It is kept in the repo to show the method and
the decomposition — explicitly labeled exploratory — and is not used for any
conclusion. The notebook is `notebooks/02_mediation_analysis.ipynb`.

## The honest bottom line

The physical link is real: wet winters reliably produce more summer hydro
(p ≈ 0.02, slope ≈ +25 GWh per +1 pp, stable under leave-one-out, confirmed by
a second data source). The market link is not established: with six price years
across two measurement regimes, no snowpack-augmented model beats persistence
out-of-sample. That asymmetry — a supported mechanism and a null downstream
result — is the actual finding, and no amount of modeling with the current data
changes it. The binding constraint is price history, not model choice.
