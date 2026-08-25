"""Small-sample inference tools: permutation tests, bootstrap CIs, power analysis.

At n = 6-8, asymptotic approximations (t CIs, normal p-values) are fragile.
This module adds distribution-free and simulation-based cross-checks:

  permutation_test   : shuffle the snowpack-year pairing many times; how often
                       does chance produce a slope at least as large as the
                       observed one? Makes no normality assumption.
  bootstrap_ci       : resample years with replacement; percentile CI for the
                       slope. Cross-checks the t-based CI.
  required_n         : simulation-based power analysis -- given the observed
                       effect size and residual noise, how many years would a
                       test with 80% power need? Reframes "not enough data"
                       as a quantified, falsifiable claim.

All randomness is seeded so every result in results/ is exactly reproducible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.regression.linear_model import OLS


def _ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Slope of the OLS fit y = a + b x (returns np.nan if degenerate)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if len(x) < 3 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return np.nan
    return float(OLS(y, np.column_stack([np.ones(len(x)), x])).fit().params[1])


# ---------------------------------------------------------------- permutation
def permutation_test(x, y, n_perm: int = 10_000, seed: int = 42) -> dict:
    """Exact-test approximation: permute the x-year pairing, refit, compare.

    H0: the pairing between snowpack and the outcome is arbitrary (no
    relationship). We shuffle x against y `n_perm` times, record each permuted
    slope, and report the two-sided probability of seeing a slope as extreme
    as the observed one. No distributional assumption beyond exchangeability.

    Returns {observed_slope, perm_p, null_mean, null_sd, n, n_perm}.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    observed = _ols_slope(x, y)
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for i in range(n_perm):
        null[i] = _ols_slope(x, rng.permutation(y))
    null = null[~np.isnan(null)]
    if np.isnan(observed) or len(null) == 0:
        return {"observed_slope": observed, "perm_p": np.nan, "n": len(x),
                "n_perm": n_perm}
    # add-one correction so p is never exactly 0
    extreme = int(np.sum(np.abs(null) >= abs(observed)))
    p = (extreme + 1) / (len(null) + 1)
    return {
        "observed_slope": float(observed),
        "perm_p": float(p),
        "null_mean": float(null.mean()),
        "null_sd": float(null.std(ddof=1)),
        "null": null.tolist(),
        "n": int(len(x)),
        "n_perm": int(n_perm),
    }


# ---------------------------------------------------------------- bootstrap
def bootstrap_ci(x, y, n_boot: int = 10_000, seed: int = 42,
                 level: float = 0.95) -> dict:
    """Percentile bootstrap CI for the OLS slope (resample years in pairs).

    Cross-check for the t-based CI: if the bootstrap and t intervals broadly
    agree, the parametric assumption is not doing the work.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)
    observed = _ols_slope(x, y)
    rng = np.random.default_rng(seed)
    slopes = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        slopes[i] = _ols_slope(x[idx], y[idx])
    slopes = slopes[~np.isnan(slopes)]
    lo, hi = np.percentile(slopes, [(1 - level) / 2 * 100,
                                    (1 + level) / 2 * 100])
    return {
        "observed_slope": float(observed),
        "boot_lo": float(lo),
        "boot_hi": float(hi),
        "level": level,
        "n": int(n),
        "n_boot": int(n_boot),
        "boot": slopes.tolist(),
    }


# ---------------------------------------------------------------- power
def power_curve(effect_slope: float, resid_sd: float, x_pool: np.ndarray,
                alpha: float = 0.05, target_power: float = 0.80,
                n_grid=None, n_sims: int = 2_000, seed: int = 42) -> dict:
    """Simulation-based power analysis for the slope t-test.

    For each candidate sample size n: simulate `n_sims` fake experiments by
    drawing x from the empirical distribution of the observed predictor
    (years), generating y = a + effect_slope * x + N(0, resid_sd), fitting the
    slope, and recording whether its p-value clears alpha. Power = fraction
    detected. Returns the curve and the smallest n reaching `target_power`.

    This turns "we don't have enough years" into a quantified claim: "with an
    effect of this size and this noise, 80% power needs ~N years."
    """
    x_pool = np.asarray([v for v in np.asarray(x_pool, float) if np.isfinite(v)])
    if len(x_pool) < 3 or not np.isfinite(effect_slope) or resid_sd <= 0:
        return {"required_n": np.nan, "curve": []}
    if n_grid is None:
        n_grid = list(range(4, 61, 2))
    rng = np.random.default_rng(seed)
    x_mean = float(x_pool.mean())
    curve = []
    required = np.nan
    for n in n_grid:
        detected = 0
        for _ in range(n_sims):
            x = rng.choice(x_pool, size=n, replace=True)
            # jitter resampled x slightly so the design is not degenerate
            x = x + rng.normal(0, 1e-9 * max(1.0, abs(x_mean)), n)
            y = x_mean + effect_slope * (x - x_mean) + rng.normal(0, resid_sd, n)
            X = np.column_stack([np.ones(n), x])
            fit = OLS(y, X).fit()
            if fit.pvalues[1] < alpha:
                detected += 1
        power = detected / n_sims
        curve.append({"n": int(n), "power": float(power)})
        if np.isnan(required) and power >= target_power:
            required = int(n)
    return {
        "effect_slope": float(effect_slope),
        "resid_sd": float(resid_sd),
        "alpha": alpha,
        "target_power": target_power,
        "required_n": required,
        "curve": curve,
        "n_sims": int(n_sims),
    }


# ---------------------------------------------------------------- link test
def hydro_price_link(panel: pd.DataFrame, target: str = "price_vol",
                     mediator: str = "hydro_gwh") -> dict:
    """Second link of the chain tested directly: hydro output -> price outcome.

    The mediation framework entangles the two legs; this regresses the price
    outcome on hydro generation alone, giving the snowpack-independent test of
    whether the physical mechanism (more hydro) moves the market variable at
    all. With 4-6 overlapping years this is exploratory by construction.
    """
    df = panel[[target, mediator]].dropna()
    if len(df) < 4:
        return {"n": len(df), "error": "not enough overlapping years"}
    X = np.column_stack([np.ones(len(df)), df[mediator].values])
    fit = OLS(df[target].values, X).fit()
    slope, se = float(fit.params[1]), float(fit.bse[1])
    t_crit = float(stats.t.ppf(0.975, df=fit.df_resid))
    return {
        "n": len(df),
        "slope": slope,
        "ci_lo": slope - t_crit * se,
        "ci_hi": slope + t_crit * se,
        "p_value": float(fit.pvalues[1]),
        "r2": float(fit.rsquared),
    }
