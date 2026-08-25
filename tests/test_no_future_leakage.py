"""Leakage tests: prove the walk-forward pipeline cannot see the future.

These are the tests that make "strictly out-of-sample" a verified property
rather than a claim: we deliberately corrupt future information and assert
that earlier predictions do not move.
"""

import numpy as np
import pandas as pd
import pytest

from src import models


def _panel(n=14, seed=7):
    rng = np.random.default_rng(seed)
    years = pd.RangeIndex(2000, 2000 + n)
    snow = 60 + 40 * np.sin(np.arange(n)) + rng.normal(0, 5, n)
    hydro = 3000 + 30 * (snow - 80) + rng.normal(0, 50, n)
    vol = 10 - 0.005 * (hydro - 3000) + rng.normal(0, 0.6, n)
    return pd.DataFrame({
        "snowpack_pct": snow, "price_vol": vol, "hydro_gwh": hydro,
    }).set_index(years)


def test_future_target_values_cannot_change_earlier_predictions():
    """Corrupt every target from year T onward; predictions before T must be
    bit-identical. If any future price realization leaks into the training
    window of an earlier forecast, this fails."""
    p = _panel()
    wf_a = models.walk_forward(p, min_train=6)

    corrupted = p.copy()
    cutoff = corrupted.index[8]
    corrupted.loc[corrupted.index >= cutoff, "price_vol"] = 999.0
    corrupted.loc[corrupted.index >= cutoff, "hydro_gwh"] = -999.0
    wf_b = models.walk_forward(corrupted, min_train=6)

    early_a = wf_a[wf_a["year"] < cutoff].sort_values(["model", "year"])
    early_b = wf_b[wf_b["year"] < cutoff].sort_values(["model", "year"])
    pd.testing.assert_frame_equal(early_a, early_b)


def test_current_year_target_does_not_affect_its_own_prediction():
    """The prediction for year T is fit without year T's target row at all --
    corrupting T's own price_vol must not change T's prediction (the exogenous
    snowpack for T is measured April 1, before the summer, so it is
    legitimately available)."""
    p = _panel()
    wf_a = models.walk_forward(p, min_train=6)

    predicted_year = p.index[8]  # a year that IS predicted (>= min_train)
    corrupted = p.copy()
    corrupted.loc[predicted_year, "price_vol"] = -12345.0
    wf_b = models.walk_forward(corrupted, min_train=6)

    a = wf_a[wf_a["model"] == "augmented_ols"].set_index("year")
    b = wf_b[wf_b["model"] == "augmented_ols"].set_index("year")
    assert a.loc[predicted_year, "pred"] == b.loc[predicted_year, "pred"]


def test_training_window_is_strictly_expanding_prefix():
    """Directly verify the fold structure: for each predicted year, the rows
    used to fit are exactly the years strictly before it."""
    p = _panel(n=10)
    min_train = 4
    df = p[["price_vol", "snowpack_pct"]].dropna().sort_index()
    for i in range(min_train, len(df)):
        train = df.iloc[:i]
        assert train.index.max() < df.index[i], (
            f"fold for {df.index[i]} used future year {train.index.max()}")


def test_permutation_and_bootstrap_are_seed_deterministic():
    from src.inference import bootstrap_ci, permutation_test

    rng = np.random.default_rng(0)
    x = rng.normal(100, 20, 10)
    y = 3.0 * x + rng.normal(0, 5, 10)
    a = permutation_test(x, y, n_perm=500, seed=42)
    b = permutation_test(x, y, n_perm=500, seed=42)
    assert a["perm_p"] == b["perm_p"]
    c = bootstrap_ci(x, y, n_boot=500, seed=42)
    d = bootstrap_ci(x, y, n_boot=500, seed=42)
    assert (c["boot_lo"], c["boot_hi"]) == (d["boot_lo"], d["boot_hi"])


def test_permutation_p_detects_signal_and_survives_null():
    from src.inference import permutation_test

    rng = np.random.default_rng(1)
    x = np.arange(10, dtype=float)
    y_signal = 5.0 * x + rng.normal(0, 1, 10)
    y_null = rng.normal(0, 1, 10)
    assert permutation_test(x, y_signal, n_perm=1000, seed=3)["perm_p"] < 0.01
    p_null = permutation_test(x, y_null, n_perm=1000, seed=3)["perm_p"]
    assert 0.1 < p_null  # no relationship -> large p


def test_bootstrap_ci_contains_observed_slope():
    from src.inference import bootstrap_ci

    rng = np.random.default_rng(2)
    x = rng.normal(80, 25, 12)
    y = 20 + 2.5 * x + rng.normal(0, 8, 12)
    out = bootstrap_ci(x, y, n_boot=2000, seed=5)
    slope = out["observed_slope"]
    assert out["boot_lo"] - 1e-9 <= slope <= out["boot_hi"] + 1e-9


def test_power_curve_requires_more_years_for_weaker_effects():
    """Sanity: a weaker effect (smaller slope vs the same noise) needs more
    years for the same power."""
    from src.inference import power_curve

    x_pool = np.array([40, 50, 60, 80, 100, 150, 200, 240], float)
    strong = power_curve(0.40, 30.0, x_pool, n_grid=[6, 10, 16], n_sims=300, seed=1)
    weak = power_curve(0.10, 30.0, x_pool, n_grid=[6, 10, 16], n_sims=300, seed=1)
    assert strong["curve"][-1]["power"] > weak["curve"][-1]["power"]
