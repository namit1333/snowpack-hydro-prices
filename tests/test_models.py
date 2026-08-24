"""Unit tests for src/models.py using synthetic data."""

import numpy as np
import pandas as pd

from src import models


def _panel(n=20, seed=0):
    rng = np.random.default_rng(seed)
    years = pd.RangeIndex(2000, 2000 + n)
    snow = 60 + 40 * np.sin(np.arange(n)) + rng.normal(0, 5, n)  # ~40-100%
    # causal chain: more snow -> more hydro -> lower summer price volatility
    hydro = 3000 + 30 * (snow - 80) + rng.normal(0, 50, n)
    vol = 10 - 0.005 * (hydro - 3000) + rng.normal(0, 0.6, n)
    return pd.DataFrame({
        "year": years, "snowpack_pct": snow, "price_vol": vol,
        "price_vol_hourly": vol + 5, "hydro_gwh": hydro,
    }).set_index("year")


def test_walk_forward_strict_oos_and_shape():
    p = _panel()
    wf = models.walk_forward(p, min_train=6)
    # years predicted: 6..19 -> 14 rows per model
    assert len(wf) == 14 * len(models.MODELS)
    assert wf["model"].nunique() == len(models.MODELS)
    assert (wf.groupby("year").size() == len(models.MODELS)).all()
    # no NaNs, finite errors
    assert wf[["pred", "actual", "error"]].notna().all().all()
    assert np.isfinite(wf["pred"]).all()


def test_augmented_ols_beats_naive_when_snowpack_is_informative():
    p = _panel(seed=3)
    wf = models.walk_forward(p, min_train=6)
    rt = models.rmse_table(wf)
    assert rt.loc["augmented_ols", "rmse"] < rt.loc["baseline_naive", "rmse"]
    assert rt.loc["augmented_ols", "rmse"] < rt.loc["baseline_mean3", "rmse"]


def test_dm_test_zero_errors():
    e1 = np.array([1.0, -2.0, 3.0, -1.0, 2.0])
    e2 = e1.copy()
    out = models.dm_test(e1, e2)
    assert out["dm"] == 0.0 and out["p"] == 1.0


def test_dm_test_favors_lower_loss():
    rng = np.random.default_rng(1)
    e2 = rng.normal(0, 3, 30)          # bigger errors
    e1 = e2 / 2                        # smaller errors
    out = models.dm_test(e1, e2)
    assert out["dm"] < 0 and out["p"] < 0.05


def test_mediation_chain_detected():
    p = _panel(seed=2)
    med = models.mediation_analysis(p)
    assert med["n"] >= 10
    # snowpack -> volatility total effect negative (high snow = low volatility)
    assert med["total_effect_c"] < 0
    assert med["a_path"] > 0           # snowpack -> hydro positive
    assert med["b_mediator"] < 0       # hydro -> lower volatility
    # controlling for hydro should shrink the direct snowpack effect (mediation)
    assert med["proportion_mediated"] > 0.5


def test_mediation_insufficient_data():
    p = _panel(n=2)
    med = models.mediation_analysis(p)
    assert "error" in med


def test_walk_forward_controls_model():
    p = _panel()
    p["temp_mean_c"] = 30 + 5 * np.sin(np.arange(len(p))) + 5
    p["demand_mean_mw"] = 30000 + 2000 * np.cos(np.arange(len(p)))
    wf = models.walk_forward(p, min_train=6, control_cols=["temp_mean_c", "demand_mean_mw"])
    assert models.CONTROL_MODEL in wf["model"].unique()
    assert wf[wf["model"] == models.CONTROL_MODEL]["pred"].notna().all()


def test_short_walk_forward_gives_multiple_oos():
    # the 2016-2018 + 2023-2025 window: 6 rows, min_train=3 -> 3 held-out years
    p = _panel(n=6)
    wf = models.walk_forward(p, min_train=3)
    n_oos = wf[wf["model"] == "augmented_ols"].shape[0]
    assert n_oos == 3


def test_run_analysis_flags_exploratory_mediation():
    p = _panel(n=6)  # too few years for inference
    res = models.run_analysis(p)
    assert any("exploratory" in v.get("status", "") for k, v in res.items()
               if k.startswith("mediation_"))
