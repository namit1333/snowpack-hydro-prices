"""Tests for the specification sensitivity analysis (synthetic data)."""

import numpy as np
import pandas as pd
import pytest

from src import sensitivity
from src.features import april1_snowpack_index, summer_hydro, summer_price_features


def _raw(n_years=10, start=2015, seed=0):
    """Synthetic raw snow / price / fuel frames spanning the needed months."""
    rng = np.random.default_rng(seed)
    # snow courses measured on ~the 1st of Jan-May
    dates, stations, values = [], [], []
    for year in range(start - 30, start + n_years):
        for month in [1, 2, 3, 4, 5, 6]:
            for day in [1, 15]:  # mid-month readings power the 15th benchmarks
                for st in ["S1", "S2", "S3"]:
                    dates.append(pd.Timestamp(year, month, day))
                    stations.append(st)
                    base = 20 + 0.4 * abs(station_effect(st)) + rng.normal(0, 2)
                    values.append(max(base, 0.5))
    snow = pd.DataFrame({"station_id": stations, "date": dates, "value": values})

    hours = pd.date_range(f"{start}-05-01", f"{start + n_years - 1}-10-31 23:00",
                          freq="h")
    years = hours.year
    months = hours.month
    price = 30 + 5 * np.sin(np.arange(len(hours)) / 200) + rng.normal(0, 3, len(hours))
    prices = pd.DataFrame({"time": hours, "hub": "TH_NP15_GEN-APND",
                           "price": price})
    mw = 3000 + 800 * np.sin(2 * np.pi * months / 12) + rng.normal(0, 100, len(hours))
    fuel = pd.DataFrame({"Time": hours, "Large Hydro": mw, "Small Hydro": mw * 0.1})
    return snow, prices, fuel


def station_effect(st):
    return {"S1": 0.0, "S2": 10.0, "S3": -5.0}[st]


def test_target_date_parameter_changes_the_index():
    snow, _, _ = _raw()
    apr1 = april1_snowpack_index(snow, target_date=(4, 1))
    may1 = april1_snowpack_index(snow, target_date=(5, 1), day_tol=7)
    assert len(apr1) > 0 and len(may1) > 0
    # different benchmark dates should generally give different indices
    common = apr1.index.intersection(may1.index)
    assert len(common) > 0
    assert not np.allclose(apr1.loc[common], may1.loc[common])


def test_summer_window_parameter_changes_volatility():
    _, prices, _ = _raw(n_years=3)
    full = summer_price_features(prices, months=[6, 7, 8, 9])
    short = summer_price_features(prices, months=[7, 8, 9])
    assert not np.allclose(full["price_vol"].values, short["price_vol"].values)


def test_sensitivity_grid_shape_and_defaults_match():
    snow, prices, fuel = _raw()
    sens = sensitivity.run_sensitivity(snow, prices, fuel)
    # 4 dates x 4 windows = 16 specification rows
    assert len(sens) == 16
    # the default specification (04-01 x Jun-Sep) must exist and be finite
    default = sens[(sens["snow_date"] == "04-01")
                   & (sens["summer_window"] == "Jun-Sep")]
    assert len(default) == 1
    row = default.iloc[0]
    assert np.isfinite(row["hydro_slope"])
    assert np.isfinite(row["price_slope"])


def test_sign_stability_counts():
    sens = pd.DataFrame({
        "hydro_slope": [1.0, 2.0, -0.5, np.nan],
        "hydro_p": [0.01, 0.20, 0.80, np.nan],
    })
    s = sensitivity.sign_stability(sens)
    assert s["specs"] == 3
    assert s["sign_positive"] == 2 and s["sign_negative"] == 1
    assert s["p_below_0.05"] == 1
