"""Specification sensitivity analysis: does the conclusion survive arbitrary
modeling choices?

The first-stage result should not hinge on one benchmark date or one definition
of "summer". This module re-runs the snowpack -> hydro regression across a grid:

  snowpack benchmark : Mar 1, Apr 1, May 1, Jun 1 (CDEC courses are measured
                       on the 1st of the month, so mid-month dates have no data)
  summer window      : Jun-Sep (default), May-Sep, Jun-Aug, Jul-Sep

and reports each specification's slope, 95% t-CI, and p-value. If the sign and
rough magnitude are stable across the grid, the headline number is not an
artifact of one arbitrary choice; if it is fragile, we want to know first.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.regression.linear_model import OLS

from src.features import april1_snowpack_index, summer_hydro, summer_price_features

SNOW_DATES: list[tuple[int, int]] = [(3, 1), (4, 1), (5, 1), (6, 1)]
SUMMER_WINDOWS: dict[str, list[int]] = {
    "Jun-Sep": [6, 7, 8, 9],   # default
    "May-Sep": [5, 6, 7, 8, 9],
    "Jun-Aug": [6, 7, 8],
    "Jul-Sep": [7, 8, 9],
}


def _slope_ci_p(x: np.ndarray, y: np.ndarray) -> dict:
    """OLS slope with small-sample t-CI and p-value (NaN-safe)."""
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(df) < 5 or df["x"].nunique() < 3:
        return {"slope": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
                "p_value": np.nan, "n": len(df)}
    X = np.column_stack([np.ones(len(df)), df["x"].values])
    fit = OLS(df["y"].values, X).fit()
    slope, se = float(fit.params[1]), float(fit.bse[1])
    t_crit = float(stats.t.ppf(0.975, df=fit.df_resid))
    return {
        "slope": slope,
        "ci_lo": slope - t_crit * se,
        "ci_hi": slope + t_crit * se,
        "p_value": float(fit.pvalues[1]),
        "n": len(df),
    }


def run_sensitivity(snow: pd.DataFrame, prices: pd.DataFrame,
                    fuel_mix: pd.DataFrame,
                    normal_window: tuple[int, int] = (1991, 2020)) -> pd.DataFrame:
    """Run the full snowpack-date x summer-window grid.

    Returns one row per specification with the hydro-leg result (snowpack ->
    hydro generation, the well-powered link) and the price-leg result
    (snowpack -> price volatility, exploratory at current coverage).
    """
    rows = []
    for win_name, months in SUMMER_WINDOWS.items():
        hydro = summer_hydro(fuel_mix, months=months).set_index("year")["hydro_gwh"]
        price_vol = summer_price_features(prices, months=months).set_index("year")["price_vol"]
        for date in SNOW_DATES:
            try:
                snow_idx = april1_snowpack_index(
                    snow, normal_window=normal_window, target_date=date, day_tol=5)
            except Exception:
                continue
            if snow_idx.empty:
                continue
            hydro_res = _slope_ci_p(snow_idx.reindex(hydro.index).values, hydro.values)
            price_res = _slope_ci_p(snow_idx.reindex(price_vol.index).values, price_vol.values)
            rows.append({
                "snow_date": f"{date[0]:02d}-{date[1]:02d}",
                "summer_window": win_name,
                # hydro leg (snowpack -> generation)
                "hydro_slope": hydro_res["slope"],
                "hydro_ci_lo": hydro_res["ci_lo"],
                "hydro_ci_hi": hydro_res["ci_hi"],
                "hydro_p": hydro_res["p_value"],
                "hydro_n": hydro_res["n"],
                # price leg (snowpack -> volatility), exploratory
                "price_slope": price_res["slope"],
                "price_p": price_res["p_value"],
                "price_n": price_res["n"],
            })
    return pd.DataFrame(rows)


def sign_stability(sens: pd.DataFrame) -> dict:
    """Summary of the grid: is the hydro-leg slope consistently positive?"""
    h = sens.dropna(subset=["hydro_slope"])
    return {
        "specs": len(h),
        "sign_positive": int((h["hydro_slope"] > 0).sum()),
        "sign_negative": int((h["hydro_slope"] <= 0).sum()),
        "slope_min": float(h["hydro_slope"].min()),
        "slope_max": float(h["hydro_slope"].max()),
        "p_below_0.05": int((h["hydro_p"] < 0.05).sum()),
        "p_below_0.10": int((h["hydro_p"] < 0.10).sum()),
    }
