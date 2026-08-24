"""Feature engineering: build the yearly analysis panel.

For each year t the panel contains:
  year               : t
  snowpack_pct       : April 1 snow water content as % of normal (median across
                       snow courses; normal = course's 1991-2020 mean April 1 SWC)
  price_mean         : summer (Jun-Sep) mean of daily-mean day-ahead LMP ($/MWh)
  price_peak         : summer maximum daily-mean day-ahead LMP ($/MWh)
  price_vol          : volatility = std of daily-mean day-ahead LMP over the summer
  price_vol_hourly   : std of ALL hourly LMP observations over the summer
  hydro_gwh          : summer hydroelectric generation (Large + Small hydro, GWh)
  hydro_gwh_eia      : EIA-reported summer CISO hydro generation (GWh), optional

The volatility targets (price_vol, price_vol_hourly) are what the models predict.

Usage:
    from src.features import build_panel
    panel = build_panel()
    panel.to_csv("data/processed/panel.csv")
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"

NORMAL_WINDOW = (1991, 2020)  # CDEC convention for "percent of normal"
SUMMER_MONTHS = [6, 7, 8, 9]
PRICE_UNIT = 1.0  # LMP in $/MWh


# ---------------------------------------------------------------- snowpack
def april1_snowpack_index(snow: pd.DataFrame,
                          normal_window: tuple[int, int] = NORMAL_WINDOW) -> pd.Series:
    """April 1 SWC as % of normal, by year, median across snow courses.

    snow: DataFrame with columns [station_id, date, value] (inches SWC).
    """
    snow = snow.copy()
    snow["year"] = snow["date"].dt.year
    snow["month"] = snow["date"].dt.month
    snow["day"] = snow["date"].dt.day

    # Some courses report April 1 slightly early/late; take Apr 1 exactly when
    # present, else the closest April observation within +-5 days.
    apr = snow[snow["month"] == 4].copy()
    apr["day_offset"] = (apr["day"] - 1).abs()
    apr = apr[apr["day_offset"] <= 5]
    idx = apr.groupby(["station_id", "year"])["day_offset"].idxmin()
    apr1 = apr.loc[idx, ["station_id", "year", "value"]].rename(columns={"value": "swc"})

    norm = (
        apr1[apr1["year"].between(*normal_window)]
        .groupby("station_id")["swc"]
        .mean()
        .rename("normal_swc")
    )
    # Require >= 10 years of April data in the normal window for a stable normal
    n_obs = apr1[apr1["year"].between(*normal_window)].groupby("station_id").size()
    keep = n_obs[n_obs >= 10].index
    norm = norm.loc[keep]

    merged = apr1.merge(norm, left_on="station_id", right_index=True, how="inner")
    merged["pct_normal"] = merged["swc"] / merged["normal_swc"] * 100.0
    # Cap extreme outliers (snow courses that went from 0 to huge) at 1st/99th
    # percentile of the cross-course distribution for each year.
    pct = merged.groupby("year")["pct_normal"]
    lo, hi = pct.transform("quantile", q=0.01), pct.transform("quantile", q=0.99)
    merged["pct_normal"] = merged["pct_normal"].clip(lo, hi)
    return merged.groupby("year")["pct_normal"].median()


# ---------------------------------------------------------------- prices
def _daily_price_stats(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily mean LMP per hub, then averaged across hubs -> daily CA-wide price."""
    prices = prices.copy()
    prices["date"] = prices["time"].dt.date
    daily = prices.groupby(["date", "hub"])["price"].mean().reset_index()
    daily = daily.groupby("date")["price"].mean().rename("daily_price").reset_index()
    daily["date"] = pd.to_datetime(daily["date"])
    return daily


def summer_price_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Per-year summer price summary: mean, peak, daily volatility, hourly volatility."""
    prices = prices.copy()
    prices["year"] = prices["time"].dt.year
    daily = _daily_price_stats(prices)
    daily["year"] = daily["date"].dt.year

    per_day = (
        daily.groupby("year")["daily_price"]
        .agg(price_mean="mean", price_peak="max", price_vol="std")
        .reset_index()
    )
    hourly = prices.groupby("year")["price"].std().rename("price_vol_hourly").reset_index()
    return per_day.merge(hourly, on="year")


# ---------------------------------------------------------------- hydro
def summer_hydro(fuel_mix: pd.DataFrame) -> pd.DataFrame:
    """Summer hydro generation (GWh) per year from CAISO fuel mix (MW x 5min)."""
    if fuel_mix.empty:
        return pd.DataFrame(columns=["year", "hydro_gwh"])
    fm = fuel_mix.copy()
    fm["Time"] = pd.to_datetime(fm["Time"])
    hydro_cols = [c for c in ["Large Hydro", "Small Hydro"] if c in fm.columns]
    fm["hydro_mw"] = fm[hydro_cols].sum(axis=1, min_count=1)
    # MW * (5/60) h = MWh per 5-minute interval
    fm["mwh"] = fm["hydro_mw"] * (5 / 60)
    fm["year"] = fm["Time"].dt.year
    return fm.groupby("year")["mwh"].sum().div(1000).rename("hydro_gwh").reset_index()


def eia_summer_hydro(hourly: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """Summer hydro (GWh) per year from EIA data.

    hourly: [period, hydro_mw] (EIA-930 hourly, MW). daily: [period, hydro_mwh].
    Prefers the daily series (directly MWh) and fills from hourly when needed.
    """
    out = pd.DataFrame(columns=["year", "hydro_gwh_eia"])
    FULL_SUMMER_DAYS = 122  # Jun 1 - Sep 30
    if not daily.empty:
        d = daily.copy()
        d["period"] = pd.to_datetime(d["period"])
        d["year"] = d["period"].dt.year
        d = d[d["period"].dt.month.isin(SUMMER_MONTHS)]
        d = d.groupby("year").agg(
            hydro_gwh_eia_d=("hydro_mwh", lambda s: s.sum() / 1000),
            n_days=("period", "nunique"),
        ).reset_index()
        out = d
    if hourly.empty:
        return out[["year", "hydro_gwh_eia_d"]].rename(columns={"hydro_gwh_eia_d": "hydro_gwh_eia"})
    h = hourly.copy()
    h["period"] = pd.to_datetime(h["period"])
    h["year"] = h["period"].dt.year
    h = h[h["period"].dt.month.isin(SUMMER_MONTHS)]
    hr = h.groupby("year").agg(
        hydro_gwh_eia_h=("hydro_mw", lambda s: s.sum() / 1000),
        n_hours=("period", "count"),
    ).reset_index()
    if out.empty:
        return hr.rename(columns={"hydro_gwh_eia_h": "hydro_gwh_eia"})[["year", "hydro_gwh_eia"]]
    merged = out.merge(hr, on="year", how="outer")
    # prefer the source that actually covers the full summer (daily starts 2020-08);
    # a partially covered summer (missing months in the EIA record) is set to NaN
    merged["use_daily"] = merged["n_days"].fillna(0) >= 100
    merged["hourly_complete"] = merged["n_hours"].fillna(0) >= 2500  # ~85% of 2928 h
    merged["hydro_gwh_eia"] = np.where(
        merged["use_daily"], merged["hydro_gwh_eia_d"],
        np.where(merged["hourly_complete"], merged["hydro_gwh_eia_h"], np.nan),
    )
    return merged[["year", "hydro_gwh_eia"]]


# ---------------------------------------------------------------- panel
def build_panel() -> pd.DataFrame:
    """Assemble the full yearly panel from cached raw data. Returns empty columns
    for any source that has no cached data (so callers can still build models on
    the subsets that exist)."""
    snow_path, price_path = RAW_DIR / "snow_courses.csv", None
    snow = (
        pd.read_csv(snow_path, parse_dates=["date"])
        if snow_path.exists() else pd.DataFrame(columns=["station_id", "date", "value"])
    )

    price_files = sorted(RAW_DIR.glob("caiso_prices_*.csv"))
    prices = (
        pd.concat([pd.read_csv(f, parse_dates=["time"]) for f in price_files], ignore_index=True)
        if price_files else pd.DataFrame(columns=["time", "hub", "price"])
    )
    mix_files = sorted(RAW_DIR.glob("caiso_fuelmix_*.csv"))
    fuel = (
        pd.concat([pd.read_csv(f, parse_dates=["Time"]) for f in mix_files], ignore_index=True)
        if mix_files else pd.DataFrame()
    )
    eia_hourly_path = RAW_DIR / "eia_ciso_hydro_hourly.csv"
    eia_daily_path = RAW_DIR / "eia_ciso_hydro_daily.csv"
    eia_hourly = (
        pd.read_csv(eia_hourly_path, parse_dates=["period"])
        if eia_hourly_path.exists() else pd.DataFrame(columns=["period", "hydro_mw"])
    )
    eia_daily = (
        pd.read_csv(eia_daily_path, parse_dates=["period"])
        if eia_daily_path.exists() else pd.DataFrame(columns=["period", "hydro_mwh"])
    )

    years = pd.RangeIndex(1980, 2026, name="year")
    panel = pd.DataFrame(index=years)

    if not snow.empty:
        idx = april1_snowpack_index(snow)
        panel["snowpack_pct"] = idx.reindex(panel.index)
    if not prices.empty:
        feat = summer_price_features(prices).set_index("year")
        panel = panel.join(feat)
    if not fuel.empty:
        panel = panel.join(summer_hydro(fuel).set_index("year"))
    if not eia_hourly.empty or not eia_daily.empty:
        panel = panel.join(eia_summer_hydro(eia_hourly, eia_daily).set_index("year"))

    PROC_DIR.mkdir(parents=True, exist_ok=True)
    panel.to_csv(PROC_DIR / "panel.csv")
    return panel


if __name__ == "__main__":
    p = build_panel()
    print(p.dropna(how="all"))
