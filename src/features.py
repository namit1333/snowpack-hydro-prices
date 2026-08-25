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

Control variables (confounders) when data is cached:
  demand_mean_mw / demand_peak_mw : summer CISO demand (EIA-930)
  temp_mean_c / heat_days_38c     : summer temperature / heat-wave days
  gas_mean                        : summer Henry Hub natural gas ($/MMBtu)

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
                          normal_window: tuple[int, int] = NORMAL_WINDOW,
                          target_date: tuple[int, int] = (4, 1),
                          day_tol: int = 5) -> pd.Series:
    """Snow water content as % of normal, by year, median across snow courses.

    snow: DataFrame with columns [station_id, date, value] (inches SWC).
    target_date: (month, day) of the measurement benchmark -- (4, 1) is the
    standard hydrological April 1 benchmark; other dates power the sensitivity
    analysis. day_tol is the +/- day tolerance around the target date.
    """
    snow = snow.copy()
    snow["year"] = snow["date"].dt.year
    snow["month"] = snow["date"].dt.month
    snow["day"] = snow["date"].dt.day

    # Courses report on/near the benchmark date; take the target date exactly
    # when present, else the closest observation within +/- day_tol days.
    tgt_m, tgt_d = target_date
    apr = snow[snow["month"] == tgt_m].copy()
    apr["day_offset"] = (apr["day"] - tgt_d).abs()
    apr = apr[apr["day_offset"] <= day_tol]
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


def summer_price_features(prices: pd.DataFrame,
                          months: list[int] | None = None) -> pd.DataFrame:
    """Per-year summer price summary: mean, peak, daily volatility, hourly volatility.

    months: summer window definition (default SUMMER_MONTHS = Jun-Sep); other
    windows power the sensitivity analysis.
    """
    months = SUMMER_MONTHS if months is None else months
    prices = prices.copy()
    prices["year"] = prices["time"].dt.year
    daily = _daily_price_stats(prices)
    daily["year"] = daily["date"].dt.year
    daily = daily[daily["date"].dt.month.isin(months)]

    per_day = (
        daily.groupby("year")["daily_price"]
        .agg(price_mean="mean", price_peak="max", price_vol="std")
        .reset_index()
    )
    hourly = prices[prices["time"].dt.month.isin(months)]
    hourly = hourly.groupby("year")["price"].std().rename("price_vol_hourly").reset_index()
    return per_day.merge(hourly, on="year")


# ---------------------------------------------------------------- hydro
def summer_hydro(fuel_mix: pd.DataFrame,
                 months: list[int] | None = None) -> pd.DataFrame:
    """Summer hydro generation (GWh) per year from CAISO fuel mix (MW x 5min).

    months: summer window definition (default SUMMER_MONTHS = Jun-Sep).
    """
    if fuel_mix.empty:
        return pd.DataFrame(columns=["year", "hydro_gwh"])
    months = SUMMER_MONTHS if months is None else months
    fm = fuel_mix.copy()
    fm["Time"] = pd.to_datetime(fm["Time"])
    fm = fm[fm["Time"].dt.month.isin(months)]
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
def _load_prices() -> pd.DataFrame:
    """Load all cached price files (modern + 2016-2018 archive) with a common
    US/Pacific-naive timestamp so the two sources join cleanly."""
    price_files = sorted(RAW_DIR.glob("caiso_prices_*.csv"))
    price_files = [f for f in price_files if f.name != "caiso_prices_historical_2016_2018.csv"]
    frames = []
    for f in price_files:
        frames.append(pd.read_csv(f, parse_dates=["time"]))
    hist_path = RAW_DIR / "caiso_prices_historical_2016_2018.csv"
    if hist_path.exists():
        frames.append(pd.read_csv(hist_path, parse_dates=["time"]))
    if not frames:
        return pd.DataFrame(columns=["time", "hub", "price"])
    prices = pd.concat(frames, ignore_index=True)
    prices["time"] = pd.to_datetime(prices["time"], utc=True).dt.tz_convert("US/Pacific")
    prices["time"] = prices["time"].dt.tz_localize(None)
    # A (timestamp, hub) must be unique: duplicates would double-count a
    # hub-hour in the daily/volatility aggregation.
    prices = prices.drop_duplicates(subset=["time", "hub"]).sort_values("time")
    return prices


def build_panel() -> pd.DataFrame:
    """Assemble the full yearly panel from cached raw data. Returns empty columns
    for any source that has no cached data (so callers can still build models on
    the subsets that exist)."""
    snow_path = RAW_DIR / "snow_courses.csv"
    snow = (
        pd.read_csv(snow_path, parse_dates=["date"])
        if snow_path.exists() else pd.DataFrame(columns=["station_id", "date", "value"])
    )

    prices = _load_prices()
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

    panel = _join_controls(panel)

    PROC_DIR.mkdir(parents=True, exist_ok=True)
    panel.to_csv(PROC_DIR / "panel.csv")
    return panel


def _join_controls(panel: pd.DataFrame) -> pd.DataFrame:
    """Join annual summer control-variable summaries when raw files are cached."""
    from src.fetch_controls import (
        demand_summary, gas_summary, temperature_summary,
    )

    demand_files = sorted(RAW_DIR.glob("eia_ciso_demand_*.csv"))
    if demand_files:
        d = pd.concat([pd.read_csv(f, parse_dates=["period"]) for f in demand_files],
                      ignore_index=True)
        panel = panel.join(demand_summary(d).set_index("year"))

    temp_files = sorted(RAW_DIR.glob("heat_temperature_*.csv"))
    if temp_files:
        t = pd.concat([pd.read_csv(f, parse_dates=["date"]) for f in temp_files],
                      ignore_index=True)
        panel = panel.join(temperature_summary(t).set_index("year"))

    gas_files = sorted(RAW_DIR.glob("gas_henryhub_*.csv"))
    if gas_files:
        g = pd.concat([pd.read_csv(g, parse_dates=["date"]) for g in gas_files],
                      ignore_index=True)
        panel = panel.join(gas_summary(g).set_index("year"))
    return panel


def build_monthly_panel() -> pd.DataFrame:
    """Monthly summer panel: one row per (year, month) of the Jun-Sep window.

    Gives ~4x the observations of the annual panel for the price legs, which is
    the natural next step for power once the price history grows. Columns:
      year, month, snowpack_pct (annual, repeated), hydro_gwh (monthly),
      price_mean, price_vol (monthly), demand_mean_mw (monthly), temp_mean_c,
      gas_mean (annual, repeated).
    """
    panel = build_panel()
    prices = _load_prices()
    if prices.empty:
        return pd.DataFrame()
    prices["year"] = prices["time"].dt.year
    prices["month"] = prices["time"].dt.month
    prices["date"] = prices["time"].dt.date

    daily = prices.groupby(["date", "hub"])["price"].mean().reset_index()
    daily = daily.groupby("date")["price"].mean().rename("daily_price").reset_index()
    daily["date"] = pd.to_datetime(daily["date"])
    daily["year"] = daily["date"].dt.year
    daily["month"] = daily["date"].dt.month
    monthly_price = daily.groupby(["year", "month"]).agg(
        price_mean=("daily_price", "mean"),
        price_vol=("daily_price", "std"),
    ).reset_index()

    out = monthly_price.copy()
    if "snowpack_pct" in panel.columns:
        out = out.merge(
            panel[["snowpack_pct"]], left_on="year", right_index=True, how="left")

    mix_files = sorted(RAW_DIR.glob("caiso_fuelmix_*.csv"))
    if mix_files:
        fuel = pd.concat([pd.read_csv(f, parse_dates=["Time"]) for f in mix_files],
                         ignore_index=True)
        hydro_cols = [c for c in ["Large Hydro", "Small Hydro"] if c in fuel.columns]
        fuel["hydro_mw"] = fuel[hydro_cols].sum(axis=1, min_count=1)
        fuel["mwh"] = fuel["hydro_mw"] * (5 / 60)
        fuel["year"] = fuel["Time"].dt.year
        fuel["month"] = fuel["Time"].dt.month
        mh = fuel.groupby(["year", "month"])["mwh"].sum().div(1000).rename("hydro_gwh").reset_index()
        out = out.merge(mh, on=["year", "month"], how="left")

    demand_files = sorted(RAW_DIR.glob("eia_ciso_demand_*.csv"))
    if demand_files:
        d = pd.concat([pd.read_csv(f, parse_dates=["period"]) for f in demand_files],
                      ignore_index=True)
        d["year"] = d["period"].dt.year
        d["month"] = d["period"].dt.month
        dm = d.groupby(["year", "month"])["demand_mw"].mean().rename("demand_mean_mw").reset_index()
        out = out.merge(dm, on=["year", "month"], how="left")

    temp_files = sorted(RAW_DIR.glob("heat_temperature_*.csv"))
    if temp_files:
        t = pd.concat([pd.read_csv(f, parse_dates=["date"]) for f in temp_files],
                      ignore_index=True)
        t["year"] = t["date"].dt.year
        t["month"] = t["date"].dt.month
        tm = t.groupby(["year", "month"])["temp_max_c"].mean().rename("temp_mean_c").reset_index()
        out = out.merge(tm, on=["year", "month"], how="left")

    gas_files = sorted(RAW_DIR.glob("gas_henryhub_*.csv"))
    if gas_files:
        g = pd.concat([pd.read_csv(f, parse_dates=["date"]) for f in gas_files],
                      ignore_index=True)
        g["year"] = g["date"].dt.year
        g["month"] = g["date"].dt.month
        gm = g.groupby(["year", "month"])["gas_price"].mean().rename("gas_mean").reset_index()
        out = out.merge(gm, on=["year", "month"], how="left")

    return out.dropna(subset=["price_vol"])


if __name__ == "__main__":
    p = build_panel()
    print(p.dropna(how="all"))
