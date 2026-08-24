"""Fetch control variables (confounders) for the price model.

Electricity prices respond to more than snowpack. This module pulls the
controls identified in the study's limitations section:

  1. Summer electricity demand (EIA-930, CISO region) - hourly, free API key
  2. Summer temperature / heat waves (Open-Meteo archive, no key)
  3. Natural-gas prices, Henry Hub (EIA) - daily, free API key

Each fetcher caches a per-year CSV under data/raw/ and returns the annual
summer summary for the panel.

Outputs (cached per year):
  data/raw/eia_ciso_demand_YYYY.csv   -> hourly CISO demand (MW)
  data/raw/heat_temperature_YYYY.csv  -> daily max temp (deg C)
  data/raw/gas_henryhub_YYYY.csv      -> daily Henry Hub price ($/MMBtu)

CLI:  python src/fetch_controls.py --start 2016 --end 2025
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"

SUMMER_MONTHS = [6, 7, 8, 9]

# A central California location for heat (Fresno)
TEMP_LAT, TEMP_LON = 36.7378, -119.7871


def _summer_range(year: int) -> tuple[str, str]:
    return f"{year}-06-01", f"{year}-09-30"


# ---------------------------------------------------------------- demand (EIA-930)
def fetch_demand(year: int, cache: bool = True) -> pd.DataFrame:
    """Hourly CISO demand (MW) for the summer from EIA-930 region-data."""
    out_path = RAW_DIR / f"eia_ciso_demand_{year}.csv"
    if cache and out_path.exists():
        return pd.read_csv(out_path, parse_dates=["period"])
    api_key = os.environ.get("EIA_API_KEY", "")
    url = "https://api.eia.gov/v2/electricity/rto/region-data/data/"
    params = {
        "frequency": "hourly",
        "data[0]": "value",
        "facets[respondent][]": "CISO",
        "facets[data-type][]": "D",  # D = demand
        "start": f"{year}-06-01",
        "end": f"{year}-09-30",
        "length": 5000,
        "offset": 0,
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "api_key": api_key,
    }
    frames = []
    while True:
        r = requests.get(url, params=params, timeout=60)
        r.raise_for_status()
        data = r.json()["response"]["data"]
        if not data:
            break
        frames.append(pd.DataFrame(data))
        if len(data) < 5000:
            break
        params["offset"] += 5000
    if not frames:
        df = pd.DataFrame(columns=["period", "value"])
    else:
        df = pd.concat(frames, ignore_index=True)
        df = df[["period", "value"]].rename(columns={"value": "demand_mw"})
        df["period"] = pd.to_datetime(df["period"])
        df = df.drop_duplicates(subset="period").sort_values("period")
    if cache:
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
    return df


# ---------------------------------------------------------------- temperature (Open-Meteo)
def fetch_temperature(year: int, cache: bool = True) -> pd.DataFrame:
    """Daily max temperature (deg C) for the summer from Open-Meteo archive."""
    out_path = RAW_DIR / f"heat_temperature_{year}.csv"
    if cache and out_path.exists():
        return pd.read_csv(out_path, parse_dates=["date"])
    start, end = _summer_range(year)
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": TEMP_LAT,
        "longitude": TEMP_LON,
        "start_date": start,
        "end_date": end,
        "daily": "temperature_2m_max",
        "timezone": "America/Los_Angeles",
    }
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()["daily"]
    df = pd.DataFrame({"date": data["time"], "temp_max_c": data["temperature_2m_max"]})
    df["date"] = pd.to_datetime(df["date"])
    if cache:
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
    return df


# ---------------------------------------------------------------- gas (EIA Henry Hub)
def fetch_gas(year: int, cache: bool = True) -> pd.DataFrame:
    """Daily Henry Hub natural gas spot price ($/MMBtu) from EIA."""
    out_path = RAW_DIR / f"gas_henryhub_{year}.csv"
    if cache and out_path.exists():
        return pd.read_csv(out_path, parse_dates=["date"])
    api_key = os.environ.get("EIA_API_KEY", "")
    url = "https://api.eia.gov/v2/natural-gas/pri/sum/data/"
    params = {
        "frequency": "daily",
        "data[0]": "value",
        "facets[series][]": "NG.RNGWHHD.D",
        "start": f"{year}-06-01",
        "end": f"{year}-09-30",
        "length": 5000,
        "api_key": api_key,
    }
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()["response"]["data"]
    if not data:
        return pd.DataFrame(columns=["date", "value"])
    df = pd.DataFrame(data)
    df = df[["period", "value"]].rename(columns={"period": "date", "value": "gas_price"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates(subset="date").sort_values("date")
    if cache:
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
    return df


# ---------------------------------------------------------------- annual summaries
def demand_summary(demand: pd.DataFrame) -> pd.DataFrame:
    """Annual summer demand summary: mean, peak, and heat proxy (days > 90F demand)."""
    if demand.empty:
        return pd.DataFrame(columns=["year", "demand_mean_mw", "demand_peak_mw"])
    d = demand.copy()
    d["year"] = d["period"].dt.year
    g = d.groupby("year")["demand_mw"].agg(demand_mean_mw="mean", demand_peak_mw="max")
    return g.reset_index()


def temperature_summary(temp: pd.DataFrame) -> pd.DataFrame:
    """Annual summer heat summary: mean daily max temp and heat-wave days (>=38C)."""
    if temp.empty:
        return pd.DataFrame(columns=["year", "temp_mean_c", "heat_days_38c"])
    t = temp.copy()
    t["year"] = t["date"].dt.year
    g = t.groupby("year").agg(
        temp_mean_c=("temp_max_c", "mean"),
        heat_days_38c=("temp_max_c", lambda s: (s >= 38).sum()),
    )
    return g.reset_index()


def gas_summary(gas: pd.DataFrame) -> pd.DataFrame:
    """Annual summer Henry Hub price summary."""
    if gas.empty:
        return pd.DataFrame(columns=["year", "gas_mean"])
    g = gas.copy()
    g["year"] = g["date"].dt.year
    return g.groupby("year")["gas_price"].mean().rename("gas_mean").reset_index()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Fetch control variables")
    ap.add_argument("--start", type=int, default=2016)
    ap.add_argument("--end", type=int, default=2025)
    args = ap.parse_args()
    for y in range(args.start, args.end + 1):
        print(f"  {y}: demand ...", flush=True)
        try:
            d = fetch_demand(y)
            print(f"    demand: {len(d):,} rows -> mean {d['demand_mw'].mean():,.0f} MW")
        except Exception as exc:
            print(f"    (demand unavailable: {exc})")
        print(f"  {y}: temperature ...", flush=True)
        try:
            t = fetch_temperature(y)
            print(f"    temperature: {len(t):,} days")
        except Exception as exc:
            print(f"    (temperature unavailable: {exc})")
        print(f"  {y}: gas ...", flush=True)
        try:
            g = fetch_gas(y)
            print(f"    gas: {len(g):,} days")
        except Exception as exc:
            print(f"    (gas unavailable: {exc})")
