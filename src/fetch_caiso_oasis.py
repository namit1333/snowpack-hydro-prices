"""Fetch California ISO (CAISO) market data via the gridstatus library.

gridstatus wraps the official OASIS API (https://oasis.caiso.com) — no API key
or registration required for the public endpoints used here.

Two datasets are fetched for the June 1 - September 30 window of each year:
  1. Day-ahead hourly Locational Marginal Prices (LMP) at the three trading hubs
     (TH_NP15_GEN-APND, TH_SP15_GEN-APND, TH_ZP26_GEN-APND). LMP history via
     OASIS goes back to ~2009, so 2010+ is safe.
  2. The real-time fuel mix, from which Large Hydro + Small Hydro generation
     (MW) is summed. This is the *mediator*: actual hydro output.

Outputs (cached per year):
  data/raw/caiso_prices_YYYY.csv    -> hourly LMP per hub
  data/raw/caiso_fuelmix_YYYY.csv   -> 5-minute fuel mix (Hydro columns)

CLI:  python src/fetch_caiso_oasis.py --start 2010 --end 2025
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from gridstatus import CAISO, Markets

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"

HUB_LOCATIONS = ["TH_NP15_GEN-APND", "TH_SP15_GEN-APND", "TH_ZP26_GEN-APND"]
HYDRO_COLUMNS = ["Large Hydro", "Small Hydro"]
SUMMER = {"start_month": 6, "start_day": 1, "end_month": 9, "end_day": 30}


def _summer_window(year: int) -> tuple[str, str]:
    start = pd.Timestamp(year, SUMMER["start_month"], SUMMER["start_day"], tz="US/Pacific")
    end = pd.Timestamp(year, SUMMER["end_month"], SUMMER["end_day"], tz="US/Pacific")
    return start.isoformat(), end.isoformat()


def fetch_prices(year: int, cache: bool = True) -> pd.DataFrame:
    """Day-ahead hourly LMP for the summer window of `year`, one row per hub-hour."""
    out_path = RAW_DIR / f"caiso_prices_{year}.csv"
    if cache and out_path.exists():
        return pd.read_csv(out_path, parse_dates=["time"])
    start, end = _summer_window(year)
    caiso = CAISO()
    df = caiso.get_lmp(
        date=start,
        end=end,
        market=Markets.DAY_AHEAD_HOURLY,
        locations=HUB_LOCATIONS,
    )
    df = df.rename(columns={"Location": "hub", "LMP": "price", "Time": "time"})
    df = df[["time", "hub", "price"]].dropna(subset=["price"])
    if cache:
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
    return df


HISTORICAL_URL = (
    "https://raw.githubusercontent.com/manukalia/CA_Electricty_Price_Prediction_Neural_Net/"
    "master/data/elec_prices_hrly.csv"
)
HISTORICAL_YEARS = [2016, 2017, 2018]  # full summers present in the archive


def fetch_historical_prices(cache: bool = True) -> dict[int, pd.DataFrame]:
    """Day-ahead hourly LMP for 2016-2018 from a public GitHub mirror of OASIS.

    CAISO's OASIS API no longer serves LMP before 2023; this archived CSV
    (Bayshore node, BAYSHOR2_1_N001) fills the 2016-2018 summers so the price
    window spans 6 years. Converted to the same [time, hub, price] schema as
    `fetch_prices`. Returns {year: df}.
    """
    import requests
    out: dict[int, pd.DataFrame] = {}
    cache_file = RAW_DIR / "caiso_prices_historical_2016_2018.csv"
    if cache and cache_file.exists():
        df = pd.read_csv(cache_file, parse_dates=["time"])
        for y in HISTORICAL_YEARS:
            out[y] = df[df["time"].dt.year == y].copy()
        return out
    print(f"    downloading historical LMP archive ({HISTORICAL_URL}) ...", flush=True)
    r = requests.get(HISTORICAL_URL, timeout=120)
    r.raise_for_status()
    df = pd.read_csv(pd.io.common.BytesIO(r.content))
    df["time"] = pd.to_datetime(df["start_datetime"], utc=True)
    df["hub"] = "BAYSHOR"
    df["price"] = df["dam_price_per_mwh"]
    df = df[["time", "hub", "price"]].dropna(subset=["price"])
    # keep only summer months to match the other price files
    df = df[df["time"].dt.month.isin([6, 7, 8, 9])].copy()
    if cache:
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_file, index=False)
    for y in HISTORICAL_YEARS:
        out[y] = df[df["time"].dt.year == y].copy()
    return out


def fetch_fuel_mix(year: int, cache: bool = True) -> pd.DataFrame:
    """5-minute CAISO fuel mix for the summer window; returns hydro columns."""
    out_path = RAW_DIR / f"caiso_fuelmix_{year}.csv"
    if cache and out_path.exists():
        return pd.read_csv(out_path, parse_dates=["Time"])
    start, end = _summer_window(year)
    caiso = CAISO()
    df = caiso.get_fuel_mix(date=start, end=end)
    cols = ["Time"] + HYDRO_COLUMNS
    df = df[[c for c in cols if c in df.columns]].copy()
    if cache:
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
    return df


def fetch_year(year: int, cache: bool = True) -> dict[str, pd.DataFrame]:
    """Fetch both price and fuel-mix data for one summer."""
    print(f"  CAISO {year} prices ...", flush=True)
    prices = fetch_prices(year, cache=cache)
    print(f"  CAISO {year} fuel mix ...", flush=True)
    try:
        fuel = fetch_fuel_mix(year, cache=cache)
    except Exception as exc:  # fuel-mix history may not cover old years
        print(f"  (fuel mix unavailable for {year}: {exc})")
        fuel = pd.DataFrame()
    return {"prices": prices, "fuel_mix": fuel}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Fetch CAISO summer prices + fuel mix")
    ap.add_argument("--start", type=int, default=2010)
    ap.add_argument("--end", type=int, default=2025)
    ap.add_argument("--no-historical", action="store_true",
                    help="skip the 2016-2018 archived LMP mirror")
    args = ap.parse_args()
    if not args.no_historical:
        print("  CAISO historical LMP (2016-2018) ...", flush=True)
        try:
            hist = fetch_historical_prices()
            for y, df in hist.items():
                print(f"    prices: {len(df):,} rows (year {y})")
        except Exception as exc:
            print(f"  (historical LMP unavailable: {exc})")
    for y in range(args.start, args.end + 1):
        res = fetch_year(y)
        print(f"    prices: {len(res['prices']):,} rows, "
              f"fuel mix: {len(res['fuel_mix']):,} rows")
