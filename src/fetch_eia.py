"""Fetch CAISO (CISO) hydroelectric generation from the EIA API.

EIA is the official independent source for CAISO hydro output:
  - hourly net generation (MW) via `electricity/rto/fuel-type-data`
    (EIA-930, available July 2018 - present)  <- primary mediator source
  - daily net generation (MWh) via `electricity/rto/daily-fuel-type-data`
    (EIA-930 daily, ~2022 - present)          <- cross-check

Requires a free EIA API key (https://www.eia.gov/opendata/) in the EIA_API_KEY
environment variable.

CLI:  EIA_API_KEY=... python src/fetch_eia.py
Outputs: data/raw/eia_ciso_hydro_hourly.csv, data/raw/eia_ciso_hydro_daily.csv
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
BASE = "https://api.eia.gov/v2/electricity/rto/"


def _key() -> str:
    k = os.environ.get("EIA_API_KEY")
    if not k:
        raise RuntimeError(
            "EIA_API_KEY not set. Get a free key at https://www.eia.gov/opendata/ "
            "then: export EIA_API_KEY=..."
        )
    return k


def _fetch(dataset: str, frequency: str, start: str, end: str) -> pd.DataFrame:
    """Paginate through one EIA dataset; returns rows as a DataFrame."""
    params = {
        "api_key": _key(),
        "frequency": frequency,
        "data[0]": "value",
        "facets[respondent][]": "CISO",
        "facets[fueltype][]": "WAT",  # hydroelectric
        "start": start,
        "end": end,
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "length": 5000,
    }
    rows, total = [], None
    while total is None or len(rows) < total:
        if rows:
            params["offset"] = len(rows)
        r = requests.get(f"{BASE}{dataset}/data/", params=params, timeout=90)
        r.raise_for_status()
        resp = r.json()["response"]
        total = int(resp["total"])
        rows += resp["data"]
        if not resp["data"]:
            break
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["period"] = pd.to_datetime(df["period"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df[["period", "value"]].dropna()


def fetch_ciso_hydro(cache: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (hourly, daily) CISO hydro net generation DataFrames."""
    hourly = _fetch("fuel-type-data", "hourly", "2018-07", "2026-12")
    hourly = hourly.rename(columns={"value": "hydro_mw"})
    daily = _fetch("daily-fuel-type-data", "daily", "2020-01", "2026-12")
    daily = daily.rename(columns={"value": "hydro_mwh"})
    # EIA-930 daily rows are repeated per data revision; keep the latest (max)
    if not daily.empty:
        daily = daily.groupby("period")["hydro_mwh"].max().reset_index()
    if cache:
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        hourly.to_csv(RAW_DIR / "eia_ciso_hydro_hourly.csv", index=False)
        if not daily.empty:
            daily.to_csv(RAW_DIR / "eia_ciso_hydro_daily.csv", index=False)
    return hourly, daily


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Fetch EIA CISO hydro generation")
    ap.add_argument("--no-cache", action="store_true")
    h, d = fetch_ciso_hydro(cache=not ap.parse_args().no_cache)
    print(f"hourly: {len(h):,} rows  {h['period'].min()} .. {h['period'].max()}")
    print(f"daily : {len(d):,} rows  {d['period'].min().date()} .. {d['period'].max().date()}")
