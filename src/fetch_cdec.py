"""Fetch California snowpack data from CDEC (California Data Exchange Center).

Source of truth: the official California Natural Resources Agency open-data mirror,
which exposes CDEC's *monthly snow course* measurements as CSVs:

    https://data.cnra.ca.gov/dataset/california-snow-data
    https://cdec.water.ca.gov/dynamicapp/req/CSVGroupServlet?GroupIds=SCX&Start=...&End=...

Why snow courses (not daily sensors)?
- Snow courses are surveyed ~monthly (Jan-Jun) every year, including the
  canonical April 1 measurement used by DWR as the seasonal hydrological benchmark.
- Sensor 3 = SNOW WC (snow water content, inches) is the standard snowpack measure.

NOTE: the CDEC server resets connections when the request has no User-Agent header,
so all requests here send one.

CLI:  python src/fetch_cdec.py --start 1980 --end 2025
Output: data/raw/snow_courses.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import requests

BASE = "https://cdec.water.ca.gov/dynamicapp/req/CSVGroupServlet"
GROUP = "SCX"  # monthly snow course water content + depth
SENSOR_SWC = 3  # snow water content (inches)
HEADERS = {"User-Agent": "snowpack-hydro-prices research script (educational)"}
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"


def _download_range(start: str, end: str, session: requests.Session) -> pd.DataFrame:
    """Download one SCX range; returns a DataFrame or empty on failure."""
    params = {"GroupIds": GROUP, "Start": start, "End": end, "download": "no"}
    r = session.get(BASE, params=params, headers=HEADERS, timeout=120)
    r.raise_for_status()
    if not r.text.strip():
        return pd.DataFrame()
    return pd.read_csv(pd.io.common.StringIO(r.text))


def fetch_snow_courses(start_year: int, end_year: int, cache: bool = True) -> pd.DataFrame:
    """Download monthly snow course data for [start_year, end_year].

    Chunks by decade to keep individual requests small. Returns a DataFrame with
    columns: station_id, date, swc_in (inches). Sensor 3 rows only.
    """
    frames = []
    with requests.Session() as s:
        for lo in range(start_year, end_year + 1, 10):
            hi = min(lo + 9, end_year)
            print(f"  CDEC SCX {lo}-{hi} ...", flush=True)
            df = _download_range(f"{lo}-01-01", f"{hi}-12-31", s)
            if df.empty:
                print(f"  (empty response for {lo}-{hi})")
                continue
            frames.append(df)
    if not frames:
        raise RuntimeError("CDEC returned no snow course data for the requested years")
    raw = pd.concat(frames, ignore_index=True)

    # Normalize column names (keep both historical spellings)
    raw = raw.rename(columns={
        "STATION_ID": "station_id",
        "ACTUAL_DATE": "date",
        "VALUE": "value",
        "SENSOR_NUM": "sensor",
    })
    raw = raw[raw["sensor"] == SENSOR_SWC].copy()
    raw["date"] = pd.to_datetime(raw["date"].astype(str).str[:8], format="%Y%m%d")
    out = raw[["station_id", "date", "value"]].dropna().copy()
    out["value"] = pd.to_numeric(out["value"], errors="coerce")

    if cache:
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        out.to_csv(RAW_DIR / "snow_courses.csv", index=False)
        print(f"  cached {len(out):,} rows -> data/raw/snow_courses.csv")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Fetch CDEC monthly snow course SWC")
    ap.add_argument("--start", type=int, default=1980)
    ap.add_argument("--end", type=int, default=2025)
    args = ap.parse_args()
    df = fetch_snow_courses(args.start, args.end)
    print(df.head())
    print(f"total rows: {len(df):,} | courses: {df['station_id'].nunique()}")
