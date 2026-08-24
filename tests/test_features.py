"""Unit tests for src/features.py using synthetic data."""

import numpy as np
import pandas as pd
import pytest

from src import features


def _snow_df(years=(1991, 2020), courses=("AAA", "BBB"), base=50.0):
    rows = []
    for c in courses:
        for y in range(years[0], years[1] + 1):
            for d in (1, 2):  # April 1 and April 2 (to exercise day tolerance)
                rows.append({"station_id": c, "date": pd.Timestamp(y, 4, d),
                             "value": base})
    return pd.DataFrame(rows)


def test_april1_snowpack_index_median_and_normal():
    snow = _snow_df()
    idx = features.april1_snowpack_index(snow)
    # All courses identical and constant -> index should be exactly 100% each year
    assert idx.between(99.5, 100.5).all()
    assert idx.index.min() == 1991 and idx.index.max() == 2020


def test_april1_index_uses_april1_not_other_months():
    rows = []
    for y in (2010, 2011, 2012):
        rows.append({"station_id": "X", "date": pd.Timestamp(y, 4, 1), "value": 50.0})
    # a May reading must be ignored
    rows.append({"station_id": "X", "date": pd.Timestamp(2010, 5, 1), "value": 999.0})
    idx = features.april1_snowpack_index(pd.DataFrame(rows))
    assert (idx == 100.0).all()


def test_summer_price_features():
    times = pd.date_range("2020-06-01", periods=120, freq="h", tz="UTC")
    hub = np.repeat(["TH_NP15_GEN-APND"], len(times))
    price = np.linspace(20, 80, len(times))
    prices = pd.DataFrame({"time": times, "hub": hub, "price": price})
    out = features.summer_price_features(prices)
    assert len(out) == 1 and out.iloc[0]["year"] == 2020
    assert out.iloc[0]["price_mean"] == pytest.approx(50.0)
    # price_peak = max *daily-mean* price, so it must sit between mean and the
    # absolute hourly peak (80)
    assert out.iloc[0]["price_mean"] < out.iloc[0]["price_peak"] < 80.0
    assert out.iloc[0]["price_vol"] > 0
    assert out.iloc[0]["price_vol_hourly"] > 0


def test_summer_price_features_drops_non_summer():
    times = pd.date_range("2020-01-01", periods=48, freq="h", tz="UTC")
    prices = pd.DataFrame({"time": times, "hub": "H", "price": 30.0})
    out = features.summer_price_features(prices)
    assert len(out) == 1 and out.iloc[0]["price_vol"] == 0.0  # constant series


def test_summer_hydro_gwh():
    # 1 MW for one hour = 1 MWh; 12 five-minute intervals of 1 MW = 1 MWh
    t = pd.date_range("2020-06-01", periods=12, freq="5min", tz="UTC")
    fm = pd.DataFrame({"Time": t, "Large Hydro": 1.0, "Small Hydro": 1.0})
    out = features.summer_hydro(fm)
    assert out.iloc[0]["year"] == 2020
    assert out.iloc[0]["hydro_gwh"] == pytest.approx(2 * 12 * 5 / 60 / 1000)


def test_summer_hydro_empty():
    out = features.summer_hydro(pd.DataFrame())
    assert out.empty and "hydro_gwh" in out.columns


def test_build_monthly_panel_schema(tmp_path, monkeypatch):
    # Point RAW_DIR at a temp dir with a minimal price file and confirm the
    # monthly panel has the right shape (one row per year-month).
    raw = tmp_path / "raw"
    raw.mkdir()
    times = pd.date_range("2020-06-01", periods=48, freq="h", tz="UTC")
    prices = pd.DataFrame({"time": times, "hub": "H", "price": np.linspace(20, 80, 48)})
    prices.to_csv(raw / "caiso_prices_2020.csv", index=False)
    monkeypatch.setattr(features, "RAW_DIR", raw)
    out = features.build_monthly_panel()
    assert not out.empty
    assert {"year", "month", "price_vol"}.issubset(out.columns)
    assert out["month"].isin([6, 7, 8, 9]).all()


def test_load_prices_no_duplicate_timestamps(tmp_path, monkeypatch):
    # The price loader must never return duplicate (timestamp, hub) rows,
    # which would double-count a hub-hour in the volatility aggregation.
    raw = tmp_path / "raw"
    raw.mkdir()
    times = pd.date_range("2020-06-01", periods=24, freq="h", tz="UTC")
    # deliberately include one duplicated (time, hub) pair
    df = pd.concat([
        pd.DataFrame({"time": times, "hub": "H1", "price": np.linspace(20, 80, 24)}),
        pd.DataFrame({"time": times[:12], "hub": "H1", "price": np.linspace(20, 80, 12)}),
    ], ignore_index=True)
    df.to_csv(raw / "caiso_prices_2020.csv", index=False)
    monkeypatch.setattr(features, "RAW_DIR", raw)
    loaded = features._load_prices()
    dupes = loaded.duplicated(subset=["time", "hub"]).sum()
    assert dupes == 0
    # both years/rows present means concat worked
    assert len(loaded) >= 24


def test_temperature_summary(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    days = pd.date_range("2020-06-01", periods=30, freq="D")
    t = pd.DataFrame({"date": days, "temp_max_c": np.concatenate([np.full(25, 35.0), np.full(5, 40.0)])})
    t.to_csv(raw / "heat_temperature_2020.csv", index=False)
    from src import fetch_controls as fc
    monkeypatch.setattr(fc, "RAW_DIR", raw)
    d = pd.read_csv(raw / "heat_temperature_2020.csv", parse_dates=["date"])
    summ = fc.temperature_summary(d)
    assert summ.iloc[0]["temp_mean_c"] == pytest.approx(35.833, abs=0.01)
    assert summ.iloc[0]["heat_days_38c"] == 5
