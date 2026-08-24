"""End-to-end pipeline: fetch -> features -> models -> plots + results.

    python src/run_pipeline.py            # uses cached data only
    python src/run_pipeline.py --fetch    # (re)fetch raw data first

Outputs:
    data/raw/      raw CDEC + CAISO (+EIA) data
    data/processed/panel.csv
    results/snowpack_vs_volatility.png    the headline plot
    results/walkforward_rmse.csv          RMSE table (baseline vs augmented)
    results/results.json                  correlations, DM tests, mediation
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.features import build_panel
from src.models import run_analysis

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def make_plot(panel: pd.DataFrame, wf: pd.DataFrame) -> str:
    """Snowpack (% of normal) vs summer price volatility by year + predictions."""
    fig, ax1 = plt.subplots(figsize=(10, 5.5))

    sub = panel[["snowpack_pct", "price_vol"]].dropna()
    ax1.bar(sub.index.astype(int), sub["snowpack_pct"],
            color="#4c9be8", alpha=0.75, label="April 1 snowpack (% of normal)")
    ax1.axhline(100, color="#4c9be8", ls="--", lw=0.8, alpha=0.6)
    ax1.set_ylabel("Snowpack % of normal", color="#4c9be8")
    ax1.set_xlabel("Year")
    ax1.tick_params(axis="y", labelcolor="#4c9be8")

    ax2 = ax1.twinx()
    ax2.plot(sub.index, sub["price_vol"], color="#c0392b", marker="o",
             lw=1.6, label="Summer price volatility (std of daily $/MWh)")
    ax2.set_ylabel("Price volatility ($/MWh)", color="#c0392b")
    ax2.tick_params(axis="y", labelcolor="#c0392b")

    # overlay out-of-sample predictions of the augmented model
    if wf is not None and not wf.empty:
        aug = wf[wf["model"] == "augmented_ols"].set_index("year")
        ax2.plot(aug.index, aug["pred"], color="black", ls=":", marker="x",
                 label="Augmented OOS forecast (walk-forward)")
    ax2.legend(loc="upper left", fontsize=8)
    ax1.set_title("California snowpack vs summer electricity price volatility")

    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "snowpack_vs_volatility.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def main(fetch: bool = False) -> None:
    if fetch:
        from src.fetch_cdec import fetch_snow_courses
        from src.fetch_caiso_oasis import fetch_historical_prices, fetch_year
        print("Fetching CDEC snow courses 1980-2025 ...")
        fetch_snow_courses(1980, 2025)
        print("Fetching CAISO historical LMP (2016-2018 archive) ...")
        try:
            fetch_historical_prices()
        except Exception as exc:
            print(f"  (historical LMP unavailable: {exc})")
        print("Fetching CAISO summer prices + fuel mix 2019-2025 ...")
        for year in range(2019, 2026):
            fetch_year(year)
        print("Fetching control variables (temperature, demand, gas) ...")
        try:
            from src.fetch_controls import fetch_demand, fetch_gas, fetch_temperature
            for year in range(2016, 2026):
                fetch_temperature(year)
                try:
                    fetch_demand(year)
                except Exception as exc:
                    print(f"  (demand {year} unavailable: {exc})")
                try:
                    fetch_gas(year)
                except Exception as exc:
                    print(f"  (gas {year} unavailable: {exc})")
        except Exception as exc:
            print(f"  (controls unavailable: {exc})")

    print("Building panel ...")
    panel = build_panel()
    print(panel.dropna(how="all").to_string())

    print("Running analysis ...")
    res = run_analysis(panel)

    wf = res.get("wf_price_vol", pd.DataFrame())
    plot_path = make_plot(panel, wf)
    print(f"plot -> {plot_path}")

    rmse = res.get("rmse_price_vol")
    if rmse is not None:
        rmse.to_csv(RESULTS / "walkforward_rmse.csv")

    # serialize what json can hold
    serializable = {k: v for k, v in res.items()
                    if not isinstance(v, pd.DataFrame)}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "results.json").write_text(
        json.dumps(serializable, indent=2, default=str))
    print(f"results -> {RESULTS / 'results.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Snowpack->hydro->prices pipeline")
    ap.add_argument("--fetch", action="store_true", help="(re)download raw data")
    main(fetch=ap.parse_args().fetch)
