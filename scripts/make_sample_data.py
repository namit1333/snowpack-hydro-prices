"""Generate a small demo dataset (data/sample/) from the real panel.

Lets someone clone the repo and run the analysis *without* downloading the
full raw data or configuring API keys. The sample is a 6-year slice with the
real snowpack, price, and hydro values so results are meaningful.

Run:  .venv/Scripts/python scripts/make_sample_data.py
Output: data/sample/sample_panel.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features import build_panel  # noqa: E402

SAMPLE_YEARS = [2016, 2017, 2018, 2023, 2024, 2025]
COLS = ["snowpack_pct", "price_mean", "price_peak", "price_vol",
        "price_vol_hourly", "hydro_gwh", "temp_mean_c"]


def main() -> None:
    panel = build_panel()
    sample = panel.loc[SAMPLE_YEARS, [c for c in COLS if c in panel.columns]]
    out_dir = ROOT / "data" / "sample"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sample_panel.csv"
    sample.to_csv(out_path)
    print(f"wrote {out_path}")
    print(sample.round(2).to_string())


if __name__ == "__main__":
    main()
