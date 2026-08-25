"""Reproducibility verification: run the analysis twice on the sample dataset
and assert the outputs are bit-identical.

CI runs this after the unit tests, so the badge means not just "tests pass"
but "the pipeline is deterministically executable end-to-end".

Run:  python scripts/check_reproducibility.py   (after scripts/make_sample_data.py)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models import run_analysis, walk_forward  # noqa: E402

SAMPLE = ROOT / "data" / "sample" / "sample_panel.csv"


def _canonical(res: dict, wf: pd.DataFrame) -> str:
    frames = {k: v.to_json(double_precision=12)
              for k, v in sorted(res.items()) if isinstance(v, pd.DataFrame)}
    plain = {k: v for k, v in res.items() if not isinstance(v, pd.DataFrame)}
    return json.dumps({"plain": plain, "frames": frames},
                      sort_keys=True, default=str)


def main() -> None:
    if not SAMPLE.exists():
        raise SystemExit(f"missing {SAMPLE} -- run scripts/make_sample_data.py first")
    panel = pd.read_csv(SAMPLE).set_index(
        pd.read_csv(SAMPLE).columns[0]).rename_axis("year")

    wf1 = walk_forward(panel, min_train=3)
    wf2 = walk_forward(panel, min_train=3)
    assert wf1.to_json(double_precision=12) == wf2.to_json(double_precision=12), \
        "walk_forward is not deterministic"

    res1 = _canonical(run_analysis(panel), wf1)
    res2 = _canonical(run_analysis(panel), wf2)
    assert res1 == res2, "run_analysis is not deterministic"

    n_keys = len(json.loads(res1)["plain"])
    print(f"reproducibility check PASSED: {n_keys} result blocks + "
          f"{len(wf1)} walk-forward rows identical across two runs")


if __name__ == "__main__":
    main()
