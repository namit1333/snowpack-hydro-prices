"""Generate the project's notebooks with nbformat.

Run:  python scripts/make_notebooks.py
Then execute them (once data/processed/panel.csv exists):
  jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"

SETUP = """import sys
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
sys.path.insert(0, str(ROOT))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 110"""

HEADER = """# Snowpack -> Hydropower -> Electricity Prices

**Question:** Does winter/spring California snowpack predict summer electricity
price behavior through hydroelectric generation?

**Panel (data/processed/panel.csv):** one row per year --
`snowpack_pct` (April 1 snow water content as % of normal), summer day-ahead
price features (`price_mean`, `price_peak`, `price_vol`, `price_vol_hourly`),
and summer hydro generation (`hydro_gwh`, `hydro_gwh_eia`)."""


def code(src: str) -> dict:
    return nbf.v4.new_code_cell(src)


def md(src: str) -> dict:
    return nbf.v4.new_markdown_cell(src)


def build_eda() -> nbf.NotebookNode:
    cells = [
        md(HEADER),
        md("## 0. Setup"),
        code(SETUP),
        md("## 1. Load the yearly panel"),
        code("""panel = pd.read_csv(ROOT / 'data' / 'processed' / 'panel.csv', index_col=0)
panel"""),
        md("## 2. Snowpack index over time"),
        code("""fig, ax = plt.subplots(figsize=(11, 4))
panel['snowpack_pct'].plot(ax=ax, marker='o', color='#4c9be8')
ax.axhline(100, color='gray', ls='--', lw=1)
ax.set_title('April 1 snow water content, % of normal (median across snow courses)')
ax.set_ylabel('% of normal')
plt.tight_layout(); plt.show()"""),
        md("## 3. Snowpack vs summer price volatility"),
        code("""from src.models import correlations

corr = correlations(panel)
print(corr.to_string(index=False))

fig, ax = plt.subplots(figsize=(7, 5))
sub = panel.dropna(subset=['snowpack_pct', 'price_vol'])
ax.scatter(sub['snowpack_pct'], sub['price_vol'], s=60, alpha=0.8)
ax.set_xlabel('April 1 snowpack (% of normal)')
ax.set_ylabel('Summer price volatility (std of daily $/MWh)')
ax.set_title('Each point is one summer')
plt.tight_layout(); plt.show()"""),
        md("""> EDA takeaway: fill in after running -- does wet years line up with calm
> summers? (Check the sign of `pearson_r` and the scatter direction.)"""),
    ]
    return nbf.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}})


def build_mediation() -> nbf.NotebookNode:
    cells = [
        md(HEADER),
        md("## Setup"),
        code(SETUP),
        md("## Mediation analysis: snowpack -> hydro output -> price volatility"),
        code("""from src.models import mediation_analysis, first_stage

panel = pd.read_csv(ROOT / 'data' / 'processed' / 'panel.csv', index_col=0)

# primary mediator: CAISO-reported summer hydro generation (no API key needed)
med = mediation_analysis(panel, target='price_vol', mediator='hydro_gwh')
med"""),
        md("### Reading the paths (Baron-Kenny regression mediation)"),
        code("""import numpy as np
rows = {
    'total effect (c): snowpack -> volatility':  med['total_effect_c'],
    'a path: snowpack -> hydro':                 med['a_path'],
    'b path: hydro -> volatility (joint)':       med['b_mediator'],
    "direct effect (c'), snowpack | hydro":      med['direct_effect_cp'],
    "proportion mediated (1 - c'/c)":           med['proportion_mediated'],
}
for k, v in rows.items():
    print(f'{k:48s} {v:+.3f}')
print(f"\\nSobel test of indirect effect: z={med['sobel_z']:.2f}, "
      f"p={med['sobel_p']:.3f} (n={med['n']})")"""),
        md("""**Causal-chain check:** if the snowpack effect runs *through* hydro,
then adding hydro to the regression should shrink the direct snowpack
coefficient (`c'` close to 0) while hydro keeps a significant `b` coefficient,
and the proportion mediated should be large.

> Mediation takeaway: fill in after running.""") ,
    ]
    return nbf.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}})


def build_walkforward() -> nbf.NotebookNode:
    cells = [
        md(HEADER),
        md("## Setup"),
        code(SETUP),
        md("## Walk-forward backtest (expanding window, strictly out-of-sample)"),
        code("""from src.models import walk_forward, rmse_table, dm_test, first_stage

panel = pd.read_csv(ROOT / 'data' / 'processed' / 'panel.csv', index_col=0)

wf = walk_forward(panel, target='price_vol', min_train=6)
rmse_table(wf)"""),
        md("### Diebold-Mariano test: augmented vs each baseline (squared errors)"),        code("""for base in ['baseline_naive', 'baseline_mean3', 'baseline_arima']:
    a = wf[wf.model == 'augmented_ols'].set_index('year').error
    b = wf[wf.model == base].set_index('year').error
    both = a.index.intersection(b.index)
    d = dm_test(a.loc[both].values, b.loc[both].values)
    print(f'{base:18s} DM={d["dm"]:+.3f}  p={d["p"]:.3f}  n={d["n"]}')

# The first stage of the causal chain has far more data (2018-2025):
print()
print('First stage: snowpack -> summer hydro output (full record)')
print(first_stage(panel, mediator='hydro_gwh'))"""),
        md("### Forecasts vs actual"),
        code("""fig, ax = plt.subplots(figsize=(11, 4.5))
for model, color in [('baseline_naive', '#888888'), ('baseline_mean3', '#b0b0b0'),
                     ('augmented_ols', '#c0392b')]:
    m = wf[wf.model == model].set_index('year')
    ax.plot(m.index, m.pred, ls='--', marker='.', color=color, label=model)
actual = wf[wf.model == 'augmented_ols'].set_index('year').actual
ax.plot(actual.index, actual, ls='-', marker='o', color='black', lw=2, label='actual')
ax.set_title('Out-of-sample summer price volatility forecasts')
ax.legend(); plt.tight_layout(); plt.show()"""),
        md("""> Backtest takeaway: fill in after running -- did adding snowpack lower
> out-of-sample RMSE vs persistence? Is the DM test significant?"""),
    ]
    return nbf.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}})


def main() -> None:
    NB_DIR.mkdir(exist_ok=True)
    (NB_DIR / "01_eda.ipynb").write_text(nbf.writes(build_eda(), version=4))
    (NB_DIR / "02_mediation_analysis.ipynb").write_text(nbf.writes(build_mediation(), version=4))
    (NB_DIR / "03_walkforward_backtest.ipynb").write_text(nbf.writes(build_walkforward(), version=4))
    print(f"wrote notebooks to {NB_DIR}")


if __name__ == "__main__":
    main()
