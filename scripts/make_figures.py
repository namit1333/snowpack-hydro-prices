"""Generate all README figures from the real analysis results.

Run:  .venv/Scripts/python scripts/make_figures.py
Outputs: results/figures/*.png

Figures:
  pipeline_flowchart.png       end-to-end architecture
  snowpack_timeseries.png      46 years of the April 1 snowpack index
  hydro_vs_snowpack.png        first-stage regression (both hydro sources)
  price_panel_2023_2025.png    prices + snowpack, the available window
  correlation_heatmap.png      Pearson/Spearman matrix
  mediation_diagram.png        Baron-Kenny path diagram with estimated paths
  equations.png                model / DM / mediation equations
  walkforward_forecasts.png    out-of-sample predictions vs actual
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FIG = ROOT / "results" / "figures"

from src.features import build_panel  # noqa: E402
from src.models import (  # noqa: E402
    correlations, first_stage, mediation_analysis, rmse_table, walk_forward,
)

BLUE, RED, GRAY = "#2c6fbb", "#c0392b", "#555555"
GOLD = "#e6a817"


def panel() -> pd.DataFrame:
    return build_panel()


# ------------------------------------------------------------------ helpers
def save(fig, name: str):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote results/figures/{name}")


# ------------------------------------------------------------------ flowchart
def flowchart(p: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(13, 8.2))
    ax.axis("off")

    def box(x, y, w, h, text, fc, ec=GRAY, fs=8.6, tc="black"):
        b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.06",
                           fc=fc, ec=ec, lw=1.4, zorder=2)
        ax.add_patch(b)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, zorder=3, color=tc)

    def arrow(x1, y1, x2, y2, label=None):
        a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                            mutation_scale=14, lw=1.4, color=GRAY, zorder=1)
        ax.add_patch(a)
        if label:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.03, label, ha="center",
                    fontsize=7.4, color=GRAY, style="italic")

    # Stage 1: raw data sources
    box(0.02, 0.62, 0.20, 0.22, "CDEC snow courses\n(SWC, 259 courses)\n1980-2025",
        "#dbe9f7")
    box(0.30, 0.62, 0.20, 0.22, "CAISO OASIS\nDAM LMP, 3 hubs\nfuel mix (5-min)",
        "#dbe9f7")
    box(0.58, 0.62, 0.20, 0.22, "EIA-930\nCISO hydro (WAT)\nhourly + daily",
        "#dbe9f7")
    box(0.86, 0.62, 0.12, 0.22, "EIA_API_KEY\n(env only)", "#f0e6d2")

    # Stage 2: ETL
    box(0.20, 0.34, 0.28, 0.18, "ETL / caching (data/raw/*.csv)\n• UA-header retry (CDEC)  • pagination (EIA)\n• revision dedup  • gap flagging",
        "#fde9d0")

    # Stage 3: features
    box(0.56, 0.34, 0.28, 0.18, "Feature engineering (features.py)\n• % of normal vs 1991-2020 base  • median across courses\n• summer (Jun-Sep) volatility: std of daily-mean LMP",
        "#e8f5e3")

    # Stage 4: modeling
    box(0.20, 0.06, 0.28, 0.18, "Models (models.py)\nbaseline: persistence, trailing-3y mean,\nARIMA(0,1,0)+drift   •   augmented: OLS + ARIMAX(exog)",
        "#e3e7f7")

    # Stage 5: validation
    box(0.56, 0.06, 0.28, 0.18, "Validation protocol\n• expanding-window walk-forward (strict OOS)\n• RMSE / MAE  • Diebold-Mariano (Newey-West HAC)\n• Baron-Kenny mediation + Sobel",
        "#f7e3e7")

    # Deliverables
    box(0.30, -0.24, 0.44, 0.18,
        "Deliverables: panel.csv • results.json • figures\n• executed notebooks • 12 unit tests • README",
        "#f9f2e0", ec=GOLD, fs=9)

    arrow(0.22, 0.62, 0.34, 0.52)
    arrow(0.50, 0.62, 0.42, 0.52)
    arrow(0.78, 0.62, 0.70, 0.52)
    arrow(0.92, 0.62, 0.70, 0.52)
    arrow(0.34, 0.52, 0.56, 0.43, "join on year")
    arrow(0.70, 0.43, 0.48, 0.34, "panel")
    arrow(0.48, 0.24, 0.48, 0.15, "fit / forecast")
    arrow(0.70, 0.24, 0.70, 0.15, "OOS errors")
    arrow(0.52, 0.12, 0.55, -0.06)
    arrow(0.52, -0.15, 0.52, -0.24)

    ax.set_title("snowpack-hydro-prices — analysis architecture", fontsize=13, pad=12)
    save(fig, "pipeline_flowchart.png")


# ------------------------------------------------------------------ snowpack
def snowpack_timeseries(p: pd.DataFrame):
    s = p["snowpack_pct"].dropna()
    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.fill_between(s.index, s.values, 100, where=s.values >= 100,
                    color=BLUE, alpha=0.25)
    ax.fill_between(s.index, s.values, 100, where=s.values < 100,
                    color=RED, alpha=0.25)
    ax.plot(s.index, s.values, color=BLUE, lw=1.8, marker="o", ms=3.5)
    ax.axhline(100, color="black", ls="--", lw=1, label="100% of normal")
    for y, lbl in [(1983, "1983 wet"), (1988, "1988 dry"), (2015, "2015 drought"),
                   (2017, "2017 wet"), (2021, "2021 dry"), (2023, "2023 wet")]:
        ax.annotate(lbl, (y, s.loc[y]), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=7.6, color=GRAY)
    ax.set_xlabel("Year"); ax.set_ylabel("April 1 snow water content (% of normal)")
    ax.set_title("Statewide snowpack index, 1980-2025 — median of 259 snow courses "
                 "vs. each course's 1991-2020 normal")
    ax.legend(loc="lower left", fontsize=8)
    ax.set_ylim(-10, 260)
    save(fig, "snowpack_timeseries.png")


# ------------------------------------------------------------------ hydro
def hydro_vs_snowpack(p: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for col, color, mk, lbl in [("hydro_gwh", BLUE, "o",
                                 "CAISO fuel mix (Large+Small hydro)"),
                                ("hydro_gwh_eia", GOLD, "s",
                                 "EIA-930 CISO (WAT)")]:
        sub = p[[col, "snowpack_pct"]].dropna()
        if sub.empty:
            continue
        ax.scatter(sub["snowpack_pct"], sub[col], color=color, marker=mk,
                   s=55, alpha=0.85, edgecolor="white", label=lbl, zorder=3)
        X = np.column_stack([np.ones(len(sub)), sub["snowpack_pct"]])
        b = np.linalg.lstsq(X, sub[col].values, rcond=None)[0]
        xs = np.linspace(sub["snowpack_pct"].min() - 5, sub["snowpack_pct"].max() + 5, 50)
        ax.plot(xs, b[0] + b[1] * xs, color=color, ls="--", lw=1.4, alpha=0.7)
        for y in sub.index:
            ax.annotate(int(y), (sub.loc[y, "snowpack_pct"], sub.loc[y, col]),
                        textcoords="offset points", xytext=(4, -8), fontsize=7, color=GRAY)
    ax.set_xlabel("April 1 snowpack (% of normal)")
    ax.set_ylabel("Summer hydro generation (GWh, Jun-Sep)")
    ax.set_title("First stage of the causal chain: snowpack → hydro output\n"
                 "(slope ≈ +25 GWh per +1 pp of normal, p = 0.022, n = 8)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    save(fig, "hydro_vs_snowpack.png")


# ------------------------------------------------------------------ prices
def price_panel(p: pd.DataFrame):
    sub = p[["snowpack_pct", "price_vol", "price_mean", "hydro_gwh"]].dropna(
        subset=["price_vol"])
    fig, ax1 = plt.subplots(figsize=(10, 4.8))
    ax1.bar(sub.index.astype(int), sub["snowpack_pct"], color=BLUE, alpha=0.55,
            width=0.45, label="April 1 snowpack (% of normal)")
    ax1.axhline(100, color=BLUE, ls="--", lw=0.8, alpha=0.6)
    ax1.set_ylabel("Snowpack % of normal", color=BLUE)
    ax1.tick_params(axis="y", labelcolor=BLUE)
    ax1.set_ylim(0, 280)
    ax2 = ax1.twinx()
    ax2.plot(sub.index, sub["price_vol"], color=RED, marker="o", lw=2,
             label="Summer price volatility (std of daily $/MWh)")
    ax2.set_ylabel("Price volatility ($/MWh)", color=RED)
    ax2.tick_params(axis="y", labelcolor=RED)
    ax2.set_ylim(0, 32)
    for y in sub.index:
        ax1.annotate(f"{sub.loc[y, 'price_vol']:.1f}", (y, sub.loc[y, "price_vol"]),
                     textcoords="offset points", xytext=(0, 9), ha="center",
                     fontsize=8, color=RED)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)
    ax1.set_title("The available price window (2023-2025): wettest year had the "
                  "highest volatility — hypothesis not supported at n = 3")
    save(fig, "price_panel_2023_2025.png")


# ------------------------------------------------------------------ heatmap
def correlation_heatmap(p: pd.DataFrame):
    corr = correlations(p, min_n=3).set_index("target")
    vars_ = list(corr.index)
    M = np.full((len(vars_), 2), np.nan)
    for i, v in enumerate(vars_):
        M[i, 0] = corr.loc[v, "pearson_r"]
        M[i, 1] = corr.loc[v, "spearman_rho"]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Pearson r", "Spearman ρ"])
    ax.set_yticks(range(len(vars_))); ax.set_yticklabels(vars_)
    for i in range(len(vars_)):
        for j in range(2):
            v = M[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=10,
                        color="white" if abs(v) > 0.5 else "black")
    ax.set_title("Snowpack vs. summer price features (n = 3, illustrative)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="correlation")
    save(fig, "correlation_heatmap.png")


# ------------------------------------------------------------------ mediation
def mediation_diagram(p: pd.DataFrame):
    fs = first_stage(p, mediator="hydro_gwh")
    med = mediation_analysis(p, target="price_vol", mediator="hydro_gwh")
    fig, ax = plt.subplots(figsize=(9.5, 5))
    ax.axis("off")

    def node(x, y, text, w=0.30, h=0.16, fc="#e3e7f7"):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                    boxstyle="round,pad=0.008,rounding_size=0.04",
                                    fc=fc, ec=GRAY, lw=1.3))
        ax.text(x, y, text, ha="center", va="center", fontsize=9.5)

    def edge(x1, y1, x2, y2, label, color=GRAY, ls="-"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=16, lw=1.6, color=color,
                                     linestyle=ls))
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.035, label, ha="center",
                fontsize=8.6, color=color)

    node(0.16, 0.72, "Snowpack\n(April 1 % normal)", fc="#dbe9f7")
    node(0.16, 0.30, "Summer hydro\n(GWh, fuel mix)", fc="#e8f5e3")
    node(0.72, 0.51, "Summer price volatility\n(std of daily $/MWh)", fc="#f7e3e7")

    edge(0.16, 0.64, 0.16, 0.39, f"a-path: +{fs['slope_gwh_per_pct']:.1f} GWh/pp\np = {fs['p_value']:.3f}  (n = {fs['n']})",
         color=BLUE)
    edge(0.31, 0.34, 0.57, 0.44, f"b-path: +{med['b_mediator']:.4f}\n(n = 3, illustrative)",
         color=GOLD)
    edge(0.31, 0.66, 0.57, 0.57, f"c'-path (direct): +{med['direct_effect_cp']:.3f}",
         color=RED, ls="--")
    edge(0.16, 0.80, 0.72, 0.60, f"c-path (total): +{med['total_effect_c']:.3f}\np = {med['p_total']:.3f}",
         color=GRAY, ls=":")

    ax.text(0.44, 0.06,
            "Baron-Kenny decomposition: proportion mediated = 1 − c′/c = "
            f"{med['proportion_mediated']:.2f}\n"
            "Sobel z = "
            f"{med['sobel_z']:.2f} (p = {med['sobel_p']:.2f}) — flagged illustrative at n = 3",
            ha="center", fontsize=8.4, color=GRAY)
    ax.set_xlim(0, 0.9); ax.set_ylim(0, 0.92)
    ax.set_title("Mediation path diagram: does hydro explain the snowpack effect?",
                 fontsize=12)
    save(fig, "mediation_diagram.png")


# ------------------------------------------------------------------ equations
def equations():
    fig = plt.figure(figsize=(11, 5.4))
    fig.patch.set_facecolor("white")
    rows = [
        (r"$\text{Volatility}_t = \frac{1}{N}\sum_{d \in \text{Jun-Sep}} "
         r"\left(\overline{P}_{d} - \overline{\overline{P}}\right)^2$", 0.84),
        (r"$\text{baseline (persistence)}: \ \hat{y}_{t} = y_{t-1} \qquad "
         r"\text{ARIMA}(0,1,0): \ \Delta y_t = \mu + \varepsilon_t$", 0.66),
        (r"$\text{augmented OLS}: \ y_t = \beta_0 + \beta_1 y_{t-1} + "
         r"\beta_2\,\text{snowpack}_t + \varepsilon_t$", 0.48),
        (r"$\text{Diebold--Mariano: } \ \mathrm{DM} = \frac{\bar{d}}{\sqrt{"
         r"\hat{V}_{NW}(\bar{d})}}, \quad d_t = (e_t^{(A)})^2 - (e_t^{(B)})^2$", 0.30),
        (r"$\text{mediation: } \ M_t = a\,\text{snowpack}_t + u_t, \quad "
         r"y_t = c'\,\text{snowpack}_t + b\,M_t + v_t, \quad "
         r"\text{mediated} = 1 - c'/c$", 0.12),
    ]
    for tex, y in rows:
        fig.text(0.5, y, tex, ha="center", va="center", fontsize=14.5)
    fig.text(0.5, 0.955, "Statistical framework", ha="center", fontsize=13,
             weight="bold")
    fig.text(0.5, 0.03,
             "Walk-forward expands the training window each year; every prediction "
             "uses only data available before the target year (strict OOS).",
             ha="center", fontsize=8.5, color=GRAY)
    save(fig, "equations.png")


# ------------------------------------------------------------------ forecasts
def walkforward_forecasts(p: pd.DataFrame):
    wf = walk_forward(p, target="price_vol", min_train=6)
    if wf.empty:
        print("  (no walk-forward rows; skipping forecast figure)")
        return
    fig, ax = plt.subplots(figsize=(10, 4.6))
    for model, color, ls in [("baseline_naive", "#999999", "--"),
                             ("baseline_mean3", "#bbbbbb", ":"),
                             ("augmented_ols", RED, "-")]:
        m = wf[wf["model"] == model].set_index("year")
        ax.plot(m.index, m["pred"], marker=".", ls=ls, color=color, lw=1.6,
                label=model)
    actual = wf[wf["model"] == "augmented_ols"].set_index("year")["actual"]
    ax.plot(actual.index, actual, marker="o", color="black", lw=2.2,
            label="actual")
    ax.set_title("Walk-forward out-of-sample forecasts vs actual (n = 1 held-out "
                 "year — illustrative)")
    ax.set_ylabel("Summer price volatility ($/MWh)"); ax.set_xlabel("Year")
    ax.legend(fontsize=8.5); ax.grid(alpha=0.25)
    save(fig, "walkforward_forecasts.png")


def main():
    print("Generating figures from real results ...")
    p = panel()
    flowchart(p)
    snowpack_timeseries(p)
    hydro_vs_snowpack(p)
    price_panel(p)
    correlation_heatmap(p)
    mediation_diagram(p)
    equations()
    walkforward_forecasts(p)
    print("done.")


if __name__ == "__main__":
    main()
