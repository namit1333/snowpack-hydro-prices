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
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.axis("off")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.35, 1.0)

    def box(x, y, w, h, text, fc, ec=GRAY, fs=8.6, tc="black"):
        b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.06",
                           fc=fc, ec=ec, lw=1.4, zorder=2)
        ax.add_patch(b)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, zorder=3, color=tc)

    def arrow(x1, y1, x2, y2, label=None, label_offset=(0, 0.03)):
        a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                            mutation_scale=14, lw=1.4, color=GRAY, zorder=1)
        ax.add_patch(a)
        if label:
            mx = (x1 + x2) / 2 + label_offset[0]
            my = (y1 + y2) / 2 + label_offset[1]
            ax.text(mx, my, label, ha="center", fontsize=7.4, color=GRAY, style="italic")

    # Stage 1: raw data sources (top row, evenly spaced)
    box(0.02, 0.72, 0.21, 0.18, "CDEC snow courses\n(SWC, 259 courses)\n1980-2025",
        "#dbe9f7")
    box(0.28, 0.72, 0.21, 0.18, "CAISO OASIS\nDAM LMP, 3 hubs\nfuel mix (5-min)",
        "#dbe9f7")
    box(0.54, 0.72, 0.21, 0.18, "EIA-930\nCISO hydro (WAT)\nhourly + daily",
        "#dbe9f7")
    box(0.80, 0.72, 0.14, 0.18, "EIA_API_KEY\n(env only)", "#f0e6d2")

    # Stage 2: ETL (middle-left)
    box(0.08, 0.40, 0.35, 0.18, "ETL / caching (data/raw/*.csv)\n"
        "\u2022 UA-header retry (CDEC)  \u2022 pagination (EIA)\n"
        "\u2022 revision dedup  \u2022 gap flagging",
        "#fde9d0")

    # Stage 3: features (middle-right)
    box(0.55, 0.40, 0.35, 0.18, "Feature engineering (features.py)\n"
        "\u2022 % of normal vs 1991-2020 base  \u2022 median across courses\n"
        "\u2022 summer (Jun-Sep) volatility: std of daily-mean LMP",
        "#e8f5e3")

    # Stage 4: modeling (bottom-left)
    box(0.08, 0.05, 0.35, 0.18, "Models (models.py)\n"
        "baseline: persistence, trailing-3y mean,\n"
        "ARIMA(0,1,0)+drift   \u2022   augmented: OLS + ARIMAX(exog)",
        "#e3e7f7")

    # Stage 5: validation (bottom-right)
    box(0.55, 0.05, 0.35, 0.18, "Validation protocol\n"
        "\u2022 expanding-window walk-forward (strict OOS)\n"
        "\u2022 RMSE / MAE  \u2022 Diebold-Mariano (Newey-West HAC)\n"
        "\u2022 Baron-Kenny mediation + Sobel",
        "#f7e3e7")

    # Deliverables (bottom center)
    box(0.20, -0.28, 0.58, 0.15,
        "Deliverables: panel.csv \u2022 results.json \u2022 figures\n"
        "\u2022 executed notebooks \u2022 12 unit tests \u2022 README",
        "#f9f2e0", ec=GOLD, fs=9)

    # Arrows: data sources -> ETL
    arrow(0.125, 0.72, 0.255, 0.58)
    arrow(0.385, 0.72, 0.355, 0.58)
    arrow(0.645, 0.72, 0.455, 0.58)
    arrow(0.87, 0.72, 0.555, 0.58, label_offset=(0.05, 0.03))

    # Arrows: ETL -> Features
    arrow(0.43, 0.49, 0.55, 0.49, "join on year")

    # Arrows: Features -> Models (via panel)
    arrow(0.35, 0.40, 0.255, 0.23, "panel")

    # Arrows: Features -> Validation
    arrow(0.725, 0.40, 0.725, 0.23, "OOS errors")

    # Arrows: Models + Validation -> Deliverables
    arrow(0.255, 0.05, 0.35, -0.13)
    arrow(0.725, 0.05, 0.63, -0.13)

    ax.set_title("snowpack-hydro-prices \u2014 analysis architecture", fontsize=14, pad=15)
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
    for y, lbl, offset in [(1983, "1983 wet", (0, 8)), (1988, "1988 dry", (0, -12)),
                           (2015, "2015 drought", (0, -12)), (2017, "2017 wet", (0, 8)),
                           (2021, "2021 dry", (0, -12)), (2023, "2023 wet", (0, 8))]:
        if y in s.index:
            ax.annotate(lbl, (y, s.loc[y]), textcoords="offset points",
                        xytext=offset, ha="center", fontsize=7.6, color=GRAY)
    ax.set_xlabel("Year"); ax.set_ylabel("April 1 snow water content (% of normal)")
    ax.set_title("Statewide snowpack index, 1980-2025 \u2014 median of 259 snow courses "
                 "vs. each course's 1991-2020 normal")
    ax.legend(loc="lower left", fontsize=8)
    ax.set_ylim(-10, 260)
    save(fig, "snowpack_timeseries.png")


# ------------------------------------------------------------------ hydro
def hydro_vs_snowpack(p: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for col, color, mk, lbl in [("hydro_gwh", BLUE, "o",
                                 "CAISO fuel mix (Large+Small hydro)"),
                                ("hydro_gwh_eia", GOLD, "s",
                                 "EIA-930 CISO (WAT)")]:
        sub = p[[col, "snowpack_pct"]].dropna()
        if sub.empty:
            continue
        ax.scatter(sub["snowpack_pct"], sub[col], color=color, marker=mk,
                   s=70, alpha=0.85, edgecolor="white", label=lbl, zorder=3)
        X = np.column_stack([np.ones(len(sub)), sub["snowpack_pct"]])
        b = np.linalg.lstsq(X, sub[col].values, rcond=None)[0]
        xs = np.linspace(sub["snowpack_pct"].min() - 5, sub["snowpack_pct"].max() + 5, 50)
        ax.plot(xs, b[0] + b[1] * xs, color=color, ls="--", lw=1.4, alpha=0.7)

        # Smart label placement to avoid overlap
        sorted_idx = sub.sort_values("snowpack_pct").index
        offsets = []
        for i, y in enumerate(sorted_idx):
            # Alternate above/below, shift left/right for close points
            if i > 0 and abs(sub.loc[y, "snowpack_pct"] - sub.loc[sorted_idx[i-1], "snowpack_pct"]) < 15:
                offsets.append((6, -10))
            else:
                offsets.append((5, 7))

        for i, y in enumerate(sub.index):
            ox, oy = offsets[i] if i < len(offsets) else (5, 7)
            ax.annotate(str(int(y)), (sub.loc[y, "snowpack_pct"], sub.loc[y, col]),
                        textcoords="offset points", xytext=(ox, oy), fontsize=8, color=GRAY,
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="none"))

    ax.set_xlabel("April 1 snowpack (% of normal)", fontsize=11)
    ax.set_ylabel("Summer hydro generation (GWh, Jun-Sep)", fontsize=11)
    ax.set_title("First stage of the causal chain: snowpack \u2192 hydro output\n"
                 "(slope \u2248 +25 GWh per +1 pp of normal, p = 0.022, n = 8)", fontsize=12)
    ax.legend(fontsize=9, loc="upper left")
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
    ax1.set_ylabel("Snowpack % of normal", color=BLUE, fontsize=11)
    ax1.tick_params(axis="y", labelcolor=BLUE)
    ax1.set_ylim(0, 280)
    ax1.set_xlabel("Year", fontsize=11)
    ax2 = ax1.twinx()
    ax2.plot(sub.index, sub["price_vol"], color=RED, marker="o", lw=2,
             label="Summer price volatility (std of daily $/MWh)")
    ax2.set_ylabel("Price volatility ($/MWh)", color=RED, fontsize=11)
    ax2.tick_params(axis="y", labelcolor=RED)
    ax2.set_ylim(0, 32)
    for y in sub.index:
        ax1.annotate(f"{sub.loc[y, 'price_vol']:.1f}", (y, sub.loc[y, "price_vol"]),
                     textcoords="offset points", xytext=(0, 10), ha="center",
                     fontsize=9, color=RED, fontweight="bold")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8, framealpha=0.9)
    ax1.set_title("The available price window (2023-2025): wettest year had the "
                  "highest volatility \u2014 hypothesis not supported at n = 3", fontsize=11)
    save(fig, "price_panel_2023_2025.png")


# ------------------------------------------------------------------ heatmap
def correlation_heatmap(p: pd.DataFrame):
    corr = correlations(p, min_n=3).set_index("target")
    vars_ = list(corr.index)
    M = np.full((len(vars_), 2), np.nan)
    for i, v in enumerate(vars_):
        M[i, 0] = corr.loc[v, "pearson_r"]
        M[i, 1] = corr.loc[v, "spearman_rho"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Pearson r", "Spearman \u03c1"], fontsize=11)
    ax.set_yticks(range(len(vars_))); ax.set_yticklabels(vars_, fontsize=10)
    for i in range(len(vars_)):
        for j in range(2):
            v = M[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=12,
                        color="white" if abs(v) > 0.5 else "black", fontweight="bold")
    ax.set_title("Snowpack vs. summer price features (n = 3, illustrative)", fontsize=12)
    fig.colorbar(im, ax=ax, shrink=0.8, label="correlation")
    save(fig, "correlation_heatmap.png")


# ------------------------------------------------------------------ mediation
def mediation_diagram(p: pd.DataFrame):
    fs = first_stage(p, mediator="hydro_gwh")
    med = mediation_analysis(p, target="price_vol", mediator="hydro_gwh")
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.axis("off")

    def node(x, y, text, w=0.28, h=0.14, fc="#e3e7f7"):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                    boxstyle="round,pad=0.008,rounding_size=0.04",
                                    fc=fc, ec=GRAY, lw=1.3, zorder=2))
        ax.text(x, y, text, ha="center", va="center", fontsize=10, zorder=3)

    def edge(x1, y1, x2, y2, label, color=GRAY, ls="-", label_pos="above"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=18, lw=1.8, color=color,
                                     linestyle=ls, zorder=1))
        mx = (x1 + x2) / 2
        my = (y1 + y2) / 2
        if label_pos == "above":
            ax.text(mx, my + 0.04, label, ha="center", fontsize=9, color=color, fontweight="bold")
        elif label_pos == "below":
            ax.text(mx, my - 0.04, label, ha="center", fontsize=9, color=color, fontweight="bold")
        elif label_pos == "left":
            ax.text(mx - 0.02, my, label, ha="right", fontsize=9, color=color, fontweight="bold")
        else:
            ax.text(mx + 0.02, my, label, ha="left", fontsize=9, color=color, fontweight="bold")

    # Nodes - positioned with more spacing
    node(0.22, 0.75, "Snowpack\n(April 1 % normal)", w=0.30, h=0.14, fc="#dbe9f7")
    node(0.22, 0.25, "Summer hydro\n(GWh, fuel mix)", w=0.30, h=0.14, fc="#e8f5e3")
    node(0.78, 0.50, "Summer price volatility\n(std of daily $/MWh)", w=0.32, h=0.14, fc="#f7e3e7")

    # a-path: Snowpack -> Hydro (vertical, left side)
    edge(0.22, 0.68, 0.22, 0.32,
         f"a-path: +{fs['slope_gwh_per_pct']:.1f} GWh/pp\np = {fs['p_value']:.3f}  (n = {fs['n']})",
         color=BLUE, label_pos="left")

    # b-path: Hydro -> Price Volatility (diagonal up-right)
    edge(0.37, 0.30, 0.62, 0.45,
         f"b-path: +{med['b_mediator']:.4f}\n(n = 3, illustrative)",
         color=GOLD, label_pos="below")

    # c'-path: Snowpack -> Price Volatility direct (horizontal)
    edge(0.37, 0.70, 0.62, 0.55,
         f"c'-path (direct): +{med['direct_effect_cp']:.3f}",
         color=RED, ls="--", label_pos="above")

    # c-path: total effect (curved arrow above)
    ax.annotate("", xy=(0.78, 0.57), xytext=(0.22, 0.82),
                arrowprops=dict(arrowstyle="-|>", color=GRAY, ls=":",
                               lw=1.5, connectionstyle="arc3,rad=0.15"))
    ax.text(0.50, 0.88, f"c-path (total): +{med['total_effect_c']:.3f}\np = {med['p_total']:.3f}",
            ha="center", fontsize=9, color=GRAY, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor=GRAY))

    # Baron-Kenny summary at bottom
    ax.text(0.50, 0.06,
            "Baron-Kenny decomposition: proportion mediated = 1 \u2212 c\u2032/c = "
            f"{med['proportion_mediated']:.2f}\n"
            "Sobel z = "
            f"{med['sobel_z']:.2f} (p = {med['sobel_p']:.2f}) \u2014 flagged illustrative at n = 3",
            ha="center", fontsize=9.5, color=GRAY,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#f9f9f9", alpha=0.9, edgecolor=GRAY))

    ax.set_xlim(0, 1.0); ax.set_ylim(0, 0.98)
    ax.set_title("Mediation path diagram: does hydro explain the snowpack effect?",
                 fontsize=13, pad=10)
    save(fig, "mediation_diagram.png")


# ------------------------------------------------------------------ equations
def equations():
    fig = plt.figure(figsize=(12, 6))
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
        fig.text(0.5, y, tex, ha="center", va="center", fontsize=15)
    fig.text(0.5, 0.96, "Statistical framework", ha="center", fontsize=14,
             weight="bold")
    fig.text(0.5, 0.03,
             "Walk-forward expands the training window each year; every prediction "
             "uses only data available before the target year (strict OOS).",
             ha="center", fontsize=9, color=GRAY)
    save(fig, "equations.png")


# ------------------------------------------------------------------ forecasts
def walkforward_forecasts(p: pd.DataFrame):
    wf = walk_forward(p, target="price_vol", min_train=6)
    if wf.empty:
        print("  (no walk-forward rows; skipping forecast figure)")
        return

    # Ensure year is numeric
    wf["year"] = wf["year"].astype(int)

    fig, ax = plt.subplots(figsize=(10, 5))
    for model, color, ls, marker in [("baseline_naive", "#999999", "--", "s"),
                                     ("baseline_mean3", "#bbbbbb", ":", "^"),
                                     ("augmented_ols", RED, "-", "D")]:
        m = wf[wf["model"] == model].set_index("year")
        ax.plot(m.index, m["pred"], marker=marker, ls=ls, color=color, lw=1.8,
                markersize=8, label=model)

    actual = wf[wf["model"] == "augmented_ols"].set_index("year")["actual"]
    ax.plot(actual.index, actual, marker="o", color="black", lw=2.5,
            markersize=10, label="actual", zorder=5)

    # Set x-axis to show only the relevant years with padding
    years = sorted(wf["year"].unique())
    if len(years) >= 2:
        ax.set_xlim(years[0] - 0.5, years[-1] + 0.5)
        ax.set_xticks(years)
        ax.set_xticklabels(years)
    else:
        ax.set_xlim(years[0] - 1, years[0] + 1)
        ax.set_xticks([years[0]])

    # Add value annotations
    for model_name in ["augmented_ols", "baseline_naive", "baseline_mean3"]:
        m = wf[wf["model"] == model_name].set_index("year")
        for yr in m.index:
            val = m.loc[yr, "pred"]
            offset_y = 8 if model_name == "augmented_ols" else -12
            ax.annotate(f"{val:.1f}", (yr, val), textcoords="offset points",
                       xytext=(0, offset_y), ha="center", fontsize=8,
                       color=RED if model_name == "augmented_ols" else GRAY)

    # Annotate actual value
    for yr in actual.index:
        ax.annotate(f"{actual[yr]:.1f}", (yr, actual[yr]), textcoords="offset points",
                   xytext=(0, 12), ha="center", fontsize=9, fontweight="bold", color="black")

    ax.set_title("Walk-forward out-of-sample forecasts vs actual (n = 1 held-out "
                 "year \u2014 illustrative)", fontsize=12)
    ax.set_ylabel("Summer price volatility ($/MWh)", fontsize=11)
    ax.set_xlabel("Year", fontsize=11)
    ax.legend(fontsize=9, loc="upper left", framealpha=0.9)
    ax.grid(alpha=0.25)
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
