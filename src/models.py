"""Models, walk-forward validation, and inference for the snowpack->prices study.

Models compared (all predict next summer's price volatility):
  baseline_naive   : persistence (this year = last year)
  baseline_mean3   : trailing 3-year mean
  baseline_arima   : ARIMA(0,1,0) with drift on the volatility series
  augmented_ols    : OLS on [lag-1 volatility, snowpack % normal]   <- the test
  augmented_arimax : ARIMA(0,1,0) with drift + snowpack as exogenous regressor

Validation is an expanding-window walk-forward: train on years t0..t, predict
t+1, repeat. Error comparison uses RMSE and the Diebold-Mariano test on squared
errors. The mediation leg asks whether controlling for actual hydro output kills
the direct snowpack->volatility effect (snowpack -> hydro output -> prices).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.regression.linear_model import OLS

MODELS = ["baseline_naive", "baseline_mean3", "baseline_arima",
          "augmented_ols", "augmented_arimax"]
CONTROL_MODEL = "augmented_controls"  # snowpack + temp + demand (confounders)


# ------------------------------------------------------------ walk-forward
def _fit_predict(model: str, train: pd.DataFrame, exog_col: str,
                  x_next: float, control_cols: list[str] | None = None,
                  next_row: pd.Series | None = None) -> float | None:
    """Fit `model` on train rows, return one-step-ahead prediction for the
    next year given the next year's exogenous value x_next (and, for the
    controls model, the next year's observed temperature/demand)."""
    y = train["y"].values
    if model == "baseline_naive":
        return y[-1]
    if model == "baseline_mean3":
        return y[-3:].mean() if len(y) >= 3 else y.mean()
    if model == "baseline_arima":
        try:
            return float(ARIMA(y, order=(0, 1, 0), trend="c").fit().forecast(1)[0])
        except Exception:
            return y[-1]
    if model == "augmented_ols":
        lr = OLS(y, np.column_stack([np.ones(len(y)), train[exog_col].values])).fit()
        return float(lr.predict(np.array([[1.0, x_next]]))[0])
    if model == "augmented_arimax":
        try:
            mod = ARIMA(y, exog=train[[exog_col]].values, order=(0, 1, 0), trend="c")
            return float(mod.fit().forecast(1, exog=np.array([[x_next]]))[0])
        except Exception:
            return y[-1]
    if model == CONTROL_MODEL:
        # snowpack + controls (temperature + demand) — the confounder check.
        # Uses the *observed* temperature/demand of the held-out year, which
        # makes this a conditional (what-if) evaluation, not a pure forecast.
        cols = [exog_col] + [c for c in (control_cols or []) if c in train.columns]
        cols = [c for c in cols if train[c].notna().all() and len(train[c].unique()) > 1]
        if len(cols) < 1 or len(train) <= len(cols) + 1:
            return y[-1]
        X = np.column_stack([np.ones(len(train))] + [train[c].values for c in cols])
        x_row = [1.0]
        for c in cols:
            if c == exog_col:
                x_row.append(float(x_next))
            elif next_row is not None and c in next_row.index:
                x_row.append(float(next_row[c]))
            else:
                x_row.append(float(train[c].iloc[-1]))
        try:
            lr = OLS(y, X).fit()
            return float(lr.predict(np.array([x_row]))[0])
        except Exception:
            return y[-1]
    raise ValueError(model)


def walk_forward(panel: pd.DataFrame, target: str = "price_vol",
                 exog_col: str = "snowpack_pct",
                 min_train: int = 3, models: list[str] | None = None,
                 control_cols: list[str] | None = None) -> pd.DataFrame:
    """Expanding-window walk-forward. Returns rows [year, model, pred, actual].

    Each year t >= min_train is predicted using only data from t0..t-1
    (strictly out-of-sample). If the sample is too short for `min_train`, the
    window shrinks to the largest size that still leaves >= 1 held-out year
    (the caller can flag such runs as illustrative). Returns an empty frame if
    fewer than 3 years are available.

    With the 2016-2018 + 2023-2025 price record (6 years), min_train=3 yields
    three genuine held-out years (2018, 2023, 2024...) rather than one.
    """
    cols = [target, exog_col] + (control_cols or [])
    df = panel[[c for c in cols if c in panel.columns]].dropna().sort_index()
    df = df.rename(columns={target: "y"})
    if len(df) < 3:
        return pd.DataFrame(columns=["year", "model", "pred", "actual", "error", "sq_error"])
    min_train = min(min_train, max(2, len(df) - 1))
    all_models = list(MODELS)
    if control_cols and any(c in df.columns for c in control_cols):
        all_models.append(CONTROL_MODEL)
    rows = []
    for model in (models or all_models):
        for i in range(min_train, len(df)):
            train = df.iloc[:i]
            next_row = df.iloc[i]
            pred = _fit_predict(model, train, exog_col, float(next_row[exog_col]),
                                control_cols=control_cols, next_row=next_row)
            rows.append({"year": df.index[i], "model": model,
                         "pred": pred, "actual": next_row["y"]})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["error"] = out["actual"] - out["pred"]
    out["sq_error"] = out["error"] ** 2
    return out


def rmse_table(wf: pd.DataFrame) -> pd.DataFrame:
    """RMSE by model (and MAE for good measure)."""
    g = wf.groupby("model").agg(
        rmse=("sq_error", lambda s: np.sqrt(s.mean())),
        mae=("error", lambda s: np.abs(s).mean()),
        n=("year", "count"),
    )
    return g.round(3).sort_values("rmse")


def dm_test(err1: np.ndarray, err2: np.ndarray) -> dict:
    """Diebold-Mariano test on squared errors (HAC / Newey-West variance).

    H0: equal predictive accuracy. Negative DM favors model 1 (lower loss).
    """
    e1, e2 = np.asarray(err1, float), np.asarray(err2, float)
    d = e1 ** 2 - e2 ** 2
    n = len(d)
    if n < 3:
        return {"dm": np.nan, "p": np.nan, "n": n}
    if np.allclose(d, 0):
        return {"dm": 0.0, "p": 1.0, "n": n}
    d_dm = d - d.mean()
    # Newey-West with lag = floor(4 * (n/100)^(2/9)) as a rough HAC choice
    L = int(np.floor(4 * (n / 100) ** (2 / 9))) if n >= 10 else 0
    gamma = np.array([np.dot(d_dm[: n - l], d_dm[l:]) / n for l in range(L + 1)])
    var = gamma[0] + 2 * np.sum(gamma[1:])
    if var <= 0:
        return {"dm": np.nan, "p": np.nan, "n": n}
    dm = d.mean() / np.sqrt(var / n)
    return {"dm": float(dm), "p": float(2 * stats.norm.cdf(-abs(dm))), "n": n}


# ------------------------------------------------------------ mediation
def mediation_analysis(panel: pd.DataFrame, target: str = "price_vol",
                       mediator: str = "hydro_gwh",
                       exog: str = "snowpack_pct") -> dict:
    """Regression-based mediation: does hydro output explain the snowpack effect?

    Paths (following Baron & Kenny):
      c   : total effect       target ~ snowpack
      a   : mediator path      hydro ~ snowpack
      b   : mediator effect    target ~ snowpack + hydro
      c'  : direct effect      coefficient on snowpack in the joint model
    Returns coefficients, p-values, and the proportion mediated 1 - c'/c.
    """
    df = panel[[target, mediator, exog]].dropna()
    res: dict = {"n": len(df)}
    if len(df) < 3:
        res["error"] = "not enough overlapping years"
        return res

    def _ols(y, xs):
        X = np.column_stack([np.ones(len(df))] + [df[x].values for x in xs])
        fit = OLS(df[y].values, X).fit()
        return fit.params, fit.pvalues, fit

    # total effect (c)
    c, pc, fit_c = _ols(target, [exog])
    # a path
    a, pa, _ = _ols(mediator, [exog])
    # joint: b (mediator) and c' (direct)
    b_joint, p_joint, fit_joint = _ols(target, [mediator, exog])

    res.update({
        "total_effect_c": float(c[1]), "p_total": float(pc[1]),
        "a_path": float(a[1]), "p_a": float(pa[1]),
        "b_mediator": float(b_joint[1]), "p_b": float(p_joint[1]),
        "direct_effect_cp": float(b_joint[2]), "p_direct": float(p_joint[2]),
        "proportion_mediated": float(1 - b_joint[2] / c[1]) if c[1] != 0 else np.nan,
        "r2_joint": float(fit_joint.rsquared),
        "r2_total": float(fit_c.rsquared),
    })
    # Sobel test of the indirect effect a*b (SEs from the two regressions)
    _, _, fit_a = _ols(mediator, [exog])
    _, _, fit_j = _ols(target, [mediator, exog])
    sa = np.sqrt(np.diag(fit_a.cov_params()))[1]
    sb = np.sqrt(np.diag(fit_j.cov_params()))[1]
    indirect = a[1] * b_joint[1]
    se_sobel = np.sqrt(a[1] ** 2 * sb ** 2 + b_joint[1] ** 2 * sa ** 2)
    res["indirect_effect"] = float(indirect)
    res["sobel_z"] = float(indirect / se_sobel) if se_sobel > 0 else np.nan
    res["sobel_p"] = float(2 * stats.norm.cdf(-abs(res["sobel_z"]))) if se_sobel > 0 else np.nan
    return res


# ------------------------------------------------------------ summary
def correlations(panel: pd.DataFrame, min_n: int = 3) -> pd.DataFrame:
    """Correlation of snowpack with each summer price feature (Pearson + Spearman)."""
    targets = [c for c in ["price_mean", "price_peak", "price_vol", "price_vol_hourly"]
               if c in panel.columns]
    rows = []
    for t in targets:
        sub = panel[[t, "snowpack_pct"]].dropna()
        if len(sub) >= min_n:
            rows.append({
                "target": t,
                "pearson_r": float(np.corrcoef(sub["snowpack_pct"], sub[t])[0, 1]),
                "spearman_rho": float(stats.spearmanr(sub["snowpack_pct"], sub[t])[0]),
                "n": len(sub),
            })
    return pd.DataFrame(rows).round(3)


def first_stage(panel: pd.DataFrame, mediator: str = "hydro_gwh") -> dict:
    """First stage of the causal chain: snowpack -> actual hydro output.

    This is the leg that has real sample size once price history is short:
    snowpack and hydro output both exist for many more years than prices.
    """
    df = panel[[mediator, "snowpack_pct"]].dropna()
    if len(df) < 5:
        return {"n": len(df), "error": "not enough years"}
    X = np.column_stack([np.ones(len(df)), df["snowpack_pct"].values])
    fit = OLS(df[mediator].values, X).fit()
    return {
        "n": len(df),
        "slope_gwh_per_pct": float(fit.params[1]),
        "p_value": float(fit.pvalues[1]),
        "r2": float(fit.rsquared),
        "pearson_r": float(np.corrcoef(df["snowpack_pct"], df[mediator])[0, 1]),
    }


def run_analysis(panel: pd.DataFrame) -> dict:
    """End-to-end analysis; returns a dict of results (also printed).

    With the currently available public price history (2023-2025), the price
    legs are explicitly flagged `illustrative`; the first-stage snowpack->hydro
    leg is estimated on the full 2018-2025 record.
    """
    out = {"correlations": correlations(panel, min_n=3).to_dict("records")}

    # Headline target: daily volatility. Also try hourly volatility.
    for target in ["price_vol", "price_vol_hourly"]:
        if target not in panel.columns or panel[target].notna().sum() < 3:
            continue
        ctrl = ["temp_mean_c", "demand_mean_mw"]
        wf = walk_forward(panel, target=target, min_train=3, control_cols=ctrl)
        out[f"wf_{target}"] = wf
        out[f"rmse_{target}"] = rmse_table(wf)
        n_oos = wf[wf["model"] == "augmented_ols"].shape[0]
        out[f"n_oos_{target}"] = n_oos
        out[f"illustrative_{target}"] = n_oos < 5

        # DM: augmented_ols vs each baseline, on common years
        dms = {}
        for base in ["baseline_naive", "baseline_mean3", "baseline_arima"]:
            a = wf[wf["model"] == "augmented_ols"].set_index("year")["error"]
            b = wf[wf["model"] == base].set_index("year")["error"]
            both = a.index.intersection(b.index)
            if len(both) >= 3:
                dms[base] = dm_test(a.loc[both].values, b.loc[both].values)
        out[f"dm_{target}"] = dms

    # First stage of the causal chain (snowpack -> hydro) on full record
    for med in ["hydro_gwh", "hydro_gwh_eia"]:
        if med in panel.columns:
            out[f"first_stage_{med}"] = first_stage(panel, mediator=med)

    # Full mediation on the headline target, with each available mediator
    # NOTE: with <=6 price years this is *exploratory* — the Sobel test needs
    # far more observations to be meaningful. Kept in the repo because the
    # decomposition is informative, but explicitly labeled insufficient for
    # inference (see README section 7).
    for med in ["hydro_gwh", "hydro_gwh_eia"]:
        if med in panel.columns and panel[med].notna().sum() >= 3:
            med_res = mediation_analysis(panel, mediator=med)
            med_res["illustrative"] = med_res.get("n", 0) < 10
            med_res["status"] = ("exploratory — insufficient observations for "
                                  "inference" if med_res.get("n", 0) < 10
                                  else "preliminary")
            out[f"mediation_{med}"] = med_res

    return out


def _fmt(d: dict) -> str:
    return ", ".join(f"{k}={v:.3f}" for k, v in d.items() if isinstance(v, float))


if __name__ == "__main__":
    from src.features import build_panel
    panel = build_panel()
    res = run_analysis(panel)
    print("\n=== Correlations (snowpack vs summer prices) ===")
    print(pd.DataFrame(res["correlations"]).to_string(index=False))
    for target in ["price_vol", "price_vol_hourly"]:
        key = f"rmse_{target}"
        if key not in res:
            continue
        print(f"\n=== Walk-forward RMSE: {target} ===")
        print(res[key].to_string())
        print("DM (augmented_ols vs baseline, squared errors):")
        for base, d in res[f"dm_{target}"].items():
            print(f"  vs {base}: DM={d['dm']:.3f}, p={d['p']:.3f}")
    print("\n=== Mediation ===")
    for k, v in res.items():
        if k.startswith("mediation_"):
            print(k, "->", _fmt(v))
