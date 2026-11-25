"""Comparison and statistics.

* paired_table      : twin vs traditional on matched units (mean difference, 95 % CI, tests)
* factorial_anova   : RCBD two-factor ANOVA for the calibration experiment
* monte_carlo_trial : many virtual seasons (different weather/noise) twin vs traditional
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats

from .config import Params, SiteConfig
from .synthetic import make_weather
from . import engine as E

METRICS = ["yield_kg_plant", "water_l_plant", "nutrient_g_plant", "drain_fraction",
           "water_productivity_kg_m3", "profit_per_plant"]


def paired_table(a: pd.DataFrame, b: pd.DataFrame, key="pair", metrics=METRICS,
                 label_a="Twin", label_b="Traditional") -> pd.DataFrame:
    """a and b must have the same `key` values (one row per matched unit)."""
    m = a.merge(b, on=key, suffixes=("_a", "_b"))
    rows = []
    n = len(m)
    for met in metrics:
        x, y = m[met + "_a"].to_numpy(float), m[met + "_b"].to_numpy(float)
        d = x - y
        sd = d.std(ddof=1) if n > 1 else np.nan
        se = sd / np.sqrt(n) if n > 1 else np.nan
        tcrit = stats.t.ppf(0.975, n - 1) if n > 1 else np.nan
        p_t = stats.ttest_rel(x, y).pvalue if n > 2 and sd > 0 else np.nan
        try:
            p_w = stats.wilcoxon(x, y).pvalue if n > 5 and np.any(d != 0) else np.nan
        except ValueError:
            p_w = np.nan
        rows.append({"metric": met, f"{label_a} mean": x.mean(), f"{label_b} mean": y.mean(),
                     "difference": d.mean(), "CI95 low": d.mean() - tcrit * se, "CI95 high": d.mean() + tcrit * se,
                     "relative %": 100 * d.mean() / y.mean() if y.mean() != 0 else np.nan,
                     "paired t p": p_t, "Wilcoxon p": p_w, "Cohen dz": d.mean() / sd if sd and sd > 0 else np.nan,
                     "n pairs": n})
    return pd.DataFrame(rows)


def factorial_anova(df: pd.DataFrame, response: str, a: str, b: str, block: str) -> pd.DataFrame:
    """Balanced RCBD factorial: y ~ block + A + B + A:B. One observation per cell per block."""
    d = df[[response, a, b, block]].dropna().copy()
    for c in (a, b, block):
        d[c] = d[c].astype(str)
    la, lb, lr = d[a].nunique(), d[b].nunique(), d[block].nunique()
    if len(d) != la * lb * lr or d.groupby([a, b, block]).size().max() != 1:
        raise ValueError("Factorial ANOVA needs a balanced design with exactly one value per A x B x block cell.")
    y = d[response].to_numpy(float)
    g = y.mean()
    ss_tot = ((y - g) ** 2).sum()
    ss_block = lb * la * ((d.groupby(block)[response].mean() - g) ** 2).sum()
    ss_a = lb * lr * ((d.groupby(a)[response].mean() - g) ** 2).sum()
    ss_b = la * lr * ((d.groupby(b)[response].mean() - g) ** 2).sum()
    cell = d.groupby([a, b])[response].mean()
    ma, mb = d.groupby(a)[response].mean(), d.groupby(b)[response].mean()
    ss_ab = 0.0
    for (ia, ib), v in cell.items():
        ss_ab += lr * (v - ma[ia] - mb[ib] + g) ** 2
    ss_err = ss_tot - ss_block - ss_a - ss_b - ss_ab
    df_ = {"Block": lr - 1, a: la - 1, b: lb - 1, f"{a} x {b}": (la - 1) * (lb - 1)}
    df_err = (la * lb - 1) * (lr - 1)
    ss = {"Block": ss_block, a: ss_a, b: ss_b, f"{a} x {b}": ss_ab}
    rows = []
    mse = ss_err / df_err if df_err > 0 else np.nan
    for k in ss:
        ms = ss[k] / df_[k]
        f = ms / mse if mse and mse > 0 else np.nan
        rows.append({"source": k, "df": df_[k], "SS": ss[k], "MS": ms, "F": f,
                     "p": stats.f.sf(f, df_[k], df_err) if np.isfinite(f) else np.nan})
    rows.append({"source": "Error", "df": df_err, "SS": ss_err, "MS": mse, "F": np.nan, "p": np.nan})
    rows.append({"source": "Total", "df": len(d) - 1, "SS": ss_tot, "MS": np.nan, "F": np.nan, "p": np.nan})
    return pd.DataFrame(rows)


def monte_carlo_trial(truth: Params, twin_params: Params, cfg: SiteConfig, n_seasons=6, days=140,
                      n_plots=3, seed=100, forecast_error=0.15, progress=None,
                      trad_frac=1.0, trad_leach=0.15, trad_fert=1.0):
    """Each season = a different weather realisation and sensor noise. Twin and traditional
    face identical 'reality' (same seed). Returns (per-plot results, paired table by season)."""
    rows = []
    for s in range(n_seasons):
        w = make_weather(days, seed=seed + s)
        for strat in ("twin", "traditional"):
            r = E.run_closed_loop(strat, truth, twin_params, cfg, w, seed=seed + s, n_plots=n_plots,
                                  forecast_error=forecast_error, trad_frac=trad_frac, trad_leach=trad_leach,
                                  trad_fert=trad_fert)["summary"]
            r["season"] = s + 1
            rows.append(r)
        if progress:
            progress((s + 1) / n_seasons)
    allp = pd.concat(rows, ignore_index=True)
    season = allp.groupby(["season", "strategy"])[METRICS].mean().reset_index()
    a = season[season.strategy == "twin"].rename(columns={"season": "pair"}).drop(columns="strategy")
    b = season[season.strategy == "traditional"].rename(columns={"season": "pair"}).drop(columns="strategy")
    return allp, paired_table(a, b)
