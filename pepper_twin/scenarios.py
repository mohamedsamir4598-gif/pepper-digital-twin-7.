"""Scenario library: results of external simulators (DSSAT, HYDRUS-1D, AquaCrop, ...).

The platform does NOT run those programs. You run them yourself, export one row per
scenario to CSV, and import it here. Two uses:

1. Benchmark   : compare the twin's season outcome with the external model for the same
                 irrigation/fertigation plan (model-to-model agreement, not validation).
2. Surrogate   : train a fast regressor on many scenarios (external or twin-generated) for
                 screening and sensitivity analysis. A surrogate trained on a simulator only
                 reproduces that simulator; it is never evidence about the real crop.

Expected CSV columns (case-insensitive), one row per scenario:
    scenario_id, source_model
    irr_frac_veg, irr_frac_flower, irr_frac_fruit, irr_frac_harvest      (fraction of reference ETc)
    fert_scale_veg, fert_scale_flower, fert_scale_fruit, fert_scale_harvest (fraction of reference recipe)
    yield_kg_plant   [required]      water_l_plant [optional]    drain_l_plant [optional]
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupKFold, KFold

from .config import Params, SiteConfig
from . import engine as E

PLAN_COLS = ["irr_frac_veg", "irr_frac_flower", "irr_frac_fruit", "irr_frac_harvest",
             "fert_scale_veg", "fert_scale_flower", "fert_scale_fruit", "fert_scale_harvest"]
TEMPLATE_COLUMNS = ["scenario_id", "source_model"] + PLAN_COLS + ["yield_kg_plant", "water_l_plant", "drain_l_plant"]


def template() -> pd.DataFrame:
    return pd.DataFrame([{c: (f"S{i}" if c == "scenario_id" else "DSSAT" if c == "source_model" else np.nan)
                          for c in TEMPLATE_COLUMNS} for i in range(1, 4)])


def load_scenarios(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={c: c.strip().lower() for c in df.columns})
    miss = [c for c in PLAN_COLS + ["yield_kg_plant"] if c not in df.columns]
    if miss:
        raise ValueError(f"Scenario file is missing columns: {miss}")
    if "scenario_id" not in df.columns:
        df["scenario_id"] = [f"S{i + 1}" for i in range(len(df))]
    if "source_model" not in df.columns:
        df["source_model"] = "external"
    return df


def run_twin_on_plans(plans: pd.DataFrame, p: Params, cfg: SiteConfig, weather: pd.DataFrame, leach=0.15):
    """Evaluate the twin on the same stage-wise plans (vectorised)."""
    X = plans[PLAN_COLS].to_numpy(float)
    pol = E.plan_policy(cfg, p, X[:, :4].T, X[:, 4:].T, leach=leach)
    r = E.simulate_season(p, cfg, weather, pol, n=len(plans))
    return pd.DataFrame({"scenario_id": plans["scenario_id"].to_numpy(), "twin_yield_kg_plant": r["yield_kg_plant"],
                         "twin_water_l_plant": r["water_l_plant"], "twin_drain_l_plant": r["drain_l_plant"]})


def benchmark(ext: pd.DataFrame, p: Params, cfg: SiteConfig, weather: pd.DataFrame, leach=0.15):
    twin = run_twin_on_plans(ext, p, cfg, weather, leach)
    m = ext.merge(twin, on="scenario_id")
    out = {}
    for ek, tk in (("yield_kg_plant", "twin_yield_kg_plant"), ("water_l_plant", "twin_water_l_plant")):
        if ek in m.columns and m[ek].notna().sum() >= 3:
            e, t = m[ek].to_numpy(float), m[tk].to_numpy(float)
            ok = ~np.isnan(e)
            e, t = e[ok], t[ok]
            out[ek] = {"n": int(ok.sum()), "RMSE": float(np.sqrt(np.mean((t - e) ** 2))),
                       "bias(twin-ext)": float(np.mean(t - e)),
                       "r": float(np.corrcoef(t, e)[0, 1]) if t.std() > 0 and e.std() > 0 else np.nan}
    return m, out


def latin_hypercube_plans(n: int, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cols = []
    for lo, hi in [(0.5, 1.4)] * 4 + [(0.2, 1.4)] * 4:
        u = (rng.permutation(n) + rng.random(n)) / n
        cols.append(lo + u * (hi - lo))
    df = pd.DataFrame(np.column_stack(cols), columns=PLAN_COLS)
    df.insert(0, "scenario_id", [f"LHS{i + 1}" for i in range(n)])
    return df


def build_twin_library(p: Params, cfg: SiteConfig, weather: pd.DataFrame, n=400, seed=0, leach=0.15):
    plans = latin_hypercube_plans(n, seed)
    res = run_twin_on_plans(plans, p, cfg, weather, leach)
    lib = plans.merge(res, on="scenario_id").rename(columns={
        "twin_yield_kg_plant": "yield_kg_plant", "twin_water_l_plant": "water_l_plant",
        "twin_drain_l_plant": "drain_l_plant"})
    lib["source_model"] = "twin"
    return lib


def train_surrogate(lib: pd.DataFrame, target="yield_kg_plant", seed=0):
    """Gradient-boosting surrogate with 5-fold cross-validated skill on *scenarios*."""
    X, y = lib[PLAN_COLS].to_numpy(float), lib[target].to_numpy(float)
    ok = ~np.isnan(y)
    X, y = X[ok], y[ok]
    if len(y) < 30:
        raise ValueError("At least 30 scenarios are needed to train a surrogate.")
    pred = np.zeros_like(y)
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(X):
        m = GradientBoostingRegressor(random_state=seed).fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    r2 = 1 - np.sum((pred - y) ** 2) / np.sum((y - y.mean()) ** 2)
    model = GradientBoostingRegressor(random_state=seed).fit(X, y)
    imp = pd.Series(model.feature_importances_, index=PLAN_COLS).sort_values(ascending=False)
    return model, {"cv_R2": float(r2), "cv_RMSE": float(np.sqrt(np.mean((pred - y) ** 2))), "n": int(len(y))}, imp
