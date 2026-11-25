"""Sequential calibration and honest validation.

Stage 1  water  : kc by stage + depletion fraction p   <- weight-derived ETa, drainage, storage
Stage 2  nutrient: uptake scale N/P/K + salt uptake    <- drainage N, P, K and EC
Stage 3  yield   : Ymax, k_w, k_n (linear in log space) <- final yields

Validation is leave-one-replicate-out (LORO): every fold calibrates on two
replicates and predicts the third *free-run* (only weather and applied
irrigation/fertigation are given). It is compared with two baselines: a
quadratic response surface and the grand mean.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
import numpy as np
import pandas as pd
from scipy.optimize import least_squares, lsq_linear

from .config import Params, SiteConfig, plant_fresh_kg
from . import models as M


# ---------------------------------------------------------------------------
# Data packing
# ---------------------------------------------------------------------------
@dataclass
class Experiment:
    units: pd.DataFrame
    weather: pd.DataFrame
    irr: np.ndarray            # (T, n) L per pot applied
    cin: np.ndarray            # (T, 3, n) mg/L applied
    ec_in: np.ndarray          # (T, n)
    eta_obs: np.ndarray        # (T, n) NaN where unknown
    drain_obs: np.ndarray      # (T, n)
    a_end_obs: np.ndarray      # (T, n) end-of-day storage from next morning's weight
    a0_obs: np.ndarray         # (n,)
    cdrain_obs: np.ndarray     # (T, 3, n)
    ecdrain_obs: np.ndarray    # (T, n)
    yield_obs: np.ndarray      # (n,) kg per plant (NaN if missing)

    @property
    def n(self):
        return len(self.units)

    @property
    def T(self):
        return len(self.weather)

    def subset(self, mask: np.ndarray) -> "Experiment":
        idx = np.where(mask)[0]
        return Experiment(
            units=self.units.iloc[idx].reset_index(drop=True), weather=self.weather,
            irr=self.irr[:, idx], cin=self.cin[:, :, idx], ec_in=self.ec_in[:, idx],
            eta_obs=self.eta_obs[:, idx], drain_obs=self.drain_obs[:, idx],
            a_end_obs=self.a_end_obs[:, idx], a0_obs=self.a0_obs[idx],
            cdrain_obs=self.cdrain_obs[:, :, idx], ecdrain_obs=self.ecdrain_obs[:, idx],
            yield_obs=self.yield_obs[idx])


def pack_experiment(units, obs, weather, yields, cfg: SiteConfig, params_for_defaults: Params | None = None,
                    biomass_fn=None) -> Experiment:
    """Pivot long tables to aligned arrays. `obs` needs `unit_id`, `day`, and any of
    pot_weight_kg, irrigation_l, drainage_l, applied_*_mg_l, applied_ec_ds_m,
    drain_n/p/k_mg_l, drain_ec_ds_m (scale totals for water; per-litre for conc)."""
    p0 = params_for_defaults or Params()
    T = len(weather)
    uids = list(units["unit_id"])
    n = len(uids)
    n_pots = cfg.substrate.pots_per_scale
    pos = {u: i for i, u in enumerate(uids)}

    def blank(*shape):
        return np.full(shape, np.nan)

    w = blank(T + 1, n)
    irr_s = blank(T, n)
    dr_s = blank(T, n)
    cin = blank(T, 3, n)
    ec_in = blank(T, n)
    cd = blank(T, 3, n)
    ecd = blank(T, n)
    for r in obs.itertuples(index=False):
        i = pos.get(r.unit_id)
        d = int(r.day) - 1
        if i is None or d < 0 or d >= T:
            continue

        def g(name):
            v = getattr(r, name, np.nan)
            return np.nan if v is None else float(v)
        w[d, i] = g("pot_weight_kg")
        irr_s[d, i] = g("irrigation_l")
        dr_s[d, i] = g("drainage_l")
        cin[d, 0, i], cin[d, 1, i], cin[d, 2, i] = g("applied_n_mg_l"), g("applied_p_mg_l"), g("applied_k_mg_l")
        ec_in[d, i] = g("applied_ec_ds_m")
        cd[d, 0, i], cd[d, 1, i], cd[d, 2, i] = g("drain_n_mg_l"), g("drain_p_mg_l"), g("drain_k_mg_l")
        ecd[d, i] = g("drain_ec_ds_m")
    # morning reading of day T+1 is unknown -> last day has no ETa
    irr = np.where(np.isnan(irr_s), 0.0, irr_s) / n_pots
    drain = dr_s / n_pots
    # fill missing applied concentrations with zero (unfertigated) and EC with water EC
    cin = np.where(np.isnan(cin), 0.0, cin)
    ec_in = np.where(np.isnan(ec_in), cfg.water_ec_ds_m, ec_in)

    eta = np.full((T, n), np.nan)
    a_end = np.full((T, n), np.nan)
    gdd_cum = np.cumsum(np.maximum(0.0, weather["Temperature_C"].to_numpy() - p0.gdd_base_c))
    for d in range(T - 1):
        eta[d] = M.eta_from_weights(w[d], w[d + 1], irr_s[d], dr_s[d], cfg)
        a_end[d] = M.storage_from_weight(w[d + 1], cfg, float(plant_fresh_kg(gdd_cum[d], cfg)))
    a0 = np.array([float(M.storage_from_weight(w[0, i], cfg)) if not np.isnan(w[0, i]) else cfg.taw_l
                   for i in range(n)])
    y = np.full(n, np.nan)
    if yields is not None and len(yields):
        for r in yields.itertuples(index=False):
            if r.unit_id in pos:
                y[pos[r.unit_id]] = float(r.yield_kg_plant)
    return Experiment(units.reset_index(drop=True), weather.reset_index(drop=True), irr, cin, ec_in, eta,
                      drain, a_end, a0, cd, ecd, y)


# ---------------------------------------------------------------------------
# Free-run replay with the *actual* applied inputs
# ---------------------------------------------------------------------------
def replay(exp: Experiment, p: Params, cfg: SiteConfig, kc_adj=1.0, water_driven=False):
    """Free-run replay with the actual applied inputs. With water_driven=True the
    measured storage, drainage and ETa replace the simulated water fluxes so only
    the nutrient/salt sub-model is being tested (what the assimilated twin does)."""
    n, T = exp.n, exp.T
    st = M.init_state(n, p, cfg, a0_frac=np.clip(exp.a0_obs / cfg.taw_l, 0, 1), kc_adj=kc_adj)
    rec = {k: [] for k in ["eta", "drain", "a_end", "c_mix", "ec_drain", "etc"]}
    for t in range(T):
        force = None
        if water_driven:
            a_prev = exp.a_end_obs[t - 1] if t > 0 else exp.a0_obs
            force = {"a_start": a_prev, "drain": exp.drain_obs[t], "eta": exp.eta_obs[t]}
        o = M.step(st, p, cfg, exp.weather["ETo_mm"].iloc[t], exp.weather["Temperature_C"].iloc[t],
                   exp.irr[t], exp.cin[t], exp.ec_in[t], force=force)
        rec["eta"].append(o["eta"]); rec["drain"].append(o["drain"]); rec["a_end"].append(o["A"])
        rec["c_mix"].append(o["c_mix"]); rec["ec_drain"].append(o["ec_drain"]); rec["etc"].append(o["etc"])
    out = {k: np.array(v) for k, v in rec.items()}
    out["state"] = st
    return out


# ---------------------------------------------------------------------------
# Stage 1: water
# ---------------------------------------------------------------------------
SIG_ETA, SIG_DRAIN, SIG_A = 0.12, 0.15, 0.60      # L, L, L (storage is soft: plant-mass correction is uncertain)


def _res_water(x, exp, p0, cfg):
    p = replace(p0, kc=list(x[:4]), p_depl=float(x[4]))
    r = replay(exp, p, cfg)
    res = []
    for sim, obs, sig in ((r["eta"], exp.eta_obs, SIG_ETA), (r["drain"], exp.drain_obs, SIG_DRAIN),
                          (r["a_end"], exp.a_end_obs, SIG_A)):
        d = (sim - obs) / sig
        res.append(np.where(np.isnan(d), 0.0, d).ravel())
    return np.concatenate(res)


def fit_water(exp, p0, cfg):
    x0 = np.array(list(p0.kc) + [p0.p_depl])
    lo = np.array([0.2] * 4 + [0.10])
    hi = np.array([1.6] * 4 + [0.70])
    sol = least_squares(_res_water, x0, bounds=(lo, hi), args=(exp, p0, cfg), x_scale=1.0,
                        diff_step=1e-3, max_nfev=60)
    return replace(p0, kc=[float(v) for v in sol.x[:4]], p_depl=float(sol.x[4])), sol


# ---------------------------------------------------------------------------
# Stage 2: nutrients
# ---------------------------------------------------------------------------
def _res_nut(x, exp, p0, cfg):
    p = replace(p0, u_scale=list(x[:3]), salt_uptake_factor=float(x[3]))
    r = replay(exp, p, cfg, water_driven=True)
    res = []
    mask = r["drain"] > 0.05
    for k in range(3):
        sim, obs = r["c_mix"][:, k, :], exp.cdrain_obs[:, k, :]
        d = (sim - obs) / (0.08 * np.nanmean(obs[~np.isnan(obs)]) + 1.0) if np.any(~np.isnan(obs)) else sim * 0
        res.append(np.where(np.isnan(d) | ~mask, 0.0, d).ravel())
    d = (r["ec_drain"] - exp.ecdrain_obs) / 0.6
    res.append(np.where(np.isnan(d) | ~mask, 0.0, d).ravel())
    return np.concatenate(res)


def fit_nutrients(exp, p0, cfg):
    x0 = np.array(list(p0.u_scale) + [p0.salt_uptake_factor])
    sol = least_squares(_res_nut, x0, bounds=([0.2, 0.2, 0.2, 0.0], [3.0, 3.0, 3.0, 1.5]),
                        args=(exp, p0, cfg), diff_step=1e-3, max_nfev=60)
    return replace(p0, u_scale=[float(v) for v in sol.x[:3]], salt_uptake_factor=float(sol.x[3])), sol


# ---------------------------------------------------------------------------
# Stage 3: yield (linear in log space)
# ---------------------------------------------------------------------------
def fit_yield(exp, p0, cfg):
    r = replay(exp, p0, cfg)
    w_idx, n_idx, sal = M.yield_indices(r["state"], p0)
    ok = ~np.isnan(exp.yield_obs)
    if ok.sum() < 6:
        raise ValueError("At least 6 units with final yield are needed to fit the yield model.")
    A = np.column_stack([np.ones(ok.sum()), w_idx[ok], n_idx[ok]])
    b = np.log(np.clip(exp.yield_obs[ok], 1e-3, None)) - np.log(sal[ok])
    sol = lsq_linear(A, b, bounds=([np.log(0.3), 0.0, 0.0], [np.log(30.0), 5.0, 5.0]))
    return replace(p0, ymax_kg_plant=float(np.exp(sol.x[0])), k_w=float(sol.x[1]), k_n=float(sol.x[2])), sol


def calibrate(exp: Experiment, cfg: SiteConfig, p0: Params | None = None):
    p0 = p0 or Params()
    p1, s1 = fit_water(exp, p0, cfg)
    p2, s2 = fit_nutrients(exp, p1, cfg)
    p3, s3 = fit_yield(exp, p2, cfg)
    return p3, {"water_cost": float(s1.cost), "nutrient_cost": float(s2.cost)}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def _metrics(sim, obs):
    sim, obs = np.asarray(sim, float).ravel(), np.asarray(obs, float).ravel()
    ok = ~(np.isnan(sim) | np.isnan(obs))
    if ok.sum() < 3:
        return {"n": int(ok.sum()), "RMSE": np.nan, "MAE": np.nan, "bias": np.nan, "NSE": np.nan, "R2": np.nan}
    s, o = sim[ok], obs[ok]
    rmse = float(np.sqrt(np.mean((s - o) ** 2)))
    nse = float(1 - np.sum((s - o) ** 2) / max(np.sum((o - o.mean()) ** 2), 1e-12))
    r2 = float(np.corrcoef(s, o)[0, 1] ** 2) if s.std() > 0 and o.std() > 0 else np.nan
    return {"n": int(ok.sum()), "RMSE": rmse, "MAE": float(np.mean(np.abs(s - o))),
            "bias": float(np.mean(s - o)), "NSE": nse, "R2": r2}


def response_surface_predict(train: Experiment, test: Experiment):
    def design(u):
        a, f = u["irrigation_level"].to_numpy(float), u["fert_level"].to_numpy(float)
        return np.column_stack([np.ones(len(u)), a, f, a * a, f * f, a * f])
    ok = ~np.isnan(train.yield_obs)
    coef, *_ = np.linalg.lstsq(design(train.units)[ok], train.yield_obs[ok], rcond=None)
    return design(test.units) @ coef


def loro_cv(exp: Experiment, cfg: SiteConfig, p0: Params | None = None):
    """Returns (summary DataFrame, per-unit predictions DataFrame, list of fold params)."""
    reps = sorted(exp.units["replicate"].unique())
    if len(reps) < 2:
        raise ValueError("Need at least two replicates for cross-validation.")
    preds, fold_params = [], []
    eta_s, eta_o, dr_s, dr_o, a_s, a_o, c_s, c_o, e_s, e_o = ([] for _ in range(10))
    for r in reps:
        te = (exp.units["replicate"] == r).to_numpy()
        train, test = exp.subset(~te), exp.subset(te)
        p, _ = calibrate(train, cfg, p0)
        fold_params.append(p)
        rr = replay(test, p, cfg)
        eta_s.append(rr["eta"]); eta_o.append(test.eta_obs)
        dr_s.append(rr["drain"]); dr_o.append(test.drain_obs)
        a_s.append(rr["a_end"]); a_o.append(test.a_end_obs)
        rn = replay(test, p, cfg, water_driven=True)      # nutrients with measured water (assimilated twin)
        m = rn["drain"] > 0.05
        c_s.append(np.where(m[:, None, :], rn["c_mix"], np.nan)); c_o.append(test.cdrain_obs)
        e_s.append(np.where(m, rn["ec_drain"], np.nan)); e_o.append(test.ecdrain_obs)
        y_twin, _ = M.predict_yield(rr["state"], p)
        preds.append(pd.DataFrame({
            "unit_id": test.units["unit_id"], "replicate": r,
            "irrigation_level": test.units["irrigation_level"], "fert_level": test.units["fert_level"],
            "yield_obs": test.yield_obs, "yield_twin": y_twin,
            "yield_surface": response_surface_predict(train, test),
            "yield_mean": np.nanmean(train.yield_obs)}))
    P = pd.concat(preds, ignore_index=True)
    cat = lambda L, ax: np.concatenate([np.asarray(x).reshape(-1) for x in L])
    rows = []
    rows.append({"target": "ETa (L/pot/day)", **_metrics(cat(eta_s, 0), cat(eta_o, 0))})
    rows.append({"target": "Drainage (L/pot/day)", **_metrics(cat(dr_s, 0), cat(dr_o, 0))})
    rows.append({"target": "Storage end of day (L)", **_metrics(cat(a_s, 0), cat(a_o, 0))})
    rows.append({"target": "Drain N (mg/L, water measured)", **_metrics(np.concatenate([x[:, 0, :].ravel() for x in c_s]),
                                                        np.concatenate([x[:, 0, :].ravel() for x in c_o]))})
    rows.append({"target": "Drain K (mg/L, water measured)", **_metrics(np.concatenate([x[:, 2, :].ravel() for x in c_s]),
                                                        np.concatenate([x[:, 2, :].ravel() for x in c_o]))})
    rows.append({"target": "Drain EC (dS/m, water measured)", **_metrics(cat(e_s, 0), cat(e_o, 0))})
    for name, col in (("Yield: twin", "yield_twin"), ("Yield: quadratic surface", "yield_surface"),
                      ("Yield: grand mean", "yield_mean")):
        rows.append({"target": name + " (kg/plant)", **_metrics(P[col], P["yield_obs"])})
    return pd.DataFrame(rows), P, fold_params


def jackknife_spread(fold_params: list[Params]) -> pd.DataFrame:
    """Parameter spread across LORO folds (a crude, honest uncertainty with few replicates)."""
    rows = []
    for name in ["p_depl", "ymax_kg_plant", "k_w", "k_n", "salt_uptake_factor"]:
        v = [getattr(p, name) for p in fold_params]
        rows.append({"parameter": name, "mean": np.mean(v), "min": np.min(v), "max": np.max(v)})
    for i, s in enumerate(["Vegetative", "Flowering", "Fruit dev.", "Harvest"]):
        v = [p.kc[i] for p in fold_params]
        rows.append({"parameter": f"kc[{s}]", "mean": np.mean(v), "min": np.min(v), "max": np.max(v)})
    for i, nme in enumerate("NPK"):
        v = [p.u_scale[i] for p in fold_params]
        rows.append({"parameter": f"u_scale[{nme}]", "mean": np.mean(v), "min": np.min(v), "max": np.max(v)})
    return pd.DataFrame(rows)
