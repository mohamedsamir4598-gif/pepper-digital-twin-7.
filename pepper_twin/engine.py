"""Digital-twin engine: state assimilation, receding-horizon decisions,
season optimisation and a closed-loop in-silico trial.

Safety: this module only *recommends*. It never opens a valve. Every decision
carries an approval flag and is range-checked against the site limits.
"""
from __future__ import annotations
from dataclasses import replace
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

from .config import Params, SiteConfig, plant_fresh_kg
from . import models as M

REF_KC = [0.60, 0.90, 1.05, 0.90]       # a grower's table Kc (traditional practice)


# ---------------------------------------------------------------------------
# Assimilated state of one unit (or one managed plot)
# ---------------------------------------------------------------------------
class UnitTwin:
    """Scalar-state twin for one scale. Fuses the model with morning weights,
    ETa derived from weights, and drainage samples."""

    def __init__(self, cfg: SiteConfig, params: Params, a0_frac=1.0, kc_adj=1.0):
        self.cfg, self.p = cfg, params
        self.state = M.init_state(1, params, cfg, a0_frac=a0_frac, kc_adj=kc_adj)
        sig = cfg.substrate.scale_sigma_kg / cfg.substrate.pots_per_scale
        self.r_obs = (max(sig, 0.01) + 0.10) ** 2        # weight noise + plant-mass uncertainty (L^2)
        self.P = 0.30 ** 2                                # storage variance (L^2)
        self.last = None
        self.day = 0
        self.gain_log = []

    # -- helpers
    @property
    def A(self):
        return float(self.state["A"][0])

    def stage(self):
        return int(M.stage_index(self.state["gdd"], self.p)[0])

    # -- measurement updates
    def morning_update(self, w_kg: float):
        """Fuse the morning scale reading (before irrigation) with the model storage."""
        bio = float(plant_fresh_kg(self.state["gdd"][0], self.cfg))
        a_obs = float(M.storage_from_weight(w_kg, self.cfg, bio))
        k = self.P / (self.P + self.r_obs)
        self.state["A"] = np.array([np.clip(self.A + k * (a_obs - self.A), 0.0, self.cfg.taw_l)])
        self.P = (1 - k) * self.P
        self.gain_log.append(k)
        return a_obs

    def adapt_kc(self, eta_obs: float, alpha=0.15):
        """Nudge the crop-coefficient multiplier when yesterday's model ETa missed the
        weight-derived ETa. Only when the model was unstressed (otherwise the miss is storage)."""
        if self.last is None:
            return
        eta_mod, ks = float(self.last["eta"][0]), float(self.last["ks"][0])
        if ks > 0.98 and eta_mod > 0.05 and np.isfinite(eta_obs):
            r = np.clip(eta_obs / eta_mod, 0.5, 1.5)
            self.state["kc_adj"] = np.clip(self.state["kc_adj"] * (1 + alpha * (r - 1)), 0.6, 1.5)

    def nutrient_update(self, c_obs=None, ec_obs=None, gain=0.5):
        """Correct root-zone mass using a drainage sample taken on the last step."""
        if self.last is None or float(self.last["drain"][0]) < 0.05:
            return
        if c_obs is not None:
            for k in range(3):
                if c_obs[k] is not None and np.isfinite(c_obs[k]) and self.last["c_mix"][k, 0] > 1e-6:
                    r = np.clip(c_obs[k] / self.last["c_mix"][k, 0], 0.5, 2.0)
                    self.state["M"][k] *= 1 + gain * (r - 1)
        if ec_obs is not None and np.isfinite(ec_obs) and self.last["ec_drain"][0] > 1e-6:
            r = np.clip(ec_obs / self.last["ec_drain"][0], 0.5, 2.0)
            self.state["S"] *= 1 + gain * (r - 1)

    # -- model advance with what was actually applied
    def advance(self, eto, temp, irr_l_pot, cin, ec_in=None):
        self.P += (0.06 * max(float(self.last["etc"][0]) if self.last else 1.0, 0.3)) ** 2
        self.last = M.step(self.state, self.p, self.cfg, eto, temp, np.array([irr_l_pot]),
                           np.asarray(cin, float).reshape(3, 1), None if ec_in is None else np.array([ec_in]))
        self.day += 1
        return self.last


# ---------------------------------------------------------------------------
# Decision engine (receding horizon, vectorised over candidate actions)
# ---------------------------------------------------------------------------
def _cin_for(fs, stage, p: Params, cfg: SiteConfig):
    return (np.asarray(p.c_ref, float)[:, None] * np.asarray(cfg.recipe_stage_mult, float)[stage][None, :]
            * np.asarray(fs, float)[None, :])


def _daily_cost(out, irr, cin, p: Params, cfg: SiteConfig):
    s = out["stage"]
    sd = np.asarray(cfg.stage_days_expected, float)
    late_days = sd[2] + sd[3]
    yv = p.ymax_kg_plant * cfg.economics.yield_price_per_kg
    loss_w = p.k_w * np.asarray(p.lam_w)[s] * (1 - out["wf"]) / sd[s]
    loss_n = p.k_n * np.asarray(p.lam_n)[s] * (1 - out["nrat"]) / sd[s]
    loss_s = (s >= 2) * p.ec_slope * np.maximum(0.0, out["ec_root"] - p.ec_threshold) / late_days
    stress = yv * (loss_w + loss_n + loss_s)
    water = cfg.economics.water_price_per_l * irr
    fert = cfg.economics.nutrient_price_per_g * (cin.sum(0) * irr / 1000.0)
    return stress, water, fert


def _follow_policy(state, p: Params, cfg: SiteConfig, eto):
    """Reference behaviour assumed after today's action: refill when depletion passes
    half of RAW, with a 10 % leaching allowance, full recipe."""
    taw = cfg.taw_l
    dr = taw - state["A"]
    raw = p.p_depl * taw
    stage = M.stage_index(state["gdd"], p)
    etc = np.asarray(p.kc)[stage] * state["kc_adj"] * eto * cfg.substrate.ground_area_per_plant_m2
    irr = np.where(dr >= 0.5 * raw, dr + 0.10 * etc, 0.0)
    irr = np.minimum(irr, cfg.max_event_l_per_plant)
    return irr, _cin_for(np.ones_like(irr), stage, p, cfg)


def recommend(twin: UnitTwin, eto_fc, temp_fc, approve=False, top_k=5):
    """Pick today's irrigation volume and fertigation strength for one plant/pot.

    eto_fc, temp_fc: forecasts for today and the next days (at least 1 value).
    """
    cfg, p = twin.cfg, twin.p
    H = min(len(eto_fc), cfg.forecast_horizon_days)
    taw = cfg.taw_l
    stage = twin.stage()
    st0 = twin.state
    etc_exp = float(p.kc[stage] * st0["kc_adj"][0] * eto_fc[0] * cfg.substrate.ground_area_per_plant_m2)
    refill = (taw - twin.A) + 0.10 * etc_exp
    vols = np.unique(np.round(np.concatenate([
        refill * np.array([0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]),
        etc_exp * np.array([0.5, 1.0, 1.5])]), 3))
    vols = vols[(vols >= 0) & (vols <= cfg.max_event_l_per_plant)]
    fss = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.25])
    V, F = np.meshgrid(vols, fss, indexing="ij")
    V, F = V.ravel(), F.ravel()
    n = len(V)

    cin0 = _cin_for(F, np.full(n, stage), p, cfg)
    ec0 = cfg.water_ec_ds_m + p.salt_per_nutrient * cin0.sum(0) / p.tds_per_ec
    feasible = ec0 <= cfg.max_ec_in_ds_m

    st = M.copy_state(st0, n)
    c_stress = np.zeros(n); c_water = np.zeros(n); c_fert = np.zeros(n)
    min_ks = np.ones(n)
    for h in range(H):
        if h == 0:
            irr, cin = V, cin0
        else:
            irr, cin = _follow_policy(st, p, cfg, eto_fc[h])
        out = M.step(st, p, cfg, eto_fc[h], temp_fc[h], irr, cin, accumulate=False)
        s_, w_, f_ = _daily_cost(out, irr, cin, p, cfg)
        c_stress += s_; c_water += w_; c_fert += f_
        min_ks = np.minimum(min_ks, out["ks"])
    total = c_stress + c_water + c_fert
    total = np.where(feasible, total, np.inf)
    order = np.argsort(total)
    b = order[0]

    table = pd.DataFrame({
        "irrigation_L_per_plant": V[order[:top_k]], "fertigation_strength": F[order[:top_k]],
        "expected_cost": total[order[:top_k]], "stress_cost": c_stress[order[:top_k]],
        "water_cost": c_water[order[:top_k]], "fertilizer_cost": c_fert[order[:top_k]],
        "min_Ks": min_ks[order[:top_k]]})

    depletion = (taw - twin.A) / taw
    reasons = [f"stage {['Vegetative','Flowering','Fruit development','Harvest'][stage]}",
               f"root-zone depletion {100 * depletion:.0f}% of available water",
               f"expected ETc today {etc_exp:.2f} L/plant"]
    if V[b] < 1e-9:
        reasons.append("no irrigation: storage covers the forecast demand")
    if F[b] < 0.5 and V[b] > 0:
        reasons.append("reduced fertigation: supply already exceeds expected uptake or EC is high")
    I = float(V[b])
    return {
        "irrigation_l_per_plant": I,
        "irrigation_total_l": I * cfg.plants_total,
        "valve_runtime_min": I / cfg.emitter_lph_per_plant * 60.0,
        "fertigation_strength": float(F[b]),
        "conc_n_mg_l": float(cin0[0, b]), "conc_p_mg_l": float(cin0[1, b]), "conc_k_mg_l": float(cin0[2, b]),
        "ec_in_ds_m": float(ec0[b]),
        "nutrient_mass_total_g": float(cin0[:, b].sum() * I * cfg.plants_total / 1000.0),
        "expected_min_ks": float(min_ks[b]),
        "stage": stage, "depletion_pct": float(100 * depletion),
        "kc_multiplier": float(st0["kc_adj"][0]),
        "reason": "; ".join(reasons),
        "alternatives": table,
        "approval_status": "approved" if approve else "pending",
        "limits_ok": bool(feasible[b] and I <= cfg.max_event_l_per_plant + 1e-9),
    }


# ---------------------------------------------------------------------------
# Open-loop season simulation (vectorised) used by the optimiser and baselines
# ---------------------------------------------------------------------------
def simulate_season(p: Params, cfg: SiteConfig, weather: pd.DataFrame, policy, n=1, kc_adj=1.0):
    """policy(t, state, eto, temp) -> (irr (n,), cin (3, n)). Returns summary arrays."""
    st = M.init_state(n, p, cfg, kc_adj=kc_adj)
    water = np.zeros(n); fert = np.zeros(n); drain = np.zeros(n); nuse = np.zeros(n)
    rows = []
    for t in range(len(weather)):
        eto, temp = float(weather["ETo_mm"].iloc[t]), float(weather["Temperature_C"].iloc[t])
        irr, cin = policy(t, st, eto, temp)
        out = M.step(st, p, cfg, eto, temp, irr, cin)
        water += out["irr"]; drain += out["drain"]
        fert += (cin * out["irr"][None, :]).sum(0) / 1000.0
        rows.append({"Day": t + 1, "irr": out["irr"].mean(), "eta": out["eta"].mean(), "etc": out["etc"].mean(),
                     "drain": out["drain"].mean(), "A": out["A"].mean(), "ec_root": out["ec_root"].mean(),
                     "stage": int(out["stage"][0])})
    y, idx = M.predict_yield(st, p)
    return {"yield_kg_plant": y, "water_l_plant": water, "nutrient_g_plant": fert, "drain_l_plant": drain,
            "eta_l_plant": st["eta_sum"].sum(0), "etc_l_plant": st["etc_sum"].sum(0),
            "indices": idx, "daily": pd.DataFrame(rows), "state": st}


def traditional_policy(cfg: SiteConfig, p: Params, frac=1.0, leach=0.15, fert_scale=1.0):
    """Grower practice: table-Kc ETc x fraction, divided by (1 - leaching), constant recipe."""
    area = cfg.substrate.ground_area_per_plant_m2

    def pol(t, st, eto, temp):
        s = M.stage_index(st["gdd"], p)
        irr = np.asarray(REF_KC)[s] * eto * area * frac / (1 - leach)
        irr = np.minimum(irr, cfg.max_event_l_per_plant)
        return irr, _cin_for(np.full_like(irr, fert_scale), s, p, cfg)
    return pol


def plan_policy(cfg: SiteConfig, p: Params, irr_frac, fert_scale, leach=0.15):
    """Open-loop plan with one irrigation fraction and one fertigation scale per stage.
    irr_frac, fert_scale: arrays (4, n)."""
    area = cfg.substrate.ground_area_per_plant_m2
    irr_frac, fert_scale = np.asarray(irr_frac, float), np.asarray(fert_scale, float)

    def pol(t, st, eto, temp):
        s = M.stage_index(st["gdd"], p)
        ar = np.arange(len(s))
        etc = np.asarray(p.kc)[s] * st["kc_adj"] * eto * area
        irr = np.minimum(etc * irr_frac[s, ar] / (1 - leach), cfg.max_event_l_per_plant)
        return irr, _cin_for(fert_scale[s, ar], s, p, cfg)
    return pol


def optimize_season(p: Params, cfg: SiteConfig, weather: pd.DataFrame, water_budget_l=None,
                    seed=0, maxiter=40, popsize=12):
    """Differential evolution over per-stage irrigation fractions and fertigation scales.
    Maximises profit; with a water budget it becomes a constrained problem (penalty)."""
    eco = cfg.economics
    bounds = [(0.5, 1.4)] * 4 + [(0.2, 1.4)] * 4

    def fun(X):
        X = np.atleast_2d(X)
        if X.shape[0] == 8:                      # vectorised call gives (8, S)
            X = X.T
        S = X.shape[0]
        pol = plan_policy(cfg, p, X[:, :4].T, X[:, 4:].T)
        r = simulate_season(p, cfg, weather, pol, n=S)
        profit = (r["yield_kg_plant"] * eco.yield_price_per_kg - r["water_l_plant"] * eco.water_price_per_l
                  - r["nutrient_g_plant"] * eco.nutrient_price_per_g)
        if water_budget_l is not None:
            profit = profit - 50.0 * np.maximum(0.0, r["water_l_plant"] - water_budget_l)
        return -profit

    res = differential_evolution(fun, bounds, seed=seed, maxiter=maxiter, popsize=popsize,
                                 vectorized=True, updating="deferred", tol=1e-6, polish=False)
    x = res.x
    r = simulate_season(p, cfg, weather, plan_policy(cfg, p, x[:4, None], x[4:, None]), n=1)
    return {"irrigation_fraction_by_stage": x[:4], "fertigation_scale_by_stage": x[4:],
            "yield_kg_plant": float(r["yield_kg_plant"][0]), "water_l_plant": float(r["water_l_plant"][0]),
            "nutrient_g_plant": float(r["nutrient_g_plant"][0]), "drain_l_plant": float(r["drain_l_plant"][0]),
            "profit": float(-res.fun), "daily": r["daily"]}


def pareto_water(p: Params, cfg: SiteConfig, weather: pd.DataFrame, budgets=None, **kw):
    """Yield-versus-water frontier by constraining the water budget (epsilon-constraint)."""
    free = optimize_season(p, cfg, weather, **kw)
    top = free["water_l_plant"]
    budgets = budgets if budgets is not None else np.linspace(0.55 * top, 1.0 * top, 6)
    rows = []
    for b in budgets:
        r = optimize_season(p, cfg, weather, water_budget_l=float(b), **kw)
        rows.append({"water_budget_l_plant": float(b), "water_used_l_plant": r["water_l_plant"],
                     "yield_kg_plant": r["yield_kg_plant"], "nutrient_g_plant": r["nutrient_g_plant"],
                     "profit": r["profit"],
                     "water_productivity_kg_m3": r["yield_kg_plant"] / max(r["water_l_plant"], 1e-9) * 1000})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Closed-loop in-silico trial: twin-managed vs traditional on the same "reality"
# ---------------------------------------------------------------------------
def run_closed_loop(strategy: str, truth: Params, twin_params: Params, cfg: SiteConfig,
                    weather: pd.DataFrame, seed=0, n_plots=3, forecast_error=0.15,
                    nutrient_sample_every=3, trad_frac=1.0, trad_leach=0.15, trad_fert=1.0):
    """strategy: 'twin' or 'traditional'. 'Reality' is simulated with `truth`; the twin only
    sees noisy scale weights and sparse drainage samples, and models with `twin_params`."""
    rng = np.random.default_rng(seed)
    kc_adj = np.exp(rng.normal(0, 0.05, n_plots))
    bio_scale = np.clip(rng.normal(1.0, 0.2, n_plots), 0.5, 1.5)
    real = M.init_state(n_plots, truth, cfg, kc_adj=kc_adj)
    sub = cfg.substrate
    T = len(weather)
    twins = [UnitTwin(cfg, twin_params) for _ in range(n_plots)] if strategy == "twin" else None
    trad = traditional_policy(cfg, truth, trad_frac, trad_leach, trad_fert)

    water = np.zeros(n_plots); drain = np.zeros(n_plots); fert = np.zeros(n_plots)
    rows = []
    prev_w = None
    for t in range(T):
        eto, temp = float(weather["ETo_mm"].iloc[t]), float(weather["Temperature_C"].iloc[t])
        bio = plant_fresh_kg(real["gdd"], cfg) * bio_scale
        w_morning = M.weight_from_storage(real["A"], cfg, bio) + rng.normal(0, sub.scale_sigma_kg, n_plots)

        if strategy == "twin":
            irr = np.zeros(n_plots); cin = np.zeros((3, n_plots))
            for i, tw in enumerate(twins):
                # yesterday's ETa from weights (needs yesterday's reading and applied water)
                if prev_w is not None:
                    eta_o = float(M.eta_from_weights(prev_w[i], w_morning[i], prev_irr[i] * sub.pots_per_scale,
                                                     prev_drain[i] * sub.pots_per_scale, cfg))
                    tw.adapt_kc(eta_o)
                tw.morning_update(w_morning[i])
                eto_fc = [eto * np.exp(rng.normal(0, forecast_error)) for _ in range(cfg.forecast_horizon_days)]
                eto_fc = [e for e in eto_fc]
                idx = np.arange(t, min(T, t + cfg.forecast_horizon_days))
                eto_fc = [float(weather["ETo_mm"].iloc[j]) * np.exp(rng.normal(0, forecast_error)) for j in idx]
                temp_fc = [float(weather["Temperature_C"].iloc[j]) for j in idx]
                dec = recommend(tw, eto_fc, temp_fc, approve=True)
                irr[i] = dec["irrigation_l_per_plant"]
                cin[:, i] = [dec["conc_n_mg_l"], dec["conc_p_mg_l"], dec["conc_k_mg_l"]]
        else:
            irr, cin = trad(t, real, eto, temp)

        out = M.step(real, truth, cfg, eto, temp, irr, cin)
        water += out["irr"]; drain += out["drain"]; fert += (cin * out["irr"][None, :]).sum(0) / 1000.0

        if strategy == "twin":
            sample = (t % nutrient_sample_every == 0)
            for i, tw in enumerate(twins):
                tw.advance(eto, temp, irr[i], cin[:, i])
                if sample and out["drain"][i] > 0.05:
                    cobs = list(out["c_mix"][:, i] * (1 + rng.normal(0, 0.05, 3)))
                    tw.nutrient_update(cobs, out["ec_drain"][i] + rng.normal(0, 0.05))
        prev_w, prev_irr, prev_drain = w_morning, out["irr"].copy(), out["drain"].copy()
        rows.append({"Day": t + 1, "irrigation_l": irr.mean(), "eta_l": out["eta"].mean(), "etc_l": out["etc"].mean(),
                     "drain_l": out["drain"].mean(), "storage_l": out["A"].mean(), "ec_root": out["ec_root"].mean(),
                     "ks": out["ks"].mean()})

    y, idx = M.predict_yield(real, truth)
    summary = pd.DataFrame({
        "strategy": strategy, "plot": np.arange(1, n_plots + 1), "yield_kg_plant": y,
        "water_l_plant": water, "drain_l_plant": drain, "nutrient_g_plant": fert,
        "drain_fraction": drain / np.maximum(water, 1e-9),
        "water_productivity_kg_m3": y / np.maximum(water, 1e-9) * 1000.0,
        "nutrient_productivity_kg_kg": y / np.maximum(fert / 1000.0, 1e-9) / 1000.0,
        "water_index": idx["water_index"], "nutrient_index": idx["nutrient_index"],
        "salinity_factor": idx["salinity_factor"]})
    eco = cfg.economics
    summary["profit_per_plant"] = (summary["yield_kg_plant"] * eco.yield_price_per_kg
                                   - summary["water_l_plant"] * eco.water_price_per_l
                                   - summary["nutrient_g_plant"] * eco.nutrient_price_per_g)
    return {"summary": summary, "daily": pd.DataFrame(rows)}
