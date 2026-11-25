"""Process model for one pot (or one scale of identical pots).

All state variables are numpy arrays of shape (n,) -- n can be experimental
units, MPC candidate actions or ensemble members -- so one implementation
serves calibration, forecasting and control.

Units: water in litres per pot, nutrients in mg per pot, concentrations mg/L,
EC in dS/m, temperature in degC, ETo in mm/day.

Daily order (matches how a scale is read):
  morning scale reading -> irrigation/fertigation -> drainage -> ET -> uptake.
"""
from __future__ import annotations
import numpy as np
from .config import Params, SiteConfig

N_STAGES = 4


def _umax(p: Params) -> np.ndarray:
    return np.array([p.u_max_N, p.u_max_P, p.u_max_K], dtype=float)   # (3, 4)


def stage_index(gdd, p: Params):
    return np.searchsorted(np.asarray(p.gdd_bounds, float), gdd, side="right")


def init_state(n: int, p: Params, cfg: SiteConfig, a0_frac=1.0, conc_frac=None,
               kc_adj=1.0, gdd0=0.0, ec0=None):
    taw, vwp = cfg.taw_l, cfg.vwp_l
    a0 = np.broadcast_to(np.asarray(a0_frac, float) * taw, (n,)).copy()
    cf = cfg.init_conc_frac if conc_frac is None else conc_frac
    c0 = np.asarray(p.c_ref, float)[:, None] * np.broadcast_to(cf, (n,))[None, :]
    ec_init = cfg.water_ec_ds_m if ec0 is None else ec0
    ec0v = np.broadcast_to(np.asarray(ec_init, float), (n,))
    vol = vwp + a0
    return {
        "A": a0,
        "M": c0 * vol[None, :],
        "S": ec0v * p.tds_per_ec * vol,
        "gdd": np.full(n, float(gdd0)),
        "kc_adj": np.broadcast_to(np.asarray(kc_adj, float), (n,)).copy(),
        "etc_sum": np.zeros((N_STAGES, n)),
        "eta_sum": np.zeros((N_STAGES, n)),
        "uact_sum": np.zeros((N_STAGES, 3, n)),
        "udem_sum": np.zeros((N_STAGES, 3, n)),
        "ec_exc": np.zeros(n),
        "ec_n": np.zeros(n),
    }


def copy_state(s: dict, n: int | None = None) -> dict:
    out = {}
    for k, v in s.items():
        v = np.array(v, copy=True)
        if n is not None:           # broadcast a scalar-state (n=1) to n copies
            v = np.repeat(v, n, axis=-1)
        out[k] = v
    return out


def step(state: dict, p: Params, cfg: SiteConfig, eto, temp, irr_l, cin,
         ec_in=None, accumulate=True, force=None):
    """Advance one day in place. Returns a dict of daily outputs.

    irr_l : (n,) litres per pot applied this morning
    cin   : (3, n) mg/L of N, P, K in the supplied solution
    ec_in : (n,) dS/m of the supplied solution (default: water EC + salts of cin)
    force : optional dict of measured water fluxes that replace the simulated ones
            (keys 'a_start', 'drain', 'eta'; NaN entries are ignored). Used to drive the
            nutrient sub-model with observed water when calibrating or assimilating.
    """
    sub = cfg.substrate
    taw, vwp = cfg.taw_l, cfg.vwp_l
    n = state["A"].shape[0]
    irr = np.broadcast_to(np.asarray(irr_l, float), (n,))
    cin = np.broadcast_to(np.asarray(cin, float).reshape(3, -1), (3, n))
    if ec_in is None:
        ec_in = cfg.water_ec_ds_m + p.salt_per_nutrient * cin.sum(0) / p.tds_per_ec
    ec_in = np.broadcast_to(np.asarray(ec_in, float), (n,))
    eto = float(eto)
    temp = float(temp)

    stage = stage_index(state["gdd"], p)
    ar = np.arange(n)

    if force is not None and "a_start" in force:
        fa = np.asarray(force["a_start"], float)
        state["A"] = np.where(np.isnan(fa), state["A"], np.clip(fa, 0.0, taw))
    a_start = state["A"].copy()
    m_start = state["M"].copy()

    # --- irrigation + drainage (complete mixing) --------------------------
    A = state["A"] + irr
    M = state["M"] + cin * irr
    S = state["S"] + ec_in * p.tds_per_ec * irr
    vol_mix = vwp + A
    c_mix = M / np.maximum(vol_mix, 1e-9)
    s_mix = S / np.maximum(vol_mix, 1e-9)
    drain = np.maximum(0.0, A - taw)
    if force is not None and "drain" in force:
        fd = np.asarray(force["drain"], float)
        drain = np.where(np.isnan(fd), drain, np.clip(fd, 0.0, 0.98 * A))
    M = M - c_mix * drain
    S = S - s_mix * drain
    A = A - drain

    # --- evapotranspiration with water-stress coefficient ------------------
    dr = taw - A
    raw = p.p_depl * taw
    ks = np.where(dr <= raw, 1.0, np.clip((taw - dr) / max(taw - raw, 1e-9), 0.0, 1.0))
    kc = np.asarray(p.kc, float)[stage] * state["kc_adj"]
    etc = kc * eto * sub.ground_area_per_plant_m2
    eta = np.minimum(etc * ks, A)
    if force is not None and "eta" in force:
        fe = np.asarray(force["eta"], float)
        eta = np.where(np.isnan(fe), eta, np.clip(fe, 0.0, 0.98 * A))
    wf = np.where(etc > 1e-9, eta / np.maximum(etc, 1e-9), 1.0)

    # --- nutrient uptake (Liebig-type availability, mass-limited) ----------
    c_root = M / np.maximum(vwp + A, 1e-9)
    cref = np.asarray(p.c_ref, float)[:, None]
    km = np.asarray(p.km, float)[:, None]
    f_av = np.minimum(1.0, (c_root / (c_root + km)) / (cref / (cref + km)))
    u_stage = _umax(p)[:, stage] * np.asarray(p.u_scale, float)[:, None]   # (3, n)
    udem = u_stage * wf[None, :]
    uact = np.minimum(udem * f_av, 0.95 * M)
    M = M - uact
    S = np.maximum(0.0, S - p.salt_uptake_factor * p.salt_per_nutrient * uact.sum(0))
    A = A - eta
    S = np.minimum(S, p.ec_cap_ds_m * p.tds_per_ec * np.maximum(vwp + A, 1e-9))

    vol_end = vwp + A
    ec_root = S / np.maximum(vol_end, 1e-9) / p.tds_per_ec
    conc_end = M / np.maximum(vol_end, 1e-9)

    if accumulate:
        state["etc_sum"][stage, ar] += etc
        state["eta_sum"][stage, ar] += eta
        state["uact_sum"][stage, :, ar] += uact.T
        state["udem_sum"][stage, :, ar] += udem.T
        late = (stage >= 2).astype(float)
        state["ec_exc"] += late * np.maximum(0.0, ec_root - p.ec_threshold)
        state["ec_n"] += late

    state["A"], state["M"], state["S"] = A, M, S
    state["gdd"] = state["gdd"] + max(0.0, temp - p.gdd_base_c)

    with np.errstate(divide="ignore", invalid="ignore"):
        nrat = np.where(udem > 1e-12, uact / np.maximum(udem, 1e-12), 1.0).min(axis=0)

    return {
        "stage": stage, "etc": etc, "eta": eta, "ks": ks, "wf": wf,
        "irr": irr.copy(), "drain": drain, "c_mix": c_mix,
        "ec_drain": s_mix / p.tds_per_ec, "ec_root": ec_root, "ec_in": ec_in.copy(),
        "A": A.copy(), "conc": conc_end, "uact": uact, "udem": udem, "nrat": nrat,
        # closure diagnostics
        "water_resid": a_start + irr - drain - eta - A,
        "n_resid": m_start + cin * irr - c_mix * drain - uact - M,
    }


# ---------------------------------------------------------------------------
# Yield: stage-weighted Jensen-type water and nutrient response + salinity
# ---------------------------------------------------------------------------
def yield_indices(state: dict, p: Params):
    with np.errstate(divide="ignore", invalid="ignore"):
        rw = np.where(state["etc_sum"] > 1e-9, state["eta_sum"] / np.maximum(state["etc_sum"], 1e-9), 1.0)
        rn_all = np.where(state["udem_sum"] > 1e-9,
                          state["uact_sum"] / np.maximum(state["udem_sum"], 1e-9), 1.0)
    rn = rn_all.min(axis=1)                                   # Liebig over N, P, K -> (4, n)
    lw = np.asarray(p.lam_w, float)[:, None]
    ln_ = np.asarray(p.lam_n, float)[:, None]
    w_idx = (lw * np.log(np.clip(rw, 0.05, 1.0))).sum(0)
    n_idx = (ln_ * np.log(np.clip(rn, 0.05, 1.0))).sum(0)
    exc = np.where(state["ec_n"] > 0, state["ec_exc"] / np.maximum(state["ec_n"], 1), 0.0)
    sal = np.clip(1.0 - p.ec_slope * exc, 0.20, 1.0)
    return w_idx, n_idx, sal


def predict_yield(state: dict, p: Params):
    """Fresh fruit yield per plant (kg) and its stress indices."""
    w_idx, n_idx, sal = yield_indices(state, p)
    y = p.ymax_kg_plant * np.exp(p.k_w * w_idx + p.k_n * n_idx) * sal
    return y, {"water_index": w_idx, "nutrient_index": n_idx, "salinity_factor": sal}


# ---------------------------------------------------------------------------
# Scale <-> storage conversions
# ---------------------------------------------------------------------------
def weight_from_storage(a_l, cfg: SiteConfig, biomass_kg_plant=0.0):
    sub = cfg.substrate
    return sub.tare_kg + sub.pots_per_scale * (cfg.vwp_l + np.asarray(a_l, float) + biomass_kg_plant)


def storage_from_weight(w_kg, cfg: SiteConfig, biomass_kg_plant=0.0):
    sub = cfg.substrate
    a = (np.asarray(w_kg, float) - sub.tare_kg) / sub.pots_per_scale - cfg.vwp_l - biomass_kg_plant
    return np.clip(a, 0.0, cfg.taw_l)


def tare_from_saturated_weight(w_fc_kg, cfg: SiteConfig, biomass_kg_plant=0.0):
    """Tare from a reading taken after saturation and free drainage (A = TAW)."""
    sub = cfg.substrate
    return float(w_fc_kg - sub.pots_per_scale * (cfg.vwp_l + cfg.taw_l + biomass_kg_plant))


def eta_from_weights(w_today, w_next, irr_scale_l, drain_scale_l, cfg: SiteConfig):
    """ETa per pot (L) between two consecutive morning readings.

    Plant fresh-weight gain is not separated (typically 1-2 % of daily ETa).
    """
    n = cfg.substrate.pots_per_scale
    return (np.asarray(w_today) + np.asarray(irr_scale_l) - np.asarray(drain_scale_l)
            - np.asarray(w_next)) / n
