"""Weather helpers and the *virtual experiment* generator.

The virtual experiment simulates the planned 4 x 4 x 3 factorial (irrigation x
fertilisation x replicate, one scale per replicate) with a hidden "true"
parameter set that differs from the defaults, then adds realistic sensor
noise. It is used to (1) run the whole platform before any real data exist and
(2) test that calibration can recover known parameters. Everything it produces
is tagged source = "synthetic" and must never be reported as measurements.
"""
from __future__ import annotations
from dataclasses import replace
import numpy as np
import pandas as pd

from .config import Params, SiteConfig, plant_fresh_kg
from . import models as M

REQUIRED_WEATHER = ["Day", "ETo_mm", "Temperature_C"]


def make_weather(days: int = 140, seed: int = 0, mean_eto: float = 4.0, mean_temp: float = 23.0):
    """Smooth seasonal greenhouse-like weather with day-to-day noise."""
    rng = np.random.default_rng(seed)
    d = np.arange(1, days + 1)
    season = np.sin(2 * np.pi * (d - 20) / max(days, 1))
    eto = mean_eto + 0.9 * season + 0.5 * np.sin(2 * np.pi * d / 9.0)
    eto = np.clip(eto * np.exp(rng.normal(0, 0.10, days)), 1.2, 8.0)
    temp = mean_temp + 3.0 * season + 1.2 * np.sin(2 * np.pi * d / 7.0) + rng.normal(0, 0.8, days)
    return pd.DataFrame({"Day": d, "ETo_mm": eto, "Temperature_C": temp})


def check_weather(w: pd.DataFrame) -> pd.DataFrame:
    miss = [c for c in REQUIRED_WEATHER if c not in w.columns]
    if miss:
        raise ValueError(f"Weather file is missing columns: {miss}")
    w = w.sort_values("Day").reset_index(drop=True)
    if w["ETo_mm"].isna().any() or (w["ETo_mm"] < 0).any():
        raise ValueError("ETo_mm must be non-negative numbers")
    return w


def true_params(base: Params | None = None) -> Params:
    """Hidden 'reality' for the virtual experiment (deliberately != defaults)."""
    p = replace(base or Params())
    p.kc = [0.55, 1.00, 1.15, 0.95]
    p.p_depl = 0.30
    p.u_scale = [1.15, 0.85, 1.25]
    p.salt_uptake_factor = 0.8
    p.ymax_kg_plant = 4.6
    p.k_w = 1.2
    p.k_n = 0.9
    return p


IRR_LEVELS = [0.60, 0.80, 1.00, 1.20]      # fraction of reference ETc
FERT_LEVELS = [0.40, 0.70, 1.00, 1.30]     # fraction of reference recipe
REF_KC = [0.60, 0.90, 1.05, 0.90]          # kc the experimenter uses to set irrigation
LEACH_FRACTION = 0.20


def make_units(irr_levels=IRR_LEVELS, fert_levels=FERT_LEVELS, reps=3, cfg: SiteConfig | None = None):
    rows = []
    n_pots = (cfg.substrate.pots_per_scale if cfg else 3)
    for i, il in enumerate(irr_levels, 1):
        for j, fl in enumerate(fert_levels, 1):
            for r in range(1, reps + 1):
                rows.append({"unit_id": f"I{i}F{j}-R{r}", "irrigation_level": il,
                             "fert_level": fl, "replicate": r, "n_pots": n_pots})
    return pd.DataFrame(rows)


def experiment_actions(units: pd.DataFrame, weather: pd.DataFrame, params_ref: Params, cfg: SiteConfig):
    """Applied irrigation (L/pot/day) and concentrations for each unit-day, as the
    experimenter would apply them: level x reference-ETc / (1 - LF)."""
    T, n = len(weather), len(units)
    gdd = np.cumsum(np.maximum(0, weather["Temperature_C"].to_numpy() - params_ref.gdd_base_c))
    gdd_prev = np.concatenate([[0.0], gdd[:-1]])
    stage = M.stage_index(gdd_prev, params_ref)
    kc = np.asarray(REF_KC)[stage]
    etc_ref = kc * weather["ETo_mm"].to_numpy() * cfg.substrate.ground_area_per_plant_m2   # (T,)
    irr = (units["irrigation_level"].to_numpy()[None, :] * etc_ref[:, None]) / (1 - LEACH_FRACTION)
    mult = np.asarray(cfg.recipe_stage_mult)[stage]
    cin = (np.asarray(params_ref.c_ref)[None, :, None] * mult[:, None, None]
           * units["fert_level"].to_numpy()[None, None, :])                     # (T, 3, n)
    return irr, cin


def virtual_experiment(cfg: SiteConfig | None = None, days: int = 140, seed: int = 1,
                       truth: Params | None = None, reps: int = 3):
    cfg = cfg or SiteConfig()
    rng = np.random.default_rng(seed)
    truth = truth or true_params()
    weather = make_weather(days, seed=seed)
    units = make_units(reps=reps, cfg=cfg)
    n = len(units)
    irr, cin = experiment_actions(units, weather, Params(), cfg)

    kc_adj = np.exp(rng.normal(0, 0.04, n))                  # unit-to-unit variability
    bio_scale = np.clip(rng.normal(1.0, 0.2, n), 0.5, 1.5)
    st = M.init_state(n, truth, cfg, kc_adj=kc_adj)
    n_pots = cfg.substrate.pots_per_scale
    sub = cfg.substrate

    rows = []
    for t in range(days):
        biomass = plant_fresh_kg(st["gdd"], cfg) * bio_scale   # unit-specific, unknown to the twin
        a_start = st["A"].copy()
        w_morning = M.weight_from_storage(a_start, cfg, biomass) + rng.normal(0, sub.scale_sigma_kg, n)
        out = M.step(st, truth, cfg, weather["ETo_mm"][t], weather["Temperature_C"][t],
                     irr[t], cin[t])
        sample_day = (t % 3 == 0)
        for u in range(n):
            has_drain = out["drain"][u] > 0.05
            rec = {
                "unit_id": units["unit_id"][u], "day": t + 1, "source": "synthetic",
                "pot_weight_kg": w_morning[u],
                "irrigation_l": irr[t, u] * n_pots * (1 + rng.normal(0, 0.01)),
                "drainage_l": out["drain"][u] * n_pots * (1 + rng.normal(0, 0.02)),
                "applied_n_mg_l": cin[t, 0, u], "applied_p_mg_l": cin[t, 1, u],
                "applied_k_mg_l": cin[t, 2, u], "applied_ec_ds_m": out["ec_in"][u],
            }
            if sample_day and has_drain:
                rec["drain_n_mg_l"] = out["c_mix"][0, u] * (1 + rng.normal(0, 0.05))
                rec["drain_p_mg_l"] = out["c_mix"][1, u] * (1 + rng.normal(0, 0.05))
                rec["drain_k_mg_l"] = out["c_mix"][2, u] * (1 + rng.normal(0, 0.05))
                rec["drain_ec_ds_m"] = out["ec_drain"][u] + rng.normal(0, 0.05)
            rows.append(rec)
    obs = pd.DataFrame(rows)

    y, idx = M.predict_yield(st, truth)
    yields = pd.DataFrame({
        "unit_id": units["unit_id"],
        "yield_kg_plant": y * np.exp(rng.normal(0, 0.05, n)),
        "source": "synthetic",
    })
    return {"cfg": cfg, "weather": weather, "units": units, "obs": obs, "yields": yields,
            "truth": truth, "true_yield_noise_free": y}
