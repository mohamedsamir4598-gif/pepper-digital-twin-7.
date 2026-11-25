"""Glue between stored data and the engine. Thin, testable, no UI code."""
from __future__ import annotations
from dataclasses import replace
import numpy as np
import pandas as pd

from .config import Params, SiteConfig
from . import models as M
from . import engine as E
from . import calibration as C


def _val(row, name):
    v = row.get(name) if isinstance(row, dict) else getattr(row, name, np.nan)
    return np.nan if v is None else float(v)


def aggregate_reference_scales(obs: pd.DataFrame, unit_ids: list[str]) -> pd.DataFrame:
    """Average several reference scales into one managed plot (platform mode: e.g. 3 scales
    in a greenhouse). Water totals and weights are averaged per scale, concentrations too."""
    g = obs[obs["unit_id"].isin(unit_ids)]
    num = [c for c in g.columns if c not in ("unit_id", "source", "created_utc", "qc", "id")]
    out = g[num].groupby("day", as_index=False).mean(numeric_only=True)
    out["unit_id"] = "PLOT"
    return out


def replay_unit(obs_unit: pd.DataFrame, weather: pd.DataFrame, cfg: SiteConfig, params: Params,
                kc_adj0=1.0) -> tuple[E.UnitTwin, pd.DataFrame]:
    """Run the assimilated twin through the recorded history of one scale/plot.

    Returns the twin positioned at the *morning after the last record* and a daily log.
    """
    n_pots = cfg.substrate.pots_per_scale
    tw = E.UnitTwin(cfg, params, kc_adj=kc_adj0)
    obs = obs_unit.sort_values("day").set_index("day")
    log = []
    prev_w = prev_irr = prev_dr = None
    first = True
    for d, r in obs.iterrows():
        if d > len(weather):
            break
        w = weather.loc[weather["Day"] == d].iloc[0]
        wk = r.get("pot_weight_kg", np.nan)
        a_model_before = tw.A
        if first and pd.notna(wk):
            tw.state["A"] = np.array([float(M.storage_from_weight(wk, cfg, float(_bio(tw, cfg))))])
        if not first and pd.notna(wk):
            if prev_w is not None and pd.notna(prev_w):
                eta_o = float(M.eta_from_weights(prev_w, wk, prev_irr * n_pots, prev_dr * n_pots, cfg))
                tw.adapt_kc(eta_o)
            tw.morning_update(float(wk))
        first = False
        irr_pot = (0.0 if pd.isna(r.get("irrigation_l", np.nan)) else float(r["irrigation_l"])) / n_pots
        dr_pot = (0.0 if pd.isna(r.get("drainage_l", np.nan)) else float(r["drainage_l"])) / n_pots
        cin = [0.0 if pd.isna(r.get(k, np.nan)) else float(r[k])
               for k in ("applied_n_mg_l", "applied_p_mg_l", "applied_k_mg_l")]
        ec_in = None if pd.isna(r.get("applied_ec_ds_m", np.nan)) else float(r["applied_ec_ds_m"])
        out = tw.advance(float(w["ETo_mm"]), float(w["Temperature_C"]), irr_pot, cin, ec_in)
        cobs = [None if pd.isna(r.get(k, np.nan)) else float(r[k]) for k in ("drain_n_mg_l", "drain_p_mg_l", "drain_k_mg_l")]
        eobs = None if pd.isna(r.get("drain_ec_ds_m", np.nan)) else float(r["drain_ec_ds_m"])
        if any(c is not None for c in cobs) or eobs is not None:
            tw.nutrient_update(cobs, eobs)
        prev_w, prev_irr, prev_dr = wk, irr_pot, dr_pot
        log.append({"day": int(d), "stage": int(out["stage"][0]), "weight_obs_kg": wk,
                    "storage_model_before_L": a_model_before, "storage_after_L": float(out["A"][0]),
                    "eta_model_L": float(out["eta"][0]), "etc_model_L": float(out["etc"][0]),
                    "ks": float(out["ks"][0]), "kc_multiplier": float(tw.state["kc_adj"][0]),
                    "drain_model_L": float(out["drain"][0]), "drain_obs_L": dr_pot,
                    "ec_root": float(out["ec_root"][0]),
                    "conc_N": float(out["conc"][0, 0]), "conc_P": float(out["conc"][1, 0]), "conc_K": float(out["conc"][2, 0])})
    return tw, pd.DataFrame(log)


def _bio(tw: E.UnitTwin, cfg: SiteConfig):
    from .config import plant_fresh_kg
    return plant_fresh_kg(tw.state["gdd"][0], cfg)


def project_remaining(tw: E.UnitTwin, weather: pd.DataFrame, start_day: int, n=100, eto_sigma=0.15,
                      kc_sigma=0.05, param_sets: list[Params] | None = None, seed=0, policy="threshold"):
    """Project yield and remaining irrigation need from the twin's current state to the end of the
    weather record, with an ensemble (weather error, Kc error, optional parameter sets).

    policy 'threshold' = refill when half the readily available water is gone (business as usual)
    policy 'fixed'     = traditional table-Kc schedule
    """
    cfg = tw.cfg
    rng = np.random.default_rng(seed)
    sets = param_sets or [tw.p]
    members = []
    rem = weather[weather["Day"] >= start_day].reset_index(drop=True)
    per = max(1, n // len(sets))
    for p in sets:
        st = M.copy_state(tw.state, per)
        st["kc_adj"] = st["kc_adj"] * np.exp(rng.normal(0, kc_sigma, per))
        water = np.zeros(per)
        for t in range(len(rem)):
            eto = float(rem["ETo_mm"].iloc[t]) * np.exp(rng.normal(0, eto_sigma))
            temp = float(rem["Temperature_C"].iloc[t])
            if policy == "fixed":
                irr, cin = E.traditional_policy(cfg, p)(t, st, eto, temp)
            else:
                irr, cin = E._follow_policy(st, p, cfg, eto)
            M.step(st, p, cfg, eto, temp, irr, cin)
            water += irr
        y, _ = M.predict_yield(st, p)
        members.append(pd.DataFrame({"yield_kg_plant": y, "water_remaining_l_plant": water}))
    df = pd.concat(members, ignore_index=True)
    q = df.quantile([0.05, 0.5, 0.95])
    return df, q


def pack_from_store(cfg: SiteConfig, units, obs, weather, yields, source: str | None = None):
    """Build the calibration Experiment from stored tables (optionally one source only)."""
    if source:
        obs = obs[obs["source"] == source]
        if len(weather) and "source" in weather.columns:
            weather = weather[weather["source"] == source]
        if len(yields) and "source" in yields.columns:
            yields = yields[yields["source"] == source]
    units = units[units["unit_id"].isin(obs["unit_id"].unique())]
    weather = weather[["Day", "ETo_mm", "Temperature_C"]].sort_values("Day").reset_index(drop=True)
    return C.pack_experiment(units, obs, weather, yields, cfg)
