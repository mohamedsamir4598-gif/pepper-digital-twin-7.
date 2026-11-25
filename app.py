"""Pepper digital-twin platform (Streamlit).  Run:  streamlit run app.py"""
from __future__ import annotations
import io
import json
from dataclasses import asdict

import numpy as np
import pandas as pd
import streamlit as st

from pepper_twin import __version__
from pepper_twin import store, services as SV, engine as E, calibration as C, compare as K, scenarios as SC
from pepper_twin.config import Params, SiteConfig, Substrate, Economics, STAGE_NAMES, params_to_json, params_from_json
from pepper_twin.synthetic import (virtual_experiment, make_weather, make_units, true_params, check_weather)

st.set_page_config(page_title="Pepper Digital Twin", layout="wide")
store.init_db()

# --------------------------------------------------------------------------- state
if "cfg" not in st.session_state:
    txt = store.kv_get("site_config")
    st.session_state.cfg = SiteConfig.from_json(txt) if txt else SiteConfig()
if "params" not in st.session_state:
    txt = store.kv_get("params")
    st.session_state.params = params_from_json(txt) if txt else Params()
    st.session_state.params_calibrated = bool(txt)
cfg: SiteConfig = st.session_state.cfg
params: Params = st.session_state.params
SOURCES = list(store.SOURCES)

# --------------------------------------------------------------------------- sidebar
st.sidebar.title("Pepper Digital Twin")
st.sidebar.caption(f"v{__version__} · decision support only: nothing here operates valves")
mode = st.sidebar.radio("Data mode", ["Manual entry", "CSV import", "Live sensors (API)", "Virtual demo (synthetic)"])
present = sorted(store.get_observations()["source"].unique()) if len(store.get_observations()) else []
default_src = {"Manual entry": "manual", "CSV import": "csv", "Live sensors (API)": "iot",
               "Virtual demo (synthetic)": "synthetic"}[mode]
active = st.sidebar.selectbox("Analyse data from source", SOURCES, index=SOURCES.index(default_src),
                              help="Synthetic data are model output, never measurements.")
if active == "synthetic":
    st.sidebar.warning("Synthetic data: results show the platform works, not that the crop behaves this way.")
st.sidebar.markdown(f"**Parameters:** {'calibrated' if st.session_state.get('params_calibrated') else 'DEFAULT ASSUMPTIONS (not calibrated)'}")

with st.sidebar.expander("Site setup"):
    s = cfg.substrate
    pots = st.number_input("Pots per scale", 1, 12, int(s.pots_per_scale))
    vol = st.number_input("Pot volume (L)", 1.0, 100.0, float(s.volume_l))
    fc = st.number_input("Container capacity θfc", 0.1, 0.9, float(s.theta_fc))
    wp = st.number_input("Lower limit θwp", 0.01, 0.6, float(s.theta_wp))
    area = st.number_input("Ground area per plant (m²)", 0.05, 3.0, float(s.ground_area_per_plant_m2))
    tare = st.number_input("Tare: scale+pots+dry substrate (kg)", 0.0, 500.0, float(s.tare_kg))
    plants = st.number_input("Plants in the managed area", 1, 200000, int(cfg.plants_total))
    emit = st.number_input("Emitter flow per plant (L/h)", 0.1, 20.0, float(cfg.emitter_lph_per_plant))
    max_ev = st.number_input("Max irrigation per event (L/plant)", 0.1, 10.0, float(cfg.max_event_l_per_plant))
    max_ec = st.number_input("Max supplied EC (dS/m)", 0.5, 8.0, float(cfg.max_ec_in_ds_m))
    wec = st.number_input("Irrigation water EC (dS/m)", 0.0, 5.0, float(cfg.water_ec_ds_m))
    yp = st.number_input("Price: yield (per kg)", 0.0, 1000.0, float(cfg.economics.yield_price_per_kg))
    wpz = st.number_input("Price: water (per L)", 0.0, 10.0, float(cfg.economics.water_price_per_l), format="%.4f")
    npz = st.number_input("Price: nutrients N+P+K (per g)", 0.0, 10.0, float(cfg.economics.nutrient_price_per_g), format="%.4f")
    if st.button("Save site setup"):
        new = SiteConfig(name=cfg.name, substrate=Substrate(volume_l=vol, theta_fc=fc, theta_wp=wp, pots_per_scale=int(pots),
                         ground_area_per_plant_m2=area, tare_kg=tare, scale_sigma_kg=s.scale_sigma_kg,
                         plant_fresh_init_kg=s.plant_fresh_init_kg, plant_fresh_max_kg=s.plant_fresh_max_kg,
                         plant_fresh_gdd_scale=s.plant_fresh_gdd_scale),
                         economics=Economics(yp, wpz, npz), stage_days_expected=cfg.stage_days_expected,
                         recipe_stage_mult=cfg.recipe_stage_mult, water_ec_ds_m=wec, init_conc_frac=cfg.init_conc_frac,
                         plants_total=int(plants), emitter_lph_per_plant=emit, max_event_l_per_plant=max_ev,
                         max_ec_in_ds_m=max_ec, forecast_horizon_days=cfg.forecast_horizon_days)
        store.kv_set("site_config", new.to_json())
        st.session_state.cfg = new
        st.rerun()
    if st.button("Reset parameters to default assumptions"):
        st.session_state.params = Params()
        st.session_state.params_calibrated = False
        st.rerun()
with st.sidebar.expander("Backup"):
    st.download_button("Download site config (JSON)", cfg.to_json(), "site_config.json")
    st.download_button("Download parameters (JSON)", params_to_json(params), "params.json")
    st.caption(f"Database file: {store.db_path()}")

tabs = st.tabs(["1 Data", "2 Twin state", "3 Decide", "4 Predict & optimise", "5 Calibrate & validate",
                "6 Compare", "7 Notes"])


def src_obs():
    o = store.get_observations(source=active)
    return o


def get_weather_df():
    w = store.get_weather()
    if len(w) and "source" in w.columns:
        ws = w[w["source"] == active]
        w = ws if len(ws) else w
    return w[["Day", "ETo_mm", "Temperature_C"]].reset_index(drop=True) if len(w) else w


# --------------------------------------------------------------------------- 1 DATA
with tabs[0]:
    st.header("Data")
    st.caption("Every row is stored with its source. Day 1 = first day after transplanting. "
               "Weights are the morning scale reading BEFORE irrigation; water volumes are scale totals (all pots on it).")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Units (scales / plots)")
        units = store.get_units()
        edited = st.data_editor(units if len(units) else pd.DataFrame(
            {"unit_id": [], "irrigation_level": [], "fert_level": [], "replicate": [], "n_pots": [], "role": []}),
            num_rows="dynamic", key="units_editor")
        if st.button("Save units"):
            store.save_units(edited.dropna(subset=["unit_id"]))
            st.rerun()
        with st.expander("Create the factorial experiment layout"):
            irr_txt = st.text_input("Irrigation levels (fraction of ETc)", "0.6,0.8,1.0,1.2")
            fert_txt = st.text_input("Fertilisation levels (fraction of recipe)", "0.4,0.7,1.0,1.3")
            reps = st.number_input("Replicates", 1, 10, 3)
            if st.button("Create units"):
                lay = make_units([float(x) for x in irr_txt.split(",")], [float(x) for x in fert_txt.split(",")],
                                 int(reps), cfg)
                store.save_units(lay)
                st.rerun()
    with c2:
        st.subheader("Weather (daily)")
        w = get_weather_df()
        if len(w):
            st.line_chart(w.set_index("Day")[["ETo_mm", "Temperature_C"]])
        else:
            st.info("No weather yet. Upload a CSV (Day, ETo_mm, Temperature_C) or generate a synthetic season.")
        wf = st.file_uploader("Weather CSV", type="csv", key="wup")
        if wf is not None and st.button("Import weather"):
            try:
                store.save_weather(check_weather(pd.read_csv(wf)), active)
                st.success("Weather imported")
                st.rerun()
            except Exception as e:
                st.error(str(e))
        st.download_button("Weather template", "Day,ETo_mm,Temperature_C\n1,3.8,22.5\n2,4.1,23.0\n", "weather_template.csv")

    st.divider()
    st.subheader(f"Mode: {mode}")
    cols = ["unit_id", "day"] + store.OBS_FIELDS
    if mode == "Manual entry":
        st.write("Type readings into the table, then save. Leave unknown cells empty.")
        tmpl = pd.DataFrame({c: pd.Series(dtype="float" if c not in ("unit_id",) else "object") for c in cols})
        rows = st.data_editor(tmpl, num_rows="dynamic", key="manual_rows")
        if st.button("Save manual readings"):
            rows = rows.dropna(subset=["unit_id", "day"])
            flagged = store.add_observations(rows, "manual")
            st.success(f"Saved {len(rows)} rows")
            if len(flagged):
                st.warning("Quality-control flags:")
                st.dataframe(flagged)
    elif mode == "CSV import":
        st.write("Upload long-format observations: one row per unit per day.")
        st.download_button("Observation template", ",".join(cols) + "\n", "observations_template.csv")
        of = st.file_uploader("Observations CSV", type="csv", key="oup")
        if of is not None and st.button("Import observations"):
            df = pd.read_csv(of)
            miss = [c for c in ("unit_id", "day") if c not in df.columns]
            if miss:
                st.error(f"Missing columns: {miss}")
            else:
                flagged = store.add_observations(df, "csv")
                st.success(f"Imported {len(df)} rows")
                if len(flagged):
                    st.warning("Quality-control flags:")
                    st.dataframe(flagged)
        yf = st.file_uploader("Final yields CSV (unit_id, yield_kg_plant)", type="csv", key="yup")
        if yf is not None and st.button("Import yields"):
            store.save_yields(pd.read_csv(yf), "csv")
            st.success("Yields imported")
    elif mode == "Live sensors (API)":
        st.write("Sensors (or a gateway) POST readings to the API; they appear here with source = iot.")
        st.code("uvicorn pepper_twin.api:app --host 0.0.0.0 --port 8000\n"
                "export TWIN_API_KEY=choose-a-long-random-secret   # required\n\n"
                "curl -X POST http://HOST:8000/api/v1/observations -H 'X-API-Key: SECRET' \\\n"
                "  -H 'Content-Type: application/json' \\\n"
                "  -d '{\"unit_id\":\"PLOT\",\"day\":12,\"pot_weight_kg\":31.4,\"irrigation_l\":4.2,\"drainage_l\":0.6}'",
                language="bash")
        st.caption("The API and this app must share the same database file (PEPPER_DB_PATH). "
                   "Approved decisions are served at /api/v1/decisions/approved for a separate, human-reviewed valve gateway.")
        st.button("Refresh")
    else:
        sd = st.number_input("Random seed", 0, 10_000, 1)
        nd = st.number_input("Season length (days)", 60, 220, 140)
        st.write("Generates the planned 4 × 4 × 3 factorial (one scale per replicate) from a hidden 'true' parameter set plus sensor noise.")
        if st.button("Generate virtual experiment"):
            ve = virtual_experiment(cfg, days=int(nd), seed=int(sd))
            store.clear_source("synthetic")
            store.save_units(ve["units"]); store.save_weather(ve["weather"], "synthetic")
            store.save_yields(ve["yields"], "synthetic"); store.add_observations(ve["obs"], "synthetic")
            st.success("Virtual experiment stored (source = synthetic)")
            st.rerun()

    st.divider()
    o = src_obs()
    st.subheader(f"Stored observations from '{active}'")
    if len(o):
        st.write(f"{len(o)} rows · {o['unit_id'].nunique()} units · days {int(o['day'].min())}–{int(o['day'].max())}")
        st.dataframe(o.tail(200))
        issues = store.qc_series(o.dropna(subset=["pot_weight_kg"]))
        if len(issues):
            st.warning("Possible scale disturbances (check before using):")
            st.dataframe(issues)
        st.download_button("Download observations (CSV)", o.to_csv(index=False), f"observations_{active}.csv")
    else:
        st.info("Nothing stored for this source yet.")

# --------------------------------------------------------------------------- 2 TWIN
with tabs[1]:
    st.header("Twin state")
    o, w = src_obs(), get_weather_df()
    if not len(o) or not len(w):
        st.info("Needs observations and weather for the selected source.")
    else:
        ids = sorted(o["unit_id"].unique())
        mode_u = st.radio("Managed object", ["One unit", "Average of reference scales (platform mode)"], horizontal=True)
        if mode_u == "One unit":
            uid = st.selectbox("Unit", ids)
            ou = o[o["unit_id"] == uid]
        else:
            refs = st.multiselect("Reference scales", ids, default=ids[:3])
            ou = SV.aggregate_reference_scales(o, refs) if refs else o.iloc[0:0]
            uid = "PLOT"
        if len(ou):
            tw, log = SV.replay_unit(ou, w, cfg, params)
            st.session_state.twin = (uid, tw, int(log["day"].max()))
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Stage", STAGE_NAMES[tw.stage()])
            c2.metric("Storage now (L/pot)", f"{tw.A:.2f} of {cfg.taw_l:.2f}")
            c3.metric("Kc multiplier (learned)", f"{float(tw.state['kc_adj'][0]):.2f}")
            c4.metric("Root-zone EC (dS/m)", f"{float(log['ec_root'].iloc[-1]):.2f}")
            st.line_chart(log.set_index("day")[["etc_model_L", "eta_model_L"]])
            st.line_chart(log.set_index("day")[["conc_N", "conc_P", "conc_K"]])
            st.line_chart(log.set_index("day")[["ec_root"]])
            st.dataframe(log.tail(60))
        else:
            st.info("Select at least one scale.")

# --------------------------------------------------------------------------- 3 DECIDE
with tabs[2]:
    st.header("Today's recommendation")
    if "twin" not in st.session_state:
        st.info("Open the 'Twin state' tab first so the twin is synchronised with the data.")
    else:
        uid, tw, last_day = st.session_state.twin
        w = get_weather_df()
        nxt = w[w["Day"] > last_day].head(cfg.forecast_horizon_days)
        st.write(f"Twin synchronised to the data of day {last_day}; recommending for day {last_day + 1}.")
        base_eto = float(nxt["ETo_mm"].iloc[0]) if len(nxt) else float(w["ETo_mm"].tail(7).mean())
        base_t = float(nxt["Temperature_C"].iloc[0]) if len(nxt) else float(w["Temperature_C"].tail(7).mean())
        use_tbl = st.checkbox("Use the weather table for the coming days (if it has them)", value=len(nxt) > 0)
        eto = st.number_input("Manual forecast ETo (mm/day)", 0.0, 15.0, round(base_eto, 2))
        tmp = st.number_input("Manual forecast mean temperature (°C)", -5.0, 55.0, round(base_t, 1))
        if st.button("Compute recommendation"):
            if use_tbl and len(nxt):
                fc_eto, fc_t = list(nxt["ETo_mm"]), list(nxt["Temperature_C"])
            else:
                fc_eto, fc_t = [eto] * cfg.forecast_horizon_days, [tmp] * cfg.forecast_horizon_days
            st.session_state.decision = E.recommend(tw, fc_eto, fc_t)
        d = st.session_state.get("decision")
        if d:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Irrigation (L/plant)", f"{d['irrigation_l_per_plant']:.2f}")
            c2.metric("Valve run time (min)", f"{d['valve_runtime_min']:.1f}")
            c3.metric("Fertigation strength", f"{d['fertigation_strength']:.2f} × recipe")
            c4.metric("Supplied EC (dS/m)", f"{d['ec_in_ds_m']:.2f}")
            st.write(f"N / P / K in solution: **{d['conc_n_mg_l']:.0f} / {d['conc_p_mg_l']:.0f} / {d['conc_k_mg_l']:.0f} mg/L** · "
                     f"total water **{d['irrigation_total_l']:.0f} L** · nutrients **{d['nutrient_mass_total_g']:.0f} g**")
            st.write("Why: " + d["reason"])
            if not d["limits_ok"]:
                st.error("This recommendation breaks a site limit. Do not apply.")
            with st.expander("Alternatives the engine considered"):
                st.dataframe(d["alternatives"])
            note = st.text_input("Note (optional)")
            a, b = st.columns(2)
            if a.button("Approve"):
                store.log_decision(uid, last_day + 1, d, "approved", note)
                st.success("Approved and logged. A separate gateway must apply it.")
            if b.button("Reject"):
                store.log_decision(uid, last_day + 1, d, "rejected", note)
                st.info("Rejected and logged.")
        log_d = store.get_decisions()
        if len(log_d):
            st.subheader("Decision log")
            st.dataframe(log_d)
            st.download_button("Download decision log", log_d.to_csv(index=False), "decisions.csv")

# --------------------------------------------------------------------------- 4 PREDICT / OPTIMISE
with tabs[3]:
    st.header("Predict and optimise")
    w = get_weather_df()
    if not len(w):
        st.info("Needs weather for the season.")
    else:
        st.subheader("Season projection from the current twin state")
        if "twin" in st.session_state:
            uid, tw, last_day = st.session_state.twin
            pol = st.radio("Future irrigation policy", ["threshold", "fixed"], horizontal=True,
                           help="threshold = refill at half of readily available water; fixed = table-Kc schedule")
            if st.button("Project yield and water need"):
                df, q = SV.project_remaining(tw, w, last_day + 1, n=150, policy=pol)
                st.session_state.proj = (df, q)
            if "proj" in st.session_state:
                df, q = st.session_state.proj
                c1, c2 = st.columns(2)
                c1.metric("Yield (kg/plant), median", f"{q.loc[0.5, 'yield_kg_plant']:.2f}",
                          f"90% range {q.loc[0.05, 'yield_kg_plant']:.2f}–{q.loc[0.95, 'yield_kg_plant']:.2f}", delta_color="off")
                c2.metric("Water still needed (L/plant), median", f"{q.loc[0.5, 'water_remaining_l_plant']:.0f}",
                          f"90% range {q.loc[0.05, 'water_remaining_l_plant']:.0f}–{q.loc[0.95, 'water_remaining_l_plant']:.0f}",
                          delta_color="off")
                hist = np.histogram(df["yield_kg_plant"], bins=15)
                st.bar_chart(pd.DataFrame({"count": hist[0]}, index=np.round(hist[1][:-1], 2)))
                st.caption("Range reflects forecast and Kc error only. Model-structure error is not included: "
                           "see cross-validation in tab 5.")
        else:
            st.info("Open 'Twin state' first.")
        st.divider()
        st.subheader("Season optimisation (operations research)")
        st.write("Chooses an irrigation fraction and a fertigation scale for each growth stage that maximise profit "
                 "under the current parameters and the weather above. Prices come from Site setup.")
        if st.button("Optimise season plan"):
            with st.spinner("Searching…"):
                st.session_state.opt = E.optimize_season(params, cfg, w)
        if "opt" in st.session_state:
            o = st.session_state.opt
            st.dataframe(pd.DataFrame({"stage": STAGE_NAMES, "irrigation (× reference ETc)": o["irrigation_fraction_by_stage"],
                                       "fertigation (× recipe)": o["fertigation_scale_by_stage"]}))
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Yield (kg/plant)", f"{o['yield_kg_plant']:.2f}")
            c2.metric("Water (L/plant)", f"{o['water_l_plant']:.0f}")
            c3.metric("Nutrients (g/plant)", f"{o['nutrient_g_plant']:.0f}")
            c4.metric("Profit/plant", f"{o['profit']:.1f}")
        if st.button("Yield–water frontier"):
            with st.spinner("Solving six water budgets…"):
                st.session_state.pareto = E.pareto_water(params, cfg, w)
        if "pareto" in st.session_state:
            pf = st.session_state.pareto
            st.line_chart(pf.set_index("water_used_l_plant")[["yield_kg_plant"]])
            st.dataframe(pf)

# --------------------------------------------------------------------------- 5 CALIBRATE
with tabs[4]:
    st.header("Calibrate and validate")
    o, w, y, u = src_obs(), get_weather_df(), store.get_yields(), store.get_units()
    if not len(o) or not len(w):
        st.info("Needs the calibration experiment's observations and weather for the selected source.")
    else:
        y = y[y["source"] == active] if len(y) else y
        st.write(f"{o['unit_id'].nunique()} units · {len(y)} final yields")
        if st.button("Calibrate and run leave-one-replicate-out validation"):
            try:
                exp = SV.pack_from_store(cfg, u, o, w, y, None)
                with st.spinner("Calibrating (water → nutrients → yield) and cross-validating…"):
                    p_all, info = C.calibrate(exp, cfg)
                    summ, P, folds = C.loro_cv(exp, cfg)
                st.session_state.calib = {"p": p_all, "summ": summ, "P": P, "folds": folds, "exp": exp}
            except Exception as e:
                st.error(f"Calibration failed: {e}")
        cal = st.session_state.get("calib")
        if cal:
            st.subheader("Cross-validated skill (predicting a replicate the model never saw)")
            st.dataframe(cal["summ"])
            P = cal["P"]
            st.scatter_chart(P, x="yield_obs", y=["yield_twin", "yield_surface"])
            st.subheader("Calibrated parameters")
            p = cal["p"]
            st.dataframe(pd.DataFrame({"parameter": [f"kc {s}" for s in STAGE_NAMES] + ["p (depletion)", "u_scale N", "u_scale P", "u_scale K",
                                                    "salt uptake factor", "Ymax (kg/plant)", "k_water", "k_nutrient"],
                                       "value": list(p.kc) + [p.p_depl] + list(p.u_scale) +
                                                [p.salt_uptake_factor, p.ymax_kg_plant, p.k_w, p.k_n]}))
            st.write("Spread across folds (with only a few replicates this is the honest uncertainty):")
            st.dataframe(C.jackknife_spread(cal["folds"]))
            if active == "synthetic":
                t = true_params()
                st.write("Parameter recovery (virtual experiment only; the truth is known here):")
                st.dataframe(pd.DataFrame({"parameter": ["kc veg", "kc flower", "kc fruit", "kc harvest", "p", "Ymax", "k_w", "k_n"],
                                           "true": list(t.kc) + [t.p_depl, t.ymax_kg_plant, t.k_w, t.k_n],
                                           "calibrated": list(p.kc) + [p.p_depl, p.ymax_kg_plant, p.k_w, p.k_n]}))
            if st.button("Use these parameters in the platform"):
                store.kv_set("params", params_to_json(p))
                st.session_state.params = p
                st.session_state.params_calibrated = True
                st.success("Saved. Decisions, projections and optimisation now use them.")
                st.rerun()
        st.divider()
        st.subheader("External simulators (DSSAT, HYDRUS-1D, AquaCrop)")
        st.write("Run them yourself, export one row per scenario, import the CSV here to benchmark the twin or to train a surrogate.")
        st.download_button("Scenario template", SC.template().to_csv(index=False), "scenarios_template.csv")
        sf = st.file_uploader("Scenario CSV", type="csv", key="scn")
        if sf is not None and len(w):
            try:
                ext = SC.load_scenarios(pd.read_csv(sf))
                m, out = SC.benchmark(ext, params, cfg, w)
                st.write("Agreement between the twin and the external model (model-to-model, not validation):")
                st.json(out)
                st.dataframe(m)
            except Exception as e:
                st.error(str(e))
        if len(w) and st.button("Build twin scenario library and surrogate"):
            lib = SC.build_twin_library(params, cfg, w, n=400)
            model, skill, imp = SC.train_surrogate(lib)
            st.json(skill)
            st.bar_chart(imp)
            st.caption("The twin itself runs in milliseconds, so a surrogate adds no speed here. "
                       "It is useful for emulating slow external simulators.")

# --------------------------------------------------------------------------- 6 COMPARE
with tabs[5]:
    st.header("Compare")
    st.subheader("A. Calibration experiment: factorial ANOVA")
    y, u = store.get_yields(), store.get_units()
    if len(y):
        y = y[y["source"] == active]
    if len(y) and len(u):
        d = y.merge(u, on="unit_id")
        try:
            st.dataframe(K.factorial_anova(d, "yield_kg_plant", "irrigation_level", "fert_level", "replicate"))
            means = d.groupby(["irrigation_level", "fert_level"])["yield_kg_plant"].mean().unstack()
            st.write("Mean yield (kg/plant) by irrigation × fertilisation")
            st.dataframe(means)
        except Exception as e:
            st.warning(str(e))
        st.caption("Replicate (scale) is the experimental unit; the pots on a scale are subsamples.")
    else:
        st.info("Needs final yields and units for the selected source.")

    st.divider()
    st.subheader("B. Twin-managed vs traditional: virtual trial")
    st.write("'Reality' is a hidden parameter set; the twin sees only noisy weights and drainage samples and uses the "
             "platform's current parameters. A positive result here shows the control logic works under these assumptions; "
             "it is not evidence about a real crop.")
    ns = st.slider("Seasons (weather realisations)", 2, 12, 5)
    fe = st.slider("Forecast error (ETo, relative sd)", 0.0, 0.4, 0.15)
    st.write("Traditional management settings (use the best fixed combination from your factorial as the fair baseline):")
    t1, t2, t3 = st.columns(3)
    tf = t1.slider("Irrigation (× table ETc)", 0.6, 1.4, 1.0)
    tl = t2.slider("Leaching fraction", 0.0, 0.4, 0.15)
    tr = t3.slider("Fertigation (× recipe)", 0.2, 1.4, 1.0)
    if st.button("Run virtual trial"):
        bar = st.progress(0.0)
        allp, tab = K.monte_carlo_trial(true_params(), params, cfg, n_seasons=ns, forecast_error=fe, progress=bar.progress,
                                        trad_frac=tf, trad_leach=tl, trad_fert=tr)
        st.session_state.trial = (allp, tab)
    if "trial" in st.session_state:
        allp, tab = st.session_state.trial
        st.dataframe(tab)
        st.dataframe(allp.groupby("strategy")[K.METRICS].mean())

    st.divider()
    st.subheader("C. Real field/greenhouse trial: twin vs traditional")
    st.write("Upload plot-level season results: columns pair, strategy (twin or traditional), yield_kg_plant, water_l_plant, "
             "nutrient_g_plant. A 'pair' is a matched block or replicate.")
    st.download_button("Template", "pair,strategy,yield_kg_plant,water_l_plant,nutrient_g_plant\n1,twin,,,\n1,traditional,,,\n",
                       "real_trial_template.csv")
    rf = st.file_uploader("Trial CSV", type="csv", key="real")
    if rf is not None:
        r = pd.read_csv(rf)
        try:
            r["water_productivity_kg_m3"] = r["yield_kg_plant"] / r["water_l_plant"] * 1000
            eco = cfg.economics
            r["profit_per_plant"] = (r["yield_kg_plant"] * eco.yield_price_per_kg - r["water_l_plant"] * eco.water_price_per_l
                                     - r["nutrient_g_plant"] * eco.nutrient_price_per_g)
            r["drain_fraction"] = r.get("drain_fraction", np.nan)
            a = r[r.strategy == "twin"].drop(columns="strategy")
            b = r[r.strategy == "traditional"].drop(columns="strategy")
            met = [m for m in K.METRICS if m != "drain_fraction"]
            st.dataframe(K.paired_table(a, b, metrics=met))
        except Exception as e:
            st.error(str(e))

# --------------------------------------------------------------------------- 7 NOTES
with tabs[6]:
    st.header("What this platform is and is not")
    st.markdown("""
**Model.** Per pot: daily water balance (FAO-56 style stress coefficient), nutrient and salt mass balance with
Liebig-type uptake limited by stage demand, and a stage-weighted Jensen-type yield function (water, nutrient, salinity).
Parameters marked ASSUMPTION in `config.py` are starting values, not validated pepper constants.

**Twin.** Morning scale weights correct stored water (scalar Kalman update); weight-derived ETa adapts a Kc multiplier;
drainage samples correct root-zone nutrient and salt mass.

**Decisions.** Receding-horizon search over irrigation volume and fertigation strength using the model, a cost made of
expected yield loss, water and nutrient prices. Always a recommendation that needs approval; site limits are hard checks.

**Known limits.**
- Salinity and nutrient effects are confounded in a fertilisation factorial; the salinity parameters are fixed assumptions.
- Plant fresh-weight gain is removed from the scale signal with an assumed curve; weigh plants to improve it.
- Few replicates mean wide parameter uncertainty; use the fold spread in tab 5.
- Everything validated on synthetic data only shows the software and calibration pipeline work.
""")
