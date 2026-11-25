import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from pepper_twin.config import Params, SiteConfig
from pepper_twin import models as M, engine as E, calibration as C, compare as K
from pepper_twin import scenarios as S, services as SV
from pepper_twin.synthetic import virtual_experiment, make_weather

_VE = {}


def ve():
    if "v" not in _VE:
        _VE["v"] = virtual_experiment(days=140, seed=1)
    return _VE["v"]


def exp_all():
    if "e" not in _VE:
        v = ve()
        _VE["e"] = C.pack_experiment(v["units"], v["obs"], v["weather"], v["yields"], v["cfg"])
    return _VE["e"]


class TestModel(unittest.TestCase):
    def test_water_and_nutrient_closure(self):
        cfg, p = SiteConfig(), Params()
        st = M.init_state(6, p, cfg)
        rng = np.random.default_rng(0)
        for _ in range(140):
            irr = rng.uniform(0, 3.0, 6)
            cin = np.array([[200.] * 6, [45.] * 6, [250.] * 6]) * rng.uniform(0, 1.3, 6)
            o = M.step(st, p, cfg, 4.5, 24.0, irr, cin)
            self.assertLess(np.abs(o["water_resid"]).max(), 1e-9)
            self.assertLess(np.abs(o["n_resid"]).max(), 1e-6)
            self.assertTrue((st["A"] >= -1e-12).all() and (st["A"] <= cfg.taw_l + 1e-9).all())

    def test_weight_roundtrip_and_tare(self):
        cfg = SiteConfig()
        a = np.array([0.5, 2.0, cfg.taw_l])
        w = M.weight_from_storage(a, cfg, 0.3)
        self.assertTrue(np.allclose(M.storage_from_weight(w, cfg, 0.3), a))
        tare = M.tare_from_saturated_weight(M.weight_from_storage(cfg.taw_l, cfg), cfg)
        self.assertAlmostEqual(tare, cfg.substrate.tare_kg, places=9)

    def test_eta_from_weights_balance(self):
        cfg = SiteConfig()
        n = cfg.substrate.pots_per_scale
        # 3 pots lose 4.5 L, receive 6 L, drain 1 L -> w_next = w + 6 - 1 - 4.5
        self.assertAlmostEqual(float(M.eta_from_weights(40.0, 40.5, 6.0, 1.0, cfg)), 4.5 / n, places=9)

    def test_stress_reduces_yield_monotonically(self):
        cfg, p = SiteConfig(), Params()
        w = make_weather(140, 3)
        ys = []
        for f in (0.5, 0.75, 1.0, 1.25):
            r = E.simulate_season(p, cfg, w, E.traditional_policy(cfg, p, frac=f))
            ys.append(float(r["yield_kg_plant"][0]))
        self.assertTrue(ys[0] < ys[1] < ys[2])


class TestAssimilation(unittest.TestCase):
    def test_weight_pulls_state_toward_observation(self):
        cfg, p = SiteConfig(), Params()
        tw = E.UnitTwin(cfg, p, a0_frac=1.0)
        target = 1.0
        w = float(M.weight_from_storage(target, cfg, 0.05))   # 0.05 kg = plant_fresh_init_kg at day 0
        before = abs(tw.A - target)
        tw.morning_update(w)
        self.assertLess(abs(tw.A - target), before)

    def test_replay_tracks_scale_and_projection_is_ordered(self):
        v = ve()
        cfg = v["cfg"]
        uid = "I3F3-R1"
        o = v["obs"][v["obs"].unit_id == uid]
        p, _ = C.calibrate(exp_all(), cfg)
        tw, log = SV.replay_unit(o.iloc[:60], v["weather"], cfg, p)
        self.assertEqual(len(log), 60)
        self.assertTrue(0.7 < log["kc_multiplier"].iloc[-1] < 1.4)
        df, q = SV.project_remaining(tw, v["weather"], start_day=61, n=60)
        self.assertLessEqual(q.loc[0.05, "yield_kg_plant"], q.loc[0.95, "yield_kg_plant"])
        self.assertGreater(q.loc[0.5, "water_remaining_l_plant"], 10)


class TestCalibration(unittest.TestCase):
    def test_parameter_recovery_on_virtual_experiment(self):
        v = ve()
        p, _ = C.calibrate(exp_all(), v["cfg"])
        t = v["truth"]
        for a, b in zip(p.kc, t.kc):
            self.assertLess(abs(a - b) / b, 0.10)
        self.assertLess(abs(p.ymax_kg_plant - t.ymax_kg_plant) / t.ymax_kg_plant, 0.12)
        self.assertLess(abs(p.p_depl - t.p_depl), 0.12)

    def test_loro_twin_beats_grand_mean(self):
        v = ve()
        summ, P, folds = C.loro_cv(exp_all(), v["cfg"])
        r = summ.set_index("target")["RMSE"]
        self.assertLess(r["Yield: twin (kg/plant)"], r["Yield: grand mean (kg/plant)"])
        self.assertEqual(len(folds), 3)


class TestEngine(unittest.TestCase):
    def test_mpc_irrigates_when_dry_not_when_wet(self):
        cfg, p = SiteConfig(), Params()
        dry = E.UnitTwin(cfg, p, a0_frac=0.15)
        wet = E.UnitTwin(cfg, p, a0_frac=1.0)
        d1 = E.recommend(dry, [5.0] * 5, [24.0] * 5)
        d2 = E.recommend(wet, [3.0] * 5, [24.0] * 5)
        self.assertGreater(d1["irrigation_l_per_plant"], 0.2)
        self.assertLess(d2["irrigation_l_per_plant"], d1["irrigation_l_per_plant"])
        self.assertEqual(d1["approval_status"], "pending")
        self.assertTrue(d1["limits_ok"])
        self.assertLessEqual(d1["ec_in_ds_m"], cfg.max_ec_in_ds_m + 1e-9)

    def test_mpc_respects_ec_limit(self):
        cfg, p = SiteConfig(), Params()
        cfg.max_ec_in_ds_m = 1.2
        tw = E.UnitTwin(cfg, p, a0_frac=0.2)
        d = E.recommend(tw, [5.0] * 5, [24.0] * 5)
        self.assertLessEqual(d["ec_in_ds_m"], 1.2 + 1e-9)

    def test_optimizer_beats_traditional_under_its_own_model(self):
        cfg, p = SiteConfig(), Params()
        w = make_weather(140, 5)
        opt = E.optimize_season(p, cfg, w, maxiter=25)
        trad = E.simulate_season(p, cfg, w, E.traditional_policy(cfg, p))
        eco = cfg.economics
        tp = trad["yield_kg_plant"][0] * eco.yield_price_per_kg - trad["water_l_plant"][0] * eco.water_price_per_l \
            - trad["nutrient_g_plant"][0] * eco.nutrient_price_per_g
        self.assertGreaterEqual(opt["profit"], tp - 1e-6)

    def test_closed_loop_runs_and_conserves(self):
        v = ve()
        p, _ = C.calibrate(exp_all(), v["cfg"])
        w = make_weather(140, 9)
        for s in ("twin", "traditional"):
            r = E.run_closed_loop(s, v["truth"], p, v["cfg"], w, seed=2, n_plots=2)
            self.assertEqual(len(r["summary"]), 2)
            self.assertTrue((r["summary"]["yield_kg_plant"] > 0).all())
            self.assertTrue((r["summary"]["drain_fraction"] <= 1).all())


class TestStatsAndScenarios(unittest.TestCase):
    def test_factorial_anova_sums_and_detects_effect(self):
        rng = np.random.default_rng(1)
        rows = []
        for a in range(4):
            for b in range(4):
                for r in range(3):
                    rows.append({"A": a, "B": b, "R": r, "y": 2.0 * a + 0.3 * b + rng.normal(0, 0.2)})
        t = K.factorial_anova(pd.DataFrame(rows), "y", "A", "B", "R").set_index("source")
        self.assertAlmostEqual(t["SS"].drop("Total").sum(), t.loc["Total", "SS"], places=8)
        self.assertLess(t.loc["A", "p"], 1e-6)
        self.assertGreater(t.loc["A x B", "p"], 0.01)

    def test_paired_table(self):
        a = pd.DataFrame({"pair": [1, 2, 3, 4], **{m: [5, 6, 7, 8.0] for m in K.METRICS}})
        b = pd.DataFrame({"pair": [1, 2, 3, 4], **{m: [4, 5, 6, 7.0] for m in K.METRICS}})
        t = K.paired_table(a, b)
        self.assertAlmostEqual(t.loc[0, "difference"], 1.0)
        self.assertEqual(int(t.loc[0, "n pairs"]), 4)

    def test_surrogate_and_benchmark(self):
        cfg, p = SiteConfig(), Params()
        w = make_weather(140, 4)
        lib = S.build_twin_library(p, cfg, w, n=200)
        model, skill, imp = S.train_surrogate(lib)
        # measured skill: ~0.80 at 200 scenarios, ~0.85 at 400, ~0.90 at 1500
        self.assertGreater(skill["cv_R2"], 0.75)
        merged, out = S.benchmark(lib.head(20), p, cfg, w)
        self.assertLess(out["yield_kg_plant"]["RMSE"], 1e-9)


class TestStore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["PEPPER_DB_PATH"] = os.path.join(self.tmp.name, "t.sqlite3")

    def tearDown(self):
        os.environ.pop("PEPPER_DB_PATH", None)
        self.tmp.cleanup()

    def test_roundtrip_qc_and_decisions(self):
        from pepper_twin import store
        qc = store.add_observation({"unit_id": "U1", "day": 1, "pot_weight_kg": 30.0}, "manual")
        self.assertEqual(qc, "")
        qc = store.add_observation({"unit_id": "U1", "day": 2, "drain_ec_ds_m": 55.0}, "iot")
        self.assertIn("drain_ec_ds_m", qc)
        obs = store.get_observations("U1")
        self.assertEqual(set(obs["source"]), {"manual", "iot"})
        with self.assertRaises(ValueError):
            store.add_observation({"unit_id": "U1", "day": 3}, "bogus")
        dec = {"irrigation_l_per_plant": 1.0, "fertigation_strength": 1.0, "conc_n_mg_l": 200, "conc_p_mg_l": 45,
               "conc_k_mg_l": 250, "ec_in_ds_m": 1.8, "reason": "t"}
        i = store.log_decision("U1", 2, dec)
        store.set_decision_status(i, "approved")
        self.assertEqual(len(store.get_decisions("approved")), 1)
        store.kv_set("a", "b")
        self.assertEqual(store.kv_get("a"), "b")

    def test_virtual_data_goes_through_store_to_calibration(self):
        from pepper_twin import store
        v = ve()
        store.save_units(v["units"]); store.save_weather(v["weather"], "synthetic")
        store.save_yields(v["yields"], "synthetic")
        store.add_observations(v["obs"].iloc[:2000], "synthetic")
        self.assertEqual(len(store.get_observations(source="synthetic")), 2000)


if __name__ == "__main__":
    unittest.main()
