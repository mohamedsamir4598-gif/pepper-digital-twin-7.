"""SQLite persistence. One schema serves every data mode: manual entry, CSV import,
live sensor API and the synthetic demo. Every observation row carries its source.

NOTE: on Streamlit Community Cloud the local file is wiped on restart. Set
PEPPER_DB_PATH to a persistent volume, or export the tables regularly.
"""
from __future__ import annotations
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

SOURCES = ("manual", "csv", "iot", "synthetic")
OBS_FIELDS = ["pot_weight_kg", "irrigation_l", "drainage_l",
              "applied_n_mg_l", "applied_p_mg_l", "applied_k_mg_l", "applied_ec_ds_m",
              "drain_n_mg_l", "drain_p_mg_l", "drain_k_mg_l", "drain_ec_ds_m", "drain_ph",
              "air_temp_c", "air_rh_pct"]
# physically plausible ranges used by quality control (low, high)
RANGES = {"pot_weight_kg": (0.5, 500), "irrigation_l": (0, 200), "drainage_l": (0, 200),
          "applied_n_mg_l": (0, 600), "applied_p_mg_l": (0, 200), "applied_k_mg_l": (0, 800),
          "applied_ec_ds_m": (0, 8), "drain_n_mg_l": (0, 1500), "drain_p_mg_l": (0, 600),
          "drain_k_mg_l": (0, 1500), "drain_ec_ds_m": (0, 20), "drain_ph": (3, 10),
          "air_temp_c": (-5, 55), "air_rh_pct": (0, 100)}


def db_path() -> Path:
    return Path(os.environ.get("PEPPER_DB_PATH", Path(__file__).resolve().parent.parent / "data" / "twin.sqlite3"))


def connect() -> sqlite3.Connection:
    p = db_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(p)
    con.row_factory = sqlite3.Row
    return con


def init_db():
    with connect() as c:
        c.executescript(f"""
        CREATE TABLE IF NOT EXISTS units(unit_id TEXT PRIMARY KEY, irrigation_level REAL, fert_level REAL,
            replicate INTEGER, n_pots INTEGER, role TEXT DEFAULT 'experiment');
        CREATE TABLE IF NOT EXISTS observations(id INTEGER PRIMARY KEY AUTOINCREMENT, unit_id TEXT NOT NULL,
            day INTEGER NOT NULL, source TEXT NOT NULL, created_utc TEXT NOT NULL, qc TEXT DEFAULT '',
            {", ".join(f + " REAL" for f in OBS_FIELDS)},
            UNIQUE(unit_id, day, source));
        CREATE TABLE IF NOT EXISTS weather(day INTEGER PRIMARY KEY, ETo_mm REAL, Temperature_C REAL, source TEXT);
        CREATE TABLE IF NOT EXISTS yields(unit_id TEXT PRIMARY KEY, yield_kg_plant REAL, source TEXT);
        CREATE TABLE IF NOT EXISTS decisions(id INTEGER PRIMARY KEY AUTOINCREMENT, unit_id TEXT, day INTEGER,
            created_utc TEXT, irrigation_l_per_plant REAL, fertigation_strength REAL, conc_n REAL, conc_p REAL,
            conc_k REAL, ec_in REAL, reason TEXT, status TEXT, note TEXT);
        CREATE TABLE IF NOT EXISTS kv(k TEXT PRIMARY KEY, v TEXT);
        """)


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# -- quality control ---------------------------------------------------------
def qc_record(rec: dict) -> str:
    flags = []
    for k, (lo, hi) in RANGES.items():
        v = rec.get(k)
        if v is not None and not pd.isna(v) and not (lo <= float(v) <= hi):
            flags.append(f"{k} out of range [{lo},{hi}]")
    return "; ".join(flags)


def qc_series(df: pd.DataFrame, max_weight_jump_kg=3.0) -> pd.DataFrame:
    """Flag implausible jumps in scale weight within a unit (e.g. someone leaning on the scale)."""
    out = []
    for uid, g in df.sort_values("day").groupby("unit_id"):
        w = g["pot_weight_kg"].astype(float)
        jump = w.diff().abs()
        for d in g.loc[jump > max_weight_jump_kg, "day"]:
            out.append({"unit_id": uid, "day": int(d), "issue": f"weight jump > {max_weight_jump_kg} kg"})
    return pd.DataFrame(out, columns=["unit_id", "day", "issue"])


# -- observations --------------------------------------------------------------
def add_observation(rec: dict, source: str, replace: bool = True) -> str:
    if source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}")
    if not rec.get("unit_id") or rec.get("day") is None:
        raise ValueError("unit_id and day are required")
    init_db()
    qc = qc_record(rec)
    cols = ["unit_id", "day", "source", "created_utc", "qc"] + OBS_FIELDS
    vals = [str(rec["unit_id"]), int(rec["day"]), source, _now(), qc] + [
        None if rec.get(f) is None or pd.isna(rec.get(f)) else float(rec[f]) for f in OBS_FIELDS]
    verb = "INSERT OR REPLACE" if replace else "INSERT"
    with connect() as c:
        c.execute(f"{verb} INTO observations({','.join(cols)}) VALUES({','.join('?' * len(cols))})", vals)
    return qc


def add_observations(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Bulk insert; returns rows that raised QC flags."""
    flagged = []
    for r in df.to_dict("records"):
        qc = add_observation(r, source)
        if qc:
            flagged.append({"unit_id": r.get("unit_id"), "day": r.get("day"), "qc": qc})
    return pd.DataFrame(flagged, columns=["unit_id", "day", "qc"])


def get_observations(unit_id: str | None = None, source: str | None = None) -> pd.DataFrame:
    init_db()
    q, args = "SELECT * FROM observations WHERE 1=1", []
    if unit_id:
        q += " AND unit_id=?"; args.append(unit_id)
    if source:
        q += " AND source=?"; args.append(source)
    with connect() as c:
        return pd.read_sql_query(q + " ORDER BY unit_id, day", c, params=args)


def clear_source(source: str):
    init_db()
    with connect() as c:
        for t in ("observations", "yields", "weather"):
            c.execute(f"DELETE FROM {t} WHERE source=?", (source,))


# -- units, weather, yields --------------------------------------------------------
def save_units(df: pd.DataFrame):
    init_db()
    with connect() as c:
        for r in df.to_dict("records"):
            c.execute("INSERT OR REPLACE INTO units(unit_id,irrigation_level,fert_level,replicate,n_pots,role) VALUES(?,?,?,?,?,?)",
                      (r["unit_id"], r.get("irrigation_level"), r.get("fert_level"), r.get("replicate"),
                       r.get("n_pots", 3), r.get("role", "experiment")))


def get_units() -> pd.DataFrame:
    init_db()
    with connect() as c:
        return pd.read_sql_query("SELECT * FROM units ORDER BY unit_id", c)


def save_weather(df: pd.DataFrame, source: str):
    init_db()
    with connect() as c:
        for r in df.to_dict("records"):
            c.execute("INSERT OR REPLACE INTO weather(day,ETo_mm,Temperature_C,source) VALUES(?,?,?,?)",
                      (int(r["Day"]), float(r["ETo_mm"]), float(r["Temperature_C"]), source))


def get_weather() -> pd.DataFrame:
    init_db()
    with connect() as c:
        df = pd.read_sql_query("SELECT day AS Day, ETo_mm, Temperature_C, source FROM weather ORDER BY day", c)
    return df


def save_yields(df: pd.DataFrame, source: str):
    init_db()
    with connect() as c:
        for r in df.to_dict("records"):
            c.execute("INSERT OR REPLACE INTO yields(unit_id,yield_kg_plant,source) VALUES(?,?,?)",
                      (str(r["unit_id"]), float(r["yield_kg_plant"]), source))


def get_yields() -> pd.DataFrame:
    init_db()
    with connect() as c:
        return pd.read_sql_query("SELECT * FROM yields ORDER BY unit_id", c)


# -- decisions ------------------------------------------------------------------------
def log_decision(unit_id: str, day: int, dec: dict, status: str = "pending", note: str = "") -> int:
    init_db()
    with connect() as c:
        cur = c.execute(
            "INSERT INTO decisions(unit_id,day,created_utc,irrigation_l_per_plant,fertigation_strength,conc_n,conc_p,conc_k,ec_in,reason,status,note) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (unit_id, day, _now(), dec["irrigation_l_per_plant"], dec["fertigation_strength"], dec["conc_n_mg_l"],
             dec["conc_p_mg_l"], dec["conc_k_mg_l"], dec["ec_in_ds_m"], dec["reason"], status, note))
        return int(cur.lastrowid)


def set_decision_status(decision_id: int, status: str, note: str = ""):
    if status not in ("pending", "approved", "rejected", "executed"):
        raise ValueError("bad status")
    with connect() as c:
        c.execute("UPDATE decisions SET status=?, note=? WHERE id=?", (status, note, decision_id))


def get_decisions(status: str | None = None) -> pd.DataFrame:
    init_db()
    with connect() as c:
        if status:
            return pd.read_sql_query("SELECT * FROM decisions WHERE status=? ORDER BY id DESC", c, params=[status])
        return pd.read_sql_query("SELECT * FROM decisions ORDER BY id DESC", c)


# -- key/value (config, calibrated parameters) -------------------------------------------
def kv_set(k: str, v: str):
    init_db()
    with connect() as c:
        c.execute("INSERT OR REPLACE INTO kv(k,v) VALUES(?,?)", (k, v))


def kv_get(k: str, default: str | None = None) -> str | None:
    init_db()
    with connect() as c:
        r = c.execute("SELECT v FROM kv WHERE k=?", (k,)).fetchone()
    return r["v"] if r else default
