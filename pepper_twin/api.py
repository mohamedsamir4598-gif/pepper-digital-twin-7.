"""Sensor gateway API.   Run:  uvicorn pepper_twin.api:app --host 0.0.0.0 --port 8000

Set TWIN_API_KEY to require the header  X-API-Key.  Without it the API refuses to start
on a non-local deployment unless TWIN_API_ALLOW_OPEN=1 (lab bench use only).

The API only RECEIVES readings and SERVES approved decisions. It never actuates anything:
a separate, human-reviewed gateway must poll /api/v1/decisions/approved and drive the valves.
"""
from __future__ import annotations
import os
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from . import store

app = FastAPI(title="Pepper digital twin sensor API", version="2.0.0")


def _auth(x_api_key: Optional[str] = Header(default=None)):
    key = os.environ.get("TWIN_API_KEY")
    if key:
        if x_api_key != key:
            raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")
    elif os.environ.get("TWIN_API_ALLOW_OPEN") != "1":
        raise HTTPException(status_code=503, detail="TWIN_API_KEY is not configured on the server")


class Reading(BaseModel):
    unit_id: str
    day: int = Field(ge=1, description="Day after transplanting (1 = first day)")
    pot_weight_kg: Optional[float] = None
    irrigation_l: Optional[float] = None
    drainage_l: Optional[float] = None
    applied_n_mg_l: Optional[float] = None
    applied_p_mg_l: Optional[float] = None
    applied_k_mg_l: Optional[float] = None
    applied_ec_ds_m: Optional[float] = None
    drain_n_mg_l: Optional[float] = None
    drain_p_mg_l: Optional[float] = None
    drain_k_mg_l: Optional[float] = None
    drain_ec_ds_m: Optional[float] = None
    drain_ph: Optional[float] = None
    air_temp_c: Optional[float] = None
    air_rh_pct: Optional[float] = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/v1/observations", dependencies=[Depends(_auth)])
def post_observation(r: Reading):
    qc = store.add_observation(r.model_dump(), source="iot")
    return {"stored": True, "qc_flags": qc}


@app.get("/api/v1/observations/{unit_id}", dependencies=[Depends(_auth)])
def get_obs(unit_id: str):
    return store.get_observations(unit_id=unit_id, source="iot").to_dict("records")


@app.get("/api/v1/decisions/approved", dependencies=[Depends(_auth)])
def approved_decisions():
    return store.get_decisions(status="approved").to_dict("records")


@app.post("/api/v1/decisions/{decision_id}/executed", dependencies=[Depends(_auth)])
def mark_executed(decision_id: int):
    store.set_decision_status(decision_id, "executed", "reported by gateway")
    return {"ok": True}
