"""Site configuration and model parameters (all JSON-serialisable).

Every number marked ASSUMPTION is a starting value that the calibration
experiment is meant to replace. None of them is a validated pepper constant.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
import json

STAGE_NAMES = ["Vegetative", "Flowering", "Fruit development", "Harvest"]
NUTRIENTS = ["N", "P", "K"]


@dataclass
class Substrate:
    volume_l: float = 15.0              # container volume per pot
    theta_fc: float = 0.45              # container capacity (m3/m3)  ASSUMPTION
    theta_wp: float = 0.15              # lower limit of available water  ASSUMPTION
    pots_per_scale: int = 3
    ground_area_per_plant_m2: float = 0.40   # 1 / plant density
    tare_kg: float = 12.0               # scale + pots + dry substrate (measure it)
    scale_sigma_kg: float = 0.02        # scale repeatability
    plant_fresh_init_kg: float = 0.05   # fresh plant mass per plant at transplanting (weigh it)
    plant_fresh_max_kg: float = 1.20    # gain by the end of the season ASSUMPTION (set 0 to ignore)
    plant_fresh_gdd_scale: float = 1200.0


@dataclass
class Economics:
    """Editable shadow prices used by the decision engine (ASSUMPTIONS)."""
    yield_price_per_kg: float = 15.0
    water_price_per_l: float = 0.01
    nutrient_price_per_g: float = 0.04   # per gram of N+P+K applied


@dataclass
class Params:
    # ---- water ----------------------------------------------------------
    kc: list = field(default_factory=lambda: [0.60, 0.90, 1.05, 0.90])
    p_depl: float = 0.35
    gdd_base_c: float = 10.0
    gdd_bounds: list = field(default_factory=lambda: [300.0, 500.0, 1000.0])
    # ---- nutrients (mg per plant per day, by stage)  ASSUMPTION ---------
    u_max_N: list = field(default_factory=lambda: [40.0, 120.0, 200.0, 140.0])
    u_max_P: list = field(default_factory=lambda: [5.0, 14.0, 24.0, 17.0])
    u_max_K: list = field(default_factory=lambda: [70.0, 200.0, 340.0, 240.0])
    u_scale: list = field(default_factory=lambda: [1.0, 1.0, 1.0])
    c_ref: list = field(default_factory=lambda: [200.0, 45.0, 250.0])   # mg/L reference recipe
    km: list = field(default_factory=lambda: [40.0, 6.0, 50.0])         # mg/L
    salt_per_nutrient: float = 1.6      # mg TDS added per mg N+P+K applied
    salt_uptake_factor: float = 1.0     # fraction of that TDS removed on uptake
    tds_per_ec: float = 640.0           # mg/L per dS/m
    ec_cap_ds_m: float = 8.0            # ceiling: precipitation/adsorption sink  ASSUMPTION
    # ---- yield ----------------------------------------------------------
    ec_threshold: float = 4.0           # dS/m of root-zone solution  ASSUMPTION
    ec_slope: float = 0.08              # relative yield loss per dS/m above threshold  ASSUMPTION
    lam_w: list = field(default_factory=lambda: [0.10, 0.35, 0.45, 0.20])   # sums to ~1.1 (ky-like)
    lam_n: list = field(default_factory=lambda: [0.08, 0.27, 0.45, 0.20])
    ymax_kg_plant: float = 4.0
    k_w: float = 1.0
    k_n: float = 1.0


@dataclass
class SiteConfig:
    name: str = "Pepper site"
    substrate: Substrate = field(default_factory=Substrate)
    economics: Economics = field(default_factory=Economics)
    stage_days_expected: list = field(default_factory=lambda: [30, 20, 50, 60])
    recipe_stage_mult: list = field(default_factory=lambda: [1.0, 1.0, 1.0, 1.0])
    water_ec_ds_m: float = 0.3
    init_conc_frac: float = 0.6         # initial root-zone conc as fraction of c_ref
    plants_total: int = 750             # plants managed by one valve group
    emitter_lph_per_plant: float = 2.0
    max_event_l_per_plant: float = 2.5
    max_ec_in_ds_m: float = 3.0         # hard safety limit on supplied solution EC
    forecast_horizon_days: int = 5

    # -- derived -----------------------------------------------------------
    @property
    def taw_l(self) -> float:
        s = self.substrate
        return s.volume_l * (s.theta_fc - s.theta_wp)

    @property
    def vwp_l(self) -> float:
        s = self.substrate
        return s.volume_l * s.theta_wp

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @staticmethod
    def from_json(text: str) -> "SiteConfig":
        d = json.loads(text)
        d["substrate"] = Substrate(**d["substrate"])
        d["economics"] = Economics(**d["economics"])
        return SiteConfig(**d)


def params_to_json(p: Params) -> str:
    return json.dumps(asdict(p), indent=2)


def params_from_json(text: str) -> Params:
    return Params(**json.loads(text))


def plant_fresh_kg(gdd, cfg: "SiteConfig"):
    """Fresh plant mass per plant (kg): a weighing nuisance that must be removed
    from the scale signal. Linear in thermal time until plant_fresh_gdd_scale."""
    import numpy as np
    s = cfg.substrate
    return s.plant_fresh_init_kg + s.plant_fresh_max_kg * np.minimum(1.0, np.asarray(gdd, float) / s.plant_fresh_gdd_scale)
