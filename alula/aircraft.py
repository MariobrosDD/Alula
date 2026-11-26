import math
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

REQUIRED_SIGN_CONVENTIONS = ("elevator", "aileron", "rudder", "throttle")

AERODYNAMIC_COEFFICIENTS_BY_SECTION = {
    "lift": ("cl_0", "cl_alpha", "cl_q", "cl_elevator"),
    "drag": ("cd_0", "oswald_efficiency"),
    "side_force": ("cy_beta", "cy_rudder"),
    "rolling_moment": (
        "c_roll_beta",
        "c_roll_p",
        "c_roll_r",
        "c_roll_aileron",
        "c_roll_rudder",
    ),
    "pitching_moment": ("cm_0", "cm_alpha", "cm_q", "cm_elevator"),
    "yawing_moment": ("cn_beta", "cn_p", "cn_r", "cn_aileron", "cn_rudder"),
}


@dataclass(frozen=True)
class Geometry:
    wing_area: float
    wingspan: float
    mean_chord: float

    @property
    def aspect_ratio(self) -> float:
        return self.wingspan**2 / self.wing_area


@dataclass(frozen=True)
class MassProperties:
    mass: float
    inertia: NDArray[np.float64]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MassProperties):
            return NotImplemented
        return self.mass == other.mass and np.array_equal(self.inertia, other.inertia)


@dataclass(frozen=True)
class AerodynamicCoefficients:
    alpha_min: float
    alpha_max: float
    beta_max: float
    cl_0: float
    cl_alpha: float
    cl_q: float
    cl_elevator: float
    cd_0: float
    oswald_efficiency: float
    cy_beta: float
    cy_rudder: float
    c_roll_beta: float
    c_roll_p: float
    c_roll_r: float
    c_roll_aileron: float
    c_roll_rudder: float
    cm_0: float
    cm_alpha: float
    cm_q: float
    cm_elevator: float
    cn_beta: float
    cn_p: float
    cn_r: float
    cn_aileron: float
    cn_rudder: float


@dataclass(frozen=True)
class PropulsionData:
    max_shaft_power: float
    propeller_efficiency: float
    static_thrust: float


@dataclass(frozen=True)
class Aircraft:
    name: str
    geometry: Geometry
    mass_properties: MassProperties
    aerodynamics: AerodynamicCoefficients
    propulsion: PropulsionData
    sign_conventions: dict[str, str]


def load_aircraft(path: str | Path) -> Aircraft:
    with open(path, "rb") as file:
        data = tomllib.load(file)

    name = data.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("Missing aircraft 'name'")

    return Aircraft(
        name=name,
        geometry=_read_geometry(_table(data, "geometry")),
        mass_properties=_read_mass_properties(_table(data, "mass")),
        aerodynamics=_read_aerodynamics(_table(data, "aerodynamics")),
        propulsion=_read_propulsion(_table(data, "propulsion")),
        sign_conventions=_read_sign_conventions(_table(data, "sign_conventions")),
    )


def _read_geometry(table: dict[str, Any]) -> Geometry:
    return Geometry(
        wing_area=_positive_number(table, "wing_area_m2", "geometry"),
        wingspan=_positive_number(table, "wingspan_m", "geometry"),
        mean_chord=_positive_number(table, "mean_chord_m", "geometry"),
    )


def _read_mass_properties(table: dict[str, Any]) -> MassProperties:
    ixx = _positive_number(table, "ixx_kg_m2", "mass")
    iyy = _positive_number(table, "iyy_kg_m2", "mass")
    izz = _positive_number(table, "izz_kg_m2", "mass")
    ixz = _number(table, "ixz_kg_m2", "mass")
    inertia = np.array(
        [
            [ixx, 0.0, -ixz],
            [0.0, iyy, 0.0],
            [-ixz, 0.0, izz],
        ]
    )
    return MassProperties(mass=_positive_number(table, "mass_kg", "mass"), inertia=inertia)


def _read_aerodynamics(table: dict[str, Any]) -> AerodynamicCoefficients:
    validity = _table(table, "validity", "aerodynamics.")
    alpha_min = math.radians(_number(validity, "alpha_min_deg", "aerodynamics.validity"))
    alpha_max = math.radians(_number(validity, "alpha_max_deg", "aerodynamics.validity"))
    beta_max = math.radians(_positive_number(validity, "beta_max_deg", "aerodynamics.validity"))
    if alpha_min >= alpha_max:
        raise ValueError("alpha_min_deg must be smaller than alpha_max_deg")

    coefficients = {}
    for section, keys in AERODYNAMIC_COEFFICIENTS_BY_SECTION.items():
        section_table = _table(table, section, "aerodynamics.")
        for key in keys:
            coefficients[key] = _number(section_table, key, f"aerodynamics.{section}")

    if not 0.0 < coefficients["oswald_efficiency"] <= 1.0:
        raise ValueError("oswald_efficiency must be in the interval (0, 1]")

    return AerodynamicCoefficients(
        alpha_min=alpha_min, alpha_max=alpha_max, beta_max=beta_max, **coefficients
    )


def _read_propulsion(table: dict[str, Any]) -> PropulsionData:
    propeller_efficiency = _positive_number(table, "propeller_efficiency", "propulsion")
    if propeller_efficiency > 1.0:
        raise ValueError("propeller_efficiency must be in the interval (0, 1]")
    return PropulsionData(
        max_shaft_power=1000.0 * _positive_number(table, "max_shaft_power_kw", "propulsion"),
        propeller_efficiency=propeller_efficiency,
        static_thrust=_positive_number(table, "static_thrust_n", "propulsion"),
    )


def _read_sign_conventions(table: dict[str, Any]) -> dict[str, str]:
    conventions = {}
    for control in REQUIRED_SIGN_CONVENTIONS:
        description = table.get(control)
        if not isinstance(description, str) or not description:
            raise ValueError(f"Missing sign convention for '{control}' in [sign_conventions]")
        conventions[control] = description
    return conventions


def _table(parent: dict[str, Any], key: str, prefix: str = "") -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Missing table [{prefix}{key}]")
    return value


def _number(table: dict[str, Any], key: str, section: str) -> float:
    value = table.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Missing or non-numeric '{key}' in [{section}]")
    return float(value)


def _positive_number(table: dict[str, Any], key: str, section: str) -> float:
    value = _number(table, key, section)
    if value <= 0.0:
        raise ValueError(f"'{key}' in [{section}] must be positive, got {value}")
    return value
