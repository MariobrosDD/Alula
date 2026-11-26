import math
from dataclasses import dataclass

from alula.aircraft import Aircraft

PER_RADIAN = "1/rad"
DIMENSIONLESS = "-"

AERODYNAMIC_SECTIONS = (
    (
        "Lift and drag",
        (
            ("cl_0", "CL0", DIMENSIONLESS),
            ("cl_alpha", "CLα", PER_RADIAN),
            ("cl_q", "CLq", DIMENSIONLESS),
            ("cl_elevator", "CLδe", PER_RADIAN),
            ("cd_0", "CD0", DIMENSIONLESS),
            ("oswald_efficiency", "Oswald efficiency e", DIMENSIONLESS),
        ),
    ),
    (
        "Side force",
        (
            ("cy_beta", "CYβ", PER_RADIAN),
            ("cy_rudder", "CYδr", PER_RADIAN),
        ),
    ),
    (
        "Rolling moment",
        (
            ("c_roll_beta", "Clβ", PER_RADIAN),
            ("c_roll_p", "Clp", DIMENSIONLESS),
            ("c_roll_r", "Clr", DIMENSIONLESS),
            ("c_roll_aileron", "Clδa", PER_RADIAN),
            ("c_roll_rudder", "Clδr", PER_RADIAN),
        ),
    ),
    (
        "Pitching moment",
        (
            ("cm_0", "Cm0", DIMENSIONLESS),
            ("cm_alpha", "Cmα", PER_RADIAN),
            ("cm_q", "Cmq", DIMENSIONLESS),
            ("cm_elevator", "Cmδe", PER_RADIAN),
        ),
    ),
    (
        "Yawing moment",
        (
            ("cn_beta", "Cnβ", PER_RADIAN),
            ("cn_p", "Cnp", DIMENSIONLESS),
            ("cn_r", "Cnr", DIMENSIONLESS),
            ("cn_aileron", "Cnδa", PER_RADIAN),
            ("cn_rudder", "Cnδr", PER_RADIAN),
        ),
    ),
)


@dataclass(frozen=True)
class SummaryRow:
    quantity: str
    value: str
    unit: str


@dataclass(frozen=True)
class SummarySection:
    title: str
    rows: tuple[SummaryRow, ...]


def format_value(value: float) -> str:
    return f"{value:.4g}"


def aircraft_summary(aircraft: Aircraft) -> list[SummarySection]:
    geometry = aircraft.geometry
    mass = aircraft.mass_properties
    propulsion = aircraft.propulsion
    aerodynamics = aircraft.aerodynamics

    sections = [
        _section(
            "Geometry",
            ("Wing area", geometry.wing_area, "m²"),
            ("Wingspan", geometry.wingspan, "m"),
            ("Mean chord", geometry.mean_chord, "m"),
            ("Aspect ratio", geometry.aspect_ratio, DIMENSIONLESS),
        ),
        _section(
            "Mass and inertia",
            ("Mass", mass.mass, "kg"),
            ("Ixx", mass.inertia[0, 0], "kg·m²"),
            ("Iyy", mass.inertia[1, 1], "kg·m²"),
            ("Izz", mass.inertia[2, 2], "kg·m²"),
            ("Ixz", -mass.inertia[0, 2], "kg·m²"),
        ),
        _section(
            "Propulsion",
            ("Maximum shaft power", propulsion.max_shaft_power / 1000.0, "kW"),
            ("Propeller efficiency", propulsion.propeller_efficiency, DIMENSIONLESS),
            ("Static thrust", propulsion.static_thrust, "N"),
        ),
        _section(
            "Linear model validity",
            ("Minimum angle of attack", math.degrees(aerodynamics.alpha_min), "deg"),
            ("Maximum angle of attack", math.degrees(aerodynamics.alpha_max), "deg"),
            ("Maximum sideslip", math.degrees(aerodynamics.beta_max), "deg"),
        ),
    ]
    for title, coefficients in AERODYNAMIC_SECTIONS:
        sections.append(
            _section(
                title,
                *((label, getattr(aerodynamics, field), unit) for field, label, unit in coefficients),
            )
        )
    sections.append(
        SummarySection(
            "Control sign conventions",
            tuple(
                SummaryRow(control.capitalize(), description, "")
                for control, description in aircraft.sign_conventions.items()
            ),
        )
    )
    return sections


def _section(title: str, *rows: tuple[str, float, str]) -> SummarySection:
    return SummarySection(
        title, tuple(SummaryRow(quantity, format_value(value), unit) for quantity, value, unit in rows)
    )
