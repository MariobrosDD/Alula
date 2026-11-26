from dataclasses import fields

import pytest

from alula.aircraft import AerodynamicCoefficients
from alula.gui.aircraft_summary import (
    AERODYNAMIC_SECTIONS,
    aircraft_summary,
    format_value,
)

VALIDITY_FIELDS = {"alpha_min", "alpha_max", "beta_max"}


def section(summary, title):
    return next(entry for entry in summary if entry.title == title)


def row(summary, title, quantity):
    return next(entry for entry in section(summary, title).rows if entry.quantity == quantity)


@pytest.fixture
def summary(light_aircraft):
    return aircraft_summary(light_aircraft)


def test_sections_appear_in_reading_order(summary):
    assert [entry.title for entry in summary] == [
        "Geometry",
        "Mass and inertia",
        "Propulsion",
        "Linear model validity",
        "Lift and drag",
        "Side force",
        "Rolling moment",
        "Pitching moment",
        "Yawing moment",
        "Control sign conventions",
    ]


@pytest.mark.parametrize(
    "value, text",
    [(16.2, "16.2"), (7.46913580, "7.469"), (100000.0, "1e+05"), (-0.74, "-0.74"), (0.0, "0")],
)
def test_values_are_shown_with_four_significant_figures(value, text):
    assert format_value(value) == text


def test_geometry_includes_derived_aspect_ratio(summary):
    assert row(summary, "Geometry", "Wing area").value == "16.2"
    assert row(summary, "Geometry", "Wing area").unit == "m²"
    assert row(summary, "Geometry", "Aspect ratio").value == "7.469"


def test_inertia_shows_product_of_inertia_with_file_sign(summary):
    assert row(summary, "Mass and inertia", "Iyy").value == "1825"
    assert row(summary, "Mass and inertia", "Ixz").value == "0"


def test_power_is_shown_in_kilowatts(summary):
    power = row(summary, "Propulsion", "Maximum shaft power")
    assert (power.value, power.unit) == ("100", "kW")


def test_validity_limits_are_shown_in_degrees(summary):
    alpha_max = row(summary, "Linear model validity", "Maximum angle of attack")
    assert (alpha_max.value, alpha_max.unit) == ("13", "deg")
    assert row(summary, "Linear model validity", "Minimum angle of attack").value == "-8"


def test_stability_derivatives_carry_symbols_and_units(summary):
    cm_alpha = row(summary, "Pitching moment", "Cmα")
    assert (cm_alpha.value, cm_alpha.unit) == ("-0.74", "1/rad")
    assert row(summary, "Rolling moment", "Clp").unit == "-"


def test_every_aerodynamic_coefficient_is_listed_exactly_once():
    listed = [field for _, coefficients in AERODYNAMIC_SECTIONS for field, _, _ in coefficients]
    expected = {field.name for field in fields(AerodynamicCoefficients)} - VALIDITY_FIELDS
    assert sorted(listed) == sorted(expected)


def test_sign_conventions_are_listed_for_every_control(summary):
    controls = [entry.quantity for entry in section(summary, "Control sign conventions").rows]
    assert controls == ["Elevator", "Aileron", "Rudder", "Throttle"]
