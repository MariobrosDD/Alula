import math

import numpy as np
import pytest

from alula.aircraft import load_aircraft


def write_variant(tmp_path, original_file, old_text, new_text):
    text = original_file.read_text(encoding="utf-8")
    assert old_text in text
    variant = tmp_path / "variant.toml"
    variant.write_text(text.replace(old_text, new_text), encoding="utf-8")
    return variant


def test_loads_name_and_geometry(light_aircraft):
    assert light_aircraft.name == "Light aircraft"
    assert light_aircraft.geometry.wing_area == 16.2
    assert light_aircraft.geometry.wingspan == 11.0
    assert light_aircraft.geometry.mean_chord == 1.49


def test_aspect_ratio_matches_primer(light_aircraft):
    assert light_aircraft.geometry.aspect_ratio == pytest.approx(7.469, abs=1e-3)


def test_builds_symmetric_inertia_matrix(light_aircraft):
    assert light_aircraft.mass_properties.mass == 1000.0
    expected = np.diag([1285.0, 1825.0, 2667.0])
    assert light_aircraft.mass_properties.inertia == pytest.approx(expected)


def test_product_of_inertia_enters_with_negative_sign(tmp_path, light_aircraft_file):
    variant = write_variant(tmp_path, light_aircraft_file, "ixz_kg_m2 = 0.0", "ixz_kg_m2 = 50.0")
    inertia = load_aircraft(variant).mass_properties.inertia
    assert inertia[0, 2] == inertia[2, 0] == -50.0


def test_converts_validity_limits_to_radians(light_aircraft):
    aerodynamics = light_aircraft.aerodynamics
    assert aerodynamics.alpha_min == pytest.approx(math.radians(-8.0))
    assert aerodynamics.alpha_max == pytest.approx(math.radians(13.0))
    assert aerodynamics.beta_max == pytest.approx(math.radians(20.0))


def test_reads_stability_derivatives(light_aircraft):
    aerodynamics = light_aircraft.aerodynamics
    assert aerodynamics.cl_alpha == 4.96
    assert aerodynamics.cm_alpha == -0.74
    assert aerodynamics.cn_beta == 0.065
    assert aerodynamics.c_roll_p == -0.47


def test_reads_propulsion_in_si_units(light_aircraft):
    propulsion = light_aircraft.propulsion
    assert propulsion.max_shaft_power == 100_000.0
    assert propulsion.propeller_efficiency == 0.7
    assert propulsion.static_thrust == 2500.0


def test_states_every_control_sign_convention(light_aircraft):
    assert set(light_aircraft.sign_conventions) == {"elevator", "aileron", "rudder", "throttle"}


@pytest.mark.parametrize(
    "old_text, new_text, message",
    [
        ("cm_alpha = -0.74\n", "", "cm_alpha"),
        ("mass_kg = 1000.0", "mass_kg = -1000.0", "mass_kg"),
        ("wingspan_m = 11.0", 'wingspan_m = "eleven"', "wingspan_m"),
        ("alpha_max_deg = 13.0", "alpha_max_deg = -10.0", "alpha_min_deg"),
        ("oswald_efficiency = 0.8", "oswald_efficiency = 1.2", "oswald_efficiency"),
        ('rudder = "positive', 'unused = "positive', "rudder"),
        ("[geometry]", "[geometry_typo]", "geometry"),
        ("[propulsion]", "[propulsion_typo]", "propulsion"),
        ("propeller_efficiency = 0.7", "propeller_efficiency = 1.5", "propeller_efficiency"),
        ("static_thrust_n = 2500.0", "static_thrust_n = 0.0", "static_thrust_n"),
    ],
)
def test_rejects_invalid_files_with_clear_message(
    tmp_path, light_aircraft_file, old_text, new_text, message
):
    variant = write_variant(tmp_path, light_aircraft_file, old_text, new_text)
    with pytest.raises(ValueError, match=message):
        load_aircraft(variant)
