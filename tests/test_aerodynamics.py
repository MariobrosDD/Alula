import math

import numpy as np
import pytest

from alula.aerodynamics import (
    AirData,
    aerodynamic_loads,
    air_data_from_body_velocity,
    induced_drag_factor,
)
from alula.controls import Controls

SEA_LEVEL_DENSITY = 1.225
CRUISE_AIRSPEED = 55.0
TRIM_ALPHA = 0.019909
TRIM_ELEVATOR = -0.05057
NO_ROTATION = np.zeros(3)


def cruise_loads(aircraft, alpha=TRIM_ALPHA, beta=0.0, rates=NO_ROTATION, controls=None, density=SEA_LEVEL_DENSITY):
    return aerodynamic_loads(
        aircraft,
        AirData(CRUISE_AIRSPEED, alpha, beta),
        rates,
        density,
        controls if controls is not None else Controls(elevator=TRIM_ELEVATOR),
    )


def lift_and_drag(force_body, alpha):
    x, _, z = force_body
    lift = x * math.sin(alpha) - z * math.cos(alpha)
    drag = -x * math.cos(alpha) - z * math.sin(alpha)
    return lift, drag


def test_air_data_of_straight_flight():
    assert air_data_from_body_velocity([55.0, 0.0, 0.0]) == pytest.approx((55.0, 0.0, 0.0))


def test_air_data_reproduces_primer_example():
    air_data = air_data_from_body_velocity([54.989, 0.0, 1.0949])
    assert air_data.airspeed == pytest.approx(55.0, abs=1e-3)
    assert math.degrees(air_data.alpha) == pytest.approx(1.14, abs=1e-2)
    assert air_data.beta == 0.0


def test_wind_from_the_right_gives_positive_sideslip():
    assert air_data_from_body_velocity([50.0, 5.0, 0.0]).beta > 0.0


def test_air_data_rejects_zero_airspeed():
    with pytest.raises(ValueError):
        air_data_from_body_velocity([0.0, 0.0, 0.0])


def test_induced_drag_factor_matches_primer(light_aircraft):
    assert induced_drag_factor(light_aircraft) == pytest.approx(0.0533, abs=1e-4)


def test_trimmed_cruise_matches_primer_chapter_12(light_aircraft):
    loads = cruise_loads(light_aircraft)
    lift, drag = lift_and_drag(loads.force_body, TRIM_ALPHA)
    assert lift == pytest.approx(9815.0, rel=1e-3)
    assert drag == pytest.approx(981.4, rel=1e-3)
    assert loads.force_body[0] == pytest.approx(-785.8, rel=1e-3)
    assert loads.force_body[2] == pytest.approx(-9832.8, rel=1e-3)
    assert loads.moment_body[1] == pytest.approx(0.0, abs=1.0)


def test_trimmed_cruise_lift_supports_the_weight(light_aircraft):
    lift, _ = lift_and_drag(cruise_loads(light_aircraft).force_body, TRIM_ALPHA)
    weight = light_aircraft.mass_properties.mass * 9.81
    assert lift == pytest.approx(weight, rel=1e-3)


def test_symmetric_flight_has_no_lateral_loads(light_aircraft):
    loads = cruise_loads(light_aircraft)
    assert loads.force_body[1] == 0.0
    assert loads.moment_body[0] == 0.0
    assert loads.moment_body[2] == 0.0


def test_more_angle_of_attack_pitches_nose_down(light_aircraft):
    nominal = cruise_loads(light_aircraft).moment_body[1]
    disturbed = cruise_loads(light_aircraft, alpha=TRIM_ALPHA + 0.02).moment_body[1]
    assert disturbed < nominal


def test_sideslip_yaws_nose_into_the_wind(light_aircraft):
    assert cruise_loads(light_aircraft, beta=0.05).moment_body[2] > 0.0


def test_sideslip_rolls_away_through_dihedral_effect(light_aircraft):
    assert cruise_loads(light_aircraft, beta=0.05).moment_body[0] < 0.0


@pytest.mark.parametrize(
    "rates, axis",
    [
        ([0.3, 0.0, 0.0], 0),
        ([0.0, 0.3, 0.0], 1),
        ([0.0, 0.0, 0.3], 2),
    ],
)
def test_rotation_is_damped(light_aircraft, rates, axis):
    still = cruise_loads(light_aircraft).moment_body[axis]
    rotating = cruise_loads(light_aircraft, rates=rates).moment_body[axis]
    assert rotating - still < 0.0


def test_trailing_edge_up_elevator_pitches_nose_up(light_aircraft):
    neutral = cruise_loads(light_aircraft, controls=Controls()).moment_body[1]
    stick_back = cruise_loads(light_aircraft, controls=Controls(elevator=-0.1)).moment_body[1]
    assert stick_back > neutral


def test_positive_aileron_rolls_right_with_adverse_yaw(light_aircraft):
    moment = cruise_loads(light_aircraft, controls=Controls(aileron=0.1)).moment_body
    assert moment[0] > 0.0
    assert moment[2] < 0.0


def test_positive_rudder_yaws_nose_left(light_aircraft):
    assert cruise_loads(light_aircraft, controls=Controls(rudder=0.1)).moment_body[2] < 0.0


@pytest.mark.parametrize("alpha_deg", [-8.0, -4.0, 0.0, 5.0, 10.0, 13.0])
def test_drag_is_always_positive(light_aircraft, alpha_deg):
    alpha = math.radians(alpha_deg)
    _, drag = lift_and_drag(cruise_loads(light_aircraft, alpha=alpha).force_body, alpha)
    assert drag > 0.0


def test_loads_scale_with_density(light_aircraft):
    full = cruise_loads(light_aircraft, beta=0.05, rates=[0.1, 0.1, 0.1])
    half = cruise_loads(light_aircraft, beta=0.05, rates=[0.1, 0.1, 0.1], density=SEA_LEVEL_DENSITY / 2)
    assert half.force_body == pytest.approx(full.force_body / 2)
    assert half.moment_body == pytest.approx(full.moment_body / 2)


@pytest.mark.parametrize("alpha_deg", [-8.5, 13.5, 20.0])
def test_rejects_angle_of_attack_outside_linear_range(light_aircraft, alpha_deg):
    with pytest.raises(ValueError, match="Angle of attack"):
        cruise_loads(light_aircraft, alpha=math.radians(alpha_deg))


def test_rejects_excessive_sideslip(light_aircraft):
    with pytest.raises(ValueError, match="Sideslip"):
        cruise_loads(light_aircraft, beta=math.radians(25.0))
