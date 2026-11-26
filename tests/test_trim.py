import math

import numpy as np
import pytest

from alula.atmosphere import SEA_LEVEL_DENSITY, isa
from alula.dynamics import ANGULAR_VELOCITY_BODY, POSITION_NED, VELOCITY_BODY, aircraft_derivatives
from alula.simulation import Termination, simulate
from alula.trim import trim

EQUIVALENT_AIRSPEED = 55.0


def true_airspeed_at(altitude):
    return EQUIVALENT_AIRSPEED * math.sqrt(SEA_LEVEL_DENSITY / isa(altitude).density)


def test_sea_level_cruise_matches_primer_chapter_15(light_aircraft):
    result = trim(light_aircraft, altitude=0.0, airspeed=55.0)
    assert math.degrees(result.alpha) == pytest.approx(1.14, abs=0.02)
    assert math.degrees(result.controls.elevator) == pytest.approx(-2.90, abs=0.02)
    assert result.controls.throttle == pytest.approx(0.77, abs=0.01)


def test_trimmed_state_has_no_accelerations(light_aircraft):
    result = trim(light_aircraft, altitude=1500.0, airspeed=50.0)
    derivatives = aircraft_derivatives(result.state, result.controls, light_aircraft)
    assert derivatives[VELOCITY_BODY] == pytest.approx(np.zeros(3), abs=1e-8)
    assert derivatives[ANGULAR_VELOCITY_BODY] == pytest.approx(np.zeros(3), abs=1e-8)
    assert result.residual < 1e-9


def test_converges_in_a_few_newton_iterations(light_aircraft):
    assert trim(light_aircraft, altitude=500.0, airspeed=50.0).iterations <= 5


def test_same_equivalent_airspeed_gives_same_alpha_and_elevator(light_aircraft):
    sea_level = trim(light_aircraft, altitude=0.0, airspeed=EQUIVALENT_AIRSPEED)
    high = trim(light_aircraft, altitude=1000.0, airspeed=true_airspeed_at(1000.0))
    assert high.alpha == pytest.approx(sea_level.alpha, abs=1e-9)
    assert high.controls.elevator == pytest.approx(sea_level.controls.elevator, abs=1e-9)
    assert high.controls.throttle > sea_level.controls.throttle


def test_trimmed_aircraft_holds_altitude_and_speed_for_a_minute(light_aircraft):
    airspeed = true_airspeed_at(1000.0)
    result = trim(light_aircraft, altitude=1000.0, airspeed=airspeed)
    flight = simulate(light_aircraft, result.state, result.controls, 60.0, 0.01)
    history = flight.history()
    assert flight.termination is Termination.COMPLETED
    assert history.altitude == pytest.approx(np.full_like(history.altitude, 1000.0), abs=1e-3)
    assert history.airspeed == pytest.approx(np.full_like(history.airspeed, airspeed), abs=1e-4)


def test_steady_climb_rises_at_airspeed_times_sine_of_climb_angle(light_aircraft):
    climb_angle = math.radians(1.5)
    level = trim(light_aircraft, altitude=0.0, airspeed=55.0)
    climb = trim(light_aircraft, altitude=0.0, airspeed=55.0, flight_path_angle=climb_angle)
    derivatives = aircraft_derivatives(climb.state, climb.controls, light_aircraft)
    assert -derivatives[POSITION_NED][2] == pytest.approx(55.0 * math.sin(climb_angle))
    assert climb.controls.throttle > level.controls.throttle


def test_steady_descent_needs_less_throttle(light_aircraft):
    level = trim(light_aircraft, altitude=1000.0, airspeed=55.0)
    descent = trim(light_aircraft, altitude=1000.0, airspeed=55.0, flight_path_angle=math.radians(-3.0))
    derivatives = aircraft_derivatives(descent.state, descent.controls, light_aircraft)
    assert derivatives[POSITION_NED][2] > 0.0
    assert descent.controls.throttle < level.controls.throttle


def test_heading_east_flies_east(light_aircraft):
    result = trim(light_aircraft, altitude=1000.0, airspeed=55.0, heading=math.radians(90.0))
    derivatives = aircraft_derivatives(result.state, result.controls, light_aircraft)
    assert derivatives[POSITION_NED] == pytest.approx([0.0, 55.0, 0.0], abs=1e-9)


def test_flying_faster_needs_more_throttle_and_less_alpha(light_aircraft):
    trims = [trim(light_aircraft, altitude=0.0, airspeed=airspeed) for airspeed in (45.0, 50.0, 55.0, 60.0)]
    throttles = [result.controls.throttle for result in trims]
    alphas = [result.alpha for result in trims]
    assert throttles == sorted(throttles)
    assert alphas == sorted(alphas, reverse=True)


def test_slow_flight_needs_high_angle_of_attack(light_aircraft):
    assert math.degrees(trim(light_aircraft, altitude=0.0, airspeed=28.0).alpha) > 12.0


@pytest.mark.parametrize("airspeed", [20.0, 27.0])
def test_rejects_airspeeds_below_linear_model_range(light_aircraft, airspeed):
    with pytest.raises(ValueError, match="Angle of attack"):
        trim(light_aircraft, altitude=0.0, airspeed=airspeed)


@pytest.mark.parametrize("airspeed", [62.0, 75.0])
def test_rejects_airspeeds_beyond_maximum_power(light_aircraft, airspeed):
    with pytest.raises(ValueError, match="throttle"):
        trim(light_aircraft, altitude=0.0, airspeed=airspeed)


def test_rejects_climb_steeper_than_excess_power_allows(light_aircraft):
    with pytest.raises(ValueError, match="throttle"):
        trim(light_aircraft, altitude=0.0, airspeed=55.0, flight_path_angle=math.radians(3.0))


def test_rejects_altitude_below_sea_level(light_aircraft):
    with pytest.raises(ValueError):
        trim(light_aircraft, altitude=-100.0, airspeed=55.0)
