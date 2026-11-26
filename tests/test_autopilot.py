import math

import numpy as np
import pytest

from alula.autopilot import (
    AILERON_LIMIT,
    ELEVATOR_LIMITS,
    RUDDER_LIMIT,
    Autopilot,
    AutopilotTargets,
    angle_difference,
)
from alula.dynamics import ANGULAR_VELOCITY_BODY, VELOCITY_BODY, State
from alula.rotations import euler_from_quaternion
from alula.simulation import Termination, simulate
from alula.trim import trim

ALTITUDE = 1000.0
AIRSPEED = 57.7
TIME_STEP = 0.02
MAX_BANK = math.radians(25.0)


def fly(aircraft, trimmed, targets, duration, start=None):
    pilot = Autopilot(targets, trimmed, TIME_STEP)
    result = simulate(aircraft, start or trimmed.state, pilot, duration, TIME_STEP)
    assert result.termination is Termination.COMPLETED
    return result, result.history()


def heading_error_deg(history, heading):
    return math.degrees(angle_difference(heading, history.yaw[-1]))


@pytest.fixture
def trimmed(light_aircraft):
    return trim(light_aircraft, ALTITUDE, AIRSPEED)


@pytest.mark.parametrize(
    "target_deg, current_deg, expected_deg",
    [(10.0, 350.0, 20.0), (350.0, 10.0, -20.0), (90.0, 0.0, 90.0), (0.0, 0.0, 0.0), (-90.0, 90.0, -180.0)],
)
def test_angle_difference_takes_the_shortest_way_round(target_deg, current_deg, expected_deg):
    difference = angle_difference(math.radians(target_deg), math.radians(current_deg))
    assert math.degrees(difference) == pytest.approx(expected_deg)


def test_angle_difference_stays_within_half_a_turn():
    for target in np.linspace(-10.0, 10.0, 41):
        assert -math.pi <= angle_difference(target, 1.234) < math.pi


@pytest.mark.parametrize("altitude, airspeed", [(-1.0, 50.0), (1000.0, 0.0)])
def test_rejects_impossible_targets(altitude, airspeed):
    with pytest.raises(ValueError):
        AutopilotTargets(altitude=altitude, heading=0.0, airspeed=airspeed)


def test_rejects_non_positive_time_step(trimmed):
    with pytest.raises(ValueError, match="Time step"):
        Autopilot(AutopilotTargets(ALTITUDE, 0.0, AIRSPEED), trimmed, time_step=0.0)


def test_rejects_calls_at_a_different_rate_than_configured(trimmed):
    pilot = Autopilot(AutopilotTargets(ALTITUDE, 0.0, AIRSPEED), trimmed, time_step=0.02)
    pilot(0.0, trimmed.state)
    with pytest.raises(ValueError, match="every 0.02 s"):
        pilot(0.05, trimmed.state)


def test_holding_the_trim_condition_keeps_trim_controls(light_aircraft, trimmed):
    result, history = fly(light_aircraft, trimmed, AutopilotTargets(ALTITUDE, 0.0, AIRSPEED), 60.0)
    assert history.altitude == pytest.approx(np.full_like(history.altitude, ALTITUDE), abs=0.1)
    final_controls = result.controls[-1]
    assert final_controls.elevator == pytest.approx(trimmed.controls.elevator, abs=1e-6)
    assert final_controls.throttle == pytest.approx(trimmed.controls.throttle, abs=1e-6)
    assert final_controls.aileron == pytest.approx(0.0, abs=1e-9)
    assert final_controls.rudder == pytest.approx(0.0, abs=1e-9)


def test_commanded_climb_starts_by_pulling_the_stick_back(trimmed):
    pilot = Autopilot(AutopilotTargets(ALTITUDE + 100.0, 0.0, AIRSPEED), trimmed, TIME_STEP)
    assert pilot(0.0, trimmed.state).elevator < trimmed.controls.elevator


@pytest.mark.parametrize("target_altitude", [1100.0, 900.0])
def test_captures_and_holds_a_new_altitude(light_aircraft, trimmed, target_altitude):
    result, history = fly(light_aircraft, trimmed, AutopilotTargets(target_altitude, 0.0, AIRSPEED), 150.0)
    trim_pitch = euler_from_quaternion(trimmed.state.quaternion).pitch
    assert history.altitude[-1] == pytest.approx(target_altitude, abs=1.0)
    assert abs(history.altitude - ALTITUDE).max() < abs(target_altitude - ALTITUDE) + 10.0
    assert np.abs(history.pitch - trim_pitch).max() < math.radians(8.0)
    assert history.alpha.max() < math.radians(10.0)


def test_climb_keeps_every_control_within_its_limits(light_aircraft, trimmed):
    result, _ = fly(light_aircraft, trimmed, AutopilotTargets(1200.0, math.radians(120.0), 50.0), 60.0)
    for controls in result.controls:
        assert ELEVATOR_LIMITS[0] <= controls.elevator <= ELEVATOR_LIMITS[1]
        assert abs(controls.aileron) <= AILERON_LIMIT
        assert abs(controls.rudder) <= RUDDER_LIMIT
        assert 0.0 <= controls.throttle <= 1.0


def test_coordinated_turn_to_a_new_heading(light_aircraft, trimmed):
    heading = math.radians(90.0)
    _, history = fly(light_aircraft, trimmed, AutopilotTargets(ALTITUDE, heading, AIRSPEED), 60.0)
    assert heading_error_deg(history, heading) == pytest.approx(0.0, abs=1.0)
    assert np.abs(history.roll).max() < MAX_BANK + math.radians(1.0)
    assert np.abs(history.beta).max() < math.radians(3.0)
    assert history.altitude == pytest.approx(np.full_like(history.altitude, ALTITUDE), abs=3.0)


@pytest.mark.parametrize("heading_deg, roll_sign", [(90.0, 1.0), (-90.0, -1.0)])
def test_banks_towards_the_side_of_the_turn(light_aircraft, trimmed, heading_deg, roll_sign):
    _, history = fly(light_aircraft, trimmed, AutopilotTargets(ALTITUDE, math.radians(heading_deg), AIRSPEED), 5.0)
    assert roll_sign * history.roll[-1] > math.radians(10.0)


def test_turns_the_short_way_across_north(light_aircraft):
    start_heading = math.radians(350.0)
    trimmed = trim(light_aircraft, ALTITUDE, AIRSPEED, heading=start_heading)
    target = math.radians(10.0)
    _, history = fly(light_aircraft, trimmed, AutopilotTargets(ALTITUDE, target, AIRSPEED), 40.0)
    assert history.roll[history.time <= 5.0].max() > math.radians(5.0)
    assert history.roll.min() > math.radians(-1.0)
    assert heading_error_deg(history, target) == pytest.approx(0.0, abs=1.0)


def test_autothrottle_captures_a_new_airspeed(light_aircraft, trimmed):
    _, history = fly(light_aircraft, trimmed, AutopilotTargets(ALTITUDE, 0.0, 50.0), 120.0)
    assert history.airspeed[-1] == pytest.approx(50.0, abs=0.2)
    assert history.altitude == pytest.approx(np.full_like(history.altitude, ALTITUDE), abs=10.0)


def test_climbs_turns_and_slows_down_at_the_same_time(light_aircraft, trimmed):
    targets = AutopilotTargets(1200.0, math.radians(120.0), 50.0)
    _, history = fly(light_aircraft, trimmed, targets, 240.0)
    assert history.altitude[-1] == pytest.approx(1200.0, abs=1.0)
    assert heading_error_deg(history, targets.heading) == pytest.approx(0.0, abs=1.0)
    assert history.airspeed[-1] == pytest.approx(50.0, abs=0.5)
    assert history.alpha.max() < math.radians(10.0)


def test_recovers_from_a_gust(light_aircraft, trimmed):
    disturbed = trimmed.state.to_array()
    disturbed[VELOCITY_BODY.start + 2] += 3.0
    disturbed[ANGULAR_VELOCITY_BODY.start + 2] += 0.05
    _, history = fly(
        light_aircraft, trimmed, AutopilotTargets(ALTITUDE, 0.0, AIRSPEED), 90.0, start=State.from_array(disturbed)
    )
    assert history.altitude[-1] == pytest.approx(ALTITUDE, abs=1.0)
    assert heading_error_deg(history, 0.0) == pytest.approx(0.0, abs=1.0)
    assert history.airspeed[-1] == pytest.approx(AIRSPEED, abs=0.2)
