import math

import numpy as np
import pytest

from alula.atmosphere import SEA_LEVEL_DENSITY, STANDARD_GRAVITY, isa
from alula.controls import Controls
from alula.dynamics import QUATERNION
from alula.propulsion import altitude_power_fraction
from alula.simulation import Termination, initial_state, simulate

ALTITUDE = 1000.0
TRIM_ALPHA = 0.019909
TRIM_ELEVATOR = -0.05057
TRIM_DRAG = 981.4


@pytest.fixture
def cruise(light_aircraft):
    density = isa(ALTITUDE).density
    airspeed = 55.0 * math.sqrt(SEA_LEVEL_DENSITY / density)
    engine = light_aircraft.propulsion
    throttle = TRIM_DRAG * airspeed / (
        engine.propeller_efficiency * engine.max_shaft_power * altitude_power_fraction(density)
    )
    return {
        "aircraft": light_aircraft,
        "airspeed": airspeed,
        "start": initial_state(ALTITUDE, airspeed, TRIM_ALPHA),
        "controls": Controls(elevator=TRIM_ELEVATOR, throttle=throttle),
    }


def with_pulse(base, start, duration, **deflections):
    def control_law(time, state):
        if start <= time < start + duration:
            return Controls(
                elevator=base.elevator + deflections.get("elevator", 0.0),
                aileron=base.aileron + deflections.get("aileron", 0.0),
                rudder=base.rudder + deflections.get("rudder", 0.0),
                throttle=base.throttle,
            )
        return base

    return control_law


def upward_crossing_times(time, signal):
    centred = signal - signal.mean()
    return [
        time[index]
        for index in range(1, len(centred))
        if centred[index - 1] < 0.0 <= centred[index]
    ]


def test_initial_state_describes_requested_flight_condition():
    state = initial_state(1500.0, 60.0, 0.05, heading=math.radians(90.0), flight_path_angle=0.1)
    assert state.altitude == 1500.0
    assert np.linalg.norm(state.velocity_body) == pytest.approx(60.0)
    assert math.atan2(state.velocity_body[2], state.velocity_body[0]) == pytest.approx(0.05)
    assert state.angular_velocity_body == pytest.approx(np.zeros(3))


@pytest.mark.parametrize("altitude, airspeed", [(0.0, 50.0), (-10.0, 50.0), (1000.0, 0.0)])
def test_initial_state_rejects_impossible_conditions(altitude, airspeed):
    with pytest.raises(ValueError):
        initial_state(altitude, airspeed, 0.0)


def test_trimmed_aircraft_keeps_flying_straight_and_level(cruise):
    result = simulate(cruise["aircraft"], cruise["start"], cruise["controls"], 60.0, 0.02)
    history = result.history()
    assert result.termination is Termination.COMPLETED
    assert result.final_time == pytest.approx(60.0)
    assert history.altitude == pytest.approx(np.full_like(history.altitude, ALTITUDE), abs=3.0)
    assert history.airspeed == pytest.approx(np.full_like(history.airspeed, cruise["airspeed"]), abs=0.5)
    assert np.all(history.roll == 0.0)
    assert np.all(history.yaw == 0.0)
    assert np.all(history.beta == 0.0)
    assert history.north[-1] == pytest.approx(cruise["airspeed"] * 60.0, rel=1e-2)


def test_elevator_pulse_excites_a_damped_phugoid(cruise):
    pilot = with_pulse(cruise["controls"], start=1.0, duration=1.0, elevator=math.radians(-2.0))
    history = simulate(cruise["aircraft"], cruise["start"], pilot, 130.0, 0.05).history()

    settled = history.time > 10.0
    crossings = upward_crossing_times(history.time[settled], history.airspeed[settled])
    periods = np.diff(crossings)
    lanchester_period = math.sqrt(2.0) * math.pi * cruise["airspeed"] / STANDARD_GRAVITY

    assert len(periods) >= 2
    assert periods.mean() == pytest.approx(32.4, rel=0.05)
    assert 0.8 < periods.mean() / lanchester_period < 1.5

    cycle_amplitudes = [
        np.ptp(history.altitude[(history.time >= start) & (history.time < start + periods.mean())])
        for start in crossings[:3]
    ]
    assert cycle_amplitudes[0] > cycle_amplitudes[1] > cycle_amplitudes[2]


def test_aircraft_glides_with_engine_off(cruise):
    controls = Controls(elevator=TRIM_ELEVATOR, throttle=0.0)
    history = simulate(cruise["aircraft"], cruise["start"], controls, 120.0, 0.05).history()
    late = history.time >= 80.0
    altitude_lost = history.altitude[late][0] - history.altitude[late][-1]
    distance_flown = history.north[late][-1] - history.north[late][0]
    assert history.altitude[-1] < ALTITUDE - 300.0
    assert 10.0 < distance_flown / altitude_lost < 14.0


def test_positive_aileron_rolls_right_and_turns_right(cruise):
    pilot = with_pulse(cruise["controls"], start=1.0, duration=1.0, aileron=math.radians(5.0))
    result = simulate(cruise["aircraft"], cruise["start"], pilot, 8.0, 0.01)
    history = result.history()
    assert history.roll.max() > math.radians(5.0)
    assert history.yaw[-1] > 0.0
    assert history.east[-1] > 0.0


def test_quaternion_stays_normalised(cruise):
    pilot = with_pulse(cruise["controls"], start=1.0, duration=2.0, aileron=0.1, rudder=-0.05)
    result = simulate(cruise["aircraft"], cruise["start"], pilot, 10.0, 0.01)
    norms = np.linalg.norm(result.states[:, QUATERNION], axis=1)
    assert norms == pytest.approx(np.ones_like(norms), abs=1e-12)


def test_dive_into_the_ground_stops_on_ground_contact(cruise):
    start = initial_state(50.0, cruise["airspeed"], TRIM_ALPHA, flight_path_angle=math.radians(-20.0))
    result = simulate(cruise["aircraft"], start, Controls(elevator=TRIM_ELEVATOR), 30.0, 0.01)
    assert result.termination is Termination.GROUND_CONTACT
    assert result.final_time < 5.0
    assert result.final_state.altitude < 1.0


def test_violent_pull_up_stops_at_model_limit(cruise):
    controls = Controls(elevator=-0.3, throttle=cruise["controls"].throttle)
    result = simulate(cruise["aircraft"], cruise["start"], controls, 30.0, 0.01)
    assert result.termination is Termination.MODEL_LIMIT
    assert "Angle of attack" in result.termination_message
    assert result.final_time < 1.0


def test_controls_are_sampled_once_per_step_and_held(cruise):
    sampled_times = []

    def recording_pilot(time, state):
        sampled_times.append(time)
        return cruise["controls"]

    result = simulate(cruise["aircraft"], cruise["start"], recording_pilot, 1.0, 0.01)
    assert len(sampled_times) == 100
    assert sampled_times == pytest.approx(0.01 * np.arange(100))
    assert len(result.controls) == len(result.times) - 1


def test_history_arrays_have_one_value_per_time(cruise):
    result = simulate(cruise["aircraft"], cruise["start"], cruise["controls"], 2.0, 0.1)
    history = result.history()
    assert all(len(values) == len(result.times) for values in history)


@pytest.mark.parametrize("time_step, duration", [(0.0, 10.0), (0.01, -1.0), (0.3, 1.0)])
def test_rejects_invalid_time_settings(cruise, time_step, duration):
    with pytest.raises(ValueError):
        simulate(cruise["aircraft"], cruise["start"], cruise["controls"], duration, time_step)
