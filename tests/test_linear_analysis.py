import math

import numpy as np
import pytest

from alula.atmosphere import STANDARD_GRAVITY
from alula.controls import Controls
from alula.dynamics import ANGULAR_VELOCITY_BODY, VELOCITY_BODY, State
from alula.integrators import integrate
from alula.linear_analysis import (
    INPUT_NAMES,
    STATE_NAMES,
    LinearModel,
    Mode,
    aircraft_modes,
    linearize,
)
from alula.simulation import simulate
from alula.trim import trim

ALTITUDE = 1000.0
AIRSPEED = 57.7


def state_index(name):
    return STATE_NAMES.index(name)


def input_index(name):
    return INPUT_NAMES.index(name)


@pytest.fixture
def trimmed(light_aircraft):
    return trim(light_aircraft, ALTITUDE, AIRSPEED)


@pytest.fixture
def model(light_aircraft, trimmed):
    return linearize(light_aircraft, trimmed)


@pytest.fixture
def modes(model):
    return aircraft_modes(model)


def derivative(model, rate_of, with_respect_to):
    return model.state_matrix[state_index(rate_of), state_index(with_respect_to)]


def control_effect(model, rate_of, control):
    return model.input_matrix[state_index(rate_of), input_index(control)]


def test_matrix_shapes(model):
    assert model.state_matrix.shape == (8, 8)
    assert model.input_matrix.shape == (8, 4)


def test_longitudinal_and_lateral_motions_are_decoupled_in_level_flight(model):
    assert model.state_matrix[:4, 4:] == pytest.approx(np.zeros((4, 4)), abs=1e-9)
    assert model.state_matrix[4:, :4] == pytest.approx(np.zeros((4, 4)), abs=1e-9)
    assert model.input_matrix[:4, 2:] == pytest.approx(np.zeros((4, 2)), abs=1e-9)
    assert model.input_matrix[4:, :2] == pytest.approx(np.zeros((4, 2)), abs=1e-9)


def test_pitch_attitude_integrates_pitch_rate(model):
    assert derivative(model, "theta", "q") == pytest.approx(1.0)
    assert derivative(model, "phi", "p") == pytest.approx(1.0)


def test_state_matrix_shows_stability_and_damping(model):
    assert derivative(model, "q", "w") < 0.0
    assert derivative(model, "q", "q") < 0.0
    assert derivative(model, "p", "p") < 0.0
    assert derivative(model, "r", "r") < 0.0
    assert derivative(model, "r", "v") > 0.0
    assert derivative(model, "p", "v") < 0.0


def test_gravity_appears_through_pitch_attitude(model):
    assert derivative(model, "u", "theta") == pytest.approx(-STANDARD_GRAVITY, rel=1e-3)


def test_input_matrix_follows_control_sign_conventions(model):
    assert control_effect(model, "q", "elevator") < 0.0
    assert control_effect(model, "u", "throttle") > 0.0
    assert control_effect(model, "p", "aileron") > 0.0
    assert control_effect(model, "r", "rudder") < 0.0


def test_all_five_modes_are_stable(modes):
    for mode in (modes.short_period, modes.phugoid, modes.roll_subsidence, modes.dutch_roll, modes.spiral):
        assert mode.is_stable


def test_modes_fall_within_typical_light_aircraft_ranges(modes):
    assert 1.0 < modes.short_period.period < 3.0
    assert 0.4 < modes.short_period.damping_ratio < 0.8
    assert 20.0 < modes.phugoid.period < 60.0
    assert 0.02 < modes.phugoid.damping_ratio < 0.2
    assert modes.roll_subsidence.time_constant < 1.0
    assert 2.0 < modes.dutch_roll.period < 6.0
    assert 0.05 < modes.dutch_roll.damping_ratio < 0.3
    assert modes.spiral.time_constant > 20.0
    assert not modes.roll_subsidence.is_oscillatory
    assert not modes.spiral.is_oscillatory


def test_phugoid_matches_the_simulated_oscillation(light_aircraft, trimmed, modes):
    pulse = Controls(elevator=trimmed.controls.elevator + math.radians(-2.0), throttle=trimmed.controls.throttle)

    def pilot(time, state):
        return pulse if 1.0 <= time < 2.0 else trimmed.controls

    history = simulate(light_aircraft, trimmed.state, pilot, 130.0, 0.05).history()
    settled = history.time > 10.0
    time, airspeed = history.time[settled], history.airspeed[settled] - history.airspeed[settled].mean()
    crossings = [time[i] for i in range(1, len(time)) if airspeed[i - 1] < 0.0 <= airspeed[i]]
    simulated_period = np.diff(crossings).mean()

    assert simulated_period == pytest.approx(modes.phugoid.period, rel=0.02)

    peaks = [
        np.abs(airspeed[(time >= start) & (time < start + simulated_period)]).max()
        for start in crossings[:3]
    ]
    expected_decay_per_cycle = math.exp(modes.phugoid.eigenvalue.real * modes.phugoid.period)
    assert peaks[1] / peaks[0] == pytest.approx(expected_decay_per_cycle, rel=0.1)
    assert peaks[2] / peaks[1] == pytest.approx(expected_decay_per_cycle, rel=0.1)


def test_linear_model_predicts_small_disturbance_of_full_model(light_aircraft, trimmed, model):
    disturbed = trimmed.state.to_array()
    disturbed[VELOCITY_BODY.start + 2] += 1.0
    disturbed[ANGULAR_VELOCITY_BODY.start + 1] += 0.02
    nonlinear = simulate(light_aircraft, State.from_array(disturbed), trimmed.controls, 10.0, 0.01)

    initial_deviation = np.zeros(8)
    initial_deviation[state_index("w")] = 1.0
    initial_deviation[state_index("q")] = 0.02
    linear = integrate(lambda time, deviation: model.state_matrix @ deviation, initial_deviation, 0.01, 10.0)

    reference = trimmed.state
    for step in (100, 300, 1000):
        state = nonlinear.states[step]
        nonlinear_w_deviation = state[VELOCITY_BODY][2] - reference.velocity_body[2]
        nonlinear_q = state[ANGULAR_VELOCITY_BODY][1]
        assert nonlinear_w_deviation == pytest.approx(linear.states[step, state_index("w")], abs=0.02)
        assert nonlinear_q == pytest.approx(linear.states[step, state_index("q")], abs=1e-3)


def test_faster_flight_lengthens_phugoid_and_speeds_up_short_period(light_aircraft):
    slow = aircraft_modes(linearize(light_aircraft, trim(light_aircraft, ALTITUDE, 40.0)))
    fast = aircraft_modes(linearize(light_aircraft, trim(light_aircraft, ALTITUDE, 60.0)))
    assert fast.phugoid.period > slow.phugoid.period
    assert fast.short_period.natural_frequency > slow.short_period.natural_frequency


def test_phugoid_is_of_the_order_of_lanchester_estimate(modes):
    lanchester = math.sqrt(2.0) * math.pi * AIRSPEED / STANDARD_GRAVITY
    assert 1.0 < modes.phugoid.period / lanchester < 1.5


def test_mode_properties_of_a_known_eigenvalue():
    mode = Mode(complex(-1.0, 2.0))
    assert mode.is_stable
    assert mode.is_oscillatory
    assert mode.period == pytest.approx(math.pi)
    assert mode.natural_frequency == pytest.approx(math.sqrt(5.0))
    assert mode.damping_ratio == pytest.approx(1.0 / math.sqrt(5.0))
    assert mode.time_constant == pytest.approx(1.0)


def test_unstable_aperiodic_mode():
    mode = Mode(complex(0.5, 0.0))
    assert not mode.is_stable
    assert not mode.is_oscillatory
    assert mode.period == math.inf
    assert mode.time_constant == pytest.approx(2.0)


def test_rejects_matrix_without_expected_mode_structure():
    model = LinearModel(state_matrix=-np.eye(8), input_matrix=np.zeros((8, 4)))
    with pytest.raises(ValueError, match="longitudinal"):
        aircraft_modes(model)
