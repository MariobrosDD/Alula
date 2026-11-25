import math

import numpy as np
import pytest

from alula.atmosphere import STANDARD_GRAVITY
from alula.dynamics import POSITION_NED, QUATERNION, VELOCITY_BODY, State, rigid_body_derivatives
from alula.integrators import euler_step, integrate, rk4_step
from alula.rotations import euler_from_quaternion, normalize_quaternion

EXACT_DECAY_AT_ONE_SECOND = math.exp(-1.0)


def exponential_decay(time, state):
    return -state


def harmonic_oscillator(time, state):
    position, velocity = state
    return np.array([velocity, -position])


def oscillator_energy(state):
    position, velocity = state
    return 0.5 * (position**2 + velocity**2)


def final_value(step, time_step, derivative=exponential_decay, initial=(1.0,), duration=1.0):
    return integrate(derivative, initial, time_step, duration, step=step).states[-1]


@pytest.mark.parametrize(
    "step, time_step, expected",
    [
        (euler_step, 0.5, 0.250000),
        (euler_step, 0.1, 0.348678),
        (rk4_step, 0.5, 0.368171),
        (rk4_step, 0.1, 0.367880),
    ],
)
def test_reproduces_primer_table_for_exponential_decay(step, time_step, expected):
    assert final_value(step, time_step)[0] == pytest.approx(expected, abs=1e-6)


def test_euler_is_first_order():
    coarse_error = abs(final_value(euler_step, 0.1)[0] - EXACT_DECAY_AT_ONE_SECOND)
    fine_error = abs(final_value(euler_step, 0.05)[0] - EXACT_DECAY_AT_ONE_SECOND)
    assert coarse_error / fine_error == pytest.approx(2.0, rel=0.1)


def test_rk4_is_fourth_order():
    coarse_error = abs(final_value(rk4_step, 0.1)[0] - EXACT_DECAY_AT_ONE_SECOND)
    fine_error = abs(final_value(rk4_step, 0.05)[0] - EXACT_DECAY_AT_ONE_SECOND)
    assert coarse_error / fine_error == pytest.approx(16.0, rel=0.1)


def test_rk4_is_exact_for_cubic_rates():
    trajectory = integrate(lambda time, state: np.array([time**3]), [0.0], 0.5, 2.0)
    assert trajectory.states[-1, 0] == pytest.approx(2.0**4 / 4.0, abs=1e-12)


def test_derivative_receives_the_current_time():
    trajectory = integrate(
        lambda time, state: np.array([math.cos(time)]),
        [0.0],
        time_step=math.pi / 200,
        duration=math.pi / 2,
        initial_time=math.pi / 2,
    )
    assert trajectory.times[0] == pytest.approx(math.pi / 2)
    assert trajectory.times[-1] == pytest.approx(math.pi)
    assert trajectory.states[-1, 0] == pytest.approx(math.sin(math.pi) - math.sin(math.pi / 2), abs=1e-9)


def test_rk4_brings_oscillator_back_after_one_period():
    period = 2.0 * math.pi
    trajectory = integrate(harmonic_oscillator, [1.0, 0.0], period / 1000, period)
    assert trajectory.states[-1] == pytest.approx([1.0, 0.0], abs=1e-9)


def test_rk4_conserves_oscillator_energy():
    period = 2.0 * math.pi
    trajectory = integrate(harmonic_oscillator, [1.0, 0.0], period / 100, 10 * period)
    energies = [oscillator_energy(state) for state in trajectory.states]
    assert max(energies) == pytest.approx(0.5, rel=1e-5)
    assert min(energies) == pytest.approx(0.5, rel=1e-5)


def test_euler_artificially_pumps_energy_into_oscillator():
    period = 2.0 * math.pi
    trajectory = integrate(harmonic_oscillator, [1.0, 0.0], period / 100, period, step=euler_step)
    assert oscillator_energy(trajectory.states[-1]) > 1.4 * oscillator_energy(trajectory.states[0])


def test_trajectory_shape_and_times():
    trajectory = integrate(harmonic_oscillator, [1.0, 0.0], 0.1, 2.0)
    assert trajectory.states.shape == (21, 2)
    assert trajectory.times == pytest.approx(np.linspace(0.0, 2.0, 21))
    assert trajectory.states[0] == pytest.approx([1.0, 0.0])


def test_zero_duration_returns_only_initial_state():
    trajectory = integrate(harmonic_oscillator, [1.0, 0.0], 0.1, 0.0)
    assert trajectory.states.shape == (1, 2)


def test_post_step_is_applied_after_every_step():
    trajectory = integrate(
        lambda time, state: np.ones(1),
        [0.0],
        time_step=0.5,
        duration=5.0,
        post_step=lambda state: np.minimum(state, 2.0),
    )
    assert trajectory.states[-1, 0] == pytest.approx(2.0)
    assert trajectory.states.max() <= 2.0


def test_initial_state_is_not_modified():
    initial = np.array([1.0, 0.0])
    integrate(harmonic_oscillator, initial, 0.1, 1.0)
    assert initial == pytest.approx([1.0, 0.0])


@pytest.mark.parametrize(
    "time_step, duration, message",
    [
        (0.0, 1.0, "Time step"),
        (-0.1, 1.0, "Time step"),
        (0.1, -1.0, "Duration"),
        (0.1, 1.05, "whole number"),
    ],
)
def test_rejects_invalid_time_settings(time_step, duration, message):
    with pytest.raises(ValueError, match=message):
        integrate(harmonic_oscillator, [1.0, 0.0], time_step, duration)


def rigid_body_without_loads(time, values):
    return rigid_body_derivatives(
        State.from_array(values), np.zeros(3), np.zeros(3), 1000.0, np.diag([1285.0, 1825.0, 2667.0])
    )


def normalize_state_quaternion(values):
    normalized = values.copy()
    normalized[QUATERNION] = normalize_quaternion(values[QUATERNION])
    return normalized


def test_rigid_body_free_fall_matches_kinematics():
    initial = State([0.0, 0.0, -1000.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    trajectory = integrate(rigid_body_without_loads, initial.to_array(), 0.01, 3.0)
    final = trajectory.states[-1]
    assert final[POSITION_NED][2] == pytest.approx(-1000.0 + 0.5 * STANDARD_GRAVITY * 3.0**2, abs=1e-9)
    assert final[VELOCITY_BODY][2] == pytest.approx(STANDARD_GRAVITY * 3.0, abs=1e-9)


def test_rigid_body_spinning_about_vertical_axis_turns_heading():
    initial = State([0.0, 0.0, -1000.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.5])
    trajectory = integrate(
        rigid_body_without_loads,
        initial.to_array(),
        0.01,
        1.0,
        post_step=normalize_state_quaternion,
    )
    final_quaternion = trajectory.states[-1][QUATERNION]
    assert np.linalg.norm(final_quaternion) == pytest.approx(1.0, abs=1e-12)
    assert euler_from_quaternion(final_quaternion).yaw == pytest.approx(0.5, abs=1e-9)
