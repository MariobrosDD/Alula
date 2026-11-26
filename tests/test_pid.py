import numpy as np
import pytest

from alula.integrators import rk4_step
from alula.pid import PID

DISTURBANCE = -0.3
REFERENCE = 1.0
TIME_STEP = 0.01


def test_proportional_term_scales_the_error():
    controller = PID(proportional_gain=2.0)
    assert controller.update(reference=10.0, measurement=7.0, time_step=0.1) == pytest.approx(6.0)


def test_integral_term_accumulates_error_over_time():
    controller = PID(proportional_gain=0.0, integral_gain=0.5)
    for _ in range(10):
        output = controller.update(reference=2.0, measurement=0.0, time_step=0.1)
    assert controller.integral == pytest.approx(2.0)
    assert output == pytest.approx(1.0)


def test_derivative_term_opposes_a_rising_measurement():
    controller = PID(proportional_gain=0.0, derivative_gain=3.0)
    assert controller.update(reference=0.0, measurement=1.0, time_step=0.1) == 0.0
    assert controller.update(reference=0.0, measurement=1.5, time_step=0.1) == pytest.approx(-15.0)


def test_reference_step_does_not_kick_the_derivative_term():
    controller = PID(proportional_gain=0.0, derivative_gain=3.0)
    controller.update(reference=0.0, measurement=5.0, time_step=0.1)
    assert controller.update(reference=100.0, measurement=5.0, time_step=0.1) == 0.0


@pytest.mark.parametrize("error, expected", [(5.0, 20.0), (-5.0, -20.0), (1.0, 10.0)])
def test_output_is_limited(error, expected):
    controller = PID(proportional_gain=10.0, output_min=-20.0, output_max=20.0)
    assert controller.update(reference=error, measurement=0.0, time_step=0.1) == expected


def test_integral_does_not_wind_up_while_saturated():
    controller = PID(proportional_gain=0.0, integral_gain=1.0, output_max=1.0)
    for _ in range(100):
        controller.update(reference=10.0, measurement=0.0, time_step=0.1)
    assert controller.integral == pytest.approx(1.0)


def test_output_leaves_saturation_as_soon_as_error_reverses():
    controller = PID(proportional_gain=0.5, integral_gain=1.0, output_min=-1.0, output_max=1.0)
    for _ in range(200):
        controller.update(reference=10.0, measurement=0.0, time_step=0.1)
    assert controller.update(reference=0.0, measurement=3.0, time_step=0.1) < 1.0


def test_integral_unwinds_while_saturated_in_the_opposite_direction():
    controller = PID(proportional_gain=100.0, integral_gain=1.0, output_max=1.0)
    controller.integral = 5.0
    controller.update(reference=0.0, measurement=0.01, time_step=0.1)
    assert controller.integral < 5.0


def test_reset_forgets_integral_and_previous_measurement():
    controller = PID(proportional_gain=1.0, integral_gain=1.0, derivative_gain=1.0)
    controller.update(reference=1.0, measurement=0.0, time_step=0.1)
    controller.update(reference=1.0, measurement=0.5, time_step=0.1)
    controller.reset()
    assert controller.integral == 0.0
    assert controller.previous_measurement is None


def test_rejects_non_positive_time_step():
    with pytest.raises(ValueError, match="Time step"):
        PID(proportional_gain=1.0).update(reference=1.0, measurement=0.0, time_step=0.0)


def test_rejects_inverted_output_limits():
    with pytest.raises(ValueError, match="output_min"):
        PID(proportional_gain=1.0, output_min=1.0, output_max=-1.0)


def run_double_integrator(controller, duration):
    state = np.array([0.0, 0.0])
    times, positions = [0.0], [0.0]
    for step in range(round(duration / TIME_STEP)):
        command = controller.update(REFERENCE, state[0], TIME_STEP)

        def plant(time, values):
            position, velocity = values
            return np.array([velocity, command + DISTURBANCE])

        state = rk4_step(plant, step * TIME_STEP, state, TIME_STEP)
        times.append((step + 1) * TIME_STEP)
        positions.append(state[0])
    return np.array(times), np.array(positions)


def test_primer_figure_17_2_proportional_only_oscillates_forever():
    times, positions = run_double_integrator(PID(proportional_gain=1.0), 40.0)
    late = positions[times > 20.0]
    assert late.max() - late.min() > 1.2
    assert late.mean() == pytest.approx(1.0 + DISTURBANCE, abs=0.1)


def test_primer_figure_17_2_derivative_damps_but_leaves_steady_error():
    times, positions = run_double_integrator(PID(proportional_gain=1.0, derivative_gain=1.4), 30.0)
    assert positions[-1] == pytest.approx(REFERENCE + DISTURBANCE, abs=0.01)
    assert np.ptp(positions[times > 20.0]) < 0.01


def test_primer_figure_17_2_integral_removes_steady_error():
    times, positions = run_double_integrator(
        PID(proportional_gain=1.0, integral_gain=0.4, derivative_gain=1.4), 60.0
    )
    assert positions[-1] == pytest.approx(REFERENCE, abs=0.01)
