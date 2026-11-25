import math
from collections.abc import Callable
from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

Derivative = Callable[[float, NDArray[np.float64]], NDArray[np.float64]]
StepFunction = Callable[[Derivative, float, NDArray[np.float64], float], NDArray[np.float64]]
PostStep = Callable[[NDArray[np.float64]], NDArray[np.float64]]

STEP_COUNT_TOLERANCE = 1e-9


class Trajectory(NamedTuple):
    times: NDArray[np.float64]
    states: NDArray[np.float64]


def euler_step(
    derivative: Derivative, time: float, state: NDArray[np.float64], time_step: float
) -> NDArray[np.float64]:
    return state + time_step * derivative(time, state)


def rk4_step(
    derivative: Derivative, time: float, state: NDArray[np.float64], time_step: float
) -> NDArray[np.float64]:
    half_step = 0.5 * time_step
    k1 = derivative(time, state)
    k2 = derivative(time + half_step, state + half_step * k1)
    k3 = derivative(time + half_step, state + half_step * k2)
    k4 = derivative(time + time_step, state + time_step * k3)
    return state + time_step / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def integrate(
    derivative: Derivative,
    initial_state: ArrayLike,
    time_step: float,
    duration: float,
    step: StepFunction = rk4_step,
    post_step: PostStep | None = None,
    initial_time: float = 0.0,
) -> Trajectory:
    steps = step_count(time_step, duration)
    state = np.array(initial_state, dtype=float)
    states = np.empty((steps + 1, state.size))
    states[0] = state

    for index in range(1, steps + 1):
        time = initial_time + (index - 1) * time_step
        state = step(derivative, time, state, time_step)
        if post_step is not None:
            state = post_step(state)
        states[index] = state

    times = initial_time + time_step * np.arange(steps + 1)
    return Trajectory(times, states)


def step_count(time_step: float, duration: float) -> int:
    if time_step <= 0.0:
        raise ValueError(f"Time step must be positive, got {time_step} s")
    if duration < 0.0:
        raise ValueError(f"Duration cannot be negative, got {duration} s")
    steps = round(duration / time_step)
    if not math.isclose(
        steps * time_step, duration, rel_tol=STEP_COUNT_TOLERANCE, abs_tol=STEP_COUNT_TOLERANCE
    ):
        raise ValueError(
            f"Duration {duration} s is not a whole number of time steps of {time_step} s"
        )
    return steps
