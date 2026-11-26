import math
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from alula.aerodynamics import air_data_from_body_velocity
from alula.aircraft import Aircraft
from alula.controls import Controls
from alula.dynamics import QUATERNION, State, aircraft_derivatives
from alula.integrators import rk4_step, step_count
from alula.rotations import euler_from_quaternion, normalize_quaternion, quaternion_from_euler

ControlLaw = Callable[[float, State], Controls]


class Termination(Enum):
    COMPLETED = "completed"
    GROUND_CONTACT = "ground contact"
    MODEL_LIMIT = "model limit"


class GroundContact(Exception):
    pass


class FlightHistory(NamedTuple):
    time: NDArray[np.float64]
    north: NDArray[np.float64]
    east: NDArray[np.float64]
    altitude: NDArray[np.float64]
    airspeed: NDArray[np.float64]
    alpha: NDArray[np.float64]
    beta: NDArray[np.float64]
    roll: NDArray[np.float64]
    pitch: NDArray[np.float64]
    yaw: NDArray[np.float64]


@dataclass(frozen=True)
class SimulationResult:
    times: NDArray[np.float64]
    states: NDArray[np.float64]
    controls: list[Controls]
    termination: Termination
    termination_message: str = ""

    @property
    def final_time(self) -> float:
        return float(self.times[-1])

    @property
    def final_state(self) -> State:
        return State.from_array(self.states[-1])

    def history(self) -> FlightHistory:
        air_data = [air_data_from_body_velocity(state[3:6]) for state in self.states]
        attitudes = [euler_from_quaternion(state[QUATERNION]) for state in self.states]
        return FlightHistory(
            time=self.times.copy(),
            north=self.states[:, 0].copy(),
            east=self.states[:, 1].copy(),
            altitude=-self.states[:, 2],
            airspeed=np.array([data.airspeed for data in air_data]),
            alpha=np.array([data.alpha for data in air_data]),
            beta=np.array([data.beta for data in air_data]),
            roll=np.array([attitude.roll for attitude in attitudes]),
            pitch=np.array([attitude.pitch for attitude in attitudes]),
            yaw=np.array([attitude.yaw for attitude in attitudes]),
        )


def initial_state(
    altitude: float,
    airspeed: float,
    alpha: float,
    heading: float = 0.0,
    flight_path_angle: float = 0.0,
) -> State:
    if altitude < 0.0:
        raise ValueError(f"Initial altitude cannot be below sea level, got {altitude} m")
    if airspeed <= 0.0:
        raise ValueError(f"Initial airspeed must be positive, got {airspeed} m/s")
    return State(
        position_ned=[0.0, 0.0, -altitude],
        velocity_body=[airspeed * math.cos(alpha), 0.0, airspeed * math.sin(alpha)],
        quaternion=quaternion_from_euler(0.0, alpha + flight_path_angle, heading),
        angular_velocity_body=[0.0, 0.0, 0.0],
    )


def simulate(
    aircraft: Aircraft,
    start: State,
    controls: Controls | ControlLaw,
    duration: float,
    time_step: float = 0.01,
) -> SimulationResult:
    steps = step_count(time_step, duration)
    control_law = _as_control_law(controls)

    state = start.to_array()
    times = [0.0]
    states = [state]
    applied_controls: list[Controls] = []
    termination = Termination.COMPLETED
    message = ""

    for index in range(steps):
        time = index * time_step
        command = control_law(time, State.from_array(state))
        try:
            next_state = rk4_step(_derivative_with(command, aircraft), time, state, time_step)
        except GroundContact:
            termination = Termination.GROUND_CONTACT
            break
        except ValueError as error:
            termination = Termination.MODEL_LIMIT
            message = str(error)
            break

        next_state[QUATERNION] = normalize_quaternion(next_state[QUATERNION])
        state = next_state
        times.append(time + time_step)
        states.append(state)
        applied_controls.append(command)

        if State.from_array(state).altitude <= 0.0:
            termination = Termination.GROUND_CONTACT
            break

    return SimulationResult(
        times=np.array(times),
        states=np.array(states),
        controls=applied_controls,
        termination=termination,
        termination_message=message,
    )


def _as_control_law(controls: Controls | ControlLaw) -> ControlLaw:
    if isinstance(controls, Controls):
        return lambda time, state: controls
    return controls


def _derivative_with(
    command: Controls, aircraft: Aircraft
) -> Callable[[float, NDArray[np.float64]], NDArray[np.float64]]:
    def derivative(time: float, values: NDArray[np.float64]) -> NDArray[np.float64]:
        state = State.from_array(values)
        if state.altitude < 0.0:
            raise GroundContact
        return aircraft_derivatives(state, command, aircraft)

    return derivative
