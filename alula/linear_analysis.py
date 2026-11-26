import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from alula.aircraft import Aircraft
from alula.controls import Controls
from alula.dynamics import ANGULAR_VELOCITY_BODY, VELOCITY_BODY, State, aircraft_derivatives
from alula.rotations import euler_angle_rates, euler_from_quaternion, quaternion_from_euler
from alula.trim import TrimResult

LONGITUDINAL_STATES = ("u", "w", "q", "theta")
LATERAL_STATES = ("v", "p", "r", "phi")
STATE_NAMES = LONGITUDINAL_STATES + LATERAL_STATES
INPUT_NAMES = ("elevator", "throttle", "aileron", "rudder")

LONGITUDINAL = slice(0, 4)
LATERAL = slice(4, 8)

PERTURBATION = 1e-6
OSCILLATION_TOLERANCE = 1e-9


@dataclass(frozen=True)
class LinearModel:
    state_matrix: NDArray[np.float64]
    input_matrix: NDArray[np.float64]

    @property
    def longitudinal_matrix(self) -> NDArray[np.float64]:
        return self.state_matrix[LONGITUDINAL, LONGITUDINAL]

    @property
    def lateral_matrix(self) -> NDArray[np.float64]:
        return self.state_matrix[LATERAL, LATERAL]


@dataclass(frozen=True)
class Mode:
    eigenvalue: complex

    @property
    def is_stable(self) -> bool:
        return self.eigenvalue.real < 0.0

    @property
    def is_oscillatory(self) -> bool:
        return abs(self.eigenvalue.imag) > OSCILLATION_TOLERANCE

    @property
    def natural_frequency(self) -> float:
        return abs(self.eigenvalue)

    @property
    def damping_ratio(self) -> float:
        return -self.eigenvalue.real / abs(self.eigenvalue)

    @property
    def period(self) -> float:
        if not self.is_oscillatory:
            return math.inf
        return 2.0 * math.pi / abs(self.eigenvalue.imag)

    @property
    def time_constant(self) -> float:
        if self.eigenvalue.real == 0.0:
            return math.inf
        return 1.0 / abs(self.eigenvalue.real)


@dataclass(frozen=True)
class AircraftModes:
    short_period: Mode
    phugoid: Mode
    roll_subsidence: Mode
    dutch_roll: Mode
    spiral: Mode


def linearize(aircraft: Aircraft, trimmed: TrimResult) -> LinearModel:
    reference = trimmed.state
    attitude = euler_from_quaternion(reference.quaternion)
    u, v, w = reference.velocity_body
    p, q, r = reference.angular_velocity_body
    reference_states = np.array([u, w, q, attitude.pitch, v, p, r, attitude.roll])
    controls = trimmed.controls
    reference_inputs = np.array(
        [controls.elevator, controls.throttle, controls.aileron, controls.rudder]
    )

    def rates(states: NDArray[np.float64], inputs: NDArray[np.float64]) -> NDArray[np.float64]:
        u, w, q, pitch, v, p, r, roll = states
        elevator, throttle, aileron, rudder = inputs
        state = State(
            position_ned=reference.position_ned,
            velocity_body=[u, v, w],
            quaternion=quaternion_from_euler(roll, pitch, attitude.yaw),
            angular_velocity_body=[p, q, r],
        )
        derivatives = aircraft_derivatives(
            state,
            Controls(elevator=elevator, aileron=aileron, rudder=rudder, throttle=throttle),
            aircraft,
        )
        u_rate, v_rate, w_rate = derivatives[VELOCITY_BODY]
        p_rate, q_rate, r_rate = derivatives[ANGULAR_VELOCITY_BODY]
        roll_rate, pitch_rate, _ = euler_angle_rates(roll, pitch, [p, q, r])
        return np.array([u_rate, w_rate, q_rate, pitch_rate, v_rate, p_rate, r_rate, roll_rate])

    state_matrix = _central_jacobian(lambda states: rates(states, reference_inputs), reference_states)
    input_matrix = _central_jacobian(lambda inputs: rates(reference_states, inputs), reference_inputs)
    return LinearModel(state_matrix, input_matrix)


def aircraft_modes(model: LinearModel) -> AircraftModes:
    longitudinal = _upper_half_plane(np.linalg.eigvals(model.longitudinal_matrix))
    if len(longitudinal) != 2 or not all(mode.is_oscillatory for mode in longitudinal):
        raise ValueError("Expected two oscillatory longitudinal modes (short period and phugoid)")
    phugoid, short_period = sorted(longitudinal, key=lambda mode: mode.natural_frequency)

    lateral = _upper_half_plane(np.linalg.eigvals(model.lateral_matrix))
    oscillatory = [mode for mode in lateral if mode.is_oscillatory]
    aperiodic = [mode for mode in lateral if not mode.is_oscillatory]
    if len(oscillatory) != 1 or len(aperiodic) != 2:
        raise ValueError("Expected one oscillatory and two aperiodic lateral modes")
    spiral, roll_subsidence = sorted(aperiodic, key=lambda mode: abs(mode.eigenvalue.real))

    return AircraftModes(
        short_period=short_period,
        phugoid=phugoid,
        roll_subsidence=roll_subsidence,
        dutch_roll=oscillatory[0],
        spiral=spiral,
    )


def _upper_half_plane(eigenvalues: NDArray[np.complex128]) -> list[Mode]:
    return [
        Mode(complex(eigenvalue))
        for eigenvalue in eigenvalues
        if eigenvalue.imag > OSCILLATION_TOLERANCE or abs(eigenvalue.imag) <= OSCILLATION_TOLERANCE
    ]


def _central_jacobian(
    function: Callable[[NDArray[np.float64]], NDArray[np.float64]], point: NDArray[np.float64]
) -> NDArray[np.float64]:
    columns = []
    for index in range(point.size):
        step = np.zeros(point.size)
        step[index] = PERTURBATION
        columns.append((function(point + step) - function(point - step)) / (2.0 * PERTURBATION))
    return np.column_stack(columns)
