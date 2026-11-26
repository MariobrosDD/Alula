import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from alula.aircraft import Aircraft
from alula.atmosphere import STANDARD_GRAVITY, dynamic_pressure, isa
from alula.controls import Controls
from alula.dynamics import ANGULAR_VELOCITY_BODY, VELOCITY_BODY, State, aircraft_derivatives
from alula.simulation import initial_state

RESIDUAL_TOLERANCE = 1e-9
MAX_ITERATIONS = 30
JACOBIAN_STEP = 1e-7
INITIAL_THROTTLE_GUESS = 0.5

Residuals = Callable[[NDArray[np.float64]], NDArray[np.float64]]


@dataclass(frozen=True)
class TrimResult:
    state: State
    controls: Controls
    alpha: float
    residual: float
    iterations: int


def trim(
    aircraft: Aircraft,
    altitude: float,
    airspeed: float,
    heading: float = 0.0,
    flight_path_angle: float = 0.0,
) -> TrimResult:
    def evaluate(alpha: float, elevator: float, throttle: float) -> NDArray[np.float64]:
        state = initial_state(altitude, airspeed, alpha, heading, flight_path_angle)
        derivatives = aircraft_derivatives(
            state, Controls(elevator=elevator, throttle=throttle), aircraft
        )
        u_rate, _, w_rate = derivatives[VELOCITY_BODY]
        _, pitch_acceleration, _ = derivatives[ANGULAR_VELOCITY_BODY]
        return np.array([u_rate, w_rate, pitch_acceleration])

    def residuals(unknowns: NDArray[np.float64]) -> NDArray[np.float64]:
        alpha, elevator, throttle = unknowns
        feasible_throttle = min(max(throttle, 0.0), 1.0)
        value = evaluate(alpha, elevator, feasible_throttle)
        if throttle != feasible_throttle:
            throttle_sensitivity = evaluate(alpha, elevator, 1.0) - evaluate(alpha, elevator, 0.0)
            value = value + (throttle - feasible_throttle) * throttle_sensitivity
        return value

    try:
        solution, residual, iterations = _newton(residuals, _initial_guess(aircraft, altitude, airspeed))
    except ValueError as error:
        raise ValueError(
            f"Cannot trim at {airspeed:.1f} m/s and {altitude:.0f} m: {error}"
        ) from error

    alpha, elevator, throttle = solution
    if not 0.0 <= throttle <= 1.0:
        raise ValueError(
            f"Cannot trim at {airspeed:.1f} m/s and {altitude:.0f} m: "
            f"it would need {throttle:.0%} throttle"
        )

    return TrimResult(
        state=initial_state(altitude, airspeed, alpha, heading, flight_path_angle),
        controls=Controls(elevator=elevator, throttle=throttle),
        alpha=alpha,
        residual=residual,
        iterations=iterations,
    )


def _initial_guess(aircraft: Aircraft, altitude: float, airspeed: float) -> NDArray[np.float64]:
    coefficients = aircraft.aerodynamics
    weight = aircraft.mass_properties.mass * STANDARD_GRAVITY
    pressure_times_area = dynamic_pressure(isa(altitude).density, airspeed) * aircraft.geometry.wing_area
    lift_coefficient = weight / pressure_times_area
    alpha = (lift_coefficient - coefficients.cl_0) / coefficients.cl_alpha
    elevator = -(coefficients.cm_0 + coefficients.cm_alpha * alpha) / coefficients.cm_elevator
    return np.array([alpha, elevator, INITIAL_THROTTLE_GUESS])


def _newton(
    residuals: Residuals, guess: NDArray[np.float64]
) -> tuple[NDArray[np.float64], float, int]:
    unknowns = guess.astype(float)
    for iteration in range(1, MAX_ITERATIONS + 1):
        error = residuals(unknowns)
        try:
            correction = np.linalg.solve(_jacobian(residuals, unknowns, error), -error)
        except np.linalg.LinAlgError as singular:
            raise ValueError("the trim equations became singular") from singular
        unknowns = unknowns + correction
        residual = float(np.linalg.norm(residuals(unknowns)))
        if residual < RESIDUAL_TOLERANCE:
            return unknowns, residual, iteration
    raise ValueError(f"Newton's method did not converge in {MAX_ITERATIONS} iterations")


def _jacobian(
    residuals: Residuals, unknowns: NDArray[np.float64], error: NDArray[np.float64]
) -> NDArray[np.float64]:
    jacobian = np.empty((error.size, unknowns.size))
    for column in range(unknowns.size):
        nudged = unknowns.copy()
        nudged[column] += JACOBIAN_STEP
        jacobian[:, column] = (residuals(nudged) - error) / JACOBIAN_STEP
    return jacobian
