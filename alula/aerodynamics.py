import math
from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from alula.aircraft import Aircraft
from alula.atmosphere import dynamic_pressure
from alula.controls import Controls


class AirData(NamedTuple):
    airspeed: float
    alpha: float
    beta: float


class AerodynamicLoads(NamedTuple):
    force_body: NDArray[np.float64]
    moment_body: NDArray[np.float64]


def air_data_from_body_velocity(velocity_body: ArrayLike) -> AirData:
    u, v, w = np.asarray(velocity_body, dtype=float)
    airspeed = math.sqrt(u**2 + v**2 + w**2)
    if airspeed == 0.0:
        raise ValueError("Angle of attack and sideslip are undefined at zero airspeed")
    return AirData(
        airspeed=airspeed,
        alpha=math.atan2(w, u),
        beta=math.asin(v / airspeed),
    )


def induced_drag_factor(aircraft: Aircraft) -> float:
    return 1.0 / (
        math.pi * aircraft.aerodynamics.oswald_efficiency * aircraft.geometry.aspect_ratio
    )


def aerodynamic_loads(
    aircraft: Aircraft,
    air_data: AirData,
    angular_velocity_body: ArrayLike,
    density: float,
    controls: Controls,
) -> AerodynamicLoads:
    coefficients = aircraft.aerodynamics
    geometry = aircraft.geometry
    airspeed, alpha, beta = air_data

    if not coefficients.alpha_min <= alpha <= coefficients.alpha_max:
        raise ValueError(
            f"Angle of attack {math.degrees(alpha):.2f} deg is outside the linear model range "
            f"[{math.degrees(coefficients.alpha_min):.1f}, "
            f"{math.degrees(coefficients.alpha_max):.1f}] deg"
        )
    if abs(beta) > coefficients.beta_max:
        raise ValueError(
            f"Sideslip {math.degrees(beta):.2f} deg exceeds the linear model limit of "
            f"{math.degrees(coefficients.beta_max):.1f} deg"
        )

    p, q, r = np.asarray(angular_velocity_body, dtype=float)
    p_hat = p * geometry.wingspan / (2.0 * airspeed)
    q_hat = q * geometry.mean_chord / (2.0 * airspeed)
    r_hat = r * geometry.wingspan / (2.0 * airspeed)

    lift_coefficient = (
        coefficients.cl_0
        + coefficients.cl_alpha * alpha
        + coefficients.cl_q * q_hat
        + coefficients.cl_elevator * controls.elevator
    )
    drag_coefficient = coefficients.cd_0 + induced_drag_factor(aircraft) * lift_coefficient**2
    side_force_coefficient = (
        coefficients.cy_beta * beta + coefficients.cy_rudder * controls.rudder
    )
    rolling_moment_coefficient = (
        coefficients.c_roll_beta * beta
        + coefficients.c_roll_p * p_hat
        + coefficients.c_roll_r * r_hat
        + coefficients.c_roll_aileron * controls.aileron
        + coefficients.c_roll_rudder * controls.rudder
    )
    pitching_moment_coefficient = (
        coefficients.cm_0
        + coefficients.cm_alpha * alpha
        + coefficients.cm_q * q_hat
        + coefficients.cm_elevator * controls.elevator
    )
    yawing_moment_coefficient = (
        coefficients.cn_beta * beta
        + coefficients.cn_p * p_hat
        + coefficients.cn_r * r_hat
        + coefficients.cn_aileron * controls.aileron
        + coefficients.cn_rudder * controls.rudder
    )

    pressure_times_area = dynamic_pressure(density, airspeed) * geometry.wing_area
    lift = pressure_times_area * lift_coefficient
    drag = pressure_times_area * drag_coefficient
    side_force = pressure_times_area * side_force_coefficient

    force_body = np.array(
        [
            lift * math.sin(alpha) - drag * math.cos(alpha),
            side_force,
            -lift * math.cos(alpha) - drag * math.sin(alpha),
        ]
    )
    moment_body = np.array(
        [
            pressure_times_area * geometry.wingspan * rolling_moment_coefficient,
            pressure_times_area * geometry.mean_chord * pitching_moment_coefficient,
            pressure_times_area * geometry.wingspan * yawing_moment_coefficient,
        ]
    )
    return AerodynamicLoads(force_body, moment_body)
