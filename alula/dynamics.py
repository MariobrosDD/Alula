from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from alula.aerodynamics import aerodynamic_loads, air_data_from_body_velocity
from alula.aircraft import Aircraft
from alula.atmosphere import STANDARD_GRAVITY, isa
from alula.controls import Controls
from alula.propulsion import thrust
from alula.rotations import body_to_ned_from_quaternion, quaternion_derivative

STATE_SIZE = 13
POSITION_NED = slice(0, 3)
VELOCITY_BODY = slice(3, 6)
QUATERNION = slice(6, 10)
ANGULAR_VELOCITY_BODY = slice(10, 13)

STATE_COMPONENT_SIZES = {
    "position_ned": 3,
    "velocity_body": 3,
    "quaternion": 4,
    "angular_velocity_body": 3,
}


@dataclass(frozen=True)
class State:
    position_ned: NDArray[np.float64]
    velocity_body: NDArray[np.float64]
    quaternion: NDArray[np.float64]
    angular_velocity_body: NDArray[np.float64]

    def __post_init__(self) -> None:
        for name, size in STATE_COMPONENT_SIZES.items():
            values = np.array(getattr(self, name), dtype=float)
            if values.shape != (size,):
                raise ValueError(f"{name} must have {size} components, got shape {values.shape}")
            object.__setattr__(self, name, values)

    @property
    def altitude(self) -> float:
        return -float(self.position_ned[2])

    def to_array(self) -> NDArray[np.float64]:
        return np.concatenate(
            [self.position_ned, self.velocity_body, self.quaternion, self.angular_velocity_body]
        )

    @classmethod
    def from_array(cls, values: ArrayLike) -> "State":
        array = np.asarray(values, dtype=float)
        if array.shape != (STATE_SIZE,):
            raise ValueError(f"A state array must have {STATE_SIZE} components, got {array.shape}")
        return cls(
            position_ned=array[POSITION_NED],
            velocity_body=array[VELOCITY_BODY],
            quaternion=array[QUATERNION],
            angular_velocity_body=array[ANGULAR_VELOCITY_BODY],
        )


class Loads(NamedTuple):
    force_body: NDArray[np.float64]
    moment_body: NDArray[np.float64]


def rigid_body_derivatives(
    state: State,
    force_body: ArrayLike,
    moment_body: ArrayLike,
    mass: float,
    inertia: ArrayLike,
    gravity: float = STANDARD_GRAVITY,
) -> NDArray[np.float64]:
    if mass <= 0.0:
        raise ValueError(f"Mass must be positive, got {mass} kg")

    inertia_matrix = np.asarray(inertia, dtype=float)
    velocity = state.velocity_body
    angular_velocity = state.angular_velocity_body
    body_to_ned = body_to_ned_from_quaternion(state.quaternion)

    gravity_body = body_to_ned.T @ np.array([0.0, 0.0, gravity])
    velocity_rate = (
        np.asarray(force_body, dtype=float) / mass
        + gravity_body
        - np.cross(angular_velocity, velocity)
    )
    angular_acceleration = np.linalg.solve(
        inertia_matrix,
        np.asarray(moment_body, dtype=float)
        - np.cross(angular_velocity, inertia_matrix @ angular_velocity),
    )
    quaternion_rate = quaternion_derivative(state.quaternion, angular_velocity)
    position_rate = body_to_ned @ velocity

    return np.concatenate([position_rate, velocity_rate, quaternion_rate, angular_acceleration])


def aircraft_loads(state: State, controls: Controls, aircraft: Aircraft) -> Loads:
    atmosphere = isa(state.altitude)
    air_data = air_data_from_body_velocity(state.velocity_body)
    aerodynamic = aerodynamic_loads(
        aircraft, air_data, state.angular_velocity_body, atmosphere.density, controls
    )
    engine_thrust = thrust(
        aircraft.propulsion, controls.throttle, air_data.airspeed, atmosphere.density
    )
    return Loads(
        force_body=aerodynamic.force_body + np.array([engine_thrust, 0.0, 0.0]),
        moment_body=aerodynamic.moment_body,
    )


def aircraft_derivatives(state: State, controls: Controls, aircraft: Aircraft) -> NDArray[np.float64]:
    loads = aircraft_loads(state, controls, aircraft)
    return rigid_body_derivatives(
        state,
        loads.force_body,
        loads.moment_body,
        aircraft.mass_properties.mass,
        aircraft.mass_properties.inertia,
    )
