from alula.aircraft import PropulsionData
from alula.atmosphere import SEA_LEVEL_DENSITY

GAGG_FERRAR_SLOPE = 1.132
GAGG_FERRAR_OFFSET = 0.132


def altitude_power_fraction(density: float) -> float:
    fraction = GAGG_FERRAR_SLOPE * density / SEA_LEVEL_DENSITY - GAGG_FERRAR_OFFSET
    if fraction <= 0.0:
        raise ValueError(
            f"Air density {density:.4f} kg/m^3 is too low for a naturally aspirated piston engine"
        )
    return fraction


def shaft_power(propulsion: PropulsionData, throttle: float, density: float) -> float:
    _check_throttle(throttle)
    return throttle * propulsion.max_shaft_power * altitude_power_fraction(density)


def thrust(propulsion: PropulsionData, throttle: float, airspeed: float, density: float) -> float:
    if airspeed < 0.0:
        raise ValueError(f"Airspeed cannot be negative, got {airspeed} m/s")
    _check_throttle(throttle)

    power_fraction = altitude_power_fraction(density)
    static_thrust = throttle * propulsion.static_thrust * power_fraction
    if airspeed == 0.0:
        return static_thrust

    propeller_thrust = (
        propulsion.propeller_efficiency * shaft_power(propulsion, throttle, density) / airspeed
    )
    return min(propeller_thrust, static_thrust)


def _check_throttle(throttle: float) -> None:
    if not 0.0 <= throttle <= 1.0:
        raise ValueError(f"Throttle must be between 0 and 1, got {throttle}")
